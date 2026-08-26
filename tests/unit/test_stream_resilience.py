"""Unit tests for streaming-interruption resilience.

A streaming model read is a blocking socket read; if the connection dies silently
(e.g. the laptop sleeps), it would park the worker thread forever — freezing the
whole (single-worker) UI. The agent guards the stream with a per-chunk idle
timeout and re-runs the turn on a dead/transient connection with backoff, instead
of hanging. These tests exercise that logic with fake models (no network/LLM).
"""

import asyncio
import threading
import time

import pytest

from mnemoai.client.agent import stream_policy
from mnemoai.client.agent.agent import (
    LangGraphAgent,
    _StreamIdleTimeout,
)

# A provider "overloaded" (529) failure — the transient error this app sees most
# under load, and the one the auxiliary calls used to give up on immediately.
_OVERLOADED = (
    "Error code: 529 - {'type': 'error', 'error': "
    "{'type': 'overloaded_error', 'message': 'Overloaded'}}"
)


def _overloaded(retry_after=None, message=_OVERLOADED):
    """An overload exception, optionally carrying an httpx-shaped retry-after."""
    exc = Exception(message)
    if retry_after is not None:
        exc.response = type("R", (), {"headers": {"retry-after": retry_after}})()
    return exc


def _agent(idle=0.2):
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._stream_idle_timeout = idle
    return a


class TestTransientNetworkClassifier:
    def test_matches_common_socket_errors(self):
        for msg in [
            "Connection reset by peer",
            "[Errno 32] Broken pipe",
            "ECONNRESET",
            "Read timed out.",
            "Server disconnected without sending a response",
            "503 Service Unavailable",
            "overloaded_error",
            "Request timed out",
        ]:
            assert LangGraphAgent._is_transient_network_error(Exception(msg)), msg

    def test_matches_500_api_error(self):
        # A streamed Anthropic 500 (the reported bug): its str() carries these
        # phrasings — must be retriable so the turn retries the STREAM (abortable)
        # rather than falling to a blocking, uncancellable non-streaming invoke.
        for msg in [
            "{'type': 'error', 'error': {'type': 'api_error', "
            "'message': 'Internal server error'}}",
            "Internal server error",
            "api_error",
        ]:
            assert LangGraphAgent._is_transient_network_error(Exception(msg)), msg

    def test_does_not_match_deterministic_errors(self):
        for msg in [
            "invalid_request_error: bad parameter",
            "401 Unauthorized",
            "model not found",
            "KeyError: 'foo'",
        ]:
            assert not LangGraphAgent._is_transient_network_error(Exception(msg)), msg


class _FakeModel:
    """A model whose .stream() yields queued chunks, optionally stalling."""

    def __init__(self, chunks, stall_before=None, stall_seconds=1.0):
        self._chunks = chunks
        self._stall_before = stall_before  # index to sleep before
        self._stall_seconds = stall_seconds

    def stream(self, messages, config=None):
        for i, c in enumerate(self._chunks):
            if self._stall_before is not None and i == self._stall_before:
                time.sleep(self._stall_seconds)
            yield c


class TestIdleTimeoutIterator:
    def test_yields_all_chunks_when_no_stall(self):
        a = _agent(idle=1.0)
        out = list(a._iter_stream_with_idle_timeout(_FakeModel(["a", "b", "c"]), [], {}))
        assert out == ["a", "b", "c"]

    def test_raises_on_stall(self):
        # Stalls 1s before the 2nd chunk; idle timeout is 0.2s → raises.
        a = _agent(idle=0.2)
        model = _FakeModel(["a", "b"], stall_before=1, stall_seconds=1.0)
        it = a._iter_stream_with_idle_timeout(model, [], {})
        assert next(it) == "a"  # first chunk arrives
        with pytest.raises(_StreamIdleTimeout):
            next(it)  # second never arrives in time

    def test_disabled_passes_through_directly(self):
        # idle=0 disables the watchdog: iterate the stream directly (no thread).
        a = _agent(idle=0)
        out = list(a._iter_stream_with_idle_timeout(_FakeModel(["x", "y"]), [], {}))
        assert out == ["x", "y"]

    def test_reader_exception_propagates(self):
        class _BoomModel:
            def stream(self, messages, config=None):
                yield "ok"
                raise ValueError("mid-stream boom")

        a = _agent(idle=1.0)
        it = a._iter_stream_with_idle_timeout(_BoomModel(), [], {})
        assert next(it) == "ok"
        with pytest.raises(ValueError, match="boom"):
            next(it)


class TestCooperativeCancel:
    """Esc/Ctrl+C must abort blocking stream/backoff waits IMMEDIATELY.

    The async KeyboardInterrupt injected into the worker thread can't preempt a
    C-level blocking wait (a `queue.get(timeout=120)` idle wait or a `time.sleep`
    backoff), so a stalled-stream cancel would otherwise stall for the full
    window. `request_cancel` sets a cooperative event the waits poll/`.wait()` on.
    """

    def _agent(self, idle=120.0):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._stream_idle_timeout = idle
        a._cancel_event = threading.Event()
        a._CANCEL_POLL_SECONDS = 0.05
        return a

    def test_request_cancel_sets_event(self):
        a = self._agent()
        assert a._cancelled() is False
        a.request_cancel()
        assert a._cancelled() is True

    def test_backoff_wakes_immediately_on_cancel(self):
        a = self._agent()
        t0 = time.time()
        threading.Timer(0.2, a.request_cancel).start()
        cancelled = a._sleep_or_cancel(30)  # would otherwise block 30s
        assert cancelled is True
        assert time.time() - t0 < 2.0

    def test_backoff_returns_false_when_delay_elapses(self):
        a = self._agent()
        assert a._sleep_or_cancel(0.05) is False  # no cancel → full (tiny) wait

    def test_idle_wait_raises_keyboardinterrupt_on_cancel(self):
        # A never-yielding stream (dropped connection): cancel must break the
        # idle wait with KeyboardInterrupt long before the 120s idle timeout.
        a = self._agent(idle=120.0)
        model = _FakeModel(["a"], stall_before=0, stall_seconds=300)
        threading.Timer(0.2, a.request_cancel).start()
        t0 = time.time()
        with pytest.raises(KeyboardInterrupt):
            list(a._iter_stream_with_idle_timeout(model, [], {}))
        assert time.time() - t0 < 2.0

    def test_sleep_or_cancel_no_event_falls_back(self):
        # Bare object with no cancel event still sleeps (short) and reports False.
        a = LangGraphAgent.__new__(LangGraphAgent)
        assert a._sleep_or_cancel(0.01) is False


class TestNetworkRetryDelay:
    def _agent_with_cfg(self, monkeypatch, delay=1.0, backoff=2.0):
        import mnemoai.client.agent.agent as mod

        monkeypatch.setattr(
            mod.config, "get",
            lambda k, d=None: {"RETRY_DELAY": delay, "RETRY_BACKOFF": backoff}
            if k == "LLM" else (d or {}),
        )
        return _agent()

    def test_exponential_backoff(self, monkeypatch):
        # Jittered: each delay is the exponential base plus up to RETRY_JITTER of it.
        a = self._agent_with_cfg(monkeypatch, delay=1.0, backoff=2.0)
        for attempt, base in ((0, 1.0), (1, 2.0), (2, 4.0)):
            d = a._network_retry_delay(attempt)
            assert base <= d <= base * (1 + stream_policy.RETRY_JITTER)

    def test_capped_at_30s(self, monkeypatch):
        a = self._agent_with_cfg(monkeypatch, delay=10.0, backoff=10.0)
        d = a._network_retry_delay(5)  # min(10*10^5, 30) + jitter
        assert 30.0 <= d <= 30.0 * (1 + stream_policy.RETRY_JITTER)

    def test_pure_helper_is_deterministic_without_jitter(self):
        # The pure function stays exact by default — jitter is opt-in, so callers
        # that need reproducible math (and these tests) aren't forced into ranges.
        assert stream_policy.network_retry_delay(0, 1.0, 2.0) == 1.0
        assert stream_policy.network_retry_delay(1, 1.0, 2.0) == 2.0
        assert stream_policy.network_retry_delay(2, 1.0, 2.0) == 4.0
        assert stream_policy.network_retry_delay(5, 10.0, 10.0) == 30.0

    def test_jitter_is_bounded_and_applied(self):
        # rand injected: 1.0 → the full jitter fraction, 0.0 → none.
        assert stream_policy.network_retry_delay(
            0, 4.0, 2.0, jitter=0.25, rand=lambda: 1.0
        ) == 5.0
        assert stream_policy.network_retry_delay(
            0, 4.0, 2.0, jitter=0.25, rand=lambda: 0.0
        ) == 4.0

    def test_prefers_provider_retry_after(self, monkeypatch):
        a = self._agent_with_cfg(monkeypatch, delay=1.0, backoff=2.0)
        assert a._transient_retry_delay(_overloaded(retry_after="7"), 0) == 7.0
        # No header → back to the jittered exponential.
        assert a._transient_retry_delay(Exception("overloaded_error"), 0) >= 1.0


class TestStreamResponseRetries:
    """_stream_response re-runs the turn on an idle/transient stream failure with
    backoff, and gives up (re-raises) after the attempt budget."""

    def _agent(self, retries):
        from langchain_core.messages import AIMessage

        a = LangGraphAgent.__new__(LangGraphAgent)
        a._empty_response_retries = retries
        a._stream_idle_timeout = 0
        a.model_with_tools = object()
        a._start_spinner = lambda *x, **k: None
        a._AIMessage = AIMessage
        return a

    def test_retries_then_succeeds(self, monkeypatch):
        from langchain_core.messages import AIMessage

        import mnemoai.client.agent.agent as mod

        monkeypatch.setattr(mod.time, "sleep", lambda s: None)  # no real backoff
        monkeypatch.setattr(mod.config, "get", lambda k, d=None: d or {})

        a = self._agent(retries=2)
        calls = {"n": 0}

        def _once(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _StreamIdleTimeout("no data for 120s")
            return AIMessage(content="recovered"), False

        a._stream_once = _once
        a._is_empty_response = lambda r: not getattr(r, "content", "")
        resp, _ = a._stream_response([], {})
        assert resp.content == "recovered"
        assert calls["n"] == 2  # failed once, retried, succeeded

    def test_gives_up_after_budget_and_reraises(self, monkeypatch):
        import mnemoai.client.agent.agent as mod

        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(mod.config, "get", lambda k, d=None: d or {})

        a = self._agent(retries=2)  # 1 + 2 = 3 attempts

        def _always_drop(*args, **kwargs):
            raise ConnectionError("Connection reset by peer")

        a._stream_once = _always_drop
        with pytest.raises(ConnectionError):
            a._stream_response([], {})

    def test_context_overflow_not_retried(self, monkeypatch):
        import mnemoai.client.agent.agent as mod
        from mnemoai.client.agent.agent import _ContextOverflow

        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(mod.config, "get", lambda k, d=None: d or {})

        a = self._agent(retries=3)
        calls = {"n": 0}

        def _overflow(*args, **kwargs):
            calls["n"] += 1
            raise _ContextOverflow("prompt is too long")

        a._stream_once = _overflow
        with pytest.raises(_ContextOverflow):
            a._stream_response([], {})
        assert calls["n"] == 1  # raised immediately, no retry


class TestNoBlockingNonStreamingFallback:
    """A mid-stream error must RE-RAISE from _stream_once (so the retry wrapper
    handles it, abortably) — NOT be swallowed by a blocking active_model.invoke()
    that can't be cancelled. That blocking fallback was the wedge behind a 500
    api_error freezing the turn and cancel not working.
    """

    def _agent(self):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._stream_idle_timeout = 0
        a.verbose = False
        a.callbacks = []
        a.styled_turn_view = False
        a.reasoning_sink = None
        a._code_formatter = None
        a._stop_spinner = lambda: None
        a._start_spinner = lambda *x, **k: None
        return a

    def test_stream_error_reraises_and_never_calls_invoke(self):
        # A model whose .stream() raises a 500 api_error, and whose .invoke()
        # explodes if ever called (it must NOT be — no blocking fallback).
        class _Model:
            invoked = False

            def stream(self, messages, config=None):
                raise RuntimeError(
                    "{'type': 'error', 'error': {'type': 'api_error', "
                    "'message': 'Internal server error'}}"
                )
                yield  # pragma: no cover

            def invoke(self, messages, config=None):
                _Model.invoked = True
                raise AssertionError("blocking non-streaming fallback must not run")

        a = self._agent()
        with pytest.raises(RuntimeError, match="api_error"):
            a._stream_once(_Model(), [], {})
        assert _Model.invoked is False

    def test_transient_stream_error_is_retried_by_wrapper(self, monkeypatch):
        # End-to-end: a 500 api_error from _stream_once is now transient, so
        # _stream_response retries it (previously it was swallowed → no retry).
        from langchain_core.messages import AIMessage

        import mnemoai.client.agent.agent as mod

        monkeypatch.setattr(mod.config, "get", lambda k, d=None: d or {})
        a = LangGraphAgent.__new__(LangGraphAgent)
        a._empty_response_retries = 2
        a._stream_idle_timeout = 0
        a.model_with_tools = object()
        a._start_spinner = lambda *x, **k: None
        a._sleep_or_cancel = lambda delay: False  # no real backoff, not cancelled
        a._is_empty_response = lambda r: not getattr(r, "content", "")
        calls = {"n": 0}

        def _once(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("api_error: Internal server error")
            return AIMessage(content="recovered"), False

        a._stream_once = _once
        resp, _ = a._stream_response([], {})
        assert resp.content == "recovered"
        assert calls["n"] == 2  # 500 retried, then succeeded


class TestAuxAttempts:
    """An auxiliary call has a working fallback, so its attempt budget is capped
    below the streamed turn's — a quick second chance, never a long stall."""

    def test_clamps_to_the_cap(self):
        assert stream_policy.aux_attempts(10) == stream_policy.AUX_RETRY_ATTEMPTS
        assert stream_policy.aux_attempts(5) == stream_policy.AUX_RETRY_ATTEMPTS

    def test_respects_a_lower_config(self):
        assert stream_policy.aux_attempts(2) == 2

    def test_never_below_one_attempt(self):
        # 0/negative retries must still make the call itself.
        assert stream_policy.aux_attempts(0) == 1
        assert stream_policy.aux_attempts(-3) == 1

    def test_junk_config_falls_back_to_the_cap(self):
        assert stream_policy.aux_attempts(None) == stream_policy.AUX_RETRY_ATTEMPTS
        assert stream_policy.aux_attempts("many") == stream_policy.AUX_RETRY_ATTEMPTS


class TestRetryAfterHeader:
    """When the provider says how long to wait, that beats our guess."""

    def test_reads_httpx_style_response_headers(self):
        assert stream_policy.retry_after_seconds(_overloaded(retry_after="12")) == 12.0

    def test_reads_botocore_style_response_dict(self):
        exc = Exception(_OVERLOADED)
        exc.response = {
            "ResponseMetadata": {"HTTPHeaders": {"retry-after": "3.5"}}
        }
        assert stream_policy.retry_after_seconds(exc) == 3.5

    def test_case_insensitive_for_plain_dicts(self):
        exc = Exception(_OVERLOADED)
        exc.response = type("R", (), {"headers": {"Retry-After": "4"}})()
        assert stream_policy.retry_after_seconds(exc) == 4.0

    def test_absent_or_unusable_values_yield_none(self):
        assert stream_policy.retry_after_seconds(Exception("boom")) is None
        assert stream_policy.retry_after_seconds(_overloaded(retry_after="0")) is None
        # HTTP-date form is legal but unusable here — never guess across a skew.
        assert stream_policy.retry_after_seconds(
            _overloaded(retry_after="Wed, 21 Oct 2026 07:28:00 GMT")
        ) is None

    def test_capped(self):
        assert stream_policy.retry_after_seconds(_overloaded(retry_after="9999")) == 60.0

    def test_transient_retry_delay_prefers_it(self):
        assert stream_policy.transient_retry_delay(
            _overloaded(retry_after="9"), 0, 1.0, 2.0
        ) == 9.0
        d = stream_policy.transient_retry_delay(_overloaded(), 1, 1.0, 2.0)
        assert 2.0 <= d <= 2.0 * (1 + stream_policy.RETRY_JITTER)


class TestCallWithTransientRetry:
    """The driver the non-streamed auxiliary calls use."""

    def test_retries_a_transient_failure_then_succeeds(self):
        calls = {"n": 0}

        def _call():
            calls["n"] += 1
            if calls["n"] == 1:
                raise _overloaded()
            return "ok"

        out = stream_policy.call_with_transient_retry(
            _call, attempts=3, base=0.0, factor=2.0
        )
        assert out == "ok"
        assert calls["n"] == 2

    def test_reraises_a_deterministic_error_immediately(self):
        calls = {"n": 0}

        def _call():
            calls["n"] += 1
            raise ValueError("invalid_request_error: bad parameter")

        with pytest.raises(ValueError):
            stream_policy.call_with_transient_retry(
                _call, attempts=3, base=0.0, factor=2.0
            )
        assert calls["n"] == 1  # not retried

    def test_reraises_the_last_transient_so_the_caller_can_fall_back(self):
        calls = {"n": 0}

        def _call():
            calls["n"] += 1
            raise _overloaded()

        with pytest.raises(Exception, match="529"):
            stream_policy.call_with_transient_retry(
                _call, attempts=3, base=0.0, factor=2.0
            )
        assert calls["n"] == 3  # whole budget spent, then the fallback path runs

    def test_reports_every_wait(self):
        seen = []

        def _call():
            raise _overloaded()

        with pytest.raises(Exception):
            stream_policy.call_with_transient_retry(
                _call, attempts=3, base=0.0, factor=2.0,
                on_retry=lambda e, d, n, t: seen.append((n, t)),
            )
        assert seen == [(1, 3), (2, 3)]  # a wait per retry, none after the last

    def test_cancel_during_backoff_aborts(self):
        event = threading.Event()
        event.set()  # already cancelled

        def _call():
            raise _overloaded()

        with pytest.raises(KeyboardInterrupt):
            stream_policy.call_with_transient_retry(
                _call, attempts=3, base=5.0, factor=2.0, cancel_event=event
            )

    def test_async_twin_retries_then_succeeds(self):
        calls = {"n": 0}

        async def _call():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _overloaded()
            return "ok"

        out = asyncio.run(
            stream_policy.acall_with_transient_retry(
                _call, attempts=3, base=0.0, factor=2.0
            )
        )
        assert out == "ok"
        assert calls["n"] == 3

    def test_async_twin_reraises_a_deterministic_error(self):
        async def _call():
            raise ValueError("model not found")

        with pytest.raises(ValueError):
            asyncio.run(
                stream_policy.acall_with_transient_retry(
                    _call, attempts=3, base=0.0, factor=2.0
                )
            )


class TestAuxiliaryCallsRetryOverload:
    """The auxiliary LLM calls each have a graceful fallback, which is why they
    used to take it on the FIRST 529 — dropping routing / the decomposition /
    a slice of the compaction summary while the streamed turn recovered beside
    them. Each must retry the overload first.
    """

    @staticmethod
    def _no_backoff(monkeypatch, mod):
        monkeypatch.setattr(
            mod.config, "get",
            lambda k, d=None: {
                "RETRY_DELAY": 0.0, "RETRY_BACKOFF": 1.0, "MAX_RETRIES": 5,
            } if k == "LLM" else (d or {}),
        )

    class _FlakyModel:
        """Fails with an overload ``fail_times`` times, then answers ``content``."""

        def __init__(self, content, fail_times=1):
            self.content = content
            self.fail_times = fail_times
            self.calls = 0

        def invoke(self, messages, config=None):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise _overloaded()
            return type("R", (), {"content": self.content})()

    def test_router_classification_retries(self, monkeypatch):
        import mnemoai.client.agent.router as mod

        self._no_backoff(monkeypatch, mod)
        router = mod.QueryRouter.__new__(mod.QueryRouter)
        router._valid_routes = set(mod.ROUTE_TOOLS.keys())
        router.usage = None
        router.usage_model_name = ""
        model = self._FlakyModel("code")

        assert router._classify_with(model, []) == "code"
        assert model.calls == 2  # 529, retried, classified

    def test_router_falls_back_to_full_once_the_budget_is_spent(self, monkeypatch):
        import mnemoai.client.agent.router as mod

        self._no_backoff(monkeypatch, mod)
        router = mod.QueryRouter.__new__(mod.QueryRouter)
        router._valid_routes = set(mod.ROUTE_TOOLS.keys())
        router.usage = None
        router.usage_model_name = ""
        router.model = self._FlakyModel("code", fail_times=99)
        monkeypatch.setattr(mod, "without_reasoning", lambda m: None)
        monkeypatch.setattr(mod, "disable_reasoning", lambda m: None)
        monkeypatch.setattr(mod, "restore_reasoning", lambda m, s: None)

        assert router.classify("do a thing", conversation_context="ctx") == "full"
        assert router.model.calls == stream_policy.AUX_RETRY_ATTEMPTS

    def test_agent_wires_its_cancel_event_into_the_router(self):
        import mnemoai.client.agent.router as mod

        class _StubModel:
            def bind_tools(self, tools):
                return self

        router = mod.QueryRouter(_StubModel())
        agent = LangGraphAgent(model=_StubModel(), tools=[], router=router)
        assert router.cancel_event is agent._cancel_event

    def test_router_backoff_is_cancellable(self, monkeypatch):
        # Classification runs inline in the turn, so Esc during its backoff must
        # abort instead of holding the worker thread for the full delay.
        import mnemoai.client.agent.router as mod

        monkeypatch.setattr(
            mod.config, "get",
            lambda k, d=None: {"RETRY_DELAY": 30.0, "RETRY_BACKOFF": 2.0}
            if k == "LLM" else (d or {}),
        )
        router = mod.QueryRouter.__new__(mod.QueryRouter)
        router.usage = None
        router.usage_model_name = ""
        router.cancel_event = threading.Event()
        router.cancel_event.set()

        with pytest.raises(KeyboardInterrupt):
            router._invoke_with_retry(self._FlakyModel("code", fail_times=99), [])

    def test_decomposition_retries(self, monkeypatch):
        import mnemoai.client.agent.agent as mod

        self._no_backoff(monkeypatch, mod)
        a = LangGraphAgent.__new__(LangGraphAgent)
        model = self._FlakyModel('[{"description": "step one", "category": "code"}]')
        a._non_reasoning = lambda: model

        subtasks = a._decompose_task("q", "orchestrator prompt", {"code", "full"})
        assert model.calls == 2  # 529, retried, decomposed
        assert [s["description"] for s in subtasks] == ["step one"]

    def test_decomposition_falls_back_to_one_subtask_after_the_budget(
        self, monkeypatch
    ):
        import mnemoai.client.agent.agent as mod

        self._no_backoff(monkeypatch, mod)
        a = LangGraphAgent.__new__(LangGraphAgent)
        model = self._FlakyModel("[]", fail_times=99)
        a._non_reasoning = lambda: model

        subtasks = a._decompose_task("q", "orchestrator prompt", {"code", "full"})
        assert model.calls == stream_policy.AUX_RETRY_ATTEMPTS
        assert subtasks == [{"description": "q", "category": "full"}]

    def test_summary_batch_retries(self, monkeypatch):
        import mnemoai.client.managers.agent_conversation_manager as mod

        self._no_backoff(monkeypatch, mod)
        mgr = mod.AgentConversationManager.__new__(mod.AgentConversationManager)
        calls = {"n": 0}

        async def _call():
            calls["n"] += 1
            if calls["n"] == 1:
                raise _overloaded()
            return "summary"

        out = asyncio.run(mgr._with_transient_retry(_call, "Summary batch 1/2"))
        assert out == "summary"
        assert calls["n"] == 2
