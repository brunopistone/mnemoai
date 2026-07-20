"""Unit tests for streaming-interruption resilience.

A streaming model read is a blocking socket read; if the connection dies silently
(e.g. the laptop sleeps), it would park the worker thread forever — freezing the
whole (single-worker) UI. The agent guards the stream with a per-chunk idle
timeout and re-runs the turn on a dead/transient connection with backoff, instead
of hanging. These tests exercise that logic with fake models (no network/LLM).
"""

import threading
import time

import pytest

from mnemoai.client.agent.agent import (
    LangGraphAgent,
    _StreamIdleTimeout,
)


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
        a = self._agent_with_cfg(monkeypatch, delay=1.0, backoff=2.0)
        assert a._network_retry_delay(0) == 1.0
        assert a._network_retry_delay(1) == 2.0
        assert a._network_retry_delay(2) == 4.0

    def test_capped_at_30s(self, monkeypatch):
        a = self._agent_with_cfg(monkeypatch, delay=10.0, backoff=10.0)
        assert a._network_retry_delay(5) == 30.0  # min(10*10^5, 30)


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
