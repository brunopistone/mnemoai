"""The MCP transport deadline must respect a tool's own timeout, and a timeout
must say so.

Two defects, both observed in a real session:

1. **A single global cap silently overrode the tool's own timeout.** Every call
   was killed at ``LLM.MCP_CALL_TIMEOUT`` (300s), so ``wait_for_task(
   timeout_seconds=1500)`` could NEVER complete — the client gave up while the
   server was still dutifully waiting. In the transcript this fired four times at
   a ~5 minute cadence while a 31 GB model was loading.

2. **The failure carried no message.** ``concurrent.futures.TimeoutError``
   stringifies to ``""``, and every log/raise site interpolates ``{e}`` — so it
   surfaced as a bare ``Tool execution error:`` with nothing after the colon,
   indistinguishable from a crash.

Deliberately NOT auto-retried: a transport timeout does not mean the tool didn't
run (the request was already delivered), and we publish no idempotency hints, so
replaying could duplicate a commit / edit / background build.

Pure logic — no server subprocess, no network.
"""

import pytest
from langchain_core.tools import ToolException

import mnemoai.client.mcp_tool_wrapper as mod
from mnemoai.client.agent import tool_formatting
from mnemoai.client.mcp_tool_wrapper import MCPCallTimeout, _call_deadline


class TestDeadlineHonorsTheToolsOwnTimeout:
    def test_long_wait_is_not_capped_at_the_default(self):
        # The exact regression: 1500 > 300 must NOT collapse to 300.
        assert _call_deadline({"timeout_seconds": 1500}) > 1500

    def test_headroom_is_added_so_the_tool_can_report_its_own_timeout(self):
        # The tool needs time to notice its deadline and ship the payload back;
        # an exact match would abort the very response we asked it to produce.
        assert _call_deadline({"timeout_seconds": 900}) == 900 + mod._TIMEOUT_HEADROOM

    def test_execute_bash_timeout_arg_is_honored(self):
        # execute_bash uses `timeout`, not `timeout_seconds`.
        assert _call_deadline({"timeout": 600}) == 600 + mod._TIMEOUT_HEADROOM

    def test_no_timeout_arg_uses_the_default(self):
        assert _call_deadline({}) == mod.MCP_CALL_TIMEOUT
        assert _call_deadline({"command": "ls"}) == mod.MCP_CALL_TIMEOUT

    def test_default_is_a_floor_not_a_ceiling(self):
        # A short tool-level timeout describes the TOOL's internal deadline, not
        # how long the round trip may take — it must not shrink the transport
        # budget below the configured default.
        assert _call_deadline({"timeout": 5}) == mod.MCP_CALL_TIMEOUT

    @pytest.mark.parametrize("bad", ["abc", None, "", [], {}, object()])
    def test_non_numeric_timeout_falls_back_to_the_default(self, bad):
        # A malformed argument is the tool's problem; it must not crash the client
        # or produce a nonsense deadline.
        assert _call_deadline({"timeout_seconds": bad}) == mod.MCP_CALL_TIMEOUT

    def test_zero_and_negative_are_ignored(self):
        assert _call_deadline({"timeout_seconds": 0}) == mod.MCP_CALL_TIMEOUT
        assert _call_deadline({"timeout_seconds": -5}) == mod.MCP_CALL_TIMEOUT

    def test_non_dict_arguments_do_not_raise(self):
        assert _call_deadline(None) == mod.MCP_CALL_TIMEOUT
        assert _call_deadline("not a dict") == mod.MCP_CALL_TIMEOUT


class TestATimeoutSaysWhatHappened:
    def test_the_exception_carries_a_message(self):
        # The whole point: the stdlib TimeoutError does not.
        assert str(TimeoutError()) == ""
        assert str(MCPCallTimeout("no response after 300s")) != ""

    def test_call_tool_sync_names_the_tool_and_the_knob(self, monkeypatch):
        w = mod.MCPClientWrapper.__new__(mod.MCPClientWrapper)

        def _boom(coro, timeout=None):
            coro.close()  # we never run it; don't leave a pending coroutine
            raise MCPCallTimeout(f"no response from the MCP server after {timeout:.0f}s")

        monkeypatch.setattr(w, "_run_coroutine", _boom, raising=False)
        with pytest.raises(MCPCallTimeout) as ei:
            w.call_tool_sync("wait_for_task", {"timeout_seconds": 900})

        msg = str(ei.value)
        assert "wait_for_task" in msg           # which tool
        assert "930" in msg                     # the deadline actually used
        assert "MCP_CALL_TIMEOUT" in msg        # the knob to change
        assert "not retried" in msg             # why it wasn't replayed

    def test_the_deadline_in_the_message_is_the_derived_one(self, monkeypatch):
        # Not the raw default — otherwise the message misreports what was waited.
        w = mod.MCPClientWrapper.__new__(mod.MCPClientWrapper)

        def _boom(coro, timeout=None):
            coro.close()
            raise MCPCallTimeout("timed out")

        monkeypatch.setattr(w, "_run_coroutine", _boom, raising=False)
        with pytest.raises(MCPCallTimeout) as ei:
            w.call_tool_sync("wait_for_task", {"timeout_seconds": 1500})
        assert "1530" in str(ei.value)


class TestTheModelNeverSeesABareError:
    def test_an_empty_exception_yields_the_class_name(self):
        # `Error: ` with nothing after it gives the model nothing to act on.
        out = tool_formatting.tool_error_message("wait_for_task", TimeoutError())
        assert out.strip() != "Error:"
        assert "TimeoutError" in out

    def test_a_real_message_is_preserved(self):
        out = tool_formatting.tool_error_message(
            "wait_for_task", MCPCallTimeout("did not respond within 930s")
        )
        assert "930s" in out

    def test_missing_field_translation_still_works(self):
        # Regression guard: the empty-message fallback must not shadow the
        # pydantic "Field required" translation.
        exc = ValueError("new_string\n  Field required [type=missing]")
        out = tool_formatting.tool_error_message("file_edit", exc)
        assert "missing required" in out and "new_string" in out


class TestToolWrapperSurfacesTheDetail:
    def test_run_does_not_emit_an_empty_reason(self, monkeypatch):
        tool = mod.MCPToolWrapper.__new__(mod.MCPToolWrapper)

        class _Client:
            def call_tool_sync(self, name, args):
                raise TimeoutError()  # the empty-str() offender

        object.__setattr__(tool, "mcp_client", _Client())
        object.__setattr__(tool, "mcp_tool", type("T", (), {"name": "execute_bash"})())
        object.__setattr__(tool, "name", "execute_bash")

        with pytest.raises(ToolException) as ei:
            tool._run()
        # Must not end at the colon with nothing after it.
        assert not str(ei.value).rstrip().endswith(":")
        assert "TimeoutError" in str(ei.value)


class TestCancelLandsDuringABlockingCall:
    """Esc must abort a tool call in progress, not at its deadline.

    ``Future.result(timeout)`` parks in ``threading.Condition.wait`` — a C-level
    acquire that only notices the ``KeyboardInterrupt`` the UI injects when it
    RETURNS. Measured: an interrupt injected 0.5s into an 8s wait landed at 8.0s.
    Since the deadline is now derived from the tool's own timeout, that window
    reached ten minutes (``execute_bash(timeout=600)`` → 630s) while the UI showed
    "(cancelling…)" and the user's next message sat queued.

    The wait is therefore sliced, and the cooperative cancel event is consulted
    between slices.
    """

    def _wrapper(self, monkeypatch):
        w = mod.MCPClientWrapper.__new__(mod.MCPClientWrapper)
        w._loop = "loop"  # only needs to be non-None
        w._cancel_probe = None
        return w

    def _never_completes(self, monkeypatch, w):
        """Patch run_coroutine_threadsafe to hand back a future that never resolves."""
        from concurrent.futures import Future

        fut = Future()
        monkeypatch.setattr(
            mod.asyncio, "run_coroutine_threadsafe", lambda coro, loop: fut
        )
        return fut

    def test_a_pending_cancel_aborts_the_wait_quickly(self, monkeypatch):
        import time as _t

        w = self._wrapper(monkeypatch)
        fut = self._never_completes(monkeypatch, w)
        w._cancel_probe = lambda: True  # cancel already requested

        t0 = _t.monotonic()
        with pytest.raises(KeyboardInterrupt):
            w._run_coroutine(object(), timeout=600)
        elapsed = _t.monotonic() - t0
        # Must land within a slice or two, NOT at the 600s deadline.
        assert elapsed < 5, f"cancel took {elapsed:.1f}s"
        assert fut.cancelled() or fut.cancel(), "orphaned coroutine not cancelled"

    def test_no_cancel_still_times_out_normally(self, monkeypatch):
        w = self._wrapper(monkeypatch)
        self._never_completes(monkeypatch, w)
        w._cancel_probe = lambda: False
        with pytest.raises(MCPCallTimeout, match="no response"):
            w._run_coroutine(object(), timeout=0.5)

    def test_teardown_is_not_cancellable(self, monkeypatch):
        # shutdown() runs right after a cancelled turn, with the flag still set.
        # If that aborted the disconnect, the server subprocess would be orphaned.
        w = self._wrapper(monkeypatch)
        self._never_completes(monkeypatch, w)
        w._cancel_probe = lambda: True
        with pytest.raises(MCPCallTimeout):  # times out, NOT KeyboardInterrupt
            w._run_coroutine(object(), timeout=0.5, cancellable=False)

    def test_a_broken_probe_does_not_break_the_call(self, monkeypatch):
        w = self._wrapper(monkeypatch)
        self._never_completes(monkeypatch, w)

        def _boom():
            raise RuntimeError("probe exploded")

        w._cancel_probe = _boom
        # Must fall through to the normal timeout rather than propagate.
        with pytest.raises(MCPCallTimeout):
            w._run_coroutine(object(), timeout=0.5)

    def test_no_probe_means_no_cancel(self, monkeypatch):
        w = self._wrapper(monkeypatch)
        self._never_completes(monkeypatch, w)
        assert w._cancel_requested() is False
        with pytest.raises(MCPCallTimeout):
            w._run_coroutine(object(), timeout=0.5)

    def test_a_completed_call_returns_its_result(self, monkeypatch):
        # The slicing must not break the happy path.
        from concurrent.futures import Future

        w = self._wrapper(monkeypatch)
        fut = Future()
        fut.set_result("the result")
        monkeypatch.setattr(
            mod.asyncio, "run_coroutine_threadsafe", lambda coro, loop: fut
        )
        assert w._run_coroutine(object(), timeout=600) == "the result"

    def test_multi_client_forwards_the_probe_to_members(self):
        # A member wrapper has no other way to learn about a cancel.
        multi = mod.MultiMCPClient.__new__(mod.MultiMCPClient)
        a = mod.MCPClientWrapper.__new__(mod.MCPClientWrapper)
        b = mod.MCPClientWrapper.__new__(mod.MCPClientWrapper)
        multi._members = [("builtin", a), ("ext", b)]
        multi._cancel_probe = None

        probe = lambda: True  # noqa: E731
        multi.set_cancel_probe(probe)
        assert a._cancel_probe is probe and b._cancel_probe is probe


class TestDerivedDeadlineIsBounded:
    """A model-supplied timeout must not become an unbounded transport wait.

    Nothing validates `timeout_seconds` / `timeout` on either side — not the
    server tool, not the client — so a hallucinated `timeout_seconds=999999999`
    would have made the transport wait ~32 years. The single 300s cap contained
    this by accident; deriving the deadline from the argument removed that safety
    net, so the bound is now explicit.
    """

    def test_an_absurd_timeout_is_clamped(self):
        assert _call_deadline({"timeout_seconds": 999999999}) == mod._MAX_DERIVED_TIMEOUT

    def test_a_days_long_timeout_is_clamped(self):
        assert _call_deadline({"timeout_seconds": 86400}) == mod._MAX_DERIVED_TIMEOUT

    def test_realistic_timeouts_are_untouched(self):
        # The whole point of deriving the deadline — these must still pass through.
        assert _call_deadline({"timeout_seconds": 1500}) == 1530
        assert _call_deadline({"timeout": 600}) == 630

    def test_the_ceiling_never_lowers_an_explicit_config(self):
        # A user who sets LLM.MCP_CALL_TIMEOUT above the ceiling means it; the
        # clamp guards the DERIVED part only and must not drag the floor down.
        big = mod._MAX_DERIVED_TIMEOUT * 2
        assert _call_deadline({"timeout": 100}, default=big) == big
        assert _call_deadline({"timeout_seconds": 999999}, default=big) == big
        assert _call_deadline({}, default=big) == big
