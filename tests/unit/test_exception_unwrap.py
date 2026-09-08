"""A failure wrapped in a task group must report — and be classified by — what
actually failed.

``str(ExceptionGroup)`` is the wrapper's own text, so every reader of an
exception saw the plumbing: the screen said ``✗ MCP server 'aws' failed to start;
skipping. (unhandled errors in a TaskGroup (1 sub-exception))``, the retry policy
matched no phrasing (a dropped socket inside a group = zero retries), and the
recovery advice had nothing to go on. The nesting is TWO deep in the MCP client
stack, so one ``.exceptions[0]`` still yields the same opaque text.

Covers the shared primitive, the screen text, both classifiers, and the two
reporting sites that produced that line — including that the traceback now has a
home (it was written nowhere, for a failure the user was shown).
"""

import logging

import pytest

from mnemoai.client.agent import stream_policy, turn_failure
from mnemoai.utils import exceptions as exc_utils
from mnemoai.utils.exceptions import (
    exception_leaves,
    exception_signature,
    exception_text,
    leaf_exception,
)
from mnemoai.utils.logger import exception_line

# anyio's own wording — what the user was shown instead of the cause.
_WRAPPER = "unhandled errors in a TaskGroup (1 sub-exception)"


class McpError(Exception):
    """Stand-in for mcp.shared.exceptions.McpError (which needs an ErrorData)."""


def _nested(leaf, depth=2):
    """``leaf`` inside ``depth`` task-group wrappers, the way anyio delivers it."""
    out = leaf
    for _ in range(depth):
        out = ExceptionGroup(_WRAPPER, [out])
    return out


# --- the shared primitive ----------------------------------------------------


class TestExceptionLeaves:
    def test_a_plain_exception_is_its_own_leaf(self):
        e = ValueError("boom")
        assert exception_leaves(e) == [e]

    def test_recurses_past_the_first_wrapper(self):
        leaf = McpError("Invalid request parameters")
        group = _nested(leaf)
        # The observed shape: one unwrap lands on ANOTHER group.
        assert isinstance(group.exceptions[0], ExceptionGroup)
        assert exception_leaves(group) == [leaf]
        assert leaf_exception(group) is leaf

    def test_every_leaf_is_kept_in_order(self):
        a, b = McpError("first"), OSError("second")
        group = ExceptionGroup("two", [ExceptionGroup("one", [a]), b])
        assert exception_leaves(group) == [a, b]
        assert leaf_exception(group) is a

    def test_depth_is_bounded(self, monkeypatch):
        # Tested by LOWERING the bound rather than building the deep nesting the
        # bound exists for; past it the group itself is returned, never nothing.
        monkeypatch.setattr(exc_utils, "_MAX_UNWRAP_DEPTH", 2)
        leaf = ValueError("deep")
        leaves = exception_leaves(_nested(leaf, depth=5))
        assert len(leaves) == 1
        assert leaves[0] is not leaf
        assert isinstance(leaves[0], ExceptionGroup)

    def test_a_group_holding_nothing_still_yields_something(self):
        # The stdlib refuses to build one, but the recursion walks data we don't
        # own — and reporting the wrapper beats reporting nothing at all.
        class _Hollow(ExceptionGroup):
            @property
            def exceptions(self):
                return ()

        hollow = _Hollow("empty", [ValueError("ignored")])
        assert exception_leaves(hollow) == [hollow]
        assert exception_line(hollow).startswith("_Hollow")


class TestClassifierText:
    def test_exception_text_is_a_drop_in_for_str(self):
        e = OSError("Connection was closed before we received a valid response")
        assert exception_text(e) == str(e)

    def test_exception_text_reads_through_the_wrapper(self):
        e = OSError("connection reset by peer")
        assert exception_text(_nested(e)) == "connection reset by peer"
        assert "TaskGroup" not in exception_text(_nested(e))

    def test_signature_carries_the_class_name(self):
        # A provider names a retryable condition after its exception CLASS.
        e = type("ThrottlingException", (Exception,), {})("slow down")
        assert exception_signature(_nested(e)) == "ThrottlingException: slow down"


# --- the screen text ---------------------------------------------------------


class TestExceptionLine:
    def test_names_the_leaf_not_the_wrapper(self):
        line = exception_line(_nested(McpError("Invalid request parameters")))
        assert line == "McpError: Invalid request parameters"
        assert "TaskGroup" not in line

    def test_siblings_are_counted_not_printed(self):
        group = ExceptionGroup("g", [McpError("first"), OSError("second"), OSError("third")])
        assert exception_line(group) == "McpError: first (+2 more)"

    def test_an_empty_message_leaves_the_class_name_alone(self):
        # A bare TimeoutError's str() is empty; "TimeoutError:" reads as truncated.
        assert exception_line(_nested(TimeoutError())) == "TimeoutError"

    def test_a_long_message_is_capped(self):
        line = exception_line(_nested(McpError("x" * 900)), limit=80)
        assert len(line) <= len("McpError: ") + 80
        assert line.endswith("…")

    def test_a_plain_exception_is_unaffected(self):
        assert exception_line(ValueError("plain")) == "ValueError: plain"


# --- the classifiers --------------------------------------------------------


class TestPolicyReadsThroughAGroup:
    def test_a_transient_failure_inside_a_group_is_still_retryable(self):
        e = OSError("Connection was closed before we received a valid response")
        assert stream_policy.is_transient_network_error(e) is True
        # Before: the wrapper matched no marker, so this was deterministic.
        assert stream_policy.is_transient_network_error(_nested(e)) is True

    def test_a_bare_wrapper_stays_deterministic(self):
        # Nothing retryable inside it — the wrapper's own text must not become
        # a marker match of its own.
        assert stream_policy.is_transient_network_error(
            _nested(McpError("Invalid request parameters"))
        ) is False

    def test_an_overflow_inside_a_group_is_still_an_overflow(self):
        e = ValueError("Input is too long for requested model")
        assert stream_policy.is_context_overflow_error(_nested(e)) is True

    def test_the_marker_names_the_real_failure(self):
        marker = turn_failure.failure_marker(_nested(McpError("nope")))
        assert "McpError" in marker
        assert "ExceptionGroup" not in marker
        assert turn_failure.is_failure_marker(marker)

    def test_recovery_advice_survives_the_wrapper(self):
        oversized = _nested(ValueError("prompt is too long: 1.2M tokens"))
        assert turn_failure.classify(oversized) == turn_failure.OVERSIZED
        assert "/compact" in turn_failure.recovery_advice(oversized)

        rejected = _nested(ValueError("ValidationException: malformed input"))
        assert turn_failure.classify(rejected) == turn_failure.REJECTED

        dropped = _nested(OSError("peer closed connection"))
        assert turn_failure.classify(dropped) == turn_failure.CONNECTION


# --- the two reporting sites ------------------------------------------------


class _FailingWrapper:
    """A member server whose connect (or tool listing) fails inside a group."""

    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        raise self._exc

    def __exit__(self, *a):
        return False

    def list_tools_sync(self):
        raise self._exc

    def shutdown(self):
        pass


class _OkWrapper:
    def __init__(self, tools=()):
        self._tools = list(tools)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def list_tools_sync(self):
        return self._tools

    def shutdown(self):
        pass


@pytest.fixture
def reported(monkeypatch):
    """Capture what the user is shown and what reaches the log file."""
    from mnemoai.client import mcp_tool_wrapper as mod

    printed, records = [], []

    class _Log:
        def error(self, msg, *args, **kwargs):
            records.append((msg % args if args else msg, kwargs))

        def __getattr__(self, _name):
            return lambda *a, **k: None

    monkeypatch.setattr(mod, "print_error", printed.append)
    monkeypatch.setattr(mod, "logger", _Log())
    monkeypatch.setattr(mod, "log_file_hint", lambda: "~/.mnemoai/logs/mnemoai.log")
    return mod, printed, records


def _multi(mod, members):
    m = mod.MultiMCPClient.__new__(mod.MultiMCPClient)
    m._members = members
    m._tools = []
    return m


class TestMemberFailureIsReported:
    def test_startup_failure_names_the_leaf(self, reported):
        mod, printed, _ = reported
        group = _nested(McpError("Invalid request parameters"))
        m = _multi(mod, [("builtin", _OkWrapper()), ("aws", _FailingWrapper(group))])

        m.__enter__()

        assert len(printed) == 1
        line = printed[0]
        assert "MCP server 'aws' failed to start; skipping." in line
        assert "McpError: Invalid request parameters" in line
        assert "TaskGroup" not in line
        # And it says where the rest of the story is.
        assert "~/.mnemoai/logs/mnemoai.log" in line
        # The healthy server is kept, the failed one dropped.
        assert [name for name, _ in m._members] == ["builtin"]

    def test_startup_failure_leaves_a_traceback_on_disk(self, reported):
        mod, _, records = reported
        m = _multi(mod, [("aws", _FailingWrapper(_nested(McpError("nope"))))])

        m.__enter__()

        assert len(records) == 1
        message, kwargs = records[0]
        assert "McpError: nope" in message
        # exc_info → the traceback reaches the FILE; console=False → the app's own
        # red line stays the only thing on screen.
        assert kwargs.get("exc_info") is True
        assert kwargs.get("extra") == {"console": False}

    def test_the_builtin_server_still_re_raises(self, reported):
        mod, printed, _ = reported
        m = _multi(mod, [("builtin", _FailingWrapper(_nested(McpError("nope"))))])
        with pytest.raises(ExceptionGroup):
            m.__enter__()
        assert printed == []

    def test_tool_listing_failure_names_the_leaf(self, reported):
        mod, printed, records = reported
        m = _multi(mod, [("aws", _FailingWrapper(_nested(McpError("boom"))))])

        assert m.list_tools_sync() == []

        assert "could not list tools; skipping." in printed[0]
        assert "McpError: boom" in printed[0]
        assert records and records[0][1].get("exc_info") is True


def test_the_console_record_is_suppressed_for_our_own_line(caplog):
    """The log call must not double-report: one failure, one line on screen."""
    from mnemoai.utils.logger import _console_filter

    record = logging.LogRecord("ai_app", logging.ERROR, __file__, 1, "x", None, None)
    record.console = False
    assert _console_filter(record) is False
