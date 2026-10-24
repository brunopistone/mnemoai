"""Stalled non-streamed routing/decomposition must not require an app restart."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from mnemoai.client.agent import stream_policy
from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.agent.router import QueryRouter
from mnemoai.client.usage_tracker import UsageTracker


def test_cancelled_auxiliary_call_never_starts():
    event = threading.Event()
    event.set()
    call = Mock()
    with pytest.raises(KeyboardInterrupt):
        stream_policy.cancellable_call(call, event)
    call.assert_not_called()


@pytest.mark.parametrize("kind", ["router", "decomposer"])
def test_stalled_auxiliary_call_cancels_promptly_and_ignores_late_reply(kind):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    event = threading.Event()
    usage = UsageTracker()
    calls = []
    def invoke(messages, config=None):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        returned.set()
        return AIMessage(content="full", usage_metadata={
            "input_tokens": 10, "output_tokens": 1, "total_tokens": 11,
        })
    model = SimpleNamespace(invoke=invoke)
    if kind == "router":
        owner = QueryRouter(model, usage=usage)
        owner.cancel_event = event
        call = lambda: owner._invoke_with_retry(model, [])
    else:
        owner = LangGraphAgent.__new__(LangGraphAgent)
        owner.usage, owner._cancel_event = usage, event
        call = lambda: owner._aux_invoke(model, [], "fixture")
    outcomes = []
    def wait_for_model():
        try:
            call()
            outcomes.append("returned")
        except KeyboardInterrupt:
            outcomes.append("cancelled")
    waiter = threading.Thread(target=wait_for_model)
    waiter.start()
    try:
        assert entered.wait(2)
        event.set()
        waiter.join(timeout=2)
        assert outcomes == ["cancelled"], "the UI must unblock before the socket returns"
        assert len(calls) == 1
        assert usage.totals()["calls"] == 1
        assert usage.totals()["calls_without_usage"] == 1
        release.set()
        assert returned.wait(2)
        assert usage.totals()["calls"] == 1  # no late double counting
    finally:
        release.set()
        waiter.join(timeout=2)


def test_auxiliary_exception_is_preserved():
    with pytest.raises(ValueError, match="original failure"):
        stream_policy.cancellable_call(
            lambda: (_ for _ in ()).throw(ValueError("original failure")), threading.Event(),
        )
