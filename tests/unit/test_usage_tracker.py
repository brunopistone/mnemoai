"""Unit tests for session token accounting (``/usage``).

The value of this feature is honesty about numbers, so the tests focus on the ways
a total could be quietly wrong: a provider that reports no usage, a partial payload,
concurrent sub-agents, and the distinction between "tokens spent this session"
(cumulative) and "how big is my context" (the latest prompt).
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from mnemoai.client import usage_tracker as ut
from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.managers.agent_conversation_manager import AgentConversationManager
from mnemoai.utils.config import config


class _Resp:
    """A response carrying whatever usage_metadata a provider might return."""

    def __init__(self, usage=None):
        self.usage_metadata = usage


def _usage(inp=0, out=0, cache_read=0, cache_write=0):
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
        "input_token_details": {
            "cache_read": cache_read,
            "cache_creation": cache_write,
        },
    }


class TestAccumulation:
    def test_a_single_call_is_recorded(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(100, 20)), "m")
        row = t.snapshot()[0]
        assert (row["input_tokens"], row["output_tokens"], row["calls"]) == (100, 20, 1)

    def test_calls_add_up(self):
        t = ut.UsageTracker()
        for _ in range(3):
            t.record(_Resp(_usage(10, 5)), "m")
        assert t.totals()["total_tokens"] == 45
        assert t.totals()["calls"] == 3

    def test_models_are_tracked_separately(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(100, 1)), "big")
        t.record(_Resp(_usage(10, 1)), "small")
        rows = t.snapshot()
        assert {r["model"] for r in rows} == {"big", "small"}
        assert t.totals()["models"] == 2

    def test_the_busiest_model_sorts_first(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(10, 0)), "quiet")
        t.record(_Resp(_usage(999, 0)), "busy")
        assert t.snapshot()[0]["model"] == "busy"

    def test_cache_tokens_are_summed(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(10, 1, cache_read=500, cache_write=200)), "m")
        row = t.snapshot()[0]
        assert row["cache_read"] == 500 and row["cache_write"] == 200

    def test_reset_clears_everything(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(10, 1)), "m")
        t.reset()
        assert t.snapshot() == []
        assert t.totals()["calls"] == 0

    def test_an_unnamed_model_is_labelled_not_dropped(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(5, 1)))
        assert t.snapshot()[0]["model"] == "unknown"


class TestAPartialTotalCannotLookComplete:
    """`usage_metadata` isn't populated by every provider. Counting a missing
    payload as zeros would silently understate the total, so those calls are
    tracked separately and the report flags them."""

    def test_a_response_with_no_usage_is_counted_as_such(self):
        t = ut.UsageTracker()
        t.record(_Resp(None), "m")
        row = t.snapshot()[0]
        assert row["calls"] == 1
        assert row["calls_without_usage"] == 1
        assert row["total_tokens"] == 0

    def test_an_empty_usage_dict_counts_as_missing(self):
        t = ut.UsageTracker()
        t.record(_Resp({}), "m")
        assert t.snapshot()[0]["calls_without_usage"] == 1

    def test_a_response_with_no_attribute_at_all_is_safe(self):
        t = ut.UsageTracker()
        t.record(object(), "m")
        assert t.snapshot()[0]["calls_without_usage"] == 1

    def test_the_report_says_the_total_is_a_lower_bound(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(10, 1)), "m")
        t.record(_Resp(None), "m")
        assert "lower bound" in ut.render(t)

    def test_a_partial_payload_keeps_what_it_has(self):
        t = ut.UsageTracker()
        t.record(_Resp({"input_tokens": 50}), "m")  # no output_tokens
        row = t.snapshot()[0]
        assert row["input_tokens"] == 50 and row["output_tokens"] == 0
        assert row["calls_without_usage"] == 0  # it DID report something

    def test_missing_cache_details_are_not_an_error(self):
        t = ut.UsageTracker()
        t.record(_Resp({"input_tokens": 1, "output_tokens": 1}), "m")
        assert t.snapshot()[0]["cache_read"] == 0


class TestGarbageIsNeverFatal:
    """Recording runs inside the model-call path — it must not break a turn."""

    def test_a_non_numeric_count_is_ignored(self):
        t = ut.UsageTracker()
        t.record(_Resp({"input_tokens": "lots", "output_tokens": None}), "m")
        assert t.snapshot()[0]["total_tokens"] == 0

    def test_a_negative_count_is_ignored(self):
        t = ut.UsageTracker()
        t.record(_Resp({"input_tokens": -5, "output_tokens": 3}), "m")
        row = t.snapshot()[0]
        assert row["input_tokens"] == 0 and row["output_tokens"] == 3

    def test_a_non_dict_usage_is_treated_as_missing(self):
        t = ut.UsageTracker()
        t.record(_Resp("not a dict"), "m")
        assert t.snapshot()[0]["calls_without_usage"] == 1

    def test_a_non_dict_cache_details_is_ignored(self):
        t = ut.UsageTracker()
        t.record(_Resp({"input_tokens": 1, "input_token_details": "x"}), "m")
        assert t.snapshot()[0]["cache_read"] == 0


class TestConcurrentRecordingIsSafe:
    """Sub-agents and orchestrator waves record from pool threads."""

    def test_no_counts_are_lost_under_concurrency(self):
        t = ut.UsageTracker()

        def worker():
            for _ in range(200):
                t.record(_Resp(_usage(1, 1)), "m")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        agg = t.totals()
        assert agg["calls"] == 1600
        assert agg["input_tokens"] == 1600 and agg["output_tokens"] == 1600


class TestTheReport:
    def test_an_empty_session_says_so_rather_than_showing_zeros(self):
        out = ut.render(ut.UsageTracker())
        assert "No tokens spent yet" in out
        assert "Ask something" in out

    def test_a_restored_conversation_still_shows_its_context(self):
        # Regression: after `--resume` or `/load` the tracker is empty (this
        # process made no calls) but a context IS loaded. Reporting "ask something
        # first" while several thousand tokens sit ready reads as a broken command.
        out = ut.render(ut.UsageTracker(), context_tokens=3907)
        assert "3,907" in out
        assert "No tokens spent yet" in out
        assert "Ask something" not in out  # there IS something loaded

    def test_the_zero_spend_is_explained_not_just_stated(self):
        out = ut.render(ut.UsageTracker(), context_tokens=1000)
        assert "restored conversation" in out

    def test_counts_are_thousands_separated(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(1234567, 0)), "m")
        assert "1,234,567" in ut.render(t)

    def test_the_model_name_appears(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(1, 1)), "claude-opus-5")
        assert "claude-opus-5" in ut.render(t)

    def test_no_dollar_cost_is_ever_shown(self):
        # Deliberate: pricing is meaningless across Ollama / SageMaker / LiteLLM,
        # so a confidently wrong figure is worse than none.
        t = ut.UsageTracker()
        t.record(_Resp(_usage(1_000_000, 500_000)), "m")
        out = ut.render(t)
        assert "$" not in out and "cost" not in out.lower().replace("no cost", "")

    def test_it_states_whose_numbers_these_are(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(1, 1)), "m")
        assert "as reported by the provider" in ut.render(t)

    def test_it_distinguishes_spend_from_context_size(self):
        # The most likely misreading: "total" is cumulative spend, NOT how big the
        # conversation is.
        t = ut.UsageTracker()
        t.record(_Resp(_usage(100, 10)), "m")
        out = ut.render(t, context_tokens=42)
        assert "Current context: 42" in out
        assert "not the size of your conversation" in out

    def test_the_context_line_is_omitted_when_unknown(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(1, 1)), "m")
        assert "Current context" not in ut.render(t, context_tokens=0)

    def test_an_all_models_line_appears_only_for_several_models(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(1, 1)), "one")
        assert "All models" not in ut.render(t)
        t.record(_Resp(_usage(1, 1)), "two")
        assert "All models" in ut.render(t)

    def test_cache_line_only_appears_when_there_is_cache_activity(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(1, 1)), "m")
        assert "cache:" not in ut.render(t)
        t.record(_Resp(_usage(1, 1, cache_read=9)), "m")
        assert "cache:" in ut.render(t)

    def test_singular_and_plural_call_wording(self):
        t = ut.UsageTracker()
        t.record(_Resp(_usage(1, 1)), "m")
        assert "1 call " in ut.render(t)
        t.record(_Resp(_usage(1, 1)), "m")
        assert "2 calls" in ut.render(t)


class TestAgentAndClientWiring:
    """The pure tracker passing proves nothing about whether calls reach it."""

    def test_the_agent_records_and_still_tracks_context_size(self):
        from mnemoai.client.agent.agent import LangGraphAgent

        a = LangGraphAgent.__new__(LangGraphAgent)
        a.usage = ut.UsageTracker()
        a.usage_model_name = "m"
        a._last_input_tokens = None
        a._capture_input_tokens(_Resp(_usage(500, 20)))
        # Cumulative spend AND the latest prompt size both land.
        assert a.usage.totals()["total_tokens"] == 520
        assert a._last_input_tokens == 500

    def test_a_quiet_worker_counts_toward_usage_but_not_context_size(self):
        # A sub-agent's prompt is a private worker context; letting it set
        # _last_input_tokens would corrupt the compaction trigger.
        from mnemoai.client.agent.agent import LangGraphAgent

        a = LangGraphAgent.__new__(LangGraphAgent)
        a.usage = ut.UsageTracker()
        a.usage_model_name = "m"
        a._last_input_tokens = 111
        a._record_usage(_Resp(_usage(90_000, 100)))
        assert a.usage.totals()["total_tokens"] == 90_100
        assert a._last_input_tokens == 111  # untouched

    def test_recording_on_a_bare_stub_does_not_raise(self):
        from mnemoai.client.agent.agent import LangGraphAgent

        a = LangGraphAgent.__new__(LangGraphAgent)  # no .usage attribute
        a._record_usage(_Resp(_usage(1, 1)))  # must be a no-op, not an error

    def test_the_router_records_its_classification_call(self):
        # Routing is a real model call the user never sees.
        from mnemoai.client.agent.router import QueryRouter

        tracker = ut.UsageTracker()

        class _Model:
            callbacks = None

            def invoke(self, messages, config=None):
                r = _Resp(_usage(300, 5))
                r.content = "code"
                return r

        r = QueryRouter(_Model(), usage=tracker)
        r.usage_model_name = "m"
        r.classify("edit this file")
        assert tracker.totals()["input_tokens"] == 300

    def test_a_router_without_a_tracker_still_works(self):
        from mnemoai.client.agent.router import QueryRouter

        class _Model:
            callbacks = None

            def invoke(self, messages, config=None):
                r = _Resp(None)
                r.content = "code"
                return r

        assert QueryRouter(_Model()).classify("edit this file") == "code"

    def test_the_client_report_handles_a_missing_agent(self):
        from mnemoai.client.client import LangGraphClient

        c = LangGraphClient.__new__(LangGraphClient)
        c.agent = None
        assert "unavailable" in c.usage_report()


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setitem(config._config_data, "LLM", {"MAX_RETRIES": 2, "RETRY_DELAY": 0})
    result = LangGraphAgent.__new__(LangGraphAgent)
    result.system_prompt = "test"
    result.callbacks = []
    result.usage_model_name = "actor"
    result.usage = ut.UsageTracker()
    result._last_input_tokens = None
    result._start_spinner = lambda *a: None
    result._stop_spinner = lambda *a: None
    result._apply_mid_turn_compaction = lambda messages: messages
    return result


def test_authoritative_empty_stream_recovery_is_counted_once(agent):
    initial = AIMessage(content="", usage_metadata=_usage(100, 10))
    recovered = AIMessage(content="Recovered", usage_metadata=_usage(100, 20))
    agent._stream_response = Mock(return_value=(initial, False))
    model = SimpleNamespace(model_name="actor", invoke=Mock(return_value=recovered))
    agent._get_route_model = lambda state: model
    result = agent._call_model({"messages": [SystemMessage(content="test"), HumanMessage(content="Hi")]})
    assert result["messages"][0].content == "Recovered"
    assert agent.usage.totals()["calls"] == 2
    assert agent.usage.totals()["total_tokens"] == 230
    assert agent._last_input_tokens == 100


def test_none_stream_and_fallback_invoke_are_distinct_attempts(agent):
    agent._stream_response = Mock(return_value=(None, False))
    model = SimpleNamespace(model_name="actor", invoke=Mock(return_value=AIMessage(
        content="Recovered", usage_metadata=_usage(100, 20),
    )))
    agent._get_route_model = lambda state: model
    agent._call_model({"messages": [SystemMessage(content="test"), HumanMessage(content="Hi")]})
    assert agent.usage.totals()["calls"] == 2
    assert agent.usage.totals()["calls_without_usage"] == 1
    assert agent.usage.totals()["total_tokens"] == 120


def test_decomposition_and_aggregation_are_attributed_without_changing_context(agent):
    model = SimpleNamespace(model_id="decomposer", callbacks=None, invoke=Mock(return_value=AIMessage(
        content='[{"description":"Do the task","category":"full"}]', usage_metadata=_usage(200, 20),
    )))
    agent.orchestrator_model = model
    agent._non_reasoning = lambda _: model
    agent._last_input_tokens = 123
    assert agent._decompose_task("Do it", "Decompose", {"full"})
    agent._stream_response = Mock(return_value=(AIMessage(content="Done", usage_metadata=_usage(100, 10)), False))
    assert agent._aggregate_results("Do it", [{"task": "Do it", "result": "Done"}], "Summarize") == "Done"
    rows = {r["model"]: r for r in agent.usage.snapshot()}
    assert rows["decomposer"]["total_tokens"] == 220
    assert rows["actor"]["total_tokens"] == 110
    assert agent._last_input_tokens == 123  # auxiliary prompts cannot stand in for parent history


def test_auxiliary_retry_records_failed_attempt_without_duplicating_success(agent):
    model = SimpleNamespace(model_name="aux", invoke=Mock(side_effect=[
        RuntimeError("503 service unavailable"),
        AIMessage(content="OK", usage_metadata=_usage(10, 2)),
    ]))
    assert agent._aux_invoke(model, [], "Fixture").content == "OK"
    assert agent.usage.totals()["calls"] == 2
    assert agent.usage.totals()["calls_without_usage"] == 1
    assert agent.usage.totals()["total_tokens"] == 12


@pytest.mark.parametrize("first", [
    AIMessage(content="", usage_metadata=_usage(50, 2)),
    RuntimeError("503 service unavailable"),
])
def test_stream_retries_count_discarded_attempt_and_final_response_once(agent, first):
    final = AIMessage(content="OK", usage_metadata=_usage(100, 5))
    agent.model_with_tools = SimpleNamespace(model_name="actor")
    agent._empty_response_retries = 1
    agent._cancelled = lambda: False
    agent._sleep_or_cancel = lambda delay: False
    agent._transient_retry_delay = lambda *a: 0
    agent._stream_once = Mock(side_effect=[
        first if isinstance(first, Exception) else (first, False),
        (final, False),
    ])
    response, _ = agent._stream_response([], {})
    agent._capture_input_tokens(response)
    totals = agent.usage.totals()
    assert totals["calls"] == 2
    assert totals["total_tokens"] == (105 if isinstance(first, Exception) else 157)
    assert totals["calls_without_usage"] == int(isinstance(first, Exception))


def test_compaction_records_real_summary_model_usage(agent):
    agent.messages = [HumanMessage(content="old"), AIMessage(content="old reply"),
                      HumanMessage(content="new"), AIMessage(content="new reply")]
    agent.session_log = None
    manager = AgentConversationManager(max_tokens=100000)
    manager._session_blocks = lambda *a: []
    model = SimpleNamespace(model_name="summary-model", ainvoke=AsyncMock(return_value=AIMessage(
        content="Earlier facts.", usage_metadata=_usage(1000, 100),
    )))
    client = SimpleNamespace(spinner=Mock(), system_prompt="test")
    assert asyncio.run(manager._compact(client, model, agent, keep_recent=2))
    assert agent.usage.snapshot()[0]["model"] == "summary-model"
    assert agent.usage.totals()["calls"] == 1
    assert agent.usage.totals()["total_tokens"] == 1100
    assert agent._last_input_tokens is None


def test_parallel_summary_maps_reduce_and_retry_are_all_counted(monkeypatch):
    monkeypatch.setitem(config._config_data, "LLM", {"MAX_RETRIES": 2, "RETRY_DELAY": 0})
    manager = AgentConversationManager(max_tokens=100000)
    manager._batch_messages = lambda messages, budget: [[{"content": "one"}], [{"content": "two"}]]
    model = SimpleNamespace(model_name="summary", ainvoke=AsyncMock(side_effect=[
        RuntimeError("503 service unavailable"),
        AIMessage(content="summary A", usage_metadata=_usage(10, 2)),
        AIMessage(content="summary B", usage_metadata=_usage(10, 2)),
        AIMessage(content="combined", usage_metadata=_usage(10, 2)),
    ]))
    tracker = ut.UsageTracker()
    assert asyncio.run(manager.generate_summary(
        [], model, record_usage=ut.recorder(tracker, model),
    ))
    assert tracker.totals()["calls"] == model.ainvoke.call_count == 4
    assert tracker.totals()["calls_without_usage"] == 1
    assert tracker.totals()["total_tokens"] == 36


def test_an_in_flight_accounting_callback_cannot_pollute_a_cleared_session():
    tracker = ut.UsageTracker()
    record = ut.recorder(tracker, SimpleNamespace(model_name="old session"))
    tracker.reset()
    record(_Resp(_usage(100, 20)))
    assert tracker.totals()["calls"] == 0
    ut.recorder(tracker, SimpleNamespace(bound=SimpleNamespace(model_id="new session")))(_Resp(_usage(10, 2)))
    assert tracker.snapshot()[0]["model"] == "new session"
    assert tracker.totals()["total_tokens"] == 12


def test_context_estimate_is_labelled_in_usage_output():
    output = ut.render(ut.UsageTracker(), context_tokens=500, estimated=True)
    assert "~500" in output and "estimated next input" in output
