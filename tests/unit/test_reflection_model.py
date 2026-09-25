"""Real extraction contract: bounded evidence, no fabricated fallback, cancellation."""

import json
import threading
from types import SimpleNamespace

import pytest

from mnemoai.client.memory import reflection
from mnemoai.client.memory.reflector import Reflector


def messages(result='{"error":true,"message":"target changed; read it again"}'):
    return [
        SimpleNamespace(type="human", content="Fix the target"),
        SimpleNamespace(type="ai", tool_calls=[
            {"name": "file_edit", "args": {"file_path": "target.py"}, "id": "call-1"},
        ]),
        SimpleNamespace(type="tool", tool_call_id="call-1", content=result),
    ]


class Model:
    def __init__(self, value=None, error=None):
        self.value = value if value is not None else {
            "lessons": [{
                "context": "editing a changed target",
                "strategy": "Read the target again before retrying an edit.",
                "evidence_ids": [0],
            }],
        }
        self.error, self.calls = error, []

    def invoke(self, prompt, config=None):
        self.calls.append((prompt, config))
        if self.error:
            raise self.error
        return SimpleNamespace(content=json.dumps(self.value))


def test_extraction_uses_model_and_links_only_observed_evidence(tmp_path):
    r, model = Reflector(str(tmp_path)), Model()
    usage = []
    entries = r.reflect_on_trajectory(
        messages(), "Fix target", model=model, scope="/project",
        source={"session_id": "s1", "turn": 4}, record_usage=usage.append,
    )
    assert len(model.calls) == len(entries) == len(usage) == 1
    entry = entries[0].to_dict()
    assert entry["provenance"] == "model" and entry["scope"] == "/project"
    assert entry["source_refs"][0]["tool_call_id"] == "call-1"
    assert entry["source_refs"][0]["turn"] == 4
    assert entry["outcome"] == "failure" and entry["confidence"] == 0.5
    assert model.calls[0][1] == {"callbacks": []}


@pytest.mark.parametrize("value", [
    {"lessons": [{"context": "c", "strategy": "s", "evidence_ids": [42]}]},
    {"lessons": [{"context": "c", "strategy": "s", "evidence_ids": [True]}]},
    {"lessons": [{"context": "c", "strategy": "s", "evidence_ids": []}]},
    {"lessons": [{"context": "c", "strategy": "", "evidence_ids": [0]}]},
    {"lessons": [{"context": "c", "strategy": "x" * 801, "evidence_ids": [0]}]},
    {"lessons": "not a list"},
    {"other": []},
])
def test_invalid_output_produces_no_partial_or_canned_lesson(tmp_path, value):
    r = Reflector(str(tmp_path))
    assert r.reflect_on_trajectory(messages(), "task", model=Model(value)) == []
    assert r.last_error and r.metrics["strategies_extracted"] == 0


def test_provider_failure_is_not_a_successful_reflection(tmp_path):
    r = Reflector(str(tmp_path))
    assert r.reflect_on_trajectory(messages(), "task", model=Model(error=OSError("outage"))) == []
    assert r.last_error


def test_no_tools_or_missing_results_make_no_model_call(tmp_path):
    r, model = Reflector(str(tmp_path)), Model()
    assert r.reflect_on_trajectory(messages()[:1], "hello", model=model) == []
    assert r.reflect_on_trajectory(messages()[:2], "task", model=model) == []
    assert model.calls == []


def test_no_lesson_is_a_valid_result(tmp_path):
    r = Reflector(str(tmp_path))
    assert r.reflect_on_trajectory(messages(), "task", model=Model({"lessons": []})) == []
    assert r.last_error is None


def test_precancelled_reflection_never_starts_a_model_request(tmp_path):
    r, model = Reflector(str(tmp_path)), Model()
    assert r.reflect_on_trajectory(
        messages(), "task", model=model, cancel=lambda: True,
    ) == []
    assert not model.calls and r.last_error


def test_timeout_prevents_late_writes_and_overlapping_calls(tmp_path):
    released, entered = threading.Event(), threading.Event()
    model = Model()
    original = model.invoke
    def wait(*args, **kwargs):
        entered.set()
        released.wait(3)
        return original(*args, **kwargs)
    model.invoke = wait
    r = Reflector(str(tmp_path))
    try:
        assert r.reflect_on_trajectory(messages(), "task", model=model, timeout=0.02) == []
        assert entered.is_set() and r._in_flight.is_set()
        assert r.reflect_on_trajectory(messages(), "task", model=model, timeout=0.02) == []
        assert r.metrics["strategies_extracted"] == 0
    finally:
        released.set()


def test_evidence_is_bounded_and_named_credentials_are_redacted():
    item = reflection.evidence_item(
        "execute_bash", {"api_key": "private-value"},
        "password=private-value " + "x" * 10_000, "call", "failure", {},
    )
    assert "private-value" not in item["args"] + item["result"]
    assert len(item["result"]) <= 1200 and item["truncated"]
    assert "result" not in item["ref"]


@pytest.mark.parametrize("result", [
    '{"password": "private-value", "content": "ordinary text"}',
    'Authorization: Bearer private-value',
    'content includes api_key="private-value"',
])
def test_named_credentials_in_structured_and_text_results_are_redacted(result):
    item = reflection.evidence_item("fs_read", {}, result, "call", "success", {})
    assert "private-value" not in item["result"]


def test_client_uses_precompaction_evidence_and_records_actual_model_usage(tmp_path):
    from mnemoai.client.client import LangGraphClient
    from mnemoai.client.memory.playbook_store import PlaybookStore
    from mnemoai.client.usage_tracker import UsageTracker

    c = LangGraphClient.__new__(LangGraphClient)
    c.reflector = Reflector(str(tmp_path))
    c.playbook = PlaybookStore(str(tmp_path))
    c.agent = SimpleNamespace(messages=[], usage=UsageTracker())
    model = Model()
    c._area_model = lambda area: model
    c._area_usage_name = lambda area: "actual-reflector"
    c._reflection_messages = messages()
    c._reflection_source = {"session_id": "s1", "turn": 1}
    c.reflect_and_learn("Fix target")
    assert len(c.playbook.snapshot()) == 1
    assert c.agent.usage.snapshot()[0]["model"] == "actual-reflector"
    assert c.agent.usage.snapshot()[0]["calls"] == 1
    c.reflect_and_learn("Fix target")
    assert len(model.calls) == 1
