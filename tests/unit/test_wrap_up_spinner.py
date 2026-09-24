"""The spinner covers the whole time the prompt is held, not just the answer.

A streamed answer stops the spinner at its first token — correct, text is on
screen from then on. But the turn is not over: ``client.query`` still builds the
summary twin, counts the whole history against the budget, evicts old tool
results, possibly runs a full LLM summary, and updates the profile; then
``ChatInterface`` stores the episode and reflects. On a large conversation that
stretch is tens of seconds with nothing moving and no end-of-turn marker yet —
the app reads as hung.

Both halves therefore carry the spinner, under ONE shared label, and both hand
it back before anything prints (the ``[Context: N tokens]`` line and the turn-end
marker) so an animating frame can't overwrite a line in the stdout spinner mode.
"""

import threading

import pytest

from mnemoai.client.client import LangGraphClient
from mnemoai.client.ui.chat_interface import ChatInterface
from mnemoai.client.ui.spinner import WRAP_UP_LABEL


class _RecordingSpinner:
    """Spinner double that keeps the order of start/stop plus the live state."""

    def __init__(self):
        self.active = False
        self.label = None
        self.events = []  # ("start", label) | ("stop", None)

    def start(self, label="Thinking"):
        self.active = True
        self.label = label
        self.events.append(("start", label))

    def set_label(self, label):
        self.label = label

    def stop(self):
        self.active = False
        self.events.append(("stop", None))

    def snapshot(self):
        return self.active, self.label


class _FakeManager:
    """Records the spinner state observed from inside the post-answer work."""

    def __init__(self, boom=False):
        self.seen = None
        self._boom = boom

    async def manage_messages(self, client, model, agent):
        self.seen = client.spinner.snapshot()
        if self._boom:
            raise RuntimeError("compaction blew up")


class _FakeMCP:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeCallbacks:
    def reset(self):
        pass


class _FakeAgent:
    """Answers without touching the spinner — as a real stream would, having
    stopped it at its first token."""

    def __init__(self, spinner):
        self.messages = []
        self._spinner = spinner
        self.events_at_answer = None

    def __call__(self, prompt):
        self.events_at_answer = len(self._spinner.events)
        return "the answer"


def _client(boom=False):
    c = LangGraphClient.__new__(LangGraphClient)
    c.spinner = _RecordingSpinner()
    c.spinner_lock = threading.Lock()
    c.callback_handler = _FakeCallbacks()
    c.mcp_client = _FakeMCP()
    c.agent = _FakeAgent(c.spinner)
    c.conversation_manager = _FakeManager(boom=boom)
    c.episodic_memory = None
    c.plan_mode_active = False
    c.status_footer_active = False  # the plain loop: the context line prints
    c._steering_reminder = lambda: ""
    c._summary_model = lambda: object()
    c._profile_turn = lambda: None
    # Called from inside the printed context line, so it records the spinner
    # state at print time.
    c.seen_at_print = []
    c._count_context_tokens = lambda: c.seen_at_print.append(c.spinner.snapshot()) or 0
    return c


@pytest.fixture
def client():
    return _client()


class TestQueryPostAnswerPhase:
    def test_compaction_check_runs_with_the_spinner_up(self, client):
        # The regression: this ran after the stream had stopped the spinner, so a
        # multi-minute compaction check looked like a hung app.
        assert client.query("hi") == "the answer"
        assert client.conversation_manager.seen == (True, WRAP_UP_LABEL)

    def test_the_wrap_up_starts_only_after_the_answer(self, client):
        client.query("hi")
        agent = client.agent
        starts = [
            i for i, (kind, label) in enumerate(client.spinner.events)
            if kind == "start" and label == WRAP_UP_LABEL
        ]
        assert starts, "the post-answer phase never started the spinner"
        # It belongs to the work AFTER the answer, not to the turn's own start.
        assert min(starts) >= agent.events_at_answer
        assert client.spinner.events[0] == ("start", "Thinking")

    def test_the_context_line_prints_with_the_spinner_down(self, client):
        # stdout mode animates on the current line, so a frame still running
        # would be written over the line the count prints.
        client.query("hi")
        assert client.seen_at_print == [(False, WRAP_UP_LABEL)]

    def test_query_returns_with_the_spinner_stopped(self, client):
        client.query("hi")
        assert client.spinner.active is False
        assert client.spinner.events[-1][0] == "stop"

    def test_a_failure_in_the_wrap_up_still_hands_the_spinner_back(self):
        c = _client(boom=True)
        out = c.query("hi")
        assert "Something went wrong" in out
        assert c.spinner.active is False


class _StubClient:
    """Enough client for _dispatch's plain-query path, recording the spinner."""

    def __init__(self):
        self.plan_mode_active = False
        self.episodic_memory = None
        self.session_id = "sess_20260101_000000"
        self.spinner = _RecordingSpinner()
        self.spinner_lock = threading.Lock()
        self.reflector = object()  # truthy → reflect_and_learn runs
        self.seen_at_reflect = None

    def query(self, q):
        # A real query leaves the spinner stopped (its own finally).
        self.spinner.stop()
        return "here is your answer"

    def reflect_and_learn(self, query):
        self.seen_at_reflect = self.spinner.snapshot()


@pytest.fixture
def ci(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path / "home"))
    c = ChatInterface.__new__(ChatInterface)
    c.client = _StubClient()
    return c


class TestLearningStepsPhase:
    def test_reflection_runs_with_the_spinner_up(self, ci):
        # Storing an episode embeds it (a provider round trip) and reflection
        # writes the playbook: still the user's wait, so still the spinner's.
        ci._dispatch("what time is it?")
        assert ci.client.seen_at_reflect == (True, WRAP_UP_LABEL)

    def test_the_turn_end_marker_prints_with_the_spinner_down(self, ci, capsys):
        ci._dispatch("what time is it?")
        assert ci.client.spinner.active is False
        assert ci.client.spinner.events[-1][0] == "stop"
        assert "done in" in capsys.readouterr().out

    def test_a_raising_learning_step_still_hands_the_spinner_back(self, ci):
        def _boom(query):
            raise RuntimeError("chroma moved")

        ci.client.reflect_and_learn = _boom
        ci._dispatch("what time is it?")
        assert ci.client.spinner.active is False

    def test_no_spinner_on_the_client_is_survivable(self, ci):
        # Off-TTY stubs and minimal clients don't have one; a missing progress
        # indicator must never fail a turn whose answer is already printed.
        ci.client.spinner = None
        ci._dispatch("what time is it?")
