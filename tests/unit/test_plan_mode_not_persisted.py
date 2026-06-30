"""Regression: the ephemeral plan-mode banner must not be persisted in history.

When plan mode is active, the client prepends a <plan-mode-active> reminder to
the prompt. That reminder is true only for the turn it's sent on — it must reach
the model that turn but NOT be stored in agent.messages (and therefore not saved
to a conversation file). Otherwise a saved/reloaded chat makes the model believe
it's still in plan mode even after /plan is toggled off.
"""

from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client.agent.agent import LangGraphAgent

BANNER = (
    "<plan-mode-active>\n"
    "Plan mode is active. You MUST NOT make any edits...\n"
    "</plan-mode-active>\n\n"
)


class _Graph:
    """Capture what the model received; echo an assistant reply."""

    def __init__(self):
        self.seen_human = []

    def invoke(self, state, config=None):
        self.seen_human = [
            m.content for m in state["messages"] if isinstance(m, HumanMessage)
        ]
        return {
            "messages": list(state["messages"]) + [AIMessage(content="done")],
            "thinking": None,
        }


def _agent():
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._messages = []
    a.system_prompt = ""
    a.recursion_limit = 50
    a._thinking = None
    a.graph = _Graph()
    a._stop_spinner = lambda: None
    a._extract_visible = lambda c: c if isinstance(c, str) else ""
    return a


class TestStripEphemeral:
    def test_strips_banner_keeps_prompt(self):
        assert (
            LangGraphAgent._strip_ephemeral(BANNER + "analyze the repo")
            == "analyze the repo"
        )

    def test_noop_on_clean_prompt(self):
        assert LangGraphAgent._strip_ephemeral("just a prompt") == "just a prompt"

    def test_strips_even_with_leading_whitespace(self):
        out = LangGraphAgent._strip_ephemeral("  " + BANNER + "go")
        assert "plan-mode-active" not in out
        assert out.endswith("go")


class TestInvokePersistsCleanTurn:
    def test_model_sees_banner_history_does_not(self):
        a = _agent()
        a.invoke(BANNER + "analyze the repo")
        # The model received the banner this turn (plan mode still enforced).
        assert any("plan-mode-active" in h for h in a.graph.seen_human)
        # But stored history holds only the clean user prompt.
        stored = [m.content for m in a._messages if isinstance(m, HumanMessage)]
        assert stored == ["analyze the repo"]
        assert not any("plan-mode-active" in s for s in stored)

    def test_no_duplicate_human_turn(self):
        a = _agent()
        a.invoke(BANNER + "do it")
        humans = [m for m in a._messages if isinstance(m, HumanMessage)]
        ais = [m for m in a._messages if isinstance(m, AIMessage)]
        assert len(humans) == 1  # the reminder-bearing turn was NOT re-added
        assert len(ais) == 1

    def test_plain_prompt_unaffected(self):
        a = _agent()
        a.invoke("hello")
        stored = [m.content for m in a._messages if isinstance(m, HumanMessage)]
        assert stored == ["hello"]
