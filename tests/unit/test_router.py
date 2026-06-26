"""Unit tests for the query router (client/agent/router.py).

Focus: route parsing and the empty-content recovery path. Reasoning models on
the Mantle Responses API (e.g. Grok) intermittently return empty content for the
short classification call; the router must retry once, then fall back to "full"
silently (no scary WARNING every turn).
"""

from langchain_core.messages import AIMessage

from mnemoai.client.agent.router import QueryRouter, is_trivial_query


class TestIsTrivialQuery:
    """Trivial, signal-free queries should bypass the orchestrator.

    Regression: short conversational prompts ("can you do it?", "please do it")
    classified as 'full' were decomposed into a single subtask in the worker
    loop, which could surface a blank answer.
    """

    def test_short_chit_chat_is_trivial(self):
        for q in ["Can you do it?", "Please do it", "what do you think?", "go ahead"]:
            assert is_trivial_query(q) is True, q

    def test_empty_is_trivial(self):
        assert is_trivial_query("") is True
        assert is_trivial_query("   ") is True

    def test_substantial_query_not_trivial(self):
        q = "Refactor the auth module to use JWT and update all the tests accordingly"
        assert is_trivial_query(q) is False

    def test_content_signal_overrides_shortness(self):
        # A short query is NOT trivial if it carries a real content signal.
        assert is_trivial_query("read config.yaml") is False
        assert is_trivial_query("open https://x.com") is False
        assert is_trivial_query("./run.py") is False


class _StubModel:
    """Returns a queued sequence of contents, one per invoke()."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0
        self.callbacks = None

    def invoke(self, messages, config=None):
        self.calls += 1
        content = self._contents.pop(0) if self._contents else ""
        return AIMessage(content=content)


def _router(contents):
    return QueryRouter(_StubModel(contents))


class TestParseRoute:
    def test_valid_route(self):
        r = _router([])
        assert r._parse_route("code") == "code"

    def test_strips_quotes_and_whitespace(self):
        r = _router([])
        assert r._parse_route("  'simple_qa'.  ") == "simple_qa"

    def test_strips_thinking_tags(self):
        r = _router([])
        assert r._parse_route("<think>hmm</think>research") == "research"

    def test_responses_content_blocks(self):
        r = _router([])
        content = [
            {"type": "reasoning", "summary": []},
            {"type": "text", "text": "full"},
        ]
        assert r._parse_route(content) == "full"

    def test_empty_returns_empty_string(self):
        r = _router([])
        assert r._parse_route("") == ""
        assert r._parse_route([]) == ""

    def test_unknown_returns_empty_string(self):
        r = _router([])
        assert r._parse_route("banana") == ""


class TestClassifyRecovery:
    def test_retries_once_on_empty_then_succeeds(self):
        # First call empty (transient null), second returns a valid route.
        r = _router(["", "code"])
        assert r.classify("edit a file") == "code"
        assert r.model.calls == 2

    def test_falls_back_to_full_after_two_empties(self):
        r = _router(["", ""])
        assert r.classify("anything") == "full"
        assert r.model.calls == 2

    def test_no_retry_when_first_succeeds(self):
        r = _router(["research"])
        assert r.classify("search the web") == "research"
        assert r.model.calls == 1


class TestFastRoute:
    """Deterministic heuristic fast-path: routes obvious cases with no LLM call."""

    def test_greeting_is_simple_qa(self):
        r = _router([])
        assert r.fast_route("hello") == "simple_qa"
        assert r.fast_route("thanks!") == "simple_qa"
        assert r.fast_route("good morning") == "simple_qa"

    def test_single_signal_routes(self):
        r = _router([])
        assert r.fast_route("Read main.py") == "code"
        assert r.fast_route("fix src/auth.py please") == "code"
        assert r.fast_route("summarize report.pdf") == "knowledge"
        assert r.fast_route("what's in data.csv") == "knowledge"
        assert r.fast_route("what's in screenshot.png?") == "knowledge"
        assert r.fast_route("read https://example.com") == "research"

    def test_multiple_signals_go_full_never_underbind(self):
        r = _router([])
        # image extension + a path = 2 signals -> full (binds everything,
        # incl. describe_image). The original bug was UNDER-binding here.
        assert r.fast_route("/Users/x/a.png what's in it?") == "full"
        assert r.fast_route("read config.yaml and fetch https://api.x.com") == "full"

    def test_no_signal_defers_to_llm(self):
        r = _router([])
        assert r.fast_route("explain recursion to me") is None
        assert r.fast_route("what is the capital of France") is None
        assert r.fast_route("") is None

    def test_classify_uses_fast_path_without_llm_call(self):
        # An obvious query must NOT spend an LLM round-trip.
        r = _router(["should-not-be-used"])
        assert r.classify("Read main.py") == "code"
        assert r.model.calls == 0

    def test_classify_with_context_skips_fast_path(self):
        # With conversation context (a follow-up), defer to the LLM so it can
        # disambiguate using that context.
        r = _router(["code"])
        assert r.classify("Read main.py", conversation_context="prior turn") == "code"
        assert r.model.calls == 1
