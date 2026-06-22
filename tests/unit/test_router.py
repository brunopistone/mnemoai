"""Unit tests for the query router (client/agent/router.py).

Focus: route parsing and the empty-content recovery path. Reasoning models on
the Mantle Responses API (e.g. Grok) intermittently return empty content for the
short classification call; the router must retry once, then fall back to "full"
silently (no scary WARNING every turn).
"""

from langchain_core.messages import AIMessage

from mnemoai.client.agent.router import QueryRouter


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
