"""Unit tests for web_search + web_crawler additions.

- web_search: optional freshness/country/ui_lang forwarded only when set (empty
  strings fail Brave validation); citation + current-year guidance in docstring.
- web_crawler: inline markdown size cap (fallback that does NOT replace the RAG
  offload) + explicit configurable page timeout.

No network: the Brave client and crawl4ai symbols are patched at the module
where they're looked up.
"""

import asyncio
import json

import pytest

import mnemoai.server.tools.web_crawler as wc
import mnemoai.server.tools.web_search as ws


class _CapturingMCP:
    def __init__(self):
        self.registered = {}

    def tool(self):
        def decorator(func):
            self.registered[func.__name__] = func
            return func

        return decorator


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# web_search
# --------------------------------------------------------------------------- #
class _FakeWeb:
    def __init__(self):
        self.results = []


class _FakeSearchResult:
    web = _FakeWeb()


class _FakeBrave:
    """Records the WebSearchRequest it was handed."""

    last_request = None

    def __init__(self, api_key=None):
        pass

    async def web(self, request):
        _FakeBrave.last_request = request
        return _FakeSearchResult()


@pytest.fixture
def web_search(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "test-key")
    monkeypatch.setattr(ws, "BraveSearch", _FakeBrave)
    # WebSearchRequest: capture kwargs it's constructed with.
    captured = {}

    def _fake_request(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ws, "WebSearchRequest", _fake_request)
    mcp = _CapturingMCP()
    ws.register_web_search_tools(mcp)
    fn = mcp.registered["web_search"]
    fn._captured = captured  # expose for assertions
    return fn


class TestWebSearch:
    def test_omits_empty_optional_params(self, web_search):
        run(web_search("hello"))
        cap = web_search._captured
        assert "freshness" not in cap
        assert "country" not in cap
        assert "ui_lang" not in cap

    def test_forwards_set_optional_params(self, web_search):
        out = json.loads(
            run(web_search("hello", freshness="pw", country="US", ui_lang="en-US"))
        )
        cap = web_search._captured
        assert cap["freshness"] == "pw"
        assert cap["country"] == "US"
        assert cap["ui_lang"] == "en-US"
        # Echoed back into the result metadata.
        assert out["freshness"] == "pw" and out["country"] == "US"

    def test_docstring_has_citation_and_year_awareness(self):
        doc = ws.register_web_search_tools.__doc__ or ""
        # The tool's own docstring (not the registrar) carries the guidance.
        mcp = _CapturingMCP()

        import os

        os.environ.setdefault("BRAVE_API_KEY", "x")
        ws.register_web_search_tools(mcp)
        tool_doc = mcp.registered["web_search"].__doc__
        assert tool_doc is not None  # must remain a real docstring (not None)
        assert "CITE" in tool_doc
        assert "current" in tool_doc.lower()

    def test_current_year_in_result_metadata(self, web_search):
        from datetime import datetime

        out = json.loads(run(web_search("hello")))
        assert out["current_year"] == datetime.now().year


# --------------------------------------------------------------------------- #
# web_crawler
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, markdown):
        self.success = True
        self.url = "https://example.com"
        self.status_code = 200
        self.markdown = markdown
        self.metadata = {"title": "x"}


class _FakeCrawler:
    """Async context manager stand-in; records the config passed to arun."""

    last_config = None

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def arun(self, url, config=None):
        _FakeCrawler.last_config = config
        return _FakeResult(_FakeCrawler._markdown)


@pytest.fixture
def web_crawler(monkeypatch):
    monkeypatch.setattr(wc, "AsyncWebCrawler", _FakeCrawler)
    # Default: RAG disabled so the inline path is exercised.
    monkeypatch.setattr(wc.config, "get", lambda key, default=None: default)
    mcp = _CapturingMCP()
    wc.register_web_crawler_tools(mcp)
    return mcp.registered["web_crawler"]


class TestWebCrawler:
    def test_inline_content_truncated_over_cap(self, web_crawler):
        _FakeCrawler._markdown = "x" * (wc._MAX_INLINE_CHARS + 5000)
        out = json.loads(run(web_crawler("https://example.com")))
        assert out["truncated"] is True
        assert len(out["content"]) <= wc._MAX_INLINE_CHARS + 300  # + marker
        assert "truncated" in out["content"]

    def test_inline_content_not_truncated_under_cap(self, web_crawler):
        _FakeCrawler._markdown = "short page"
        out = json.loads(run(web_crawler("https://example.com")))
        assert out["truncated"] is False
        assert out["content"] == "short page"

    def test_page_timeout_default(self, web_crawler, monkeypatch):
        _FakeCrawler._markdown = "hi"
        _FakeCrawler.last_config = None
        run(web_crawler("https://example.com"))
        assert _FakeCrawler.last_config.page_timeout == 60000

    def test_page_timeout_from_config_override(self, monkeypatch):
        monkeypatch.setattr(wc, "AsyncWebCrawler", _FakeCrawler)

        def _cfg(key, default=None):
            if key == "WEB_CRAWL":
                return {"PAGE_TIMEOUT_MS": 15000}
            return default

        monkeypatch.setattr(wc.config, "get", _cfg)
        mcp = _CapturingMCP()
        wc.register_web_crawler_tools(mcp)
        _FakeCrawler._markdown = "hi"
        run(mcp.registered["web_crawler"]("https://example.com"))
        assert _FakeCrawler.last_config.page_timeout == 15000

    def test_invalid_url_rejected(self, web_crawler):
        out = json.loads(run(web_crawler("not-a-url")))
        assert out["error"] is True
