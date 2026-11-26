"""Provider isolation, real SDK request contracts, tool schemas and cancellation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import yaml
from mcp.server.fastmcp import FastMCP

from mnemoai.server.tools.web_search import register_web_search_tools
from mnemoai.server.tools.web_search_controller import WebSearchController
from mnemoai.server.tools.web_search_providers import tavily as T
from mnemoai.utils import configurator as C
from mnemoai.utils.web_search_config import api_key_for, provider_for


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def tavily(monkeypatch):
    state = SimpleNamespace(
        calls=[],
        closed=0,
        key=None,
        error=None,
        response={
            "results": [
                {
                    "title": "Source",
                    "url": "https://example.com",
                    "content": "Evidence",
                    "score": 0.95,
                    "published_date": "2026-11-01",
                }
            ],
            "request_id": "request-1",
            "usage": {"credits": 1},
        },
    )

    class Client:
        def __init__(self, api_key):
            state.key = api_key

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            state.closed += 1

        def __getattr__(self, name):
            async def call(**kwargs):
                state.calls.append((name, kwargs))
                if state.error:
                    raise state.error
                return json.loads(json.dumps(state.response))

            return call

    monkeypatch.setattr(
        T, "import_module", lambda name: SimpleNamespace(AsyncTavilyClient=Client)
    )
    monkeypatch.setattr(T, "classify_url", lambda url: SimpleNamespace(blocked=False))
    state.provider = WebSearchController("tavily", "tavily-test-key")
    return state


def test_legacy_selection_and_key_precedence(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "env-brave")
    monkeypatch.setenv("TAVILY_API_KEY", "env-tavily")
    assert provider_for({}) == "brave"
    assert provider_for({"WEB_SEARCH_PROVIDER": " Tavily "}) == "tavily"
    for provider in ("brave", "tavily"):
        name = provider.upper() + "_API_KEY"
        assert api_key_for(provider, {}) == "env-" + provider
        assert (
            api_key_for(provider, {name: "your_" + provider + "_api_key"})
            == "env-" + provider
        )
        assert api_key_for(provider, {name: "yaml-key"}) == "yaml-key"
    with pytest.raises(ValueError, match="brave or tavily"):
        provider_for({"WEB_SEARCH_PROVIDER": "unsupported"})
    with pytest.raises(ValueError):
        WebSearchController("unsupported")


def test_search_normalizes_evidence_and_keeps_usage(tavily):
    result = json.loads(run(tavily.provider.search("  query  ")))
    method, params = tavily.calls[0]
    assert method == "search" and params["query"] == "query"
    assert params["search_depth"] == "basic" and params["auto_parameters"] is False
    assert not params["include_answer"] and not params["include_raw_content"]
    assert params["language"] == "en"
    assert result["results"][0] == {
        "title": "Source",
        "link": "https://example.com",
        "description": "Evidence",
        "position": 1,
        "score": 0.95,
        "date": "2026-11-01",
    }
    assert result["search_provider"] == "Tavily"
    assert result["usage"] == {"credits": 1} and result["request_id"] == "request-1"
    assert tavily.closed == 1 and tavily.key == "tavily-test-key"


@pytest.mark.parametrize("value,expected", [(0, 1), (-10, 1), (10, 10), (99, 20)])
def test_result_count_bounds(tavily, value, expected):
    run(tavily.provider.search("query", num_results=value))
    assert tavily.calls[0][1]["max_results"] == expected


@pytest.mark.parametrize(
    "freshness,expected",
    [
        ("pd", {"time_range": "day"}),
        ("pw", {"time_range": "week"}),
        ("pm", {"time_range": "month"}),
        ("py", {"time_range": "year"}),
        (
            "2026-01-01to2026-02-01",
            {"start_date": "2026-01-01", "end_date": "2026-02-01"},
        ),
    ],
)
def test_freshness_mapping(tavily, freshness, expected):
    run(tavily.provider.search("query", freshness=freshness))
    assert expected.items() <= tavily.calls[0][1].items()


def test_advanced_parameters_and_extra_result_fields(tavily):
    tavily.response.update(
        answer="Generated summary", images=["https://example.com/a.jpg"]
    )
    tavily.response["results"][0].update(
        raw_content="Full text", favicon="https://example.com/favicon"
    )
    result = json.loads(
        run(
            tavily.provider.search(
                "query",
                search_depth="advanced",
                search_lang="it",
                filter_by_language=True,
                country="IT",
                start_date="2026-01-01",
                include_domains=["example.com"],
                exclude_domains=["other.example"],
                include_domains_mode="prefer",
                include_answer="advanced",
                include_raw_content="markdown",
                exact_match=True,
                include_images=True,
                include_image_descriptions=True,
                include_favicon=True,
                max_age_hours=0,
                chunks_per_source=2,
            )
        )
    )
    params = tavily.calls[0][1]
    assert params["country"] == "italy" and params["language"] == "it"
    assert params["include_domains_mode"] == "prefer" and params["max_age_hours"] == 0
    assert params["search_depth"] == "advanced" and params["chunks_per_source"] == 2
    assert params["include_answer"] == "advanced" and params["filter_by_language"]
    assert result["answer"] == "Generated summary"
    assert result["results"][0]["raw_content"] == "Full text" and result["images"]


@pytest.mark.parametrize("topic", ["news", "finance"])
def test_specialist_topics(tavily, topic):
    run(tavily.provider.search("query", topic=topic, freshness="pw"))
    assert tavily.calls[0][1]["topic"] == topic and "country" not in tavily.calls[0][1]


def test_ultra_fast_omits_unsupported_chunks_and_ui_language_is_explicit(tavily):
    result = json.loads(
        run(tavily.provider.search("query", search_depth="ultra-fast", ui_lang="it-IT"))
    )
    assert "chunks_per_source" not in tavily.calls[0][1]
    assert "ui_lang" not in tavily.calls[0][1] and result["warnings"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": " "},
        {"freshness": "yesterday"},
        {"freshness": "2026-02-30to2026-03-01"},
        {"freshness": "2026-04-01to2026-03-01"},
        {"start_date": "20261101"},
        {"freshness": "pw", "start_date": "2026-11-01"},
        {"country": "US", "topic": "news"},
        {"country": "ZZ"},
        {"include_domains_mode": "prefer"},
        {"include_domains": ["x"] * 301},
        {"exclude_domains": ["x"] * 151},
        {"include_domains": [""]},
        {"include_image_descriptions": True},
        {"search_lang": "", "filter_by_language": True},
        {"chunks_per_source": 0},
        {"max_age_hours": 721},
        {"timeout": float("nan")},
    ],
)
def test_invalid_search_inputs_never_contact_provider(tavily, kwargs):
    result = json.loads(run(tavily.provider.search(**{"query": "query", **kwargs})))
    assert result["error"] and not tavily.calls


def test_content_budget_preserves_sources_and_structured_json(tavily):
    tavily.response["results"] *= 2
    tavily.response["results"][0]["content"] = "x" * 200_000
    result = json.loads(run(tavily.provider.search("query")))
    assert result["truncated"]
    assert len(result["results"][0]["description"]) < 100_100
    assert result["results"][1]["link"] == "https://example.com"
    assert result["total_results"] == 2


@pytest.mark.parametrize(
    "exception", [ValueError("tavily-test-key rejected"), asyncio.CancelledError()]
)
def test_failure_redaction_cleanup_and_cancellation(tavily, exception, caplog):
    tavily.error = exception
    if isinstance(exception, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            run(tavily.provider.search("query"))
    else:
        result = run(tavily.provider.search("query"))
        assert json.loads(result)["error"]
        assert "tavily-test-key" not in result + caplog.text
    assert tavily.closed == 1


@pytest.mark.parametrize(
    "method,args",
    [
        (
            "tavily_extract",
            {
                "urls": ["https://example.com"],
                "query": "evidence",
                "chunks_per_source": 2,
            },
        ),
        (
            "tavily_crawl",
            {"url": "https://example.com", "instructions": "Find docs", "limit": 3},
        ),
        ("tavily_map", {"url": "https://example.com", "select_paths": ["/docs/.*"]}),
    ],
)
def test_page_tools_forward_options_and_close(tavily, method, args):
    result = json.loads(run(getattr(tavily.provider.provider, method)(**args)))
    assert result["search_provider"] == "Tavily" and tavily.closed == 1
    endpoint, params = tavily.calls[0]
    assert endpoint == method.removeprefix("tavily_")
    assert args.items() <= params.items() and params["include_usage"]
    if endpoint != "extract":
        assert params["allow_external"] is False


def test_extract_preserves_partial_failures_and_marks_total_failure(tavily):
    tavily.response["failed_results"] = [
        {"url": "https://bad.example", "error": "unavailable"}
    ]
    result = json.loads(
        run(tavily.provider.provider.tavily_extract(["https://example.com"]))
    )
    assert not result.get("error") and result["failed_results"]
    tavily.response["results"] = []
    result = json.loads(
        run(tavily.provider.provider.tavily_extract(["https://example.com"]))
    )
    assert result["error"] and result["failed_results"]


@pytest.mark.parametrize(
    "method,args",
    [
        ("tavily_extract", {"urls": []}),
        ("tavily_extract", {"urls": ["https://example.com"] * 21}),
        ("tavily_extract", {"urls": ["https://example.com"], "chunks_per_source": 1}),
        ("tavily_crawl", {"url": "https://example.com", "limit": 101}),
        ("tavily_crawl", {"url": "https://example.com", "max_depth": 6}),
        ("tavily_map", {"url": "https://example.com", "max_breadth": 0}),
    ],
)
def test_web_tool_limits_before_requests(tavily, method, args):
    result = json.loads(run(getattr(tavily.provider.provider, method)(**args)))
    assert result["error"] and not tavily.calls


def test_all_urls_are_checked_before_extract_request(tavily, monkeypatch):
    monkeypatch.setattr(
        T,
        "classify_url",
        lambda url: SimpleNamespace(
            blocked="localhost" in url, reason="private destination"
        ),
    )
    result = json.loads(
        run(
            tavily.provider.provider.tavily_extract(
                [
                    "https://example.com",
                    "http://localhost/secrets",
                ]
            )
        )
    )
    assert result["error"] and not tavily.calls


@pytest.mark.parametrize(
    "provider,extra", [("brave", False), ("tavily", False), ("tavily", True)]
)
def test_real_mcp_schema_and_optional_web_tools(provider, extra):
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from mnemoai.client.mcp_tool_wrapper import MCPToolWrapper

    mcp = FastMCP("provider-schema")
    register_web_search_tools(mcp, provider, "test-key", include_web_tools=extra)
    tools = {t.name: t for t in run(mcp.list_tools())}
    assert set(tools) == (
        {"web_search", *T.TavilySearchProvider.web_tools} if extra else {"web_search"}
    )
    properties = tools["web_search"].inputSchema["properties"]
    assert (
        "query" in properties
        and "self" not in properties
        and "kwargs" not in properties
    )
    assert ("search_depth" in properties) == (provider == "tavily")
    # Follow the same MCP -> LangChain -> LLM tool schema path as the app.
    llm_tool = convert_to_openai_tool(MCPToolWrapper(tools["web_search"], object()))
    description = llm_tool["function"]["description"]
    assert description == tools["web_search"].description
    for instruction in (
        "Search the INTERNET",
        "do NOT rely on training knowledge",
        "current calendar year",
        "ground every factual claim",
        "CITE your sources",
        "include the result title and link",
        "Do not answer web questions from memory",
    ):
        assert instruction in description


@pytest.mark.parametrize(
    "method,params",
    [
        (
            "search",
            {
                "query": "SDK contract",
                "search_depth": "advanced",
                "country": "US",
                "search_lang": "it",
            },
        ),
        (
            "tavily_extract",
            {"urls": ["https://example.com"], "extract_depth": "advanced"},
        ),
        ("tavily_crawl", {"url": "https://example.com", "limit": 2}),
        ("tavily_map", {"url": "https://example.com", "limit": 2}),
    ],
)
def test_real_sdk_http_contract(monkeypatch, sdk_http_modules, method, params):
    # Exercise the installed SDK and its async context manager, not a look-alike.
    from tavily import async_tavily

    calls = []
    original = httpx.AsyncClient
    clients = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"results": [], "usage": {"credits": 1}})

    def client(**kwargs):
        instance = original(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(instance)
        return instance

    monkeypatch.setattr(async_tavily.httpx, "AsyncClient", client)
    monkeypatch.setattr(T, "classify_url", lambda url: SimpleNamespace(blocked=False))
    adapter = T.TavilySearchProvider("test-auth")
    result = json.loads(run(getattr(adapter, method)(**params)))
    assert not result.get("error")
    request = calls[0]
    assert request.url.host == "api.tavily.com"
    assert request.headers["authorization"] == "Bearer test-auth"
    body = json.loads(request.content)
    if method == "search":
        assert body["country"] == "united states" and body["language"] == "it"
        assert body["search_depth"] == "advanced" and body["include_usage"]
    else:
        assert request.url.path == "/" + method.removeprefix("tavily_")
    assert all(c.is_closed for c in clients)


@pytest.fixture
def search_config(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        "# preserve\nMODEL_ID: {NAME: chat, TYPE: ollama}\nBRAVE_API_KEY: original-brave\n"
    )
    monkeypatch.setattr(C, "_config_to_edit", lambda: path)
    monkeypatch.setattr(C, "_ask_choice", lambda *a, **k: "tavily")
    monkeypatch.setattr(C, "_ask", lambda *a, **k: "tavily-key")
    monkeypatch.setattr(C, "_ask_bool", lambda *a, **k: False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    return path


def test_search_config_switch_preserves_models_comments_and_both_keys(search_config):
    assert C.run_search_settings() == search_config
    data = yaml.safe_load(search_config.read_text())
    assert data["WEB_SEARCH_PROVIDER"] == "tavily" and data["ENABLE_WEB_SEARCH"]
    assert (
        data["TAVILY_API_KEY"] == "tavily-key"
        and data["BRAVE_API_KEY"] == "original-brave"
    )
    assert data["MODEL_ID"] == {"NAME": "chat", "TYPE": "ollama"}
    assert search_config.read_text().startswith("# preserve")


def test_search_config_cancellation_and_concurrent_edit_preserve_file(
    search_config, monkeypatch
):
    original = search_config.read_text()
    monkeypatch.setattr(C, "_ask", Mock(side_effect=C._Cancelled()))
    assert C.run_search_settings() is None and search_config.read_text() == original

    def edit(*a, **k):
        search_config.write_text(original + "CONCURRENT: true\n")
        return "key"

    monkeypatch.setattr(C, "_ask", edit)
    assert C.run_search_settings() is None and "CONCURRENT" in search_config.read_text()


def test_missing_key_disables_feature_and_env_key_avoids_prompt(
    search_config, monkeypatch
):
    monkeypatch.setattr(C, "_ask", lambda *a, **k: "")
    assert C.run_search_settings()
    assert yaml.safe_load(search_config.read_text())["ENABLE_WEB_SEARCH"] is False
    monkeypatch.setenv("TAVILY_API_KEY", "env-key")
    monkeypatch.setattr(
        C, "_ask", Mock(side_effect=AssertionError("must not prompt for key"))
    )
    assert C.run_search_settings()
    data = yaml.safe_load(search_config.read_text())
    assert data["ENABLE_WEB_SEARCH"] and "TAVILY_API_KEY" not in data


def test_research_and_readonly_routes_include_tavily():
    from mnemoai.client.agent.plan_policy import PLAN_ALLOWED_TOOLS
    from mnemoai.client.agent.router import ROUTE_TOOLS
    from mnemoai.client.agent.subagents import _READONLY_TOOLS

    tools = set(T.TavilySearchProvider.web_tools)
    assert tools <= set(ROUTE_TOOLS["research"])
    assert tools <= PLAN_ALLOWED_TOOLS and tools <= set(_READONLY_TOOLS)


@pytest.mark.parametrize("written", [True, False])
def test_search_settings_dispatch_restarts_only_after_save(monkeypatch, written):
    from mnemoai.client.ui import chat_interface

    ui = chat_interface.ChatInterface.__new__(chat_interface.ChatInterface)
    ui.client = SimpleNamespace()
    ui._restart_in_place = Mock()
    monkeypatch.setattr(
        chat_interface,
        "run_search_settings",
        lambda: "config.yaml" if written else None,
    )
    ui._dispatch("/config search")
    assert ui._restart_in_place.call_count == int(written)


@pytest.mark.parametrize(
    "provider,env,expected_key",
    [
        ("tavily", {}, "TAVILY_API_KEY"),
        ("tavily", {"BRAVE_API_KEY": "brave-only"}, "TAVILY_API_KEY"),
        ("brave", {"TAVILY_API_KEY": "tavily-only"}, "BRAVE_API_KEY"),
        ("tavily", {"TAVILY_API_KEY": "configured"}, None),
        ("invalid", {}, "WEB_SEARCH_PROVIDER"),
    ],
)
def test_doctor_checks_selected_provider_only(monkeypatch, provider, env, expected_key):
    from mnemoai.client import doctor

    for key in ("BRAVE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        doctor, "config", {"ENABLE_WEB_SEARCH": True, "WEB_SEARCH_PROVIDER": provider}
    )
    checks = doctor._feature_checks()
    if expected_key:
        assert len(checks) == 1 and expected_key in checks[0].detail
    else:
        assert checks == []


@pytest.mark.parametrize(
    "provider,search,crawl,key,expected",
    [
        ("brave", True, True, True, {"web_search", "web_crawler"}),
        (
            "tavily",
            True,
            True,
            True,
            {
                "web_search",
                "web_crawler",
                "tavily_extract",
                "tavily_crawl",
                "tavily_map",
            },
        ),
        ("tavily", True, False, True, {"web_search"}),
        ("tavily", False, True, True, {"web_crawler"}),
        ("tavily", True, True, False, {"web_crawler"}),
        ("invalid", True, False, True, set()),
    ],
)
def test_actual_registration_honors_provider_key_and_feature_gates(
    monkeypatch, provider, search, crawl, key, expected
):
    from mnemoai.server.tools import tools_manager

    settings = {
        "ENABLE_WEB_SEARCH": search,
        "ENABLE_WEB_CRAWL": crawl,
        "WEB_SEARCH_PROVIDER": provider,
    }
    if key:
        settings.update(
            BRAVE_API_KEY="brave-test-key", TAVILY_API_KEY="tavily-test-key"
        )
    for name in ("BRAVE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tools_manager, "config", settings)
    mcp = FastMCP("registration-test")
    tools_manager.ToolManager().register_tools(mcp)
    names = {t.name for t in run(mcp.list_tools())}
    web_names = {
        "web_search",
        "web_crawler",
        "tavily_extract",
        "tavily_crawl",
        "tavily_map",
    }
    assert names & web_names == expected
