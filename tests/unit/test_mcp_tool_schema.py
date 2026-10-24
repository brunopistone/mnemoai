"""Preserve full external MCP schemas through LangChain tool binding."""

import asyncio
from types import SimpleNamespace

import pytest
from mcp.types import ListToolsResult, Tool

from mnemoai.client.agent.confirmation_gate import _tool_accepts
from mnemoai.client.mcp_tool_wrapper import MCPClientWrapper, MCPToolWrapper


def test_nested_schema_unions_items_and_enums_survive():
    schema = {
        "type": "object",
        "properties": {
            "value": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            "items": {"type": "array", "items": {"type": "integer"}},
            "mode": {"type": "string", "enum": ["read", "write"]},
        },
        "required": ["value"],
    }
    calls = []
    client = SimpleNamespace(call_tool_sync=lambda name, args: calls.append(args) or "ok")
    tool = MCPToolWrapper(mcp_tool=Tool(name="external", inputSchema=schema), mcp_client=client)
    assert tool.args_schema == schema
    assert tool.invoke({"value": {"nested": True}}) == "ok"
    assert calls == [{"value": {"nested": True}}]


def test_confirmation_override_field_is_visible_in_json_schema():
    tool = SimpleNamespace(args_schema={"properties": {"allow_dangerous": {"type": "boolean"}}})
    assert _tool_accepts(tool, "allow_dangerous")
    assert not _tool_accepts(tool, "missing")


@pytest.mark.parametrize("first_cursor", ["second", ""])
def test_discovery_follows_pages_including_an_empty_first_page(first_cursor):
    requested = []
    async def list_tools(cursor=None):
        requested.append(cursor)
        if cursor is None:
            return ListToolsResult(tools=[], nextCursor=first_cursor)
        if cursor == first_cursor:
            return ListToolsResult(tools=[Tool(name="one", inputSchema={})], nextCursor="third")
        return ListToolsResult(tools=[Tool(name="two", inputSchema={})])
    client = MCPClientWrapper.__new__(MCPClientWrapper)
    client._session = SimpleNamespace(list_tools=list_tools)
    tools = asyncio.run(client._list_tools())
    assert [tool.name for tool in tools] == ["one", "two"]
    assert requested == [None, first_cursor, "third"]


def test_discovery_rejects_cursor_cycles_without_publishing_partial_tools():
    async def list_tools(cursor=None):
        return ListToolsResult(tools=[Tool(name="one", inputSchema={})], nextCursor="repeat")
    client = MCPClientWrapper.__new__(MCPClientWrapper)
    client._session = SimpleNamespace(list_tools=list_tools)
    client._tools = ["previous-tools"]
    with pytest.raises(RuntimeError, match="repeated pagination cursor"):
        asyncio.run(client._list_tools())
    assert client._tools == ["previous-tools"]
