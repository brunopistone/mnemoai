"""Preserve full external MCP schemas through LangChain tool binding."""

from types import SimpleNamespace

from mcp.types import Tool

from mnemoai.client.agent.confirmation_gate import _tool_accepts
from mnemoai.client.mcp_tool_wrapper import MCPToolWrapper


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
