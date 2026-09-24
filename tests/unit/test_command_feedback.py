"""Compact command output must show useful status without setup boilerplate."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemoai.client.ui.command_feedback import render_mcp_status, render_model_update
from mnemoai.utils.configurator import ModelOverride

_ANSI = re.compile(r"\033\[[0-9;]*m")


def plain(text):
    return _ANSI.sub("", text)


def tool(name, owner, original=None):
    return SimpleNamespace(
        name=name, mcp_client=owner, mcp_tool=SimpleNamespace(name=original or name)
    )


def test_mcp_counts_ownership_not_namespace_prefixes():
    builtin = SimpleNamespace(_connected=True)
    playwright = SimpleNamespace(_connected=True)
    aws = SimpleNamespace(_connected=True)
    members = [("builtin", builtin), ("playwright", playwright), ("aws", aws)]
    tools = [
        tool("fs_read", builtin),
        tool("playwright__diagnostics", builtin),
        tool("browser_navigate", playwright),
        tool("aws__fs_read", aws, original="fs_read"),
    ]
    output = plain(render_mcp_status(members, tools))
    assert "builtin: connected (2 tools)" in output
    assert "playwright: connected (1 tool)" in output
    assert "aws: connected (1 tool)" in output
    assert "4 tools available." in output
    assert "/mcp verbose" in output
    assert "browser_navigate" not in output
    assert "namespaced" not in output
    assert "mcpServers" not in output
    assert "Config:" not in output


def test_mcp_verbose_preserves_tools_renames_and_setup_details():
    owner = SimpleNamespace(_connected=True)
    tools = [tool("aws__fs_read", owner, original="fs_read"), tool("lookup", owner)]
    path = Path("/some/custom/mcp.json")
    output = plain(
        render_mcp_status(
            [("aws", owner)], tools, verbose=True, config_path=path, width=48
        )
    )
    assert "fs_read → aws__fs_read" in output
    assert "lookup" in output
    assert f"Config: {path}" in output
    assert '"mcpServers"' in output


def test_mcp_disconnected_and_empty_states_are_explicit():
    disconnected = SimpleNamespace(_connected=False)
    output = plain(render_mcp_status([("playwright", disconnected)], []))
    assert "playwright: disconnected (0 tools)" in output
    output = plain(render_mcp_status([], []))
    assert "No MCP servers connected." in output


def test_external_labels_cannot_emit_terminal_control_sequences():
    owner = SimpleNamespace(_connected=True)
    output = render_mcp_status(
        [("server\033[2J", owner)], [tool("tool\033[2J", owner)], verbose=True
    )
    assert "\033[2J" not in output


def test_verbose_tool_list_wraps_without_dropping_names():
    owner = SimpleNamespace(_connected=True)
    names = ["browser_navigate", "browser_snapshot", "browser_console_messages"]
    output = plain(
        render_mcp_status(
            [("playwright", owner)],
            [tool(name, owner) for name in names],
            verbose=True,
            config_path=Path("/config/mcp.json"),
            width=45,
        )
    )
    assert all(name in output for name in names)
    tool_lines = [line for line in output.splitlines() if line.startswith("    ")]
    assert len(tool_lines) >= 2
    assert all(len(line) <= 45 for line in tool_lines)


@pytest.mark.parametrize(
    "label",
    [
        "Chat model",
        "Vision model",
        "Embeddings model",
        "Router model",
        "Orchestrator model",
        "Summary model",
    ],
)
def test_all_model_roles_share_a_compact_confirmation(label):
    change = ModelOverride(
        Path("/private/config.yaml"),
        "MODEL_ID",
        label=label,
        model_name="selected-model",
        model_type="bedrock",
    )
    output = plain(render_model_update(change, applied=True))
    assert output == f"• {label} changed to selected-model (bedrock)"
    assert "/private" not in output
    assert "=" * 10 not in output
    assert "Note:" not in output
    assert "reset" not in output


def test_restart_notice_says_saved_not_already_applied():
    change = ModelOverride(
        Path("/config.yaml"),
        "MODEL_ID",
        label="Chat model",
        model_name="selected-model",
        parameters_reset=True,
    )
    output = plain(render_model_update(change, applied=False))
    assert "Chat model saved: selected-model" in output
    assert "Restarting to apply" in output
    assert "Parameters reset to defaults · /params to adjust." in output
    assert "changed to" not in output
    assert len(output.splitlines()) == 3


def test_follow_chat_is_not_described_as_a_separate_model():
    change = ModelOverride(
        Path("/config.yaml"),
        "ROUTER",
        label="Router model",
        model_name="main-model",
        follows_chat=True,
    )
    output = plain(render_model_update(change, applied=True))
    assert output == "• Router model now follows Chat: main-model"
