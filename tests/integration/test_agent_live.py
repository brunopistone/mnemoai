"""Integration tests: real agent + Ollama + MCP subprocess.

Auto-skipped unless a runtime config.yaml exists and Ollama is reachable
(see conftest.py). These verify the end-to-end paths that unit tests cannot:
routing, tool invocation, and that no query returns a silent empty turn.
"""

import pytest

pytestmark = pytest.mark.integration


class TestAgentEndToEnd:
    def test_greeting_returns_nonempty(self, live_client):
        # Regression for the "router returned ''" / silent empty-turn bug:
        # a greeting must produce a visible answer.
        resp = live_client.query("Hello")
        assert resp is not None
        assert resp.strip() != ""

    def test_simple_factual_question(self, live_client):
        resp = live_client.query(
            "What is the capital of France? Answer in one short sentence."
        )
        assert "paris" in resp.lower()

    def test_tool_backed_query_lists_files(self, live_client):
        # Should route to a tool-using path and actually invoke a file tool.
        resp = live_client.query(
            "Using a tool, list the python files in the current directory."
        )
        assert resp.strip() != ""
        assert ".py" in resp.lower()

    def test_no_silent_empty_turn_on_followup(self, live_client):
        # A short ambiguous follow-up should still yield a response, not silence.
        live_client.query("Pick a number between 1 and 10.")
        resp = live_client.query("Why that one?")
        assert resp is not None
        assert resp.strip() != ""


class TestBashTimeoutLive:
    def test_bash_timeout_does_not_hang(self, live_client):
        # The agent should report back rather than hang when a command exceeds
        # its timeout (process-group kill + prompt return).
        resp = live_client.query(
            "Run the bash command 'sleep 8' with a 2 second timeout and tell me what happened."
        )
        assert resp is not None
        assert resp.strip() != ""
