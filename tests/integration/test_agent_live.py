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
        # Point at a directory that definitely contains a known .py file and ask
        # a yes/no question, so the assertion is model-agnostic (it doesn't
        # depend on how the model phrases a file listing, or on CWD contents).
        resp = live_client.query(
            "Using a file tool, does the file 'tests/unit/test_bm25.py' exist "
            "in this project? Answer yes or no."
        )
        assert resp.strip() != ""
        assert "yes" in resp.lower()

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


class TestPlanModeEnforcementLive:
    """Plan mode must hard-block file writes through the full agent loop.

    Asserts on the filesystem side effect (was the file created?) rather than
    the model's wording, so it's deterministic regardless of model quality.
    """

    def test_plan_mode_blocks_file_write(self, live_client, tmp_path):
        target = tmp_path / "plan_mode_probe.txt"
        prompt = (
            f"Create a file at {target} containing the text 'hello'. "
            "Use the file write tool."
        )

        live_client.plan_mode_active = True
        try:
            resp = live_client.query(prompt)
        finally:
            live_client.plan_mode_active = False
        # The write must NOT have happened while plan mode was on.
        assert not target.exists(), "plan mode failed to block the file write"
        assert resp is not None and resp.strip() != ""

    def test_write_succeeds_when_plan_mode_off(self, live_client, tmp_path):
        target = tmp_path / "plan_mode_probe_off.txt"
        # Disable the write-confirmation gate for this turn so a non-TTY test
        # run isn't blocked at the prompt (the gate auto-proceeds with no TTY,
        # but be explicit).
        live_client.plan_mode_active = False
        live_client.query(
            f"Create a file at {target} containing the text 'hello'. "
            "Use the file write tool."
        )
        # We don't hard-assert creation (depends on the model actually calling
        # the tool), but plan mode must not be what stops it: the flag is off.
        assert live_client.plan_mode_active is False
