"""Guards that this tier does not write into the developer's real environment.

These are the meta-tests for conftest.py's own isolation. They matter because
this tier drives a REAL client and a REAL MCP subprocess, so every side effect
is a real one — and the two halves are isolated by DIFFERENT mechanisms:

  * the app home by ``$MNEMOAI_HOME`` (an env var the subprocess inherits), and
  * project-scoped lookups by the process cwd, which no env var redirects.

Each guard below is a regression for isolation that was actually broken. They
make no model call, so they stay cheap enough to run with the tier.
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestAppHomeIsRedirected:
    def test_resolved_paths_are_not_in_the_real_home(self, _isolated_app_home):
        from mnemoai.utils import paths

        real = Path.home() / ".mnemoai"
        for resolved in (paths.app_home(), paths.profile_dir(), paths.sessions_dir()):
            assert real not in resolved.parents and resolved != real, resolved

    def test_no_project_steering_is_discovered(self, _isolated_app_home):
        # Run from the checkout, discovery finds this repo's own CLAUDE.md (tens
        # of KB) and prepends it to every live query: real tokens, and the
        # routing assertions stop testing what they were written for (a long
        # prefix makes "Hello" no longer look trivial).
        from mnemoai.utils import paths

        assert paths.steering_files() == []


class TestServerSubprocessIsIsolated:
    """The MCP server is a separate process; env vars reach it, cwd does not.

    It inherits cwd ONCE at spawn and keeps it for life, so the per-test chdir
    cannot reach back into it — ``live_client`` has to start it from the neutral
    dir. Without that the server ran with cwd = the checkout, and a
    relative-path ``fs_write`` created files in the developer's repo (observed).
    """

    @staticmethod
    def _tool(client, name):
        tools = {t.name: t for t in client.tools}
        if name not in tools:
            pytest.skip(f"{name} not enabled in this config")
        return tools[name]

    def test_server_cwd_is_not_the_checkout(self, live_client, _neutral_cwd):
        result = self._tool(live_client, "execute_bash").invoke({"command": "pwd"})
        payload = json.loads(result) if isinstance(result, str) else result
        server_cwd = Path(payload["stdout"].strip()).resolve()
        assert server_cwd != _REPO_ROOT
        assert _REPO_ROOT not in server_cwd.parents
        assert server_cwd == Path(os.getcwd()).resolve()

    def test_a_relative_write_lands_outside_the_checkout(
        self, live_client, _neutral_cwd
    ):
        result = self._tool(live_client, "fs_write").invoke(
            {
                "command": "create",
                "path": "tier_isolation_probe.txt",
                "content": "probe\n",
            }
        )
        payload = json.loads(result) if isinstance(result, str) else result
        assert payload.get("success"), payload
        written = Path(payload["path"]).resolve()
        assert _REPO_ROOT not in written.parents, f"wrote into the repo: {written}"
        assert not (_REPO_ROOT / "tier_isolation_probe.txt").exists()
