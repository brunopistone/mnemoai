"""Unit tests for the /doctor self-check (client/doctor.py).

``render`` is pure and every collector is a small function over a stubbed config
or a stubbed client, so the whole report is testable with no model, no MCP
subprocess and no config.yaml. The checks worth pinning are the ones where a
wrong answer is worse than no answer: a report that says everything is fine when
a binary is missing, or that calls a *failed* MCP server "not configured", sends
the user looking in the wrong place.
"""

import socket

import pytest

from mnemoai.client import doctor
from mnemoai.client.doctor import FAIL, INFO, OK, WARN, Check


class FakeConfig:
    """Stand-in for the Config singleton: dotted lookups over a plain dict."""

    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def _resolve_config_path(self):
        return self.values.get("__config_path__")

    def _resolve_prompts_path(self):
        return self.values.get("__prompts_path__")


@pytest.fixture
def cfg(monkeypatch):
    """Install a FakeConfig into the doctor module and hand it back."""
    fake = FakeConfig()
    monkeypatch.setattr(doctor, "config", fake)
    return fake


class TestRender:
    def test_header_counts_problems_and_warnings(self):
        checks = [
            Check("A", "one", FAIL),
            Check("A", "two", WARN),
            Check("A", "three", OK),
        ]
        assert "1 problem, 1 warning" in doctor.render(checks, color=False)

    def test_header_is_clean_when_nothing_is_wrong(self):
        text = doctor.render([Check("A", "one", OK), Check("A", "two", INFO)], color=False)
        assert "everything checks out" in text

    def test_warnings_only(self):
        text = doctor.render([Check("A", "one", WARN), Check("A", "two", WARN)], color=False)
        assert "2 warnings" in text and "problem" not in text

    def test_sections_group_and_fixes_follow_their_check(self):
        checks = [
            Check("Install", "python", INFO, "3.12"),
            Check("Tools", "rg", FAIL, "not on PATH", "Install rg."),
        ]
        lines = doctor.render(checks, color=False).splitlines()
        assert "  Install" in lines and "  Tools" in lines
        rg = next(i for i, line in enumerate(lines) if "rg" in line)
        assert "→ Install rg." in lines[rg + 1]

    def test_color_false_emits_no_ansi(self):
        text = doctor.render([Check("A", "one", FAIL, "d", "f")], color=False)
        assert "\033[" not in text

    def test_color_true_paints_the_marks(self):
        assert "\033[" in doctor.render([Check("A", "one", FAIL)], color=True)


class TestConfigChecks:
    def test_no_config_anywhere_is_a_failure(self, cfg):
        checks = doctor._config_checks()
        config_check = next(c for c in checks if c.name == "config.yaml")
        assert config_check.status == FAIL and config_check.fix

    def test_reports_the_file_that_was_actually_loaded(self, cfg, tmp_path, monkeypatch):
        loaded = tmp_path / "elsewhere.yaml"
        loaded.write_text("MODEL_ID: {}", encoding="utf-8")
        cfg.values["__config_path__"] = loaded
        cfg.values["MODEL_ID"] = {"TYPE": "ollama"}
        monkeypatch.setattr(doctor, "config_path", lambda: tmp_path / "config" / "config.yaml")
        monkeypatch.delenv("MNEMOAI_CONFIG", raising=False)

        detail = next(c for c in doctor._config_checks() if c.name == "config.yaml").detail
        # "I edited config.yaml and nothing changed" — say which file is live.
        assert "elsewhere.yaml" in detail and "NOT" in detail

    def test_env_override_is_named_as_such(self, cfg, tmp_path, monkeypatch):
        loaded = tmp_path / "env.yaml"
        loaded.write_text("MODEL_ID: {}", encoding="utf-8")
        cfg.values.update({"__config_path__": loaded, "MODEL_ID": {"TYPE": "ollama"}})
        monkeypatch.setenv("MNEMOAI_CONFIG", str(loaded))
        detail = next(c for c in doctor._config_checks() if c.name == "config.yaml").detail
        assert "$MNEMOAI_CONFIG" in detail

    def test_a_config_without_a_model_section_is_a_failure(self, cfg, tmp_path, monkeypatch):
        loaded = tmp_path / "config.yaml"
        loaded.write_text("{}", encoding="utf-8")
        cfg.values["__config_path__"] = loaded
        monkeypatch.setattr(doctor, "config_path", lambda: loaded)
        names = {c.name: c for c in doctor._config_checks()}
        assert names["MODEL_ID"].status == FAIL

    def test_resolved_ignores_a_path_that_is_not_a_file(self, tmp_path):
        assert doctor._resolved(lambda: tmp_path / "nope.yaml") is None
        assert doctor._resolved(lambda: None) is None

    def test_resolved_tolerates_a_resolver_that_raises(self):
        def boom():
            raise RuntimeError("no")

        assert doctor._resolved(boom) is None


class TestProviderChecks:
    def test_no_provider_is_a_single_failure(self, cfg):
        checks = doctor._provider_checks()
        assert len(checks) == 1 and checks[0].status == FAIL

    def test_model_label_includes_the_protocol(self, cfg):
        cfg.values["MODEL_ID"] = {
            "TYPE": "mantle",
            "NAME": "claude-opus-5",
            "API_PROTOCOL": "anthropic",
        }
        detail = next(c for c in doctor._provider_checks() if c.name == "model").detail
        assert "mantle" in detail and "claude-opus-5" in detail and "anthropic" in detail

    def test_prompt_cache_off_explains_where_it_works(self, cfg):
        cfg.values["MODEL_ID"] = {"TYPE": "ollama", "NAME": "qwen3"}
        cache = next(c for c in doctor._provider_checks() if c.name == "prompt cache")
        assert cache.status == INFO and "bedrock" in cache.fix

    def test_prompt_cache_on_reports_the_ttl(self, cfg):
        cfg.values["MODEL_ID"] = {
            "TYPE": "bedrock",
            "NAME": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "PROMPT_CACHE_TTL": "1h",
        }
        cache = next(c for c in doctor._provider_checks() if c.name == "prompt cache")
        assert cache.status == OK and "1h" in cache.detail

    def test_a_missing_api_key_is_a_failure_with_the_variable_named(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        check = doctor._credentials_check("openai", {})
        assert check.status == FAIL and "OPENAI_API_KEY" in check.fix

    def test_a_local_openai_endpoint_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        check = doctor._credentials_check("openai", {"API_BASE": "http://localhost:8080/v1"})
        assert check.status == OK and "localhost:8080" in check.detail

    def test_a_key_in_the_config_is_never_printed(self):
        check = doctor._credentials_check("anthropic", {"API_KEY": "sk-secret-value"})
        assert check.status == OK and "sk-secret-value" not in check.detail

    def test_litellm_manages_its_own_auth(self):
        assert doctor._credentials_check("litellm", {}).status == INFO


class TestProbePort:
    def test_a_closed_port_fails_with_the_fix(self):
        with socket.socket() as s:  # bind, don't listen: nothing will connect
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        check = doctor._probe_port("Provider", "ollama server", "127.0.0.1", port, "Start it.")
        assert check.status == FAIL and check.fix == "Start it."

    def test_an_open_port_is_ok(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            check = doctor._probe_port("Provider", "srv", "127.0.0.1", port, "")
        assert check.status == OK


class TestToolChecks:
    def test_a_missing_required_binary_fails(self, monkeypatch):
        monkeypatch.setattr(doctor.shutil, "which", lambda _b: None)
        checks = {c.name: c for c in doctor._tool_checks(None)}
        assert checks["rg"].status == FAIL  # grep_search has no fallback
        assert checks["git"].status == WARN
        assert checks["bash"].status == WARN

    def test_a_client_with_no_tools_is_a_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor, "mcp_config_path", lambda: tmp_path / "mcp.json")

        class Client:
            tools = []

        checks = {c.name: c for c in doctor._tool_checks(Client())}
        assert checks["MCP tools"].status == FAIL

    def test_no_client_means_the_mcp_line_is_skipped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor, "mcp_config_path", lambda: tmp_path / "mcp.json")
        assert not [c for c in doctor._tool_checks(None) if c.name == "MCP tools"]


class TestExternalMcpChecks:
    def _mcp_json(self, tmp_path, monkeypatch, servers):
        import json

        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
        monkeypatch.setattr(doctor, "mcp_config_path", lambda: path)
        return path

    def test_nothing_declared_says_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(doctor, "mcp_config_path", lambda: tmp_path / "absent.json")
        assert doctor._external_mcp_checks(object()) == []

    def test_a_declared_but_unconnected_server_warns(self, tmp_path, monkeypatch):
        self._mcp_json(tmp_path, monkeypatch, {"weather": {"command": "x"}})

        class Client:
            class mcp_client:
                _members = [("builtin", object())]

        check = doctor._external_mcp_checks(Client())[0]
        # A failed server must not read as "not configured": that's the whole point.
        assert check.status == WARN and "weather" in check.detail

    def test_all_connected_is_ok(self, tmp_path, monkeypatch):
        self._mcp_json(tmp_path, monkeypatch, {"weather": {"command": "x"}})

        class Client:
            class mcp_client:
                _members = [("builtin", object()), ("weather", object())]

        check = doctor._external_mcp_checks(Client())[0]
        assert check.status == OK and "weather" in check.detail

    def test_without_a_live_client_it_says_not_checked(self, tmp_path, monkeypatch):
        self._mcp_json(tmp_path, monkeypatch, {"weather": {"command": "x"}})
        check = doctor._external_mcp_checks(None)[0]
        # Not a warning: we simply have nothing to compare against.
        assert check.status == INFO and "not checked" in check.detail

    def test_a_disabled_entry_is_not_expected_to_connect(self, tmp_path, monkeypatch):
        self._mcp_json(tmp_path, monkeypatch, {"weather": {"command": "x", "disabled": True}})
        assert doctor._declared_mcp_servers() == []

    def test_unparsable_mcp_json_is_reported_not_raised(self, tmp_path, monkeypatch):
        path = tmp_path / "mcp.json"
        path.write_text("{oops", encoding="utf-8")
        monkeypatch.setattr(doctor, "mcp_config_path", lambda: path)
        assert doctor._declared_mcp_servers() == ["(unparsable mcp.json)"]


class TestFeatureChecks:
    def test_a_feature_that_is_on_gets_its_dependency_checked(self, cfg):
        cfg.values.update({"ENABLE_RAG": True, "RAG.VECTOR_STORE": "faiss"})
        names = [c.name for c in doctor._feature_checks()]
        assert any("faiss" in n for n in names)

    def test_everything_off_checks_nothing(self, cfg):
        assert doctor._feature_checks() == []

    def test_web_search_without_a_key_warns(self, cfg, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        cfg.values["ENABLE_WEB_SEARCH"] = True
        check = next(c for c in doctor._feature_checks() if c.name == "web search")
        assert check.status == WARN

    def test_web_search_with_a_key_is_silent(self, cfg, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        cfg.values.update({"ENABLE_WEB_SEARCH": True, "BRAVE_API_KEY": "abc"})
        assert not [c for c in doctor._feature_checks() if c.name == "web search"]

    def test_import_check_reports_a_missing_module(self):
        check = doctor._import_check("Features", "nope", "definitely_not_a_module_xyz")
        assert check.status == FAIL and "pip install" in check.fix


class TestSizeChecks:
    def _stub_stores(self, monkeypatch, memory_text, steering=()):
        monkeypatch.setattr(doctor, "MemoryStore", lambda: type("S", (), {"read": lambda _s: memory_text})())
        monkeypatch.setattr(doctor, "SteeringStore", lambda: type("S", (), {"sizes": lambda _s: list(steering)})())

    def test_a_nearly_full_memory_file_warns(self, cfg, monkeypatch):
        # Silently trimming at the cap is the failure mode; 99% is one fact away.
        cfg.values["MEMORY.MAX_CHARS"] = 100
        self._stub_stores(monkeypatch, "x" * 95)
        check = next(c for c in doctor._size_checks() if c.name == "MEMORY.md")
        assert check.status == WARN and "Nearly full" in check.fix

    def test_a_small_memory_file_is_ok(self, cfg, monkeypatch):
        cfg.values["MEMORY.MAX_CHARS"] = 100
        self._stub_stores(monkeypatch, "x" * 10)
        check = next(c for c in doctor._size_checks() if c.name == "MEMORY.md")
        assert check.status == OK and not check.fix

    def test_over_the_cap_says_it_is_being_trimmed(self, cfg, monkeypatch):
        cfg.values["MEMORY.MAX_CHARS"] = 100
        self._stub_stores(monkeypatch, "x" * 150)
        check = next(c for c in doctor._size_checks() if c.name == "MEMORY.md")
        assert check.status == WARN and "trims" in check.fix

    def test_steering_files_are_listed_with_their_injected_size(self, cfg, monkeypatch):
        from pathlib import Path

        self._stub_stores(monkeypatch, "", [(Path("/repo/STEERING.md"), "abcde")])
        steering = [c for c in doctor._size_checks() if c.name == "steering"]
        assert len(steering) == 1 and "5 chars" in steering[0].detail

    def test_a_broken_store_does_not_break_the_report(self, cfg, monkeypatch):
        def boom():
            raise RuntimeError("no store")

        monkeypatch.setattr(doctor, "MemoryStore", boom)
        monkeypatch.setattr(doctor, "SteeringStore", boom)
        assert doctor._size_checks() == []


class TestCommandChecks:
    """A rejected command file is exactly the invisible failure /doctor is for:
    you type ``/deploy``, the line goes to the model as prose, and nothing ever
    said the file was skipped."""

    def _store(self, monkeypatch, tmp_path, files):
        from mnemoai.client import user_commands

        root = tmp_path / "commands"
        root.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (root / name).write_text(text)
        user_commands._SCAN_CACHE.clear()
        monkeypatch.setattr(
            doctor, "UserCommandStore", lambda: user_commands.UserCommandStore(root=root)
        )

    def test_loaded_commands_are_named(self, monkeypatch, tmp_path):
        self._store(monkeypatch, tmp_path, {"deploy.md": "ship it", "review.md": "look"})
        check = next(c for c in doctor._command_checks() if c.name == "your commands")
        assert check.status == OK
        assert "/deploy" in check.detail and "/review" in check.detail

    def test_a_long_list_is_counted_not_dumped(self, monkeypatch, tmp_path):
        self._store(monkeypatch, tmp_path, {f"c{i}.md": "body" for i in range(9)})
        check = next(c for c in doctor._command_checks() if c.name == "your commands")
        assert "+3" in check.detail

    def test_a_rejected_file_warns_with_the_reason_and_a_fix(self, monkeypatch, tmp_path):
        self._store(monkeypatch, tmp_path, {"compact.md": "shadowing a built-in"})
        skipped = [c for c in doctor._command_checks() if c.name == "command skipped"]
        assert len(skipped) == 1
        assert skipped[0].status == WARN
        assert "built-in" in skipped[0].detail and skipped[0].fix

    def test_no_commands_means_no_row(self, monkeypatch, tmp_path):
        self._store(monkeypatch, tmp_path, {})
        assert doctor._command_checks() == []

    def test_a_broken_store_does_not_break_the_report(self, monkeypatch):
        def boom():
            raise RuntimeError("no commands dir")

        monkeypatch.setattr(doctor, "UserCommandStore", boom)
        assert doctor._command_checks() == []


class TestLogCheck:
    def test_reports_the_path_size_and_retention(self, cfg, tmp_path, monkeypatch):
        cfg.values["LOG_MAX_AGE_DAYS"] = 14
        log = tmp_path / "logs" / "mnemoai.log"
        log.parent.mkdir()
        log.write_bytes(b"x" * 4096)
        monkeypatch.setattr(doctor, "app_log_path", lambda: log)
        check = doctor._log_check()
        assert check.status == INFO
        assert "4 KB" in check.detail and "kept 14 days" in check.detail

    def test_zero_says_the_sweep_is_off(self, cfg, tmp_path, monkeypatch):
        cfg.values["LOG_MAX_AGE_DAYS"] = 0
        monkeypatch.setattr(doctor, "app_log_path", lambda: tmp_path / "nope.log")
        assert "never expired" in doctor._log_check().detail

    def test_a_missing_log_is_not_a_problem(self, cfg, tmp_path, monkeypatch):
        # delay=True — a run that never logged has no file, which is the good case.
        monkeypatch.setattr(doctor, "app_log_path", lambda: tmp_path / "nope.log")
        check = doctor._log_check()
        assert check.status == INFO and "empty" in check.detail

    def test_a_junk_retention_value_falls_back(self, cfg, tmp_path, monkeypatch):
        cfg.values["LOG_MAX_AGE_DAYS"] = "soon"
        monkeypatch.setattr(doctor, "app_log_path", lambda: tmp_path / "nope.log")
        assert "kept 7 days" in doctor._log_check().detail


class TestSteeringEntry:
    def test_path_and_text(self):
        assert doctor._steering_entry(("/a/STEERING.md", "abc")) == ("/a/STEERING.md", 3)

    def test_path_and_count(self):
        assert doctor._steering_entry(("/a/STEERING.md", 42)) == ("/a/STEERING.md", 42)

    def test_anything_else_is_skipped(self):
        assert doctor._steering_entry("nope") == (None, 0)


class TestCollectAndReport:
    def test_collect_runs_with_no_client_and_no_config(self, cfg, monkeypatch, tmp_path):
        monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
        monkeypatch.setattr(doctor, "mcp_config_path", lambda: tmp_path / "mcp.json")
        checks = doctor.collect(None)
        assert checks and all(isinstance(c, Check) for c in checks)
        assert {"Install", "Configuration", "Provider", "Tools"} <= {c.section for c in checks}

    def test_report_survives_a_check_that_breaks(self, monkeypatch):
        def boom(_client=None):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(doctor, "collect", boom)
        text = doctor.report(None)
        # A diagnostic that dies becomes the problem being diagnosed.
        assert "could not complete" in text and "kaboom" in text
