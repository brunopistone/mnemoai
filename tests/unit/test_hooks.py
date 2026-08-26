"""Unit tests for user tool hooks (client/hooks.py).

The loader is pure and the runner shells out to short commands, so the whole
module is testable without an LLM, an MCP server, or a terminal. What matters
most here is the *gate* semantics — a hook that can widen what the app may do, or
one that can wedge a turn, is a security/reliability bug rather than a cosmetic
one — so those get a test each: deny wins over allow regardless of file order,
a broken or slow hook never blocks, and nothing outside the app home is read.
"""

import json
import sys

import pytest

from mnemoai.client import hooks


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point the app home at a temp dir and drop the cached snapshot."""
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    hooks.reset_cache()
    yield tmp_path
    hooks.reset_cache()


def _write(tmp_path, config: dict):
    """Write a hooks.json into the temp app home and return its path."""
    d = tmp_path / "hooks"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "hooks.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    return p


def _one(event: str, matcher: str, command: str, **extra) -> dict:
    """A config with a single hook."""
    entry = {"type": "command", "command": command, **extra}
    return {"hooks": {event: [{"matcher": matcher, "hooks": [entry]}]}}


# A python one-liner is the only portable way to write a hook body in a test.
_PY = f'"{sys.executable}"'


class TestLoad:
    def test_missing_file_is_not_an_error(self, _isolated_home):
        reg = hooks.load(_isolated_home / "hooks" / "hooks.json")
        assert reg.hooks == () and reg.errors == ()
        assert not reg  # falsy when nothing is configured

    def test_parses_a_hook(self, _isolated_home):
        p = _write(_isolated_home, _one("PreToolUse", "fs_write", "true", timeout=5))
        reg = hooks.load(p)
        assert reg.errors == ()
        assert [(h.event, h.matcher, h.command, h.timeout) for h in reg.hooks] == [
            ("PreToolUse", "fs_write", "true", 5.0)
        ]

    def test_event_map_without_the_hooks_wrapper(self, _isolated_home):
        p = _write(
            _isolated_home,
            {"PostToolUse": [{"matcher": "*", "hooks": [{"command": "true"}]}]},
        )
        reg = hooks.load(p)
        assert len(reg.hooks) == 1 and reg.hooks[0].event == "PostToolUse"

    def test_defaults(self, _isolated_home):
        p = _write(_isolated_home, {"hooks": {"PreToolUse": [{"hooks": [{"command": "true"}]}]}})
        hook = hooks.load(p).hooks[0]
        assert hook.matcher == "*"  # no matcher = every tool
        assert hook.timeout == hooks.DEFAULT_TIMEOUT

    @pytest.mark.parametrize(
        "value,expected",
        [(0, hooks.DEFAULT_TIMEOUT), (-5, hooks.DEFAULT_TIMEOUT),
         ("nope", hooks.DEFAULT_TIMEOUT), (10_000, hooks.MAX_TIMEOUT), (12, 12.0)],
    )
    def test_timeout_is_clamped(self, _isolated_home, value, expected):
        p = _write(_isolated_home, _one("PreToolUse", "*", "true", timeout=value))
        assert hooks.load(p).hooks[0].timeout == expected

    def test_broken_json_is_reported_not_raised(self, _isolated_home):
        d = _isolated_home / "hooks"
        d.mkdir(parents=True)
        (d / "hooks.json").write_text("{not json", encoding="utf-8")
        reg = hooks.load(d / "hooks.json")
        assert reg.hooks == () and len(reg.errors) == 1

    def test_unknown_event_and_type_are_reported(self, _isolated_home):
        p = _write(
            _isolated_home,
            {
                "hooks": {
                    "Whenever": [{"matcher": "*", "hooks": [{"command": "true"}]}],
                    "PreToolUse": [
                        {"matcher": "*", "hooks": [{"type": "wasm", "command": "x"}]},
                        {"matcher": "*", "hooks": [{"command": "true"}]},
                    ],
                }
            },
        )
        reg = hooks.load(p)
        # Reported, never silently ignored — and the valid sibling still loads.
        assert len(reg.errors) == 2
        assert len(reg.hooks) == 1

    def test_comment_keys_are_tolerated(self, _isolated_home):
        p = _write(
            _isolated_home,
            {
                "//": ["explanation"],
                "hooks": {"PreToolUse": [{"//": "note", "matcher": "*",
                                          "hooks": [{"command": "true"}]}]},
            },
        )
        reg = hooks.load(p)
        assert reg.errors == () and len(reg.hooks) == 1

    def test_bundled_example_is_valid(self):
        # The example is what users copy: a parse error in it is a shipped bug.
        from pathlib import Path

        import mnemoai

        example = Path(mnemoai.__file__).parent / "utils" / "hooks.json.example"
        reg = hooks.load(example)
        assert reg.errors == ()
        assert reg.hooks  # and it actually declares some


class TestMatching:
    def test_glob_on_the_tool_name(self, _isolated_home):
        p = _write(_isolated_home, _one("PreToolUse", "fs_*", "true"))
        reg = hooks.load(p)
        assert hooks.matching(reg, "PreToolUse", "fs_write")
        assert not hooks.matching(reg, "PreToolUse", "execute_bash")

    def test_event_is_part_of_the_match(self, _isolated_home):
        p = _write(_isolated_home, _one("PreToolUse", "*", "true"))
        reg = hooks.load(p)
        assert not hooks.matching(reg, "PostToolUse", "fs_write")


class TestRunEvent:
    def test_no_hooks_means_no_decision(self, _isolated_home):
        out = hooks.run_event("PreToolUse", "fs_write", {}, registry=hooks.Registry())
        assert out.decision is None and not out.denied and not out.allowed

    def test_exit_2_denies_with_stderr_as_the_reason(self, _isolated_home):
        cmd = f'{_PY} -c "import sys; sys.stderr.write(\'nope: policy\'); sys.exit(2)"'
        reg = hooks.load(_write(_isolated_home, _one("PreToolUse", "*", cmd)))
        out = hooks.run_event("PreToolUse", "fs_write", {}, registry=reg)
        assert out.denied and "policy" in out.reason

    def test_json_allow_is_honored(self, _isolated_home):
        cmd = f"{_PY} -c \"print('{{\\\"decision\\\": \\\"allow\\\"}}')\""
        reg = hooks.load(_write(_isolated_home, _one("PreToolUse", "*", cmd)))
        assert hooks.run_event("PreToolUse", "execute_bash", {}, registry=reg).allowed

    def test_deny_beats_allow_whatever_the_order(self, _isolated_home):
        # The safe answer must not depend on which hook the user wrote first.
        allow = f"{_PY} -c \"print('{{\\\"decision\\\": \\\"allow\\\"}}')\""
        deny = f'{_PY} -c "import sys; sys.exit(2)"'
        for first, second in ((allow, deny), (deny, allow)):
            reg = hooks.load(
                _write(
                    _isolated_home,
                    {
                        "hooks": {
                            "PreToolUse": [
                                {"matcher": "*", "hooks": [{"command": first},
                                                           {"command": second}]}
                            ]
                        }
                    },
                )
            )
            assert hooks.run_event("PreToolUse", "fs_write", {}, registry=reg).denied

    def test_additional_context_comes_back(self, _isolated_home):
        cmd = f"{_PY} -c \"print('{{\\\"additionalContext\\\": \\\"note this\\\"}}')\""
        reg = hooks.load(_write(_isolated_home, _one("PostToolUse", "*", cmd)))
        out = hooks.run_event("PostToolUse", "fs_write", {}, registry=reg)
        assert out.context == "note this" and out.decision is None

    def test_plain_stdout_is_a_notice_not_model_context(self, _isolated_home):
        # A formatter's chatter belongs in the user's scrollback, not in the prompt.
        reg = hooks.load(_write(_isolated_home, _one("PostToolUse", "*", "echo reformatted")))
        out = hooks.run_event("PostToolUse", "fs_write", {}, registry=reg)
        assert out.context == ""
        assert any("reformatted" in n for n in out.notices)

    def test_a_failing_hook_does_not_block(self, _isolated_home):
        reg = hooks.load(_write(_isolated_home, _one("PreToolUse", "*", "exit 7")))
        out = hooks.run_event("PreToolUse", "fs_write", {}, registry=reg)
        assert not out.denied and out.notices  # reported, not blocking

    def test_a_timeout_does_not_block(self, _isolated_home):
        reg = hooks.load(_write(_isolated_home, _one("PreToolUse", "*", "sleep 5", timeout=1)))
        out = hooks.run_event("PreToolUse", "fs_write", {}, registry=reg)
        assert not out.denied
        assert any("timed out" in n for n in out.notices)

    def test_the_payload_reaches_the_hook_on_stdin(self, _isolated_home):
        cmd = (
            f'{_PY} -c "import json,sys; d=json.load(sys.stdin); '
            "sys.exit(2 if d['tool_input']['path'] == '/x' "
            "and d['hook_event_name'] == 'PreToolUse' "
            "and d['tool_name'] == 'fs_write' else 0)\""
        )
        reg = hooks.load(_write(_isolated_home, _one("PreToolUse", "*", cmd)))
        out = hooks.run_event("PreToolUse", "fs_write", {"path": "/x"}, registry=reg)
        assert out.denied

    def test_event_env_vars_are_exported(self, _isolated_home):
        cmd = 'test "$MNEMOAI_HOOK_EVENT" = PreToolUse && exit 2'
        reg = hooks.load(_write(_isolated_home, _one("PreToolUse", "*", cmd)))
        assert hooks.run_event("PreToolUse", "fs_write", {}, registry=reg).denied

    def test_deny_short_circuits_the_rest(self, _isolated_home, tmp_path):
        marker = tmp_path / "second-ran"
        reg = hooks.load(
            _write(
                _isolated_home,
                {
                    "hooks": {
                        "PreToolUse": [
                            {
                                "matcher": "*",
                                "hooks": [
                                    {"command": "exit 2"},
                                    {"command": f"touch {marker}"},
                                ],
                            }
                        ]
                    }
                },
            )
        )
        assert hooks.run_event("PreToolUse", "fs_write", {}, registry=reg).denied
        assert not marker.exists()  # the blocked call has nothing left to hook

    def test_huge_input_values_are_clipped(self, _isolated_home):
        # An fs_write body can be megabytes; a hook must still get a usable payload.
        payload = hooks._payload("PreToolUse", "fs_write",
                                 {"content": "x" * 100_000, "path": "/x"}, None, "s", "/tmp")
        body = json.loads(payload)
        assert len(body["tool_input"]["content"]) == hooks._MAX_INPUT_VALUE_CHARS
        assert body["tool_input"]["_truncated"] == ["content"]
        assert body["tool_input"]["path"] == "/x"


class TestSnapshot:
    def test_active_is_snapshotted_not_re_read(self, _isolated_home):
        _write(_isolated_home, _one("PreToolUse", "fs_write", "true"))
        assert len(hooks.active().hooks) == 1
        # Editing mid-session must NOT change what is running: hooks are code.
        _write(_isolated_home, _one("PreToolUse", "*", "rm -rf /"))
        assert hooks.active().hooks[0].matcher == "fs_write"
        hooks.reset_cache()
        assert hooks.active().hooks[0].matcher == "*"  # a restart picks it up


class TestRender:
    def test_no_hooks_points_at_the_example(self, _isolated_home):
        text = hooks.render(hooks.Registry(path=str(_isolated_home / "hooks/hooks.json")))
        assert "No hooks configured" in text and "hooks.json.example" in text

    def test_lists_hooks_and_surfaces_errors(self, _isolated_home):
        reg = hooks.Registry(
            hooks=(hooks.Hook("PreToolUse", "fs_write", "ruff format", 5.0),),
            path="/tmp/hooks.json",
            errors=("hooks.json: unknown event 'Nope' — skipped.",),
        )
        text = hooks.render(reg)
        assert "PreToolUse" in text and "fs_write" in text and "ruff format" in text
        assert "unknown event" in text
