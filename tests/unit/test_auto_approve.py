"""Unit tests for auto-approve mode — the tiers, and the gate they feed.

`/auto` raises how much runs without a confirmation prompt. Because it is the one
feature whose whole job is to STOP asking, the tests are mostly about what it
still refuses to cover: the safety-override category, a write outside the working
directory at the scoped tier, a shell command below the top tier, and anything
already blocked above the gate (plan mode, a hook deny).

Four layers, deliberately separate: the pure ladder (`auto_approve`), the real
`_confirm_tool` ladder driven end-to-end on a bare agent, the mutual exclusion
with plan mode on the client, and the `--auto` launch flag.
"""

import builtins
import os
import sys

import pytest

from mnemoai import main as main_mod
from mnemoai.client import hooks
from mnemoai.client.agent import auto_approve, tool_loop
from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.client import LangGraphClient


class TestTheLadder:
    """The pure mode ladder: normalize, cycle, badge, notice."""

    def test_modes_are_ordered_off_first(self):
        assert auto_approve.MODES[0] == auto_approve.DEFAULT_MODE == "off"

    def test_cycle_walks_the_ladder_and_wraps(self):
        seen = [auto_approve.DEFAULT_MODE]
        for _ in range(len(auto_approve.MODES)):
            seen.append(auto_approve.cycle(seen[-1]))
        assert seen == list(auto_approve.MODES) + ["off"]

    def test_an_unknown_mode_degrades_to_off_not_to_a_wider_tier(self):
        for junk in ("yolo", "", "  ", "ALL_THE_THINGS", None, 3, object()):
            assert auto_approve.normalize(junk) == "off"

    def test_a_known_mode_is_case_and_space_insensitive(self):
        assert auto_approve.normalize("  WRITES ") == "writes"

    def test_cycle_of_junk_starts_from_off(self):
        assert auto_approve.cycle("nonsense") == "edits"

    def test_off_has_no_badge_and_the_others_do(self):
        assert auto_approve.badge("off") is None
        for mode in ("edits", "writes", "all"):
            label, color = auto_approve.badge(mode)
            assert label and color.startswith("ansi")

    def test_every_mode_has_a_notice(self):
        for mode in auto_approve.MODES:
            assert auto_approve.notice(mode)


class TestWhatEachTierCovers:
    """covers(): the category matrix, per tier."""

    def test_off_covers_nothing(self):
        for category in ("write", "memory", "bash", "git"):
            assert auto_approve.covers("off", category, target=os.getcwd()) is False

    def test_edits_covers_only_writes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert auto_approve.covers("edits", "write", target=str(tmp_path / "a.py"))
        assert auto_approve.covers("edits", "memory") is False
        assert auto_approve.covers("edits", "bash") is False

    def test_writes_adds_memory_but_not_bash(self, tmp_path):
        assert auto_approve.covers("writes", "write", target=str(tmp_path / "a.py"))
        assert auto_approve.covers("writes", "memory") is True
        assert auto_approve.covers("writes", "bash") is False

    def test_all_adds_bash(self, tmp_path):
        assert auto_approve.covers("all", "write", target=str(tmp_path / "a.py"))
        assert auto_approve.covers("all", "memory") is True
        assert auto_approve.covers("all", "bash") is True

    def test_the_safety_override_category_is_never_covered(self):
        for mode in auto_approve.MODES:
            assert auto_approve.covers(mode, "git") is False
        assert "git" in auto_approve.NEVER_AUTO

    def test_an_unknown_category_is_not_covered_at_any_tier(self):
        for mode in auto_approve.MODES:
            assert auto_approve.covers(mode, "network") is False


class TestWorkspaceScoping:
    """The `edits` tier is scoped to the working directory; the wider ones aren't."""

    def test_a_write_inside_the_tree_is_covered(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pkg").mkdir()
        assert auto_approve.covers("edits", "write", target="pkg/mod.py")

    def test_the_working_directory_itself_counts_as_inside(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert auto_approve.covers("edits", "write", target=str(tmp_path))

    def test_a_write_outside_the_tree_still_asks(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert auto_approve.covers("edits", "write", target="/etc/hosts") is False
        assert auto_approve.covers("edits", "write", target="../escape.py") is False

    def test_a_tilde_path_outside_the_tree_still_asks(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert auto_approve.covers("edits", "write", target="~/.ssh/authorized_keys") is (
            False
        )

    def test_a_symlink_out_of_the_tree_cannot_smuggle_a_write(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside"
        outside.mkdir()
        inside = tmp_path / "work"
        inside.mkdir()
        monkeypatch.chdir(inside)
        (inside / "link").symlink_to(outside, target_is_directory=True)
        # Textually inside the cwd; resolves outside it, so it must still ask.
        assert auto_approve.covers("edits", "write", target="link/secret.py") is False

    def test_a_sibling_sharing_a_name_prefix_is_not_inside(self, tmp_path, monkeypatch):
        work = tmp_path / "repo"
        work.mkdir()
        sibling = tmp_path / "repo-old"
        sibling.mkdir()
        monkeypatch.chdir(work)
        assert auto_approve.covers("edits", "write", target=str(sibling / "f.py")) is False

    def test_an_unknown_target_asks_rather_than_assuming(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for target in (None, "", "   "):
            assert auto_approve.covers("edits", "write", target=target) is False

    def test_the_wider_tiers_are_not_workspace_scoped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert auto_approve.covers("writes", "write", target="/etc/hosts") is True
        assert auto_approve.covers("all", "write", target="/etc/hosts") is True

    def test_a_wide_tier_still_needs_no_target(self):
        # `writes` is path-agnostic, so a missing path must not make it ask.
        assert auto_approve.covers("writes", "write", target=None) is True

    def test_an_explicit_cwd_is_honored(self, tmp_path):
        assert auto_approve.covers(
            "edits", "write", target=str(tmp_path / "f.py"), cwd=str(tmp_path)
        )
        assert (
            auto_approve.covers("edits", "write", target="/etc/hosts", cwd=str(tmp_path))
            is False
        )


def _agent(monkeypatch, mode, answers=()):
    """A bare agent whose gate sees a TTY, an enabled toggle, and a mode.

    Input is queued rather than stubbed away so that an unexpected PROMPT is a
    hard failure (IndexError) instead of a silently-passing test — the thing
    being asserted is usually that nothing asked.
    """
    a = LangGraphAgent.__new__(LangGraphAgent)
    a._trusted_confirm_categories = set()
    a._stop_spinner = lambda: None
    a._auto_approve_provider = lambda: mode

    import mnemoai.client.agent.agent as mod

    monkeypatch.setattr(mod.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(mod.config, "get", lambda k, d=None: True)

    queued = list(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": queued.pop(0))
    return a


class TestThroughTheRealGate:
    """_confirm_tool end-to-end: the mode reaches the ladder and stops there."""

    def test_a_scoped_edit_runs_without_a_prompt(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        a = _agent(monkeypatch, "edits")
        args = {"path": str(tmp_path / "mod.py"), "command": "create"}
        assert a._confirm_tool("fs_write", args) is True

    def test_an_out_of_tree_edit_still_prompts(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        a = _agent(monkeypatch, "edits", ["n"])
        assert a._confirm_tool("fs_write", {"path": "/etc/hosts", "command": "create"}) is (
            False
        )

    def test_both_write_tools_are_covered(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        a = _agent(monkeypatch, "edits")
        assert a._confirm_tool("fs_write", {"path": "a.py", "command": "create"}) is True
        assert a._confirm_tool("file_edit", {"file_path": "a.py"}) is True

    def test_bash_still_prompts_below_the_top_tier(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        a = _agent(monkeypatch, "writes", ["n"])
        assert a._confirm_tool("execute_bash", {"command": "rm -rf build"}) is False

    def test_bash_runs_at_the_top_tier(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        a = _agent(monkeypatch, "all")
        assert a._confirm_tool("execute_bash", {"command": "rm -rf build"}) is True

    def test_a_memory_write_runs_from_the_writes_tier(self, monkeypatch):
        a = _agent(monkeypatch, "writes")
        assert a._confirm_tool("memory", {"action": "add", "text": "x"}) is True

    def test_a_memory_write_still_prompts_at_the_edits_tier(self, monkeypatch):
        a = _agent(monkeypatch, "edits", ["n"])
        assert a._confirm_tool("memory", {"action": "add", "text": "x"}) is False

    def test_off_prompts_for_everything_as_before(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        a = _agent(monkeypatch, "off", ["n", "n", "n"])
        assert a._confirm_tool("fs_write", {"path": "a.py", "command": "create"}) is False
        assert a._confirm_tool("execute_bash", {"command": "ls"}) is False
        assert a._confirm_tool("memory", {"action": "add", "text": "x"}) is False

    def test_a_safety_override_still_prompts_at_the_top_tier(self, monkeypatch):
        # allow_dangerous=True IS the request to override a server-side refusal;
        # no tier may wave it through.
        a = _agent(monkeypatch, "all", ["n"])
        assert a._confirm_tool("git_commit_safe", {"allow_dangerous": True}) is False

    def test_an_agent_with_no_provider_behaves_exactly_as_before(self, monkeypatch):
        a = _agent(monkeypatch, "all", ["n"])
        del a._auto_approve_provider
        assert a._confirm_tool("execute_bash", {"command": "ls"}) is False

    def test_a_raising_provider_falls_back_to_prompting(self, monkeypatch):
        a = _agent(monkeypatch, "all", ["n"])

        def _boom():
            raise RuntimeError("provider is broken")

        a._auto_approve_provider = _boom
        assert a._confirm_tool("execute_bash", {"command": "ls"}) is False

    def test_the_toggle_still_wins_over_the_mode(self, monkeypatch):
        # A disabled REQUIRE_* toggle already proceeds; the mode must not change
        # that ordering (it sits below the toggle).
        a = _agent(monkeypatch, "off")
        import mnemoai.client.agent.agent as mod

        monkeypatch.setattr(mod.config, "get", lambda k, d=None: False)
        assert a._confirm_tool("execute_bash", {"command": "ls"}) is True

    def test_the_mode_is_read_per_call(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        a = _agent(monkeypatch, "off", ["n"])
        mode = {"v": "off"}
        a._auto_approve_provider = lambda: mode["v"]
        assert a._confirm_tool("execute_bash", {"command": "ls"}) is False
        mode["v"] = "all"  # a mid-session /auto must apply to the next call
        assert a._confirm_tool("execute_bash", {"command": "ls"}) is True


class _Tool:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return f"{self.name}-out"


class TestItCannotWidenAnythingAboveTheGate:
    """The mode replaces a keypress, not a permission: the blocks above still hold."""

    def _loop_agent(self, tool, blocked=False, hook=None):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.verbose = False
        a.tools = [tool]
        a.tools_by_route = None
        a._start_spinner = lambda *x, **k: None
        a._stop_spinner = lambda *x, **k: None
        a._effective_route = lambda state: None
        a._run_spawn_batch = lambda tool_calls: {}
        a._is_blocked_by_plan_mode = lambda *x: blocked
        a._auto_approve_provider = lambda: "all"
        a._trusted_confirm_categories = set()
        a._run_hooks = hook or (lambda *x, **k: hooks.Outcome())
        a._invoke_tool = lambda t, name, args, quiet=False: t.invoke(args)
        a._confirm_tool = lambda name, args: LangGraphAgent._confirm_tool(a, name, args)
        return a

    def _run(self, agent, tool_name, args):
        calls = [{"name": tool_name, "args": args, "id": "t1"}]
        msgs = []
        tool_loop.run_tool_calls(agent, calls, agent.tools, msgs)
        return msgs

    def test_plan_mode_still_blocks_at_the_top_tier(self, monkeypatch):
        tool = _Tool("execute_bash")
        agent = self._loop_agent(tool, blocked=True)
        out = self._run(agent, "execute_bash", {"command": "rm -rf build"})
        assert tool.calls == []  # never ran
        assert len(out) == 1 and out[0].tool_call_id == "t1"

    def test_a_hook_deny_still_blocks_at_the_top_tier(self, monkeypatch):
        tool = _Tool("execute_bash")
        agent = self._loop_agent(
            tool,
            hook=lambda *x, **k: hooks.Outcome(
                decision="deny", reason="denied by policy"
            ),
        )
        out = self._run(agent, "execute_bash", {"command": "rm -rf build"})
        assert tool.calls == []
        assert len(out) == 1 and "denied by policy" in out[0].content


class TestMutualExclusionWithPlanMode:
    """Raising a tier drops plan mode: both on would mean one doing nothing."""

    def _client(self):
        c = LangGraphClient.__new__(LangGraphClient)
        c.plan_mode_active = False
        c.auto_approve_mode = auto_approve.DEFAULT_MODE
        return c

    def test_raising_a_tier_turns_plan_mode_off(self):
        c = self._client()
        c.plan_mode_active = True
        assert c.set_auto_approve_mode("edits") == "edits"
        assert c.plan_mode_active is False

    def test_switching_the_mode_off_leaves_plan_mode_alone(self):
        c = self._client()
        c.plan_mode_active = True
        assert c.set_auto_approve_mode("off") == "off"
        assert c.plan_mode_active is True

    def test_an_unknown_tier_is_normalized_and_leaves_plan_mode_alone(self):
        c = self._client()
        c.plan_mode_active = True
        assert c.set_auto_approve_mode("yolo") == "off"
        assert c.plan_mode_active is True


class TestLaunchFlag:
    """`mnemoai --auto` — the tier as a launch flag, deliberately without tiers.

    The mode is session state, so a run meant to go uninterrupted still opened at
    `off` and had to be raised by hand on the first prompt. The flag is a bare
    switch that lands on the widest tier; the granularity stays in `/auto`.
    """

    class _Loader:
        def start(self, *_):
            return self

        def set_phase(self, *_):
            pass

        def stop(self):
            pass

    def _run_main(self, monkeypatch, **kwargs):
        """Drive `main()` with the heavy pieces stubbed; return the stub client."""
        made = {}

        class _Client:
            def __init__(self, verbose=False):
                self.mode = auto_approve.DEFAULT_MODE
                made["client"] = self

            def start(self, verbose):
                pass

            def set_auto_approve_mode(self, mode):
                self.mode = mode
                return mode

        class _UI:
            def __init__(self, client):
                pass

            def run_chat_loop(self, welcome=True):
                pass

        monkeypatch.setattr(main_mod, "StartupLoader", lambda: self._Loader())
        monkeypatch.setattr(main_mod, "_client", None)
        monkeypatch.setattr("mnemoai.client.client.LangGraphClient", _Client)
        monkeypatch.setattr("mnemoai.client.ui.chat_interface.ChatInterface", _UI)
        main_mod.main(**kwargs)
        return made["client"]

    def test_the_flag_starts_the_session_at_the_widest_tier(self, monkeypatch):
        assert self._run_main(monkeypatch, auto=True).mode == "all"

    def test_without_it_the_session_starts_at_off(self, monkeypatch):
        assert self._run_main(monkeypatch).mode == auto_approve.DEFAULT_MODE

    def _parse(self, monkeypatch, argv):
        """Run `cli()` on argv and return the kwargs it hands `main()`."""
        seen = {}
        monkeypatch.setattr(main_mod, "main", lambda **kw: seen.update(kw))
        monkeypatch.setattr(main_mod, "enable_file_logging", lambda: None)
        monkeypatch.setattr(main_mod, "seed_example_files", lambda: None)
        monkeypatch.setattr(main_mod, "config_exists", lambda: True)
        monkeypatch.setattr(sys, "argv", ["mnemoai", *argv])
        main_mod.cli()
        return seen

    def test_cli_threads_the_flag_through(self, monkeypatch):
        assert self._parse(monkeypatch, ["--auto"])["auto"] is True

    def test_cli_defaults_to_asking(self, monkeypatch):
        assert self._parse(monkeypatch, [])["auto"] is False

    def test_it_composes_with_the_session_flags(self, monkeypatch):
        got = self._parse(monkeypatch, ["--auto", "--continue", "--no-verbose"])
        assert got["auto"] is True
        assert got["resume"] == "latest"
        assert got["verbose"] is False

    def test_it_takes_no_tier_argument(self, monkeypatch):
        # A bare switch on purpose: a tier argument would need validating, and a
        # typo normalized to a wider tier is worse than being told it's wrong.
        with pytest.raises(SystemExit):
            self._parse(monkeypatch, ["--auto", "edits"])
