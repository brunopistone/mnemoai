"""Unit tests for how a multi-step plan reports progress.

The scheduler has two ways to show a plan: a LIVE checklist in the pinned region
(one block whose rows tick from [ ] to [✓] as work lands) and, when there is no
transient region — the plain off-TTY loop — a block printed per wave plus one
line per completion, because a printed block is frozen the moment it lands.
These drive :meth:`_run_subtasks_scheduled` with stub subtasks (no LLM, no MCP).
"""

import pytest

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.ui.turn_view import StepStatus


class _Recorder(StepStatus):
    """A live sink that keeps what the UI would have shown at each tick."""

    def __init__(self):
        super().__init__()
        self.snapshots = []

    def mark_done(self, index):
        super().mark_done(index)
        self.snapshots.append(self.render())


def _agent(sink=None, workers=1, boom=False):
    a = LangGraphAgent.__new__(LangGraphAgent)
    a.verbose = True
    a._max_subagent_concurrency = workers
    if sink is not None:
        a.steps_sink = sink
    a._start_spinner = lambda *_a, **_k: None
    a._stop_spinner = lambda *_a, **_k: None
    a._set_headless = lambda *_a, **_k: None

    def _run(i, subtasks, results, history=None):
        if boom:
            raise RuntimeError("worker died")
        return {"task": subtasks[i]["description"], "result": "ok"}

    a._run_subtask = _run
    return a


def _plan(n=3, chain=False):
    steps = [{"description": f"Produce artifact {i}"} for i in range(n)]
    if chain:
        for i in range(1, n):
            steps[i]["depends_on"] = [i - 1]
    return steps


def _rows(block: str) -> list:
    return [ln for ln in block.split("\n")[1:] if "[" in ln]


class TestLiveChecklist:
    """With a sink the EXISTING rows tick; nothing is appended per completion."""

    def test_finished_steps_are_checked_in_place(self, capsys):
        sink = _Recorder()
        agent = _agent(sink, workers=3)
        agent._run_subtasks_scheduled(_plan(3))
        # One row per step at every tick — the whole plan stays one block.
        assert [len(_rows(s)) for s in sink.snapshots] == [3, 3, 3]
        # …and the checks accumulate inside it.
        assert [s.count("[✓]") for s in sink.snapshots] == [1, 2, 3]

    def test_no_tick_lines_are_printed(self, capsys):
        # The bug: a wave of 5 printed five green "[ ]" rows and then appended
        # "[✓] 1/5 …", "[✓] 2/5 …" below them instead of checking those rows.
        agent = _agent(_Recorder(), workers=3)
        agent._run_subtasks_scheduled(_plan(3))
        out = capsys.readouterr().out
        assert "1/3 Produce" not in out and "2/3 Produce" not in out

    def test_the_finished_plan_is_still_committed_to_scrollback(self, capsys):
        # The live block is transient, so the closing one is the plan's only
        # permanent record — exactly one block, every step checked.
        agent = _agent(_Recorder(), workers=3)
        agent._run_subtasks_scheduled(_plan(3))
        out = capsys.readouterr().out
        assert out.count("Steps") == 1
        assert "3/3" in out and out.count("[✓]") == 3

    def test_a_sequential_wave_runs_one_step_at_a_time(self, capsys):
        # Three independent steps, one worker: they run one after another, so
        # marking the whole wave green would claim work that hasn't started.
        sink = _Recorder()
        agent = _agent(sink, workers=1)
        agent._run_subtasks_scheduled(_plan(3))
        first = sink.snapshots[0]
        assert first.count("[✓]") == 1
        assert first.count("\033[32m[ ]") == 0  # steps 2-3 are still pending

    def test_the_region_is_handed_back_when_the_plan_ends(self, capsys):
        sink = _Recorder()
        agent = _agent(sink, workers=3)
        agent._run_subtasks_scheduled(_plan(3))
        assert sink.active is False and sink.render() == ""

    def test_a_failed_step_also_hands_the_region_back(self, capsys):
        # Else the checklist stays pinned above the prompt for the rest of the
        # session, frozen on a plan that is no longer running.
        sink = _Recorder()
        agent = _agent(sink, workers=1, boom=True)
        with pytest.raises(RuntimeError):
            agent._run_subtasks_scheduled(_plan(2, chain=True))
        assert sink.active is False and sink.render() == ""


class TestPinnedRegion:
    """The live block has to actually reach the screen: a sink nothing renders
    would silently replace the printed checklist with nothing at all."""

    def _controls(self, app):
        from prompt_toolkit.formatted_text import ANSI
        from prompt_toolkit.layout.controls import FormattedTextControl

        out = []
        for window in app.layout.find_all_windows():
            control = window.content
            if isinstance(control, FormattedTextControl):
                text = control.text() if callable(control.text) else control.text
                if isinstance(text, ANSI):
                    out.append(str(text.value))
        return out

    def test_the_checklist_provider_is_rendered_above_the_prompt(self):
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from mnemoai.client.ui.tui import PinnedPromptReader

        sink = StepStatus()
        with create_pipe_input() as inp, create_app_session(
            input=inp, output=DummyOutput()
        ):
            reader = PinnedPromptReader(
                prompt_text=lambda: "> ",
                commands=[],
                dispatch=lambda line: None,
                steps_text=sink.render,
            )
            app = reader._build_app()
            assert not any(self._controls(app))  # idle: the region is empty
            sink.start(["Read the config", "Run the tests"])
            sink.set_running([0])
            assert any("Steps" in text for text in self._controls(app))

    def test_the_pinned_loop_connects_the_agent_to_the_region(self):
        # Both halves live in one place, and half a chain fails silently: a sink
        # nothing renders shows no plan at all, and a region nothing feeds
        # replaces the printed checklist with an empty row.
        import inspect

        from mnemoai.client.ui.chat_interface import ChatInterface

        source = inspect.getsource(ChatInterface._run_pinned_loop)
        assert "steps_sink" in source and "steps_text" in source


class TestPrintedChecklist:
    """Without a sink (off-TTY plain loop) the per-wave block + ticks stay."""

    def test_a_wave_ticks_each_completion_on_its_own_line(self, capsys):
        agent = _agent(workers=3)
        agent._run_subtasks_scheduled(_plan(3))
        out = capsys.readouterr().out
        assert "1/3 Produce" in out and "2/3 Produce" in out
        # Wave block + closing block.
        assert out.count("Steps") == 2

    def test_a_lone_step_is_not_ticked(self, capsys):
        # The next block (or the closing one) marks it a moment later.
        agent = _agent(workers=3)
        agent._run_subtasks_scheduled(_plan(2, chain=True))
        out = capsys.readouterr().out
        assert "1/2 Produce" not in out
        assert "2/2" in out  # the closing block still says the plan finished
