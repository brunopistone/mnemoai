"""Unit tests for the during-tool progress spinner (UX: never look stuck).

While a tool runs (e.g. executing Python via execute_bash, a web fetch, a file
write), the agent shows an animated spinner with a per-tool label so a slow
``tool.invoke()`` never presents a frozen, blank terminal after the user
confirms the command. These tests cover the label builder and that
``_invoke_tool`` starts the spinner, runs the tool, and always stops it.
"""

from mnemoai.client.agent.agent import LangGraphAgent


def _agent():
    return LangGraphAgent.__new__(LangGraphAgent)


class TestToolProgressLabel:
    def test_bash_label_includes_command(self):
        a = _agent()
        label = a._tool_progress_label("execute_bash", {"command": "python run.py"})
        assert "python run.py" in label

    def test_bash_label_truncates_long_command(self):
        a = _agent()
        long_cmd = "python " + "x" * 100
        label = a._tool_progress_label("execute_bash", {"command": long_cmd})
        assert len(label) < 70
        assert label.endswith("…")

    def test_bash_label_without_command(self):
        a = _agent()
        assert a._tool_progress_label("execute_bash", {}) == "Running command"

    def test_write_label_includes_path(self):
        a = _agent()
        label = a._tool_progress_label("fs_write", {"path": "/tmp/x.py"})
        assert "/tmp/x.py" in label

    def test_known_tool_labels(self):
        a = _agent()
        assert a._tool_progress_label("web_search", {}) == "Searching the web"
        assert a._tool_progress_label("describe_image", {}) == "Analyzing image"

    def test_unknown_tool_falls_back_to_name(self):
        a = _agent()
        assert a._tool_progress_label("some_tool", {}) == "Running some_tool"


class _FakeTool:
    def __init__(self, result="ok", boom=False):
        self.result = result
        self.boom = boom
        self.invoked_with = None

    def invoke(self, args):
        self.invoked_with = args
        if self.boom:
            raise RuntimeError("tool failed")
        return self.result


class TestInvokeToolSpinner:
    """_invoke_tool must start the spinner, run the tool, and ALWAYS stop it."""

    def _spy_agent(self):
        a = _agent()
        events = []
        a._start_spinner = lambda label="Thinking": events.append(("start", label))
        a._stop_spinner = lambda: events.append(("stop", None))
        return a, events

    def test_runs_tool_and_returns_result(self):
        a, events = self._spy_agent()
        tool = _FakeTool(result="42")
        out = a._invoke_tool(tool, "execute_bash", {"command": "echo 42"})
        assert out == "42"
        assert tool.invoked_with == {"command": "echo 42"}

    def test_spinner_started_with_label_then_stopped(self):
        a, events = self._spy_agent()
        a._invoke_tool(_FakeTool(), "execute_bash", {"command": "ls"})
        assert events[0][0] == "start"
        assert "ls" in events[0][1]
        assert events[-1] == ("stop", None)

    def test_spinner_stopped_even_on_error(self):
        a, events = self._spy_agent()
        tool = _FakeTool(boom=True)
        try:
            a._invoke_tool(tool, "execute_bash", {"command": "boom"})
        except RuntimeError:
            pass
        # The finally clause must still have stopped the spinner.
        assert ("stop", None) in events
