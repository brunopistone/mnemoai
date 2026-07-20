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
        # web_crawler shows a generic label (no URL — it can be very long).
        assert a._tool_progress_label("web_crawler", {"url": "http://x"}) == (
            "Crawling the page"
        )

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

    def test_web_crawler_animates_spinner(self):
        # web_crawler no longer prints its own terminal progress (its stderr goes
        # to the MCP log), so it must animate the spinner like any slow tool —
        # started with the crawl label, then stopped.
        a, events = self._spy_agent()
        out = a._invoke_tool(_FakeTool(result="page"), "web_crawler", {"url": "http://x"})
        assert out == "page"
        assert events[0] == ("start", "Crawling the page")
        assert events[-1] == ("stop", None)

    def test_no_self_reporting_tools(self):
        # The self-reporting carve-out is empty now (web_crawler was removed).
        assert LangGraphAgent._SELF_REPORTING_TOOLS == set()


class TestConfirmRestoresSpinner:
    """A confirmation prompt borrows the terminal (stops the spinner). In the
    QUIET worker path (a sequential orchestrator step / foreground sub-agent)
    nothing else restarts it, so `_prompt_confirm` must hand it back — otherwise
    the spinner stays dead for the rest of the subtask and the terminal looks
    frozen at a bare `>` after the first confirmation (the reported bug).
    """

    def _agent(self, active, label="step 1/2: analyze"):
        a = LangGraphAgent.__new__(LangGraphAgent)
        events = []
        a._spinner_snapshot = lambda: (active, label)
        a._start_spinner = lambda lbl="Thinking": events.append(("start", lbl))
        a._stop_spinner = lambda: events.append(("stop", None))
        a._confirm_ui = lambda h, d, c: "yes"
        a._trusted_confirm_categories = set()
        return a, events

    def test_spinner_restored_after_confirm_when_it_was_running(self):
        a, events = self._agent(active=True, label="step 1/2: analyze")
        assert a._prompt_confirm("Run shell command?", "ls", "bash") is True
        # It stopped to prompt, then restarted with the SAME label.
        assert ("stop", None) in events
        assert events[-1] == ("start", "step 1/2: analyze")

    def test_spinner_not_restarted_when_it_was_idle(self):
        # Foreground _execute_tools already stopped the spinner before the tool
        # loop, so it's idle here — _invoke_tool restarts it, not _prompt_confirm.
        a, events = self._agent(active=False)
        a._prompt_confirm("Run shell command?", "ls", "bash")
        assert all(e[0] != "start" for e in events)  # never restarted

    def test_decline_still_restores_spinner(self):
        a, events = self._agent(active=True, label="step 2/2: write")
        a._confirm_ui = lambda h, d, c: "no"
        assert a._prompt_confirm("Write to file?", "x.py", "write") is False
        assert events[-1] == ("start", "step 2/2: write")

    def test_allow_all_restores_spinner(self):
        a, events = self._agent(active=True, label="step 1/2: analyze")
        a._confirm_ui = lambda h, d, c: "all"
        assert a._prompt_confirm("Run shell command?", "ls", "bash") is True
        assert "bash" in a._trusted_confirm_categories
        assert events[-1] == ("start", "step 1/2: analyze")


class _AnswerResponse:
    """Minimal stand-in for an AIMessage with visible content, no tool calls."""

    def __init__(self, content="done"):
        self.content = content
        self.tool_calls = []
        self.additional_kwargs = {}


class TestCallModelStartsSpinner:
    """The final answer turn must spin while waiting for the model's first token.

    Regression: _call_model used to rely on the preceding tool node leaving the
    spinner running. Once each tool call stopped its own spinner on completion,
    the wait between the last tool result and the final answer showed a frozen
    terminal. _call_model now starts the spinner at entry itself.
    """

    def test_spinner_started_before_streaming(self):
        a = _agent()
        events = []
        a.system_prompt = None
        a.callbacks = []
        a._start_spinner = lambda label="Thinking": events.append("start")
        a._get_route_model = lambda state: object()

        def _fake_stream(messages, config, model=None, mark_answer=False):
            # The spinner must already have been started by the time we stream.
            events.append("stream")
            return _AnswerResponse(), False

        a._stream_response = _fake_stream
        a._extract_thinking = lambda r: None
        a._extract_visible = lambda c: "done"

        a._call_model({"messages": [], "route": None})

        assert events[0] == "start"
        assert "stream" in events
        assert events.index("start") < events.index("stream")


class _Chunk:
    """Minimal streaming-chunk stand-in for _stream_once.

    ``content`` is a plain string (Ollama-style). ``tool_call_chunks`` mimics
    langchain's streamed tool-call argument fragments. Chunks add together (the
    accumulation `response = response + chunk` in _stream_once), so `__add__`
    just returns the right-hand chunk — enough for these tests.
    """

    def __init__(self, content="", tool_call_chunks=None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []
        self.additional_kwargs = {}

    def __add__(self, other):
        return other


class _StubModel:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, messages, config=None):
        yield from self._chunks


class TestToolArgStreamingSpinner:
    """A tool call streams its args as content-less chunks; for a large arg
    (fs_write's file_text of a big document) that is a long silent stretch.
    After reasoning stopped the spinner, _stream_once must re-raise a
    "Preparing tool call" spinner so the terminal never looks frozen.
    """

    def _agent_recording(self, chunks):
        a = _agent()
        a.verbose = True
        a.callbacks = [object()]  # non-empty so the spinner logic engages
        a._code_formatter = None  # not touched (no visible content in these tests)
        events = []
        a._start_spinner = lambda label="Thinking": events.append(("start", label))
        a._stop_spinner = lambda: events.append(("stop", None))
        return a, events

    def test_spinner_restarts_while_building_tool_args(self):
        # reasoning chunk (stops spinner) → several content-less tool-arg chunks.
        # Reasoning chunk: Ollama-style reasoning arrives via additional_kwargs.
        r = _Chunk(content="")
        r.additional_kwargs = {"reasoning_content": "thinking..."}
        arg1 = _Chunk(content="", tool_call_chunks=[{"args": '{"file_text": "'}])
        arg2 = _Chunk(content="", tool_call_chunks=[{"args": "big doc..."}])
        a, events = self._agent_recording([r, arg1, arg2])

        a._stream_once(_StubModel([r, arg1, arg2]), [], {}, print_reasoning=True)

        # The spinner must have been (re)started with the preparing label after
        # reasoning stopped it — exactly once, not per-chunk.
        starts = [e for e in events if e[0] == "start"]
        assert ("start", "Preparing tool call") in starts
        assert sum(1 for e in starts if e[1] == "Preparing tool call") == 1

    def test_no_preparing_spinner_without_tool_calls(self):
        # A plain reasoning-then-answer stream must NOT trigger the preparing
        # spinner (no tool_call_chunks anywhere).
        r = _Chunk(content="")
        r.additional_kwargs = {"reasoning_content": "thinking..."}
        ans = _Chunk(content="hello")
        a, events = self._agent_recording([r, ans])
        # process_chunk is called on visible content; stub the formatter.
        a._code_formatter = type("F", (), {"process_chunk": lambda self, c: None,
                                            "flush": lambda self: None})()

        a._stream_once(_StubModel([r, ans]), [], {}, print_reasoning=True,
                       mark_answer=True)

        assert not any(e[1] == "Preparing tool call" for e in events if e[0] == "start")


class TestToolMarkerStyled:
    """_print_tool_marker renders the styled block in pinned mode and the plain
    [⚙ …] marker otherwise — shared by _execute_tools AND the worker loop so a
    'full'-route turn doesn't fall back to the old marker (UX regression)."""

    def _agent(self, styled):
        a = LangGraphAgent.__new__(LangGraphAgent)
        a.styled_turn_view = styled
        return a

    def test_styled_renders_name_and_arg_block(self, capsys):
        a = self._agent(styled=True)
        a._print_tool_marker({"name": "fs_read", "args": {"path": "/x.pdf", "mode": "PDF"}})
        out = capsys.readouterr().out
        assert "fs_read" in out
        assert "↳ path=/x.pdf" in out and "↳ mode=PDF" in out
        assert "[⚙" not in out  # NOT the old marker

    def test_unstyled_uses_gear_marker(self, capsys):
        a = self._agent(styled=False)
        a._print_tool_marker({"name": "fs_read", "args": {"path": "/x.pdf"}})
        out = capsys.readouterr().out
        assert "[⚙" in out
        assert "fs_read" in out

    def test_missing_styled_attr_defaults_to_plain(self, capsys):
        a = LangGraphAgent.__new__(LangGraphAgent)  # no styled_turn_view set
        a._print_tool_marker({"name": "web_search", "args": {}})
        assert "[⚙" in capsys.readouterr().out
