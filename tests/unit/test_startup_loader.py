"""Unit tests for the boot progress spinner (utils/startup_loader.py).

Dependency-free: it must import without pulling any heavy lib, animate only on a
TTY, and be a silent no-op off-TTY (pipes/CI) so captured output stays clean.
"""

import sys
import time

from mnemoai.utils.startup_loader import StartupLoader


class _FakeTTY:
    """A minimal stdout stand-in that reports as a TTY and records writes."""

    def __init__(self):
        self.buf = []

    def isatty(self):
        return True

    def write(self, s):
        self.buf.append(s)

    def flush(self):
        pass


def test_off_tty_is_silent_noop(monkeypatch, capsys):
    # Default test stdout is not a TTY → no thread, no output, no crash.
    loader = StartupLoader()
    assert loader._tty is False
    loader.start("Loading libraries")
    assert loader._thread is None  # never spawned a thread off-TTY
    loader.set_phase("Connecting model")
    loader.stop()
    assert capsys.readouterr().out == ""


def test_stop_is_idempotent_off_tty():
    loader = StartupLoader()
    loader.start()
    loader.stop()
    loader.stop()  # second stop must not raise


def test_on_tty_animates_and_clears(monkeypatch):
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    loader = StartupLoader(interval=0.01)
    assert loader._tty is True
    loader.start("Loading libraries")
    import time

    time.sleep(0.05)  # let the daemon thread emit a few frames
    loader.set_phase("Connecting model")
    time.sleep(0.03)
    loader.stop()
    out = "".join(fake.buf)
    # It animated the phases and cleared the line on stop.
    assert "Loading libraries" in out
    assert "Connecting model" in out
    assert "\r\033[K" in out  # line-clear sequence emitted (frames + final stop)
    assert loader._thread is None  # joined/cleared


def test_trailing_dots_animate(monkeypatch):
    # The trailing dots cycle 0→1→2→3 (like the main-loop spinner) so a long
    # phase reads as live progress, not a static "…".
    import re
    import time

    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    loader = StartupLoader(interval=0.01)
    loader.start("Connecting model")
    time.sleep(0.2)  # cover several dot cycles
    loader.stop()
    seen = set()
    for chunk in fake.buf:
        m = re.search(r"Connecting model(\.*)", chunk)
        if m is not None:
            seen.add(len(m.group(1)))
    assert {0, 1, 2, 3} <= seen  # all four dot counts appeared


def test_context_manager_stops_on_exit(monkeypatch):
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    with StartupLoader(interval=0.01) as loader:
        loader.set_phase("Starting tools server")
        import time

        time.sleep(0.03)
    assert loader._thread is None  # stopped on __exit__
    assert "Starting tools server" in "".join(fake.buf)


class TestWriteAboveDoesNotCollideWithTheSpinner:
    """A startup warning must not be glued onto the live spinner line.

    Observed: an external MCP server failing to start printed
    ``⠸ Connecting model.✗ MCP server 'time' failed to start`` — the error was
    appended straight onto the animating line with no erase, so the message read
    as if the MODEL had failed to connect, and a stale spinner line was left
    stranded above the welcome banner.
    """

    def _run(self, monkeypatch, emit):
        fake = _FakeTTY()
        monkeypatch.setattr(sys, "stdout", fake)
        loader = StartupLoader(interval=0.005).start("Connecting model")
        try:
            time.sleep(0.05)  # let some frames land
            emit(loader)
            time.sleep(0.05)  # let the spinner resume
        finally:
            loader.stop()
        return "".join(fake.buf)

    def test_message_is_not_appended_to_a_spinner_frame(self, monkeypatch):
        out = self._run(monkeypatch, lambda ldr: ldr.write_above("BOOM"))
        at = out.index("BOOM")
        # Everything from the message to its newline must be the message alone —
        # no phase text trailing it on the same line.
        tail = out[at : out.index("\n", at)]
        assert "Connecting model" not in tail

    def test_the_spinner_line_is_erased_before_the_message(self, monkeypatch):
        out = self._run(monkeypatch, lambda ldr: ldr.write_above("BOOM"))
        assert "\r\033[KBOOM" in out

    def test_the_message_survives_in_scrollback(self, monkeypatch):
        out = self._run(monkeypatch, lambda ldr: ldr.write_above("BOOM"))
        assert "BOOM\n" in out

    def test_print_error_routes_through_the_active_loader(self, monkeypatch):
        # The real call site is utils.console.print_error, deep inside
        # client.start() — it must get the same treatment without knowing about
        # the loader.
        from mnemoai.utils.console import print_error

        out = self._run(
            monkeypatch,
            lambda ldr: print_error("MCP server 'time' failed to start; skipping."),
        )
        at = out.index("failed to start")
        assert "Connecting model" not in out[at : out.index("\n", at)]

    def test_console_is_unhooked_after_stop(self, monkeypatch):
        from mnemoai.utils import console

        fake = _FakeTTY()
        monkeypatch.setattr(sys, "stdout", fake)
        loader = StartupLoader(interval=0.005).start("x")
        assert console._active_loader is loader
        loader.stop()
        # A dangling loader would swallow every later message into a spinner
        # that no longer paints.
        assert console._active_loader is None

    def test_write_above_off_tty_is_a_plain_print(self, capsys):
        # Off-TTY there is no thread and no animation; the message must still
        # reach stdout exactly once.
        loader = StartupLoader()
        loader.start("x")
        loader.write_above("plain message")
        loader.stop()
        assert capsys.readouterr().out == "plain message\n"

    def test_print_error_without_a_loader_still_prints(self, capsys):
        from mnemoai.utils.console import print_error

        print_error("standalone")
        assert "standalone" in capsys.readouterr().out
