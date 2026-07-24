"""Unit tests for the boot progress spinner (utils/startup_loader.py).

Dependency-free: it must import without pulling any heavy lib, animate only on a
TTY, and be a silent no-op off-TTY (pipes/CI) so captured output stays clean.
"""

import sys

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
