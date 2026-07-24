"""Minimal startup progress spinner (stdlib only, zero heavy deps).

Shown while ``mnemoai`` boots — importing the LLM/agent stack (transformers via
langchain-core is a multi-second unavoidable floor) and spawning the MCP server
subprocess take ~10-30s, during which the terminal would otherwise sit frozen.
This animates a single ``⠦ <phase>…`` line on a daemon thread so the user sees
what's happening, then clears itself before the welcome banner prints.

Deliberately dependency-free (only ``sys``/``threading``/``time``/``itertools``):
it must be importable and start animating BEFORE the heavy imports run, so it
can't pull any of them in. No-ops off a TTY (pipes/CI) so output stays clean.
"""

import itertools
import sys
import threading
import time

_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class StartupLoader:
    """A single animated status line for the boot sequence.

    Usage::

        loader = StartupLoader()
        loader.start("Loading libraries")
        ...heavy import...
        loader.set_phase("Starting tools server")
        ...client.start()...
        loader.stop()  # clears the line

    Or as a context manager (``stop`` runs on exit, even on error)::

        with StartupLoader() as loader:
            loader.set_phase(...)
    """

    def __init__(self, interval: float = 0.08) -> None:
        self._interval = interval
        self._phase = "Starting Mnemo AI"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Only animate on an interactive terminal; off-TTY (pipe/CI) stays silent
        # so captured output isn't polluted with spinner frames / escape codes.
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def start(self, phase: str | None = None) -> "StartupLoader":
        if phase:
            self._phase = phase
        if not self._tty or self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def set_phase(self, phase: str) -> None:
        """Update the phase text shown next to the spinner (thread-safe)."""
        with self._lock:
            self._phase = phase

    def stop(self) -> None:
        """Stop the animation and clear the line (idempotent)."""
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=1.0)
        if self._tty:
            # Erase the whole line and return the cursor to column 0.
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def _spin(self) -> None:
        # Braille glyph rotates every tick; the trailing dots cycle 0→3 roughly
        # every ~0.3s (same feel as the main-loop spinner), so a long phase reads
        # as live progress rather than a static "…".
        for i in itertools.count():
            if self._stop.is_set():
                return
            with self._lock:
                phase = self._phase
            frame = _FRAMES[i % len(_FRAMES)]
            dots = "." * ((i // 3) % 4)
            sys.stdout.write(f"\r\033[K\033[38;5;63m{frame}\033[0m {phase}{dots}")
            sys.stdout.flush()
            time.sleep(self._interval)

    def __enter__(self) -> "StartupLoader":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
