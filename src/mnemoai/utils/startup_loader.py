"""Minimal startup progress spinner (stdlib only, zero heavy deps).

Shown while ``mnemoai`` boots — importing the LLM/agent stack (transformers via
langchain-core is a multi-second unavoidable floor) and spawning the MCP server
subprocess take ~10-30s, during which the terminal would otherwise sit frozen.
This animates a single ``⠦ <phase>…`` line on a daemon thread so the user sees
what's happening, then clears itself before the welcome banner prints.

Deliberately dependency-free (stdlib plus ``utils.console``, which is itself
import-free): it must be importable and start animating BEFORE the heavy imports
run, so it can't pull any of them in. No-ops off a TTY (pipes/CI) so output stays
clean.

A message printed by anything else while this is animating has to go through
``write_above`` (``utils.console`` routes ``print_error``/``print_success`` there
automatically) — a bare ``print`` gets chased by the next frame and leaves a
stale spinner line behind.
"""

import itertools
import sys
import threading
import time

from mnemoai.utils import console

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
        self._paused = False
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
        # Startup warnings reach the terminal via utils.console; point it here so
        # they suspend the animation instead of racing it.
        console.set_active_loader(self)
        return self

    def set_phase(self, phase: str) -> None:
        """Update the phase text shown next to the spinner (thread-safe)."""
        with self._lock:
            self._phase = phase

    def write_above(self, text: str) -> None:
        """Print ``text`` as a permanent line without the spinner clobbering it.

        A plain ``print()`` from under the spinner leaves the animation running:
        the message ends with a newline, the spinner's next frame repaints
        ``\\r\\033[K<phase>`` on that NEW line, and ``stop()`` only erases the
        final line — so a stale ``⠿ Connecting model…`` is stranded in the
        scrollback, which reads as "the model never connected" even though the
        real message was an unrelated warning.

        So: suspend the animation, erase the spinner line, emit the text, then
        resume. Off-TTY (or when not animating) this is just a ``print``.
        """
        if not self._tty or self._thread is None:
            print(text)
            return
        # Hold the paint lock so no frame can land between the erase and the
        # write; _spin() takes the same lock around its own output.
        with self._lock:
            self._paused = True
            sys.stdout.write("\r\033[K")
            sys.stdout.write(text.rstrip("\n") + "\n")
            sys.stdout.flush()
            self._paused = False

    def stop(self) -> None:
        """Stop the animation and clear the line (idempotent)."""
        self._stop.set()
        # Unhook FIRST: once the thread is going away, console output must go
        # straight to stdout rather than through a loader that no longer paints.
        console.set_active_loader(None)
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
            frame = _FRAMES[i % len(_FRAMES)]
            dots = "." * ((i // 3) % 4)
            # Paint under the lock so a write_above() erase+emit can't be split
            # by a frame — otherwise the frame lands on the new line and the
            # message scrolls away with a spinner glued to it.
            with self._lock:
                if not self._paused:
                    sys.stdout.write(
                        f"\r\033[K\033[38;5;63m{frame}\033[0m {self._phase}{dots}"
                    )
                    sys.stdout.flush()
            time.sleep(self._interval)

    def __enter__(self) -> "StartupLoader":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
