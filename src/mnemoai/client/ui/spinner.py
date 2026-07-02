import sys
import threading
import time


class SpinnerStatus:
    """Shared, thread-safe status for the pinned-input spinner.

    In pinned-input mode the spinner can't write ``\\r`` to stdout — that fights
    prompt_toolkit's redraw of the pinned prompt. Instead the ``Spinner`` updates
    this holder (active flag + label) and the prompt's ``bottom_toolbar`` renders
    the animated frame from it. A single instance is shared between the callback
    handler's ``Spinner`` (writer, worker thread) and the toolbar (reader, UI
    thread); reads/writes are trivially atomic here but guarded for clarity.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = False
        self.label = "Thinking"

    def set(self, active: bool, label: str = None) -> None:
        with self._lock:
            self.active = active
            if label is not None:
                self.label = label

    def snapshot(self) -> tuple:
        """Return (active, label) atomically for rendering."""
        with self._lock:
            return self.active, self.label


class Spinner:
    """Processing-status spinner.

    Two rendering modes:
    - **Default (stdout):** animates ``⠋ Thinking…`` on the current line via
      ``\\r`` writes — used by the non-TTY plain loop where nothing else owns the
      terminal.
    - **Sink mode:** when a :class:`SpinnerStatus` sink is attached (the pinned
      TTY UI), ``start``/``set_label``/``stop`` only flip the sink's state — no
      thread, no stdout writes — and the pinned status line draws the frame. This
      keeps the spinner from colliding with the pinned prompt's redraw.
    """

    def __init__(self, sink: SpinnerStatus = None) -> None:
        """Initialize spinner.

        Args:
            sink: Optional shared status holder. When provided, the spinner runs
                in state-only mode (no stdout writes); otherwise it animates on
                stdout as before.
        """
        self.spinning = False
        self.thread = None
        self.label = "Thinking"
        self._sink = sink

    def start(self, label: str = "Thinking") -> None:
        """Start the spinner.

        Args:
            label: Text shown next to the animated glyph (e.g. a phase like
                "Summarizing 12 older messages"). Defaults to "Thinking".
        """
        self.label = label
        if self._sink is not None:
            # State-only: the toolbar animates; nothing to write here.
            self._sink.set(True, label)
            return
        if self.spinning:
            return
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def set_label(self, label: str) -> None:
        """Update the label on a running spinner (e.g. to show a new phase)."""
        self.label = label
        if self._sink is not None:
            self._sink.set(True, label)

    def stop(self) -> None:
        """Stop the spinner."""
        if self._sink is not None:
            self._sink.set(False)
            return
        self.spinning = False
        if self.thread:
            self.thread.join()
            self.thread = None
        # Clear the entire line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _spin(self) -> None:
        """Spinner animation (stdout mode only)."""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self.spinning:
            dots = "." * ((i // 3) % 4)  # 0, 1, 2, 3 dots cycling
            # Clear the line first so a shorter label doesn't leave stale chars.
            sys.stdout.write(f"\r\033[K{chars[i % len(chars)]} {self.label}{dots}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1


# Braille frames shared by the stdout animation and the pinned-toolbar renderer.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def spinner_toolbar_text(status: SpinnerStatus) -> str:
    """Render the current spinner frame for a prompt ``bottom_toolbar``.

    Time-based animation so it advances with the prompt's ``refresh_interval``
    without needing its own thread: the braille glyph rotates AND the trailing
    dots cycle 0→1→2→3 (``Thinking`` → ``Thinking...``), matching the classic
    stdout spinner. Returns an empty string when idle (toolbar shows nothing).
    """
    active, label = status.snapshot()
    if not active:
        return ""
    tick = int(time.time() * 10)
    frame = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
    dots = "." * ((tick // 3) % 4)  # 0,1,2,3 dots cycling, like the old spinner
    return f"{frame} {label}{dots} (esc to cancel)"
