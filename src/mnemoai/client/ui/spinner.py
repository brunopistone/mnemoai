import sys
import threading
import time


class SpinnerStatus:
    """Thread-safe status shared between the ``Spinner`` (worker thread) and the
    pinned status line (UI thread), which renders the animated frame from it."""

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
    """Processing-status spinner with two rendering modes.

    Default (stdout): animates ``⠋ Thinking…`` via ``\\r`` writes (non-TTY plain
    loop). Sink mode: with a :class:`SpinnerStatus` attached (pinned TTY UI), the
    methods only flip sink state and the pinned status line draws the frame,
    avoiding collision with prompt_toolkit's redraw.
    """

    def __init__(self, sink: SpinnerStatus = None) -> None:
        self.spinning = False
        self.thread = None
        self.label = "Thinking"
        self._sink = sink

    def start(self, label: str = "Thinking") -> None:
        """Start the spinner; ``label`` shows next to the glyph."""
        self.label = label
        if self._sink is not None:
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
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _spin(self) -> None:
        """Spinner animation (stdout mode only)."""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self.spinning:
            dots = "." * ((i // 3) % 4)
            # Clear the line first so a shorter label doesn't leave stale chars.
            sys.stdout.write(f"\r\033[K{chars[i % len(chars)]} {self.label}{dots}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1


# Braille frames shared by the stdout animation and the pinned-toolbar renderer.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def spinner_toolbar_text(status: SpinnerStatus) -> str:
    """Render the current spinner frame for the pinned status line.

    Time-based (advances with the app's ``refresh_interval``, no own thread):
    the glyph rotates and the dots cycle 0→3. Empty string when idle.
    """
    active, label = status.snapshot()
    if not active:
        return ""
    tick = int(time.time() * 10)
    frame = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
    dots = "." * ((tick // 3) % 4)
    return f"{frame} {label}{dots} (esc to cancel)"
