import sys
import threading
import time


class Spinner:
    """Simple spinner for showing processing status."""

    def __init__(self) -> None:
        """Initialize spinner."""
        self.spinning = False
        self.thread = None
        self.label = "Thinking"

    def start(self, label: str = "Thinking") -> None:
        """Start the spinner.

        Args:
            label: Text shown next to the animated glyph (e.g. a phase like
                "Summarizing 12 older messages"). Defaults to "Thinking".
        """
        self.label = label
        if self.spinning:
            return
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def set_label(self, label: str) -> None:
        """Update the label on a running spinner (e.g. to show a new phase)."""
        self.label = label

    def stop(self) -> None:
        """Stop the spinner."""
        self.spinning = False
        if self.thread:
            self.thread.join()
        # Clear the entire line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _spin(self) -> None:
        """Spinner animation."""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self.spinning:
            dots = "." * ((i // 3) % 4)  # 0, 1, 2, 3 dots cycling
            # Clear the line first so a shorter label doesn't leave stale chars.
            sys.stdout.write(f"\r\033[K{chars[i % len(chars)]} {self.label}{dots}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
