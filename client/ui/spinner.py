import sys
import threading
import time


class Spinner:
    """Simple spinner for showing processing status."""

    def __init__(self) -> None:
        """Initialize spinner."""
        self.spinning = False
        self.thread = None

    def start(self) -> None:
        """Start the spinner."""
        if self.spinning:
            return
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

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
            sys.stdout.write(f"\r{chars[i % len(chars)]} Thinking{dots}   ")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
