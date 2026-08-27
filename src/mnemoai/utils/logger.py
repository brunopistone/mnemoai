"""Logging utilities for the AI application.

Two destinations with deliberately different jobs. **stderr** shares the screen
with the conversation, so a record reads as one line of the interface — a mark,
the message, nothing else — because a stack trace (or a timestamp and a level
name) printed there scrolls the answer away and asks the user to debug the app.
**``~/.mnemoai/logs/mnemoai.log``** carries the whole record, traceback included,
which is what debugging actually needs; it is rotated by size and swept by age so
it can't grow without bound. See :class:`_ConsoleFormatter` and
:func:`enable_file_logging`.
"""

import logging
import logging.handlers
import os
import sys
import threading
import warnings
from typing import Optional

# ANSI colors per level. ERROR/CRITICAL red, WARNING yellow; DEBUG/INFO dim.
_LEVEL_COLORS = {
    logging.WARNING: "\033[93m",   # yellow
    logging.ERROR: "\033[91m",     # red
    logging.CRITICAL: "\033[91m",  # red
}
_DIM = "\033[90m"
_RESET = "\033[0m"

# The mark that opens a console line, in the app's own vocabulary (``✗`` as in
# console.print_error, ``!``/``·`` as in /doctor): the severity is the glyph, so
# the line doesn't spend a word — or a timestamp — spelling it out.
_LEVEL_MARKS = {
    logging.WARNING: "!",
    logging.ERROR: "✗",
    logging.CRITICAL: "✗",
}

# Longest console line before it is cut. A provider error can carry a JSON body
# or a rendered request in its message; unwrapped, one record then fills the
# window as effectively as the traceback we just moved to the file.
_MAX_CONSOLE_CHARS = 500


def one_line(text: str, limit: int = _MAX_CONSOLE_CHARS) -> str:
    """First line of ``text``, capped — the console's share of a long message.

    Used for the console record and for any UI message built from an exception:
    the whole thing is in the log file, so the screen gets the headline.
    """
    head = str(text or "").strip().split("\n", 1)[0].strip()
    if len(head) > limit:
        head = head[: limit - 1].rstrip() + "…"
    return head


class _CursorTracker:
    """Wraps a text stream to remember whether the cursor is mid-line.

    The chat UI streams answer chunks to stdout without a trailing newline, so a
    log record written to stderr afterwards lands ON THE SAME visual line. By
    tracking whether the last character written to stdout was a newline, the log
    handler can prepend one when needed — keeping logs on their own lines.
    """

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        # Start "at line start" so a log before any output doesn't get a blank
        # line prepended.
        self.at_line_start = True

    def write(self, s):
        n = self._wrapped.write(s)
        if s:
            self.at_line_start = s.endswith("\n")
        return n

    def __getattr__(self, name):
        # Delegate everything else (flush, isatty, fileno, encoding, …).
        return getattr(self._wrapped, name)


# Install the tracker on stdout once, so log handlers can consult it. Only wrap
# a real stream (skip when stdout is already wrapped or missing).
if not isinstance(sys.stdout, _CursorTracker) and sys.stdout is not None:
    sys.stdout = _CursorTracker(sys.stdout)


class _NewlineGuardHandler(logging.StreamHandler):
    """StreamHandler that ensures a log record starts on a fresh line.

    If stdout is mid-line (the chat UI streamed text without a trailing
    newline), emit a leading newline to stderr first so the log message isn't
    appended to the user-facing output. No-op when stdout isn't a TTY (piped
    output stays clean) or when already at line start.
    """

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        out = sys.stdout
        try:
            mid_line = (
                isinstance(out, _CursorTracker)
                and not out.at_line_start
                and hasattr(self.stream, "isatty")
                and self.stream.isatty()
            )
            if mid_line:
                self.stream.write("\n")
                self.stream.flush()
                out.at_line_start = True
        except Exception:
            pass
        super().emit(record)


class _ColorFormatter(logging.Formatter):
    """Formatter that colors the whole record by level when writing to a TTY.

    Colors are applied only when the stream is a terminal (``use_color``), so
    redirected/piped logs stay free of ANSI escape codes.
    """

    def __init__(self, fmt: str, use_color: bool) -> None:
        super().__init__(fmt)
        self.use_color = use_color

    def _paint(self, text: str, color: Optional[str]) -> str:
        """Wrap ``text`` in ``color``, or leave it plain off a terminal."""
        return f"{color}{text}{_RESET}" if self.use_color and color else text

    def format(self, record: logging.LogRecord) -> str:
        return self._paint(super().format(record), _LEVEL_COLORS.get(record.levelno))


class _ConsoleFormatter(_ColorFormatter):
    """Renders a record as ONE line of the interface, not as a log line.

    ``✗ Query failed: division by zero`` — the mark carries the severity, so the
    timestamp, logger name and level word that belong in a log file are left in
    the log file. The traceback isn't dropped either, it's moved: the file
    handler writes it in full and the console line ends with a dim pointer to
    that file. ``LOG_LEVEL=DEBUG`` opts back into the whole record on screen — at
    that point the terminal IS the debugger.

    Never formats the record itself outside verbose mode: handlers run in order
    over the SAME object, so clearing ``exc_info`` in place (or letting
    ``Formatter.format`` cache ``exc_text``) would strip the traceback from the
    file handler that runs after this one.
    """

    def __init__(self, fmt: str, use_color: bool, verbose: bool = False) -> None:
        super().__init__(fmt, use_color)
        self.verbose = verbose
        self.hint = ""  # set by enable_file_logging once there's a file to point at

    def format(self, record: logging.LogRecord) -> str:
        if self.verbose:
            return super().format(record)
        message = record.getMessage().strip()
        head = one_line(message)
        buried = bool(record.exc_info or record.exc_text or record.stack_info)
        buried = buried or head != message
        if not head:
            exc = record.exc_info[0] if record.exc_info else None
            head = exc.__name__ if exc is not None else "error"
        mark = _LEVEL_MARKS.get(record.levelno, "·")
        color = _LEVEL_COLORS.get(record.levelno, _DIM)
        line = self._paint(f"{mark} {head}", color)
        if buried and self.hint:
            line += self._paint(self.hint, _DIM)
        return line


class _FileFormatter(logging.Formatter):
    """The whole record for the log file, plus a captured warning's origin.

    ``origin`` (``RuntimeWarning from …/bedrock_converse.py:1270``) is metadata
    about where a warning came from, in the same class as the timestamp and the
    thread name: it belongs on disk, and the console omits it for the same reason
    it omits those — not because it was buried, but because it isn't the message.
    """

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        origin = getattr(record, "origin", "")
        return f"{text}\n  {origin}" if origin else text


def _console_filter(record: logging.LogRecord) -> bool:
    """``extra={"console": False}`` keeps a record out of the interface.

    For a failure the app already reports in the chat in its own words: the
    diagnostic (and its traceback) still belong in the log file, but a second
    line about the same exception is noise, not information.
    """
    return getattr(record, "console", True) is not False


def setup_logger(name: str = "ai_app", level: int = None) -> logging.Logger:
    """Set up and configure a logger.

    Operational diagnostics (model init, tool loading, summary generation,
    etc.) go through this logger to stderr and are **off by default** (level
    WARNING) so the chat UI stays clean; set ``LOG_LEVEL=INFO`` or
    ``LOG_LEVEL=DEBUG`` to surface them for troubleshooting. User-facing output
    (results, prompts, status the user asked for) should use ``print()``
    instead of this logger.

    Args:
        name: The name of the logger
        level: The logging level (defaults to WARNING, or the LOG_LEVEL env var)

    Returns:
        The configured logger
    """
    # Get log level from environment variable if not specified
    if level is None:
        log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, log_level_str, logging.WARNING)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Create console handler and set level. The newline-guard handler ensures a
    # log line never lands inline with streamed chat output (which has no
    # trailing newline) — it prepends a newline when stdout is mid-line.
    if not logger.handlers:
        console_handler = _NewlineGuardHandler(sys.stderr)
        console_handler.setLevel(level)

        # Color the record by level on a TTY (red for errors, yellow for
        # warnings); stay plain when stderr is redirected to a file/pipe. The
        # console formatter renders it as one line of the interface (the
        # traceback goes to the log file) unless LOG_LEVEL=DEBUG asked for it —
        # in which case the fmt below is what gets used.
        use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        formatter = _ConsoleFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            use_color=use_color,
            verbose=level <= logging.DEBUG,
        )

        # Add formatter to handler
        console_handler.setFormatter(formatter)
        console_handler.addFilter(_console_filter)

        # Add handler to logger
        logger.addHandler(console_handler)

    # Suppress Brave Search client logs
    logging.getLogger("brave_search_python_client").setLevel(logging.WARNING)
    logging.getLogger("brave_search_python_client.boot").setLevel(logging.WARNING)

    return logger


# Create a default logger instance
logger = setup_logger()

# A child of ours, so a captured warning inherits the console + file handlers
# (``ai_app`` doesn't propagate) while the file line still names its source.
_warnings_logger = logging.getLogger("ai_app.warnings")

# The file handler, once installed. Module-level so a second call is a no-op:
# every entry point may ask for file logging without coordinating.
_file_handler: Optional[logging.Handler] = None


def _short(path) -> str:
    """``/Users/me/.mnemoai/logs/mnemoai.log`` → ``~/.mnemoai/logs/mnemoai.log``."""
    text = str(path)
    home = os.path.expanduser("~")
    return "~" + text[len(home) :] if home and text.startswith(home) else text


def log_file_hint() -> str:
    """``~/.mnemoai/logs/mnemoai.log``, or ``""`` when nothing is written there.

    For a UI message that reports a failure in its own words and wants to say
    where the details are — the same pointer the console formatter appends.
    """
    if _file_handler is None:
        return ""
    return _short(getattr(_file_handler, "baseFilename", ""))


def enable_file_logging(level: int = None) -> Optional[logging.Handler]:
    """Write diagnostics to ``~/.mnemoai/logs/mnemoai.log`` and keep the screen clean.

    Called once from the entry point (not at import time — a unit test must not
    touch the real app home). Three attachments, each closing a path by which a
    stack trace reached the chat window:

    * **our logger** — it has ``propagate = False``, so the root handler below
      would never see its records;
    * **the ROOT logger** — stdlib and third-party loggers (``asyncio``,
      ``concurrent.futures``, botocore) propagate there, and a root with NO
      handler falls back to :data:`logging.lastResort`, which prints to stderr.
      That is how a cancelled turn's ``Exception in worker`` traceback (the
      thread pool logging a ``KeyboardInterrupt`` we injected on purpose) landed
      in the middle of the conversation. Third-party records are kept at
      ``level`` so the file gets their warnings without their INFO chatter;
    * **the two excepthooks** — a thread that dies or an unhandled crash bypasses
      logging entirely and prints a raw traceback of its own;
    * **``warnings.showwarning``** — same bypass, from a dependency (see
      :func:`_install_warning_capture`).

    Best-effort: an unwritable app home must not stop the app from running, so a
    failure here leaves the console-only setup in place.
    """
    global _file_handler
    if _file_handler is not None:
        return _file_handler
    if level is None:
        name = os.getenv("LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, name, logging.WARNING)
    # Lazy import: paths imports this module at top level, so the dependency can
    # only run in this direction at call time.
    from mnemoai.utils.paths import (
        APP_LOG_BACKUPS,
        APP_LOG_MAX_BYTES,
        app_log_path,
        logs_dir,
    )

    try:
        logs_dir()  # app_log_path deliberately doesn't create it
        handler = logging.handlers.RotatingFileHandler(
            app_log_path(),
            maxBytes=APP_LOG_MAX_BYTES,
            backupCount=APP_LOG_BACKUPS,
            encoding="utf-8",
            delay=True,  # a run that never logs leaves no file behind
        )
    except Exception as exc:  # noqa: BLE001 — logging must not break startup
        logger.debug(f"File logging unavailable: {exc}")
        return None

    # Our own records are worth INFO detail on disk even when the console is
    # quiet at WARNING; the console handler keeps its own (higher) level.
    file_level = min(level, logging.INFO)
    handler.setLevel(file_level)
    handler.setFormatter(
        _FileFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] %(message)s"
        )
    )
    _file_handler = handler

    logger.addHandler(handler)
    if logger.level > file_level:
        logger.setLevel(file_level)
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    # Now that there IS a file, point the console line at it.
    hint = f" (traceback → {_short(app_log_path())})"
    for h in logger.handlers:
        if isinstance(h.formatter, _ConsoleFormatter):
            h.formatter.hint = hint
    _install_excepthooks()
    _install_warning_capture()
    return handler


# Console lines already spent on a captured warning. The stdlib registry dedupes
# per calling module, which a library that resets the filters (or warns from a
# fresh module each turn) defeats — and the same RuntimeWarning printed twice in
# one turn reads as two separate problems.
_seen_warnings: set = set()
_MAX_SEEN_WARNINGS = 256


def _install_warning_capture() -> None:
    """Route ``warnings.warn`` through the logger instead of raw stderr.

    ``warnings.showwarning`` prints four lines of its own — path, line number,
    category, message, then the offending source line — straight to stderr. It is
    the same raw diagnostic in the middle of the conversation the excepthooks
    close, except it comes from a dependency (langchain, botocore), so it can't be
    fixed at the call site. Here it becomes one ``!`` line like any other warning,
    with its origin kept in the log file, and a repeat is file-only.

    Not :func:`logging.captureWarnings`: that routes to the ``py.warnings``
    logger, which propagates to the root — where only the FILE handler lives — so
    the warning would vanish from the screen entirely, and it logs the whole
    pre-formatted multi-line block as the message.

    Best-effort and idempotent: a warning must never be the thing that raises.
    """
    if getattr(sys, "_mnemoai_warning_capture", False):
        return
    sys._mnemoai_warning_capture = True

    def show(message, category, filename, lineno, file=None, line=None) -> None:
        try:
            name = getattr(category, "__name__", None) or str(category)
            text = str(message)
            origin = f"{name} from {filename}:{lineno}"
            first = (origin, text) not in _seen_warnings
            if first and len(_seen_warnings) < _MAX_SEEN_WARNINGS:
                _seen_warnings.add((origin, text))
            _warnings_logger.warning(
                text, extra={"origin": origin, "console": first}
            )
        except Exception:  # noqa: BLE001 — never raise from a warning
            pass

    warnings.showwarning = show


def _install_excepthooks() -> None:
    """Route uncaught exceptions through the logger instead of raw stderr.

    ``threading.excepthook`` and ``sys.excepthook`` both print a traceback
    directly, which is the one thing the console must not show. A
    ``KeyboardInterrupt`` is exempt: it's how a turn is cancelled here, not a
    crash. Idempotent, and falls back to the stdlib hook if logging itself fails.
    """
    if getattr(sys, "_mnemoai_excepthooks", False):
        return
    sys._mnemoai_excepthooks = True

    def thread_hook(args) -> None:
        if args.exc_type is None or issubclass(
            args.exc_type, (KeyboardInterrupt, SystemExit)
        ):
            return
        name = getattr(args.thread, "name", "?")
        try:
            logger.error(
                f"Unhandled error in thread {name}: {args.exc_value}",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        except Exception:  # noqa: BLE001 — never raise from an excepthook
            pass

    def sys_hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            return
        try:
            logger.critical(
                f"Unhandled error: {exc_value}",
                exc_info=(exc_type, exc_value, exc_tb),
            )
        except Exception:  # noqa: BLE001
            sys.__excepthook__(exc_type, exc_value, exc_tb)

    threading.excepthook = thread_hook
    sys.excepthook = sys_hook
