"""Unit tests for the two-destination logging split (utils/logger.py).

The rule under test: the terminal is a conversation, so a record reaches stderr
as ONE line, while the whole record — traceback included — goes to
``~/.mnemoai/logs/mnemoai.log``. Includes the case that motivated it: a
``KeyboardInterrupt`` we inject to cancel a turn escaping into the thread pool,
which logged ``Exception in worker`` through the ROOT logger and printed a full
traceback into the middle of the chat.
"""

import logging
import sys
import threading

import pytest

from mnemoai.utils import logger as logger_mod


@pytest.fixture
def isolated_logging(tmp_path, monkeypatch):
    """Give the test its own app home AND restore all global logging state.

    ``enable_file_logging`` mutates process-wide state (handlers on two loggers,
    both excepthooks, a module global), so a test that didn't undo it would leak
    a handler writing into a deleted tmp dir for the rest of the suite.
    """
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    app = logging.getLogger("ai_app")
    root = logging.getLogger()
    saved = (
        list(app.handlers),
        app.level,
        list(root.handlers),
        root.level,
        sys.excepthook,
        threading.excepthook,
        logger_mod._file_handler,
        getattr(sys, "_mnemoai_excepthooks", False),
        [getattr(h.formatter, "hint", None) for h in app.handlers],
    )
    logger_mod._file_handler = None
    sys._mnemoai_excepthooks = False
    yield tmp_path
    handler = logger_mod._file_handler
    if handler is not None:
        handler.close()
    (
        app.handlers[:],
        app.level,
        root.handlers[:],
        root.level,
        sys.excepthook,
        threading.excepthook,
        logger_mod._file_handler,
        sys._mnemoai_excepthooks,
        hints,
    ) = saved
    for h, hint in zip(app.handlers, hints):
        if hint is not None:
            h.formatter.hint = hint


def _record(message="boom", exc_info=None, level=logging.ERROR):
    return logging.LogRecord("ai_app", level, __file__, 1, message, None, exc_info)


def _exc_info():
    try:
        raise ZeroDivisionError("division by zero")
    except ZeroDivisionError:
        return sys.exc_info()


class TestConsoleFormatter:
    def test_traceback_is_replaced_by_a_pointer(self):
        fmt = logger_mod._ConsoleFormatter("%(message)s", use_color=False)
        fmt.hint = " (traceback → ~/.mnemoai/logs/mnemoai.log)"
        out = fmt.format(_record("Query failed: division by zero", _exc_info()))
        assert out == (
            "✗ Query failed: division by zero "
            "(traceback → ~/.mnemoai/logs/mnemoai.log)"
        )
        assert "Traceback" not in out

    def test_the_line_is_the_interface_not_a_log_line(self):
        # No timestamp, no logger name, no level word: the mark carries the
        # severity, the way console.print_error and /doctor already do.
        fmt = logger_mod._ConsoleFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", use_color=False
        )
        out = fmt.format(_record("Query failed: division by zero"))
        assert out == "✗ Query failed: division by zero"
        assert "ai_app" not in out and "ERROR" not in out

    def test_warning_and_info_get_their_own_marks(self):
        fmt = logger_mod._ConsoleFormatter("%(message)s", use_color=False)
        warn = fmt.format(_record("retrying in 1.0s", level=logging.WARNING))
        info = fmt.format(_record("Loaded 42 tools", level=logging.INFO))
        assert warn == "! retrying in 1.0s"
        assert info == "· Loaded 42 tools"

    def test_color_paints_the_body_by_level_and_the_pointer_dim(self):
        fmt = logger_mod._ConsoleFormatter("%(message)s", use_color=True)
        fmt.hint = " (traceback → log)"
        out = fmt.format(_record("boom", _exc_info()))
        assert out.startswith("\033[91m✗ boom\033[0m")  # red body
        assert out.endswith("\033[90m (traceback → log)\033[0m")  # dim pointer

    def test_multi_line_message_is_collapsed_to_its_first_line(self):
        fmt = logger_mod._ConsoleFormatter("%(message)s", use_color=False)
        out = fmt.format(_record("headline\nline two\nline three"))
        assert out == "✗ headline"

    def test_a_very_long_message_is_capped(self):
        # A provider error can carry a JSON body; unwrapped it fills the window
        # as effectively as the traceback we just moved to the file.
        fmt = logger_mod._ConsoleFormatter("%(message)s", use_color=False)
        fmt.hint = " (see log)"
        out = fmt.format(_record("x" * 2000))
        assert len(out) < 600 and out.endswith("… (see log)")

    def test_plain_record_gets_no_pointer(self):
        fmt = logger_mod._ConsoleFormatter("%(message)s", use_color=False)
        fmt.hint = " (see log)"
        # Nothing was dropped, so nothing is added — the pointer would be noise.
        assert fmt.format(_record("loaded 42 tools")) == "✗ loaded 42 tools"

    def test_verbose_keeps_the_traceback(self):
        # LOG_LEVEL=DEBUG opts back in: at that point the terminal IS the debugger.
        fmt = logger_mod._ConsoleFormatter("%(message)s", use_color=False, verbose=True)
        assert "ZeroDivisionError" in fmt.format(_record("boom", _exc_info()))

    def test_formatting_does_not_consume_the_record(self):
        # Handlers run in order over the SAME record: stripping exc_info in place
        # would take the traceback away from the file handler that runs next.
        record = _record("boom", _exc_info())
        logger_mod._ConsoleFormatter("%(message)s", use_color=False).format(record)
        assert record.exc_info is not None
        assert record.exc_text is None
        assert "ZeroDivisionError" in logging.Formatter("%(message)s").format(record)

    def test_empty_message_falls_back_to_the_exception_type(self):
        fmt = logger_mod._ConsoleFormatter("%(message)s", use_color=False)
        assert fmt.format(_record("", _exc_info())) == "✗ ZeroDivisionError"


class TestOneLine:
    def test_first_line_only(self):
        assert logger_mod.one_line("head\nbody\nmore") == "head"

    def test_capped_with_an_ellipsis(self):
        out = logger_mod.one_line("y" * 50, limit=10)
        assert out == "yyyyyyyyy…" and len(out) == 10

    def test_none_and_blank_are_empty(self):
        assert logger_mod.one_line(None) == ""
        assert logger_mod.one_line("   \n  ") == ""


class TestFileLogging:
    def test_traceback_goes_to_the_file_not_the_console(self, isolated_logging, capsys):
        logger_mod.enable_file_logging()
        try:
            raise RuntimeError("kaboom")
        except RuntimeError as e:
            logger_mod.logger.error(f"Query failed: {e}", exc_info=True)

        err = capsys.readouterr().err
        assert "Query failed: kaboom" in err
        assert "Traceback" not in err and "RuntimeError" not in err

        text = (isolated_logging / "logs" / "mnemoai.log").read_text()
        assert "Traceback (most recent call last)" in text
        assert "RuntimeError: kaboom" in text

    def test_root_records_are_captured_and_never_printed(self, isolated_logging, capsys):
        # The reported bug: the thread pool logs the KeyboardInterrupt we injected
        # to cancel a turn, and with no root handler logging.lastResort printed the
        # whole traceback into the conversation.
        logger_mod.enable_file_logging()
        logging.getLogger("concurrent.futures").critical(
            "Exception in worker", exc_info=(KeyboardInterrupt, KeyboardInterrupt(), None)
        )
        assert capsys.readouterr().err == ""
        text = (isolated_logging / "logs" / "mnemoai.log").read_text()
        assert "Exception in worker" in text
        assert "KeyboardInterrupt" in text

    def test_our_info_records_reach_the_file_while_the_console_stays_quiet(
        self, isolated_logging, capsys
    ):
        logger_mod.enable_file_logging(level=logging.WARNING)
        logger_mod.logger.info("Loaded 42 tools from MCP server")
        assert capsys.readouterr().err == ""
        assert "Loaded 42 tools" in (isolated_logging / "logs" / "mnemoai.log").read_text()

    def test_idempotent(self, isolated_logging):
        first = logger_mod.enable_file_logging()
        before = len(logging.getLogger().handlers)
        assert logger_mod.enable_file_logging() is first
        assert len(logging.getLogger().handlers) == before

    def test_unwritable_home_does_not_raise(self, isolated_logging, monkeypatch):
        # Logging must never be the reason the app won't start.
        monkeypatch.setattr(
            logger_mod, "logger", logging.getLogger("ai_app_test_quiet"), raising=True
        )
        monkeypatch.setattr(
            "mnemoai.utils.paths.logs_dir", lambda: (_ for _ in ()).throw(OSError("ro"))
        )
        assert logger_mod.enable_file_logging() is None

    def test_no_file_until_something_is_logged(self, isolated_logging):
        # delay=True: a run that logs nothing leaves no file behind.
        logger_mod.enable_file_logging()
        assert not (isolated_logging / "logs" / "mnemoai.log").exists()

    def test_console_false_keeps_a_record_off_screen_but_in_the_file(
        self, isolated_logging, capsys
    ):
        # The escape hatch for a failure the UI already reports in its own words:
        # one report on screen, the full record (traceback included) on disk.
        logger_mod.enable_file_logging()
        try:
            raise RuntimeError("kaboom")
        except RuntimeError as e:
            logger_mod.logger.error(
                f"Query failed: {e}", exc_info=True, extra={"console": False}
            )
        assert capsys.readouterr().err == ""
        text = (isolated_logging / "logs" / "mnemoai.log").read_text()
        assert "Query failed: kaboom" in text
        assert "RuntimeError: kaboom" in text

    def test_log_file_hint_is_empty_until_there_is_a_file(self, isolated_logging):
        assert logger_mod.log_file_hint() == ""
        logger_mod.enable_file_logging()
        assert logger_mod.log_file_hint().endswith("logs/mnemoai.log")


class TestExceptHooks:
    def test_thread_crash_is_logged_not_printed(self, isolated_logging, capsys):
        logger_mod.enable_file_logging()

        def boom():
            raise RuntimeError("thread died")

        t = threading.Thread(target=boom, name="worker-1")
        t.start()
        t.join()

        err = capsys.readouterr().err
        assert "Unhandled error in thread worker-1: thread died" in err
        assert "Traceback" not in err
        text = (isolated_logging / "logs" / "mnemoai.log").read_text()
        assert "RuntimeError: thread died" in text

    def test_cancelling_a_turn_is_not_a_crash(self, isolated_logging, capsys):
        # Esc injects KeyboardInterrupt into the worker on purpose; reporting it
        # as an unhandled error would turn every cancel into an error message.
        logger_mod.enable_file_logging()
        hook_args = threading.ExceptHookArgs(
            (KeyboardInterrupt, KeyboardInterrupt(), None, threading.current_thread())
        )
        threading.excepthook(hook_args)
        assert capsys.readouterr().err == ""

    def test_sys_excepthook_logs_one_line(self, isolated_logging, capsys):
        logger_mod.enable_file_logging()
        sys.excepthook(*_exc_info())
        err = capsys.readouterr().err
        assert "Unhandled error: division by zero" in err
        assert "Traceback" not in err
        assert "ZeroDivisionError" in (
            isolated_logging / "logs" / "mnemoai.log"
        ).read_text()
