"""Main entry point for the LangGraph chat application."""

import argparse
import sys
from typing import Any, Optional

# Only LIGHTWEIGHT modules at top level. The heavy LLM/agent stack
# (LangGraphClient/ChatInterface → langchain-core → transformers, multi-second)
# is imported INSIDE main() so the startup spinner can animate during that cost
# instead of the terminal sitting frozen. configurator/console/paths/
# startup_loader are all dependency-free.
from mnemoai.utils.configurator import config_exists, run_first_run_setup
from mnemoai.utils.console import print_error
from mnemoai.utils.logger import enable_file_logging
from mnemoai.utils.paths import seed_example_files
from mnemoai.utils.startup_loader import StartupLoader

# Global client reference for cleanup (typed loosely to avoid a heavy top import).
_client: Optional[Any] = None


def main(
    verbose: bool = False, resume: Optional[str] = None, auto: bool = False
) -> None:
    """Initialize the application and start the chat loop.

    Args:
        verbose: Enable verbose mode to show thinking process
        resume: Resume a previous session from THIS directory — a session id (or
            file path) to restore directly, ``"latest"`` for the most recent, or
            ``"pick"`` to choose from a list.
        auto: Start with auto-approve at its widest tier, as ``/auto all`` does.

    Returns:
        None
    """
    global _client

    loader = StartupLoader().start("Loading libraries")
    try:
        # Heavy imports (the multi-second cost) happen here, under the spinner.
        from mnemoai.client.client import LangGraphClient
        from mnemoai.client.ui.chat_interface import ChatInterface

        # LangGraphClient() spawns the MCP server subprocess (its own cold import
        # of the tool stack); start() connects it, builds the model, inits memory.
        loader.set_phase("Starting tools server")
        _client = LangGraphClient(verbose=verbose)

        loader.set_phase("Connecting model")
        _client.start(verbose)

        if auto:
            # Through the same setter `/auto` uses (which also drops plan mode):
            # a launch flag must not become a second way into the tier, or the
            # two can disagree. Nothing else is needed for the badge — the input
            # line reads the mode off the client on every repaint.
            _client.set_auto_approve_mode("all")

        chat_interface = ChatInterface(_client)
    finally:
        # Clear the spinner line before the welcome banner prints (or on error).
        loader.stop()

    # Resume AFTER the spinner stops: the picker is interactive and replaying a
    # transcript prints to scrollback, neither of which can run under it.
    # Cancelling the picker exits instead of falling through to a fresh session —
    # `--resume` means "resume", so starting a new chat would be a surprise.
    resumed = False
    if resume:
        outcome = _resume_session(_client, resume, chat_interface)
        if outcome == "exit":
            _discard_empty_session(_client)
            return
        resumed = outcome == "resumed"  # "fresh" → let the loop show the banner

    # Register cleanup function using chat interface method. Enable if you need to save conversation automatically on closure
    # atexit.register(lambda: chat_interface.client.save_conversation(chat_interface.chat_timestamp))

    try:
        # On a resume the banner was already printed before the transcript, so the
        # restored conversation ends up directly above the prompt.
        chat_interface.run_chat_loop(welcome=not resumed)
    finally:
        # A launch nobody typed into leaves a turn-less transcript; drop it so
        # empty files don't accumulate until they age out.
        _discard_empty_session(_client)


def _discard_empty_session(client: Any) -> None:
    """Drop this run's session file if nothing was ever asked (best-effort)."""
    log = getattr(getattr(client, "agent", None), "session_log", None)
    if log is None:
        return
    try:
        log.discard_if_empty()
    except Exception:  # noqa: BLE001 — cleanup must never mask a real exit
        pass


def _format_session_label(entry: dict) -> str:
    """One picker row: how long ago, turn count, and the name or opening prompt.

    A ``/rename`` title wins over the prompt preview when there is one — that is
    the entire point of naming a session. A ``/branch`` fork is tagged, because it
    INHERITS its parent's opening prompt — so the preview alone renders a branch
    and the conversation it came from as two identical rows.
    """
    import time

    age = max(0, int(time.time() - entry.get("modified", 0)))
    if age < 3600:
        when = f"{age // 60}m ago"
    elif age < 86400:
        when = f"{age // 3600}h ago"
    else:
        when = f"{age // 86400}d ago"
    # Size the row by the whole RESTORABLE conversation, not just the turns typed
    # in this file: a resumed session inherits its history, so `turns` reported the
    # longest conversation in the list as "1 turn".
    count = entry.get("exchanges") or entry.get("turns", 0)
    forked = entry.get("branched_from") or {}
    if forked.get("through_turn"):
        tag = f" (branch @ turn {forked['through_turn']})"
    elif entry.get("resumed_from"):
        tag = " (continued)"
    else:
        tag = ""
    title = (entry.get("label") or "").strip() or entry.get("preview", "")
    return (
        f"{when:>8}  {count:>3} turn{'s' if count != 1 else ''}  {title}{tag}"
    )


def _resume_session(client: Any, resume: str, chat_interface: Any = None) -> str:
    """Restore a previous session from this directory into the running client.

    ``resume`` is ``"pick"`` (choose from a list), ``"latest"`` (most recent), or
    a session id / file path.

    Returns one of three outcomes, because the caller needs to know both whether
    to continue AND whether the banner was already shown:

    * ``"exit"`` — the user cancelled the picker or named a session that doesn't
      exist. `--resume` is a request to resume, so silently starting a new
      conversation would surprise the user and leave an empty session behind.
    * ``"resumed"`` — restored; the banner was printed here, BEFORE the
      transcript, so the conversation ends up next to the prompt.
    * ``"fresh"`` — nothing to resume (no sessions yet); carry on normally and
      let the caller print the banner.
    """
    from mnemoai.client.session_log import list_sessions
    from mnemoai.client.ui import turn_view
    from mnemoai.client.ui.tui import select_from_list

    sessions = list_sessions()
    if not sessions:
        print_error("No previous sessions found for this directory.")
        return "fresh"  # nothing to resume, but a fresh session is still useful

    target = None
    if resume == "latest":
        target = sessions[0]
    elif resume == "pick":
        chosen = select_from_list(
            "Resume a session in this directory",
            [(s["path"], _format_session_label(s)) for s in sessions],
        )
        if not chosen:
            # Cancelling the picker ABORTS: the user launched the app purely to
            # resume, so silently starting a fresh session would be a surprise
            # (and could add an unwanted empty session).
            return "exit"
        target = next((s for s in sessions if s["path"] == chosen), None)
    else:
        # An explicit id (or a path) — match the id, then fall back to a suffix
        # match so a partial/abbreviated id still resolves. Resolved against ALL
        # sessions, including links the menu collapses away: naming an id asks for
        # that exact point in a resume chain, so hiding a row from the picker must
        # not make it unreachable.
        candidates = list_sessions(limit=10_000, collapse_chains=False)
        target = next(
            (s for s in candidates if resume in (s["session_id"], s["path"])), None
        ) or next((s for s in candidates if s["session_id"].endswith(resume)), None)
        if target is None:
            print_error(f"No session matching '{resume}' in this directory.")
            return "exit"

    # Banner FIRST, then the transcript: the restored conversation must end up
    # directly above the prompt (reading top-to-bottom: logo → your past
    # conversation → prompt), not scrolled off above the logo.
    if chat_interface is not None:
        chat_interface.show_welcome()

    if target and client.resume_session(target["path"]):
        # Full id, not a prefix: `--resume <id>` matches by suffix, so this line is
        # also how you copy the id of the session you're now in.
        print(turn_view.render_session_notice(f"resumed  {target['session_id']}"))
    return "resumed"


def cli() -> None:
    """Console-script entry point (used by the ``mnemoai`` command).

    Parses CLI args and starts the app. Kept zero-arg so it can be referenced
    as ``main:cli`` in pyproject's [project.scripts].
    """
    parser = argparse.ArgumentParser(
        prog="mnemoai", description="Mnemo AI — local agentic AI assistant"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose mode (hide thinking process)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Start with auto-approve on (same as /auto all): file writes, memory "
        "updates and shell commands run without asking, for this session only",
    )
    # `--resume` with no value opens the picker; `--resume <id>` restores that
    # session directly. Scoped to the current directory either way.
    parser.add_argument(
        "--resume",
        nargs="?",
        const="pick",
        metavar="SESSION_ID",
        help="Resume a previous session from this directory "
        "(no value: choose from a list)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_latest",
        action="store_true",
        help="Resume the most recent session from this directory (no prompt)",
    )
    args = parser.parse_args()

    # Before anything can fail: from here on a traceback goes to
    # ~/.mnemoai/logs/mnemoai.log and the terminal gets one line.
    enable_file_logging()

    seed_example_files()

    if not config_exists() and sys.stdin.isatty():
        if run_first_run_setup() is not None:
            from mnemoai.utils.config import config

            config.reload()
        elif not config_exists():
            # Setup was declined/cancelled and there's still no config to run
            # with — exit cleanly rather than crashing deep in client init.
            print_error("No config available. Exiting.")
            return

    # Default is verbose=True, unless --no-verbose is specified
    verbose = not args.no_verbose
    # --continue is the non-interactive form of --resume (most recent session);
    # an explicit --resume value wins if both are given.
    resume = args.resume or ("latest" if args.continue_latest else None)
    main(verbose=verbose, resume=resume, auto=args.auto)


if __name__ == "__main__":
    cli()
