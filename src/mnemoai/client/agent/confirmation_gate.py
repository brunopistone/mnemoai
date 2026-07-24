"""Client-side destructive-tool confirmation gate (agent-arg helpers).

The hard client-side gate that asks the user before running a destructive tool —
shell (``execute_bash``), file writes (``fs_write``/``file_edit``), and memory
writes — each behind its ``REQUIRE_*`` toggle. It must live client-side: the MCP
server is a piped subprocess and can't prompt the terminal. Handles session-trust
("a" = allow this category), headless auto-deny (a background sub-agent has no
TTY), non-TTY auto-proceed, cross-thread prompt serialization, and the
pre-approved-bash bypass (a plan's ``allowed_bash``).

Pure of the prompt mechanics: the functions take the agent as the first arg and
dispatch back through its methods (``_prompt_confirm``, ``_is_headless``,
``_is_preapproved_bash``, spinner) so a test that overrides one on a ``__new__``
stub still intercepts. The agent keeps thin delegating methods and re-exports the
category frozensets as class attributes (the ``plan_policy`` alias pattern). Reads
the shared ``config`` singleton and the real ``sys.stdin`` (the tests' patch
points), so the extraction is transparent to them.
"""

import sys

from mnemoai.utils.config import config

# Destructive-tool confirmation categories (each gated behind a REQUIRE_* toggle).
CONFIRM_BASH_TOOLS = {"execute_bash"}
CONFIRM_WRITE_TOOLS = {"fs_write", "file_edit"}
CONFIRM_MEMORY_TOOLS = {"memory"}


def is_preapproved_bash(agent, command: str) -> bool:
    """True if ``command`` was pre-approved via a plan's ``allowed_bash``.

    A command matches when it equals, or begins with, one of the pre-approved
    entries (so ``pytest`` pre-approves ``pytest tests/unit``). Set only after
    the user approves a plan that declared commands; empty otherwise.
    """
    approved = getattr(agent, "_preapproved_bash", None)
    if not approved:
        return False
    cmd = (command or "").strip()
    if not cmd:
        return False
    return any(cmd == a or cmd.startswith(a + " ") for a in approved)


def confirm(agent, tool_name: str, tool_args: dict) -> bool:
    """Ask the user to approve a destructive tool before it runs.

    Returns True to proceed. Gates shell (``execute_bash``), file writes
    (``fs_write``/``file_edit``), and memory writes, each behind its
    ``REQUIRE_*`` toggle; every other tool proceeds. Enforced client-side (the
    MCP subprocess can't prompt); non-TTY runs auto-proceed.
    """
    if tool_name in CONFIRM_BASH_TOOLS:
        # A command the plan pre-declared (via exit_plan_mode allowed_bash)
        # runs without a prompt — approving the plan approved these.
        if agent._is_preapproved_bash(tool_args.get("command", "")):
            return True
        category, toggle, toggle_default, header, detail = (
            "bash",
            "REQUIRE_BASH_CONFIRMATION",
            True,
            "▶ Run shell command?",
            tool_args.get("command", ""),
        )
    elif tool_name in CONFIRM_WRITE_TOOLS:
        path = tool_args.get("path", "")
        op = tool_args.get("command", "edit")  # fs_write: create/str_replace/…
        category, toggle, toggle_default, header, detail = (
            "write",
            "REQUIRE_WRITE_CONFIRMATION",
            True,
            "▶ Write to file?",
            f"{op} {path}".strip(),
        )
    elif tool_name in CONFIRM_MEMORY_TOOLS:
        # Only the write actions touch the file; a bad/read action proceeds.
        action = (tool_args.get("action") or "").strip().lower()
        if action not in ("add", "replace", "remove"):
            return True
        text = tool_args.get("text") or tool_args.get("old_text") or ""
        category, toggle, toggle_default, header, detail = (
            "memory",
            "REQUIRE_MEMORY_CONFIRMATION",
            False,
            "▶ Update memory?",
            f"{action}: {text[:60]}",
        )
    else:
        return True

    if not config.get(toggle, toggle_default):
        return True
    # Already trusted this session (user answered "a" earlier). A background
    # sub-agent inherits these — a category the user pre-approved runs.
    trusted = getattr(agent, "_trusted_confirm_categories", None)
    if trusted is not None and category in trusted:
        return True
    # Background sub-agent (no TTY of its own): it CANNOT prompt, so an
    # untrusted destructive tool auto-DENIES (the safe direction — never
    # silently run something unattended). It proceeds only via a pre-trusted
    # category above. Keyed thread-local so only the background daemon thread
    # is headless; the foreground turn still prompts normally.
    if agent._is_headless():
        return False
    if not sys.stdin.isatty():
        return True  # non-interactive: can't prompt, don't block

    # Serialize the actual prompt across threads: with concurrent sub-agents
    # two tool calls could otherwise fight for the terminal at once. The lock
    # is absent on bare test objects (built via __new__) — degrade to no lock.
    lock = getattr(agent, "_confirm_lock", None)
    if lock is None:
        return agent._prompt_confirm(header, detail, category)
    with lock:
        # Re-check trust inside the lock: while we waited, a concurrent
        # sub-agent's "a" may have trusted this category — don't re-prompt.
        if category in getattr(agent, "_trusted_confirm_categories", set()):
            return True
        return agent._prompt_confirm(header, detail, category)


def prompt_confirm(agent, header: str, detail: str, category: str) -> bool:
    """Show the actual confirmation prompt and return True to proceed.

    Split out of :func:`confirm` so the interactive part can run under the
    confirm lock (serializing concurrent sub-agent prompts)."""
    # We borrow the terminal for the prompt, so stop the spinner — but
    # remember whether it was running (and its label) so we can put it back
    # afterward. This matters for a QUIET worker that can prompt (a sequential
    # orchestrator step / a foreground sub-agent): nothing else restarts the
    # spinner in that path, so without restoring it here it would stay dead
    # for the rest of the subtask after the first confirmation (the terminal
    # then looks frozen at a bare `>` while work continues). In the foreground
    # `_execute_tools` path the spinner is already stopped before the tool
    # loop, so `was_active` is False and `_invoke_tool` restarts it as before.
    was_active, prev_label = agent._spinner_snapshot()
    agent._stop_spinner()

    def _finish(proceed: bool) -> bool:
        # Hand the spinner back exactly as it was (label preserved).
        if was_active:
            agent._start_spinner(prev_label)
        return proceed

    # The pinned-input UI installs a `_confirm_ui` hook (in-app y/N/a keypress
    # → yes|no|all) since a plain input() would fight the live app for stdin.
    # Absent (plain loop / unit-test bare object) → legacy print()+input().
    confirm_ui = getattr(agent, "_confirm_ui", None)
    if confirm_ui is not None:
        answer = confirm_ui(header, detail, category)
        if answer == "all":
            agent._trusted_confirm_categories.add(category)
            return _finish(True)
        return _finish(answer == "yes")

    # "a" = allow this whole category for the rest of the session.
    print(f"\n\033[93m{header}\033[0m\n  \033[1m{detail}\033[0m")
    try:
        answer = input("  Proceed? (y/N/a=allow all this session): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        answer = ""
    if answer in ("a", "all", "always"):
        if not hasattr(agent, "_trusted_confirm_categories"):
            agent._trusted_confirm_categories = set()
        agent._trusted_confirm_categories.add(category)
        return _finish(True)
    return _finish(answer in ("y", "yes"))
