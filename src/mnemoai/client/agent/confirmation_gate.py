"""Client-side destructive-tool confirmation gate (agent-arg helpers).

The hard client-side gate that asks the user before running a destructive tool —
shell (``execute_bash``), file writes (``fs_write``/``file_edit``), and memory
writes — each behind its ``REQUIRE_*`` toggle. It must live client-side: the MCP
server is a piped subprocess and can't prompt the terminal. Handles session-trust
("a" = allow this category), the session's :mod:`auto_approve` mode, headless
auto-deny (a background sub-agent has no TTY), non-TTY auto-proceed, cross-thread
prompt serialization, and the pre-approved-bash bypass (a plan's ``allowed_bash``).

Pure of the prompt mechanics: the functions take the agent as the first arg and
dispatch back through its methods (``_prompt_confirm``, ``_is_headless``,
``_is_preapproved_bash``, spinner) so a test that overrides one on a ``__new__``
stub still intercepts. The agent keeps thin delegating methods and re-exports the
category frozensets as class attributes (the ``plan_policy`` alias pattern). Reads
the shared ``config`` singleton and the real ``sys.stdin`` (the tests' patch
points), so the extraction is transparent to them.
"""

import json
import sys

from mnemoai.client.agent import auto_approve, plan_policy
from mnemoai.utils.config import config

# Destructive-tool confirmation categories (each gated behind a REQUIRE_* toggle).
CONFIRM_BASH_TOOLS = {"execute_bash"}
CONFIRM_WRITE_TOOLS = {"fs_write", "file_edit"}
CONFIRM_MEMORY_TOOLS = {"memory"}

# Some tools can only tell a call is dangerous once they inspect the world (is
# this branch pushed? is this a protected branch?), so they refuse and answer
# ``requires_confirmation`` instead of acting. That payload used to go to the
# MODEL, whose only documented next step was to re-call with
# ``allow_dangerous=True`` — i.e. the model approved its own dangerous call and
# no human was ever in the loop. Both halves are gated here instead: the flag is
# confirmed before the call, and the refusal payload is confirmed after it.
CONFIRM_RESULT_CATEGORY = "git"
_RESULT_APPROVAL_REASON = "User approved this operation at the confirmation prompt."
_RESULT_TOGGLE = "REQUIRE_GIT_CONFIRMATION"


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
    # Arg-driven and tool-agnostic, so it comes first: ``allow_dangerous=True``
    # IS the request to override a server-side safety refusal. The MCP server
    # can see the flag but not the human, so the flag itself has to be confirmed
    # here or the model can wave through anything it was just refused.
    if tool_args.get("allow_dangerous"):
        return _gated_prompt(
            agent,
            CONFIRM_RESULT_CATEGORY,
            _RESULT_TOGGLE,
            True,
            "▶ Override safety check?",
            f"{tool_name} {tool_args.get('command') or ''}".strip(),
        )

    target = None  # the path a write would touch, for auto-approve scoping
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
        # Both spellings: file_edit calls it file_path, fs_write calls it path.
        # Reading one made the prompt read a bare "edit" with no filename.
        path = plan_policy.write_target(tool_args)
        target = path
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

    return _gated_prompt(
        agent, category, toggle, toggle_default, header, detail, target=target
    )


def _auto_approved(agent, category: str, target) -> bool:
    """True when the session's auto-approve mode covers this call.

    Reads the mode through the agent's provider (the ``plan_mode_provider``
    pattern: a lambda onto ``client.auto_approve_mode``), so the live value is
    picked up per call and a bare test object without one simply has no mode.
    Non-raising — a broken provider must not take a tool call down, and the
    prompt is the safe answer.
    """
    provider = getattr(agent, "_auto_approve_provider", None)
    if provider is None:
        return False
    try:
        mode = provider()
    except Exception:
        return False
    return auto_approve.covers(mode, category, target=target)


def _gated_prompt(
    agent,
    category: str,
    toggle: str,
    toggle_default: bool,
    header: str,
    detail: str,
    target=None,
) -> bool:
    """Run the toggle → trust → auto → headless → TTY ladder, then prompt.

    True to proceed. Shared by the pre-call gate (:func:`confirm`) and the
    post-call one (:func:`confirm_result`) so the two can't drift apart on who
    gets asked. ``target`` is the path a write would touch — only the
    workspace-scoped auto-approve tier reads it.
    """
    if not config.get(toggle, toggle_default):
        return True
    # Already trusted this session (user answered "a" earlier). A background
    # sub-agent inherits these — a category the user pre-approved runs.
    trusted = getattr(agent, "_trusted_confirm_categories", None)
    if trusted is not None and category in trusted:
        return True
    # Auto-approve mode: the same kind of standing session trust as "a" above,
    # just revocable per tier — so it sits at the same rung, ABOVE the headless
    # branch (a background sub-agent inherits it for exactly the reason it
    # inherits "a"). It can never widen anything: the safety floors, the
    # plan-mode block and a hook's deny are all decided in `tool_loop` before
    # this gate runs, and `auto_approve.NEVER_AUTO` keeps the safety-override
    # category (``git``) asking in every mode.
    if _auto_approved(agent, category, target):
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


def parse_confirmation_request(result):
    """Return the payload when a tool result is asking for confirmation, else None."""
    if not isinstance(result, str) or "requires_confirmation" not in result:
        return None
    try:
        payload = json.loads(result)
    except ValueError:
        return None
    if isinstance(payload, dict) and payload.get("requires_confirmation"):
        return payload
    return None


def _request_detail(payload: dict) -> str:
    """One-line summary of why the tool refused, for the prompt's detail row."""
    reasons = []
    for warning in payload.get("warnings") or []:
        text = warning.get("message", "") if isinstance(warning, dict) else str(warning)
        if text:
            reasons.append(text)
    reasons.extend(str(issue) for issue in payload.get("issues") or [])
    if not reasons:
        reasons.append(str(payload.get("message") or "Flagged as risky."))
    detail = " ".join(reasons)
    command = payload.get("command") or ""
    return (f"{command} — {detail}" if command else detail)[:300]


def _tool_accepts(tool, field: str) -> bool:
    """True when ``tool``'s arg schema exposes ``field``, so a retry can set it."""
    fields = getattr(getattr(tool, "args_schema", None), "model_fields", None)
    return bool(fields) and field in fields


def confirm_result(agent, tool, tool_name: str, tool_args: dict, result):
    """Resolve a ``requires_confirmation`` tool result with the user.

    Called for every tool result; returns ``result`` untouched unless the tool
    refused and asked for confirmation. When it did, the user is shown the real
    warnings and either the call is retried with the override set, or a refusal
    is returned that tells the model NOT to override it itself.
    """
    payload = parse_confirmation_request(result)
    if payload is None:
        return result
    # An already-approved retry (see below), or a tool with no override
    # parameter: nothing to ask, and nothing we could re-run differently.
    if tool_args.get("allow_dangerous") or not _tool_accepts(tool, "allow_dangerous"):
        return result

    if not _gated_prompt(
        agent,
        CONFIRM_RESULT_CATEGORY,
        _RESULT_TOGGLE,
        True,
        "▶ Proceed with flagged operation?",
        _request_detail(payload),
    ):
        return json.dumps(
            {
                "error": True,
                "declined_by_user": True,
                "message": (
                    "The user was shown the risks of this operation and declined "
                    "it. Do NOT retry with allow_dangerous=True — ask them how "
                    "they want to proceed instead."
                ),
                "command": payload.get("command", ""),
            }
        )

    retry_args = dict(tool_args)
    retry_args["allow_dangerous"] = True
    if _tool_accepts(tool, "reason") and not str(tool_args.get("reason") or "").strip():
        retry_args["reason"] = _RESULT_APPROVAL_REASON
    return tool.invoke(retry_args)


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
