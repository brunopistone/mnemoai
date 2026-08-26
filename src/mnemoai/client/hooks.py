"""User-scriptable hooks around tool calls (``~/.mnemoai/hooks/hooks.json``).

A hook is a shell command run before or after a tool call. It exists for the
things a prompt cannot reliably enforce: format every file the agent writes,
refuse writes under a path you never want touched, auto-approve the read-only
commands you are tired of confirming, hand the model a hint when a tool fails.

**Where a hook sits in the gate order** — below the floors, never above them::

    server-side safety floor (bash_policy / path_policy / read_state)
      -> plan mode hard block
        -> PreToolUse hook            <- here
          -> _confirm_tool (the confirmation prompt)
            -> the tool runs

So a hook's ``deny`` is honored anywhere, but its ``allow`` reaches exactly one
gate: it satisfies the confirmation prompt for that one call. It cannot unblock a
tool that plan mode blocked, and it cannot touch the server-side floors, which
run inside the MCP subprocess and would still refuse a catastrophic command. A
config file must not be a way to widen what the app is allowed to do.

**App home only, and snapshotted at startup.** Unlike ``STEERING.md`` — read-only
text, re-read every turn, happily per-project — a hooks file is arbitrary code
that fires on tool calls. A ``hooks.json`` arriving with a ``git clone`` would be
remote code execution on the first edit in that repo, so nothing outside the app
home is read (:func:`~mnemoai.utils.paths.hooks_config_path`). The snapshot is
the same argument one step on: hooks are re-read only at startup, so editing the
file mid-session cannot change what is already running (restart to apply). Both
choices are deliberate departures from steering, not oversights.

**Never blocking, never interactive.** A hook that exits non-zero (other than the
deny code), crashes, or overruns its timeout is *reported and skipped* — the tool
proceeds. A slow or broken hook must not be able to wedge the agent, which is the
failure mode that makes people delete a hooks feature rather than debug it. And
hooks fire from tool-calling threads (parallel orchestrator waves and background
sub-agents included), so they must never wait for input: there is no terminal to
prompt on, the same rule ``ask_user_question`` follows when headless.

Hooks run under real bash (``executable=bash_path()``), not ``/bin/sh`` — the
tools document bash syntax and so does this, so ``[[ ]]`` and ``source`` behave
as written rather than depending on the host's ``/bin/sh``.

Pure file/parse logic plus one ``subprocess`` call: no LLM, no MCP, no config
dependency, unit-testable on its own.
"""

import fnmatch
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from mnemoai.server.tools.shell_state import bash_path
from mnemoai.utils.console import print_error
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import hooks_config_path

PRE_TOOL_USE = "PreToolUse"
POST_TOOL_USE = "PostToolUse"
POST_TOOL_USE_FAILURE = "PostToolUseFailure"
EVENTS = (PRE_TOOL_USE, POST_TOOL_USE, POST_TOOL_USE_FAILURE)

# `exit 2` means "block this call" — every other non-zero code is a hook that
# broke, which must not block anything.
DENY_EXIT_CODE = 2

DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 600.0

# Caps. `context` is fed to the model (so it is per-turn cost); `reason` is shown
# to the model as the block reason; a notice only reaches the user's scrollback.
_MAX_CONTEXT_CHARS = 4000
_MAX_REASON_CHARS = 500
_MAX_NOTICE_CHARS = 300
# A single tool_input value (an fs_write body can be megabytes, and a hook that
# needs it is rare; one that chokes on it is not).
_MAX_INPUT_VALUE_CHARS = 20000
_MAX_RESPONSE_CHARS = 20000


class Hook(NamedTuple):
    """One command to run for one event, matched against the tool name."""

    event: str
    matcher: str
    command: str
    timeout: float

    @property
    def label(self) -> str:
        """Short single-line form of the command, for notices and ``/hooks``."""
        flat = " ".join(self.command.split())
        return flat if len(flat) <= 60 else flat[:57] + "…"


class Registry(NamedTuple):
    """The loaded hooks, where they came from, and what didn't parse."""

    hooks: Tuple[Hook, ...] = ()
    path: Optional[str] = None
    errors: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.hooks)


class Outcome(NamedTuple):
    """What the hooks for one event decided."""

    decision: Optional[str] = None  # "allow" | "deny" | None
    reason: str = ""
    context: str = ""  # additionalContext — the only part the model sees
    notices: Tuple[str, ...] = ()  # user-facing lines (never sent to the model)

    @property
    def denied(self) -> bool:
        return self.decision == "deny"

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


# The startup snapshot. Read from tool-calling threads, so lock-guarded.
_lock = threading.Lock()
_snapshot: Optional[Registry] = None


def load(path: Path) -> Registry:
    """Parse a hooks file, collecting problems instead of raising.

    Tolerant on purpose, the way ``mcp.json`` is: one malformed entry must not
    cost the user the rest of their hooks, and a broken file must not stop the
    app from starting. Problems come back as strings so the caller decides when
    (and whether) to show them — nothing here prints.
    """
    if not path.is_file():
        return Registry(path=str(path))

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return Registry(path=str(path), errors=(f"hooks.json is not valid JSON: {e}",))
    except OSError as e:
        return Registry(path=str(path), errors=(f"hooks.json could not be read: {e}",))

    if not isinstance(raw, dict):
        return Registry(path=str(path), errors=("hooks.json must contain a JSON object.",))

    # Either nested under "hooks" or the event map at the top level — both read
    # naturally, and guessing wrong here costs the user every hook they wrote.
    events = raw.get("hooks", raw)
    if not isinstance(events, dict):
        return Registry(path=str(path), errors=('hooks.json: "hooks" must be an object.',))

    hooks: List[Hook] = []
    errors: List[str] = []
    for event, groups in events.items():
        if event.startswith("//"):  # comment key
            continue
        if event not in EVENTS:
            errors.append(
                f"hooks.json: unknown event '{event}' (expected {', '.join(EVENTS)}) — skipped."
            )
            continue
        if not isinstance(groups, list):
            errors.append(f"hooks.json: '{event}' must be a list of matcher groups — skipped.")
            continue
        for group in groups:
            _parse_group(event, group, hooks, errors)

    return Registry(hooks=tuple(hooks), path=str(path), errors=tuple(errors))


def _parse_group(event: str, group: Any, hooks: List[Hook], errors: List[str]) -> None:
    """Append every valid hook in one matcher group; report the rest."""
    if not isinstance(group, dict):
        errors.append(f"hooks.json: '{event}' contains a non-object entry — skipped.")
        return

    matcher = group.get("matcher") or "*"
    if not isinstance(matcher, str):
        errors.append(f"hooks.json: '{event}' has a non-string matcher — skipped.")
        return

    entries = group.get("hooks")
    if not isinstance(entries, list):
        errors.append(f"hooks.json: '{event}' matcher '{matcher}' has no \"hooks\" list — skipped.")
        return

    for entry in entries:
        if not isinstance(entry, dict):
            errors.append(f"hooks.json: '{event}' matcher '{matcher}' has a non-object hook.")
            continue
        kind = entry.get("type", "command")
        if kind != "command":
            errors.append(
                f"hooks.json: '{event}' matcher '{matcher}' uses type '{kind}'; "
                "only 'command' is supported — skipped."
            )
            continue
        command = entry.get("command")
        if not isinstance(command, str) or not command.strip():
            errors.append(f"hooks.json: '{event}' matcher '{matcher}' has no command — skipped.")
            continue
        hooks.append(Hook(event, matcher, command.strip(), _timeout(entry.get("timeout"))))


def _timeout(value: Any) -> float:
    """Clamp a configured timeout into (0, MAX_TIMEOUT]; junk falls back."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT
    if seconds <= 0:
        return DEFAULT_TIMEOUT
    return min(seconds, MAX_TIMEOUT)


def active() -> Registry:
    """The startup snapshot, loading (and reporting) it on first use."""
    global _snapshot
    with _lock:
        if _snapshot is None:
            _snapshot = load(hooks_config_path())
            for err in _snapshot.errors:
                print_error(err)
            if _snapshot.hooks:
                logger.info(f"Loaded {len(_snapshot.hooks)} tool hook(s) from {_snapshot.path}")
        return _snapshot


def reset_cache() -> None:
    """Drop the snapshot (tests, and the in-place re-exec paths)."""
    global _snapshot
    with _lock:
        _snapshot = None


def matching(registry: Registry, event: str, tool_name: str) -> List[Hook]:
    """Hooks for ``event`` whose matcher globs ``tool_name``, in file order."""
    return [
        h for h in registry.hooks if h.event == event and fnmatch.fnmatchcase(tool_name, h.matcher)
    ]


def run_event(
    event: str,
    tool_name: str,
    tool_input: Optional[Dict[str, Any]] = None,
    *,
    tool_response: Optional[str] = None,
    session_id: str = "",
    cwd: Optional[str] = None,
    registry: Optional[Registry] = None,
) -> Outcome:
    """Run every hook matching ``event``/``tool_name`` and fold the results.

    The first ``deny`` short-circuits (the call is blocked, so later hooks have
    nothing to act on); ``additionalContext`` from the hooks that did run is
    concatenated. An ``allow`` does not short-circuit — a later ``deny`` still
    wins, because the safe answer must not depend on file order.
    """
    reg = active() if registry is None else registry
    hooks = matching(reg, event, tool_name)
    if not hooks:
        return Outcome()

    payload = _payload(event, tool_name, tool_input, tool_response, session_id, cwd)
    decision: Optional[str] = None
    reason = ""
    contexts: List[str] = []
    notices: List[str] = []

    for hook in hooks:
        result = _run_one(hook, payload, cwd, notices)
        if result is None:
            continue
        if result.get("context"):
            contexts.append(result["context"])
        verdict = result.get("decision")
        if verdict == "deny":
            decision, reason = "deny", result.get("reason", "")
            break
        if verdict == "allow" and decision is None:
            decision, reason = "allow", result.get("reason", "")

    return Outcome(
        decision=decision,
        reason=reason,
        context="\n".join(contexts)[:_MAX_CONTEXT_CHARS],
        notices=tuple(notices),
    )


def _payload(
    event: str,
    tool_name: str,
    tool_input: Optional[Dict[str, Any]],
    tool_response: Optional[str],
    session_id: str,
    cwd: Optional[str],
) -> str:
    """The JSON handed to a hook on stdin."""
    body: Dict[str, Any] = {
        "session_id": session_id,
        "cwd": cwd or os.getcwd(),
        "hook_event_name": event,
        "tool_name": tool_name,
        "tool_input": _clip_input(tool_input or {}),
    }
    if tool_response is not None:
        body["tool_response"] = tool_response[:_MAX_RESPONSE_CHARS]
    return json.dumps(body, default=str)


def _clip_input(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Cap long string values — an fs_write body can be megabytes."""
    out: Dict[str, Any] = {}
    truncated: List[str] = []
    for key, value in tool_input.items():
        if isinstance(value, str) and len(value) > _MAX_INPUT_VALUE_CHARS:
            out[key] = value[:_MAX_INPUT_VALUE_CHARS]
            truncated.append(key)
        else:
            out[key] = value
    if truncated:
        out["_truncated"] = truncated  # a hook can tell a clipped value from a short one
    return out


def _run_one(
    hook: Hook, payload: str, cwd: Optional[str], notices: List[str]
) -> Optional[Dict[str, Any]]:
    """Execute one hook. Returns its verdict, or None when it had nothing to say.

    Every failure path here is non-blocking by design: the worst a broken hook
    can do is add a line to the scrollback and a warning to the log.
    """
    env = dict(os.environ, MNEMOAI_HOOK_EVENT=hook.event, MNEMOAI_TOOL_NAME=hook.matcher)
    try:
        proc = subprocess.run(
            hook.command,
            shell=True,
            executable=bash_path(),
            input=payload,
            capture_output=True,
            text=True,
            timeout=hook.timeout,
            cwd=cwd or None,
            env=env,
        )
    except subprocess.TimeoutExpired:
        notices.append(f"hook timed out after {hook.timeout:g}s: {hook.label}")
        logger.warning(f"Hook timed out ({hook.event} {hook.matcher}): {hook.command}")
        return None
    except OSError as e:
        notices.append(f"hook could not run: {hook.label} ({e})")
        logger.warning(f"Hook failed to start ({hook.event} {hook.matcher}): {e}")
        return None

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode == DENY_EXIT_CODE:
        reason = (stderr or stdout or "blocked by a hook")[:_MAX_REASON_CHARS]
        notices.append(f"hook blocked {hook.event}: {reason}")
        logger.info(f"Hook denied {hook.event}/{hook.matcher}: {reason}")
        return {"decision": "deny", "reason": reason}

    if proc.returncode != 0:
        detail = (stderr or stdout or f"exit {proc.returncode}")[:_MAX_NOTICE_CHARS]
        notices.append(f"hook error (non-blocking): {hook.label} — {detail}")
        logger.warning(f"Hook error ({hook.event}/{hook.matcher}) exit {proc.returncode}: {detail}")
        return None

    verdict = _parse_stdout(stdout)
    if verdict is None:
        # Plain output: the user's, not the model's. Silence stays silent — a
        # formatter that runs on every write should not narrate itself.
        if stdout:
            notices.append(f"hook: {stdout.splitlines()[0][:_MAX_NOTICE_CHARS]}")
        # stderr from a hook that SUCCEEDED is debug output, not a problem — kept
        # out of the scrollback but logged, since it's how you debug a hook.
        logger.debug(f"Hook ok ({hook.event}/{hook.matcher}): {stderr or 'no stderr'}")
        return None

    if verdict.get("decision") == "deny":
        notices.append(f"hook blocked {hook.event}: {verdict.get('reason', '')}"[:_MAX_NOTICE_CHARS])
    elif verdict.get("decision") == "allow":
        notices.append(f"hook approved this call: {hook.label}")
    if verdict.get("context"):
        notices.append(f"hook added context: {hook.label}")
    return verdict


def _parse_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    """A hook's structured answer, or None if it didn't print one."""
    if not stdout.startswith("{"):
        return None
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    out: Dict[str, Any] = {}
    raw = data.get("decision") or data.get("permissionDecision")
    if isinstance(raw, str) and raw.lower() in ("allow", "deny"):
        out["decision"] = raw.lower()
    for key in ("reason", "permissionDecisionReason", "systemMessage"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            out["reason"] = value.strip()[:_MAX_REASON_CHARS]
            break
    context = data.get("additionalContext")
    if isinstance(context, str) and context.strip():
        out["context"] = context.strip()[:_MAX_CONTEXT_CHARS]
    return out or None


def render(registry: Optional[Registry] = None) -> str:
    """The ``/hooks`` report: what is loaded, from where, and how to add more."""
    reg = active() if registry is None else registry
    lines = ["Tool hooks"]
    lines.append(f"  config: {_tilde(reg.path)}")

    if not reg.hooks:
        example = _tilde(str(Path(reg.path or "").with_name("hooks.json.example")))
        lines += [
            "",
            "  No hooks configured.",
            f"  Copy {example} to hooks.json and restart to add some.",
        ]
    else:
        for event in EVENTS:
            group = [h for h in reg.hooks if h.event == event]
            if not group:
                continue
            lines += ["", f"  {event}"]
            for hook in group:
                lines.append(f"    {hook.matcher:<20} {hook.label}  ({hook.timeout:g}s)")
        lines += [
            "",
            f"  {len(reg.hooks)} hook(s), read at startup — restart to apply an edit.",
        ]

    if reg.errors:
        lines += [""] + [f"  ! {err}" for err in reg.errors]
    return "\n".join(lines)


def _tilde(path: Optional[str]) -> str:
    """``~``-shortened path for display."""
    if not path:
        return "(none)"
    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except ValueError:
        return path
