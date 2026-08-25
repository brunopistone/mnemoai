"""Self-check for an install (``/doctor``).

Answers one question: *is anything about this setup broken or about to be?* Every
other report describes the conversation (``/context``, ``/usage``, ``/hooks``);
this one describes the machine — the config that resolved, the provider that will
be called, the binaries the tools shell out to, the optional dependencies a
feature that is switched ON actually needs.

It exists because this app fails in places a user cannot see. The MCP server is a
piped subprocess, so a missing ``rg`` surfaces as one tool erroring mid-task; a
feature toggle is a line in a YAML file, so ``ENABLE_RAG: true`` with no vector
store installed looks like the model ignoring instructions; prompt caching is
silently provider-gated, so a config that cannot cache reads as a config that
does. Each of those is a one-line check here, with the fix on the next line.

**Every check is local, cheap, and read-only.** No LLM call, no model warm-up, no
write anywhere: running ``/doctor`` must never be the thing that changes the state
it is reporting on. The one exception is a TCP connect to a configured local
provider (Ollama), bounded to a couple of seconds — "the server isn't running" is
the single most common failure and a probe is the only way to know.

Findings are ``Check`` records with a status, so ``render`` is pure and the whole
report is unit-testable without a terminal, a model, or a config file — the
``context_report`` split (``collect``/``report`` take the client, everything else
is pure).
"""

import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple

from mnemoai.client import hooks
from mnemoai.client.memory.memory_store import MemoryStore
from mnemoai.client.memory.steering_store import SteeringStore
from mnemoai.models import prompt_cache
from mnemoai.utils.config import config
from mnemoai.utils.logger import logger
from mnemoai.utils.paths import (
    app_home,
    config_path,
    hooks_config_path,
    mcp_config_path,
    memory_file_path,
    prompts_path,
    sessions_dir,
)

OK = "ok"
WARN = "warn"
FAIL = "fail"
INFO = "info"

_MARKS = {OK: ("✓", "\033[92m"), WARN: ("!", "\033[93m"), FAIL: ("✗", "\033[91m"), INFO: ("·", "\033[90m")}
_GRAY = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_PROBE_TIMEOUT = 2.0
# Binaries the tools shell out to. Only `rg` is a hard requirement (grep_search
# has no fallback); the rest degrade to a subset of the tools.
_BINARIES = (
    ("rg", "grep_search", True),
    ("git", "the git tools", False),
    ("bash", "the shell tools (they fall back to /bin/sh)", False),
)


class Check(NamedTuple):
    """One line of the report: what was checked, how it went, how to fix it."""

    section: str
    name: str
    status: str
    detail: str = ""
    fix: str = ""


def _short(path: Any) -> str:
    """``~``-shortened path for display."""
    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except (ValueError, TypeError):
        return str(path)


def _version() -> str:
    """Installed version, or a note that this is a checkout."""
    try:
        from importlib.metadata import version

        return version("mnemoai-assistant")
    except Exception:  # noqa: BLE001 — PackageNotFoundError and anything odder
        return "(not installed — running from a checkout)"


def _install_checks() -> List[Check]:
    """Version, interpreter, and whether the app home is usable."""
    out = [
        Check("Install", "mnemoai", INFO, _version()),
        Check(
            "Install",
            "python",
            INFO,
            f"{platform.python_version()} ({sys.executable})",
        ),
        Check("Install", "platform", INFO, f"{platform.system()} {platform.release()}"),
    ]

    home = app_home()
    if not home.is_dir():
        out.append(
            Check(
                "Install",
                "app home",
                FAIL,
                f"{_short(home)} does not exist",
                "Start the app once (it seeds the directory), or check $MNEMOAI_HOME.",
            )
        )
    elif not os.access(home, os.W_OK):
        out.append(
            Check(
                "Install",
                "app home",
                FAIL,
                f"{_short(home)} is not writable",
                "Fix its permissions — sessions, memory and indexes all live there.",
            )
        )
    else:
        out.append(Check("Install", "app home", OK, _short(home)))
    return out


def _config_checks() -> List[Check]:
    """Which config and prompts files are actually LOADED, and from which tier.

    Deliberately the resolved file rather than the expected one: config resolution
    has four tiers ($MNEMOAI_CONFIG, the app home, the legacy flat path, the
    packaged fallback), and "I edited config.yaml and nothing changed" is what
    happens when the file being read is not the file being edited.
    """
    out: List[Check] = []
    loaded = _resolved(config._resolve_config_path)
    if loaded is None:
        out.append(
            Check(
                "Configuration",
                "config.yaml",
                FAIL,
                "no config file found anywhere",
                "Copy config/config.yaml.example to config/config.yaml, or run /config.",
            )
        )
    else:
        detail = _short(loaded)
        expected = config_path()
        if os.environ.get("MNEMOAI_CONFIG"):
            detail += "  (via $MNEMOAI_CONFIG)"
        elif Path(loaded) != Path(expected):
            detail += f"  (NOT {_short(expected)})"
        out.append(Check("Configuration", "config.yaml", OK, detail))

        if not config.get("MODEL_ID"):
            out.append(
                Check(
                    "Configuration",
                    "MODEL_ID",
                    FAIL,
                    "missing from the loaded config — no chat model is configured",
                    "Run /config (or /model) to write a model section.",
                )
            )

    prompts = _resolved(config._resolve_prompts_path)
    if prompts is None:
        out.append(
            Check(
                "Configuration",
                "prompts.yaml",
                WARN,
                "not found — the packaged prompts are in use",
                "Harmless; the app re-seeds it at startup if you want to edit prompts.",
            )
        )
    else:
        detail = _short(prompts)
        if Path(prompts) != Path(prompts_path()) and not os.environ.get("MNEMOAI_PROMPTS"):
            detail += f"  (NOT {_short(prompts_path())})"
        out.append(Check("Configuration", "prompts.yaml", OK, detail))
    return out


def _resolved(resolver) -> Optional[Path]:
    """Run one of Config's path resolvers, tolerating a failure."""
    try:
        found = resolver()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"/doctor could not resolve a config path: {e}")
        return None
    return found if found and Path(found).is_file() else None


def _provider_checks() -> List[Check]:
    """The configured model, its credentials or reachability, and caching."""
    section = config.get("MODEL_ID") or {}
    provider = str(section.get("TYPE", "") or "").strip().lower()
    name = str(section.get("NAME", "") or "")
    out: List[Check] = []

    if not provider:
        return [
            Check(
                "Provider",
                "model",
                FAIL,
                "no MODEL_ID.TYPE configured",
                "Run /config to pick a provider.",
            )
        ]

    label = f"{provider}: {name or '(no NAME)'}"
    protocol = str(section.get("API_PROTOCOL", "") or "")
    if protocol:
        label += f" ({protocol})"
    out.append(Check("Provider", "model", OK if name else WARN, label,
                     "" if name else "Set MODEL_ID.NAME."))

    out.append(_credentials_check(provider, section))

    policy = prompt_cache.policy(section)
    if policy.control:
        out.append(
            Check("Provider", "prompt cache", OK, f"on, {prompt_cache.ttl(section)} TTL")
        )
    else:
        out.append(
            Check(
                "Provider",
                "prompt cache",
                INFO,
                "off for this provider/model",
                f"Supported on {', '.join(prompt_cache.CACHEABLE_TYPES)} with a Claude/Nova "
                "model (mantle needs API_PROTOCOL: anthropic).",
            )
        )
    return out


def _credentials_check(provider: str, section: dict) -> Check:
    """Can this provider actually be reached — creds present, or port open."""
    if provider == "ollama":
        host = str(section.get("HOST", "localhost") or "localhost")
        port = int(section.get("PORT", 11434) or 11434)
        return _probe_port("Provider", "ollama server", host, port,
                           "Start it with `ollama serve`.")

    if provider in ("bedrock", "mantle", "sagemaker"):
        return _aws_credentials_check(provider, section)

    env_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "litellm": "",
    }
    env_key = env_keys.get(provider, "")
    has_key = bool(section.get("API_KEY")) or (bool(env_key) and bool(os.environ.get(env_key)))
    if provider == "openai" and (section.get("API_BASE") or section.get("ENDPOINT_URL")):
        # A local OpenAI-compatible server usually wants no key at all.
        return Check("Provider", "endpoint", OK,
                     str(section.get("API_BASE") or section.get("ENDPOINT_URL")))
    if has_key:
        return Check("Provider", "credentials", OK, "an API key is set (not shown)")
    if not env_key:
        return Check("Provider", "credentials", INFO,
                     f"not checked for {provider} — it manages its own auth")
    return Check(
        "Provider",
        "credentials",
        FAIL,
        f"no API key found for {provider}",
        f"Export {env_key}, or set MODEL_ID.API_KEY.",
    )


def _aws_credentials_check(provider: str, section: dict) -> Check:
    """Whether botocore can resolve credentials for the AWS-backed providers."""
    region = str(section.get("REGION", "") or os.environ.get("AWS_REGION", "") or "")
    try:
        import boto3

        creds = boto3.Session(region_name=region or None).get_credentials()
    except ImportError:
        return Check(
            "Provider",
            "aws credentials",
            FAIL,
            "boto3 is not installed",
            f"`pip install boto3` — {provider} cannot be called without it.",
        )
    except Exception as e:  # noqa: BLE001 — a profile/config error, not a crash
        return Check("Provider", "aws credentials", WARN, f"could not be resolved: {e}",
                     "Run `aws configure`.")
    if creds is None:
        return Check(
            "Provider",
            "aws credentials",
            FAIL,
            "none found",
            "Run `aws configure` (or export AWS_PROFILE / the AWS_* variables).",
        )
    detail = "resolved"
    if region:
        detail += f", region {region}"
    else:
        return Check("Provider", "aws credentials", WARN, "resolved, but no REGION is set",
                     "Set MODEL_ID.REGION (or AWS_REGION).")
    return Check("Provider", "aws credentials", OK, detail)


def _probe_port(section: str, name: str, host: str, port: int, fix: str) -> Check:
    """TCP-connect to a local provider — the only network call in the report."""
    try:
        with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT):
            return Check(section, name, OK, f"reachable at {host}:{port}")
    except OSError as e:
        return Check(section, name, FAIL, f"not reachable at {host}:{port} ({e})", fix)


def _tool_checks(client: Any) -> List[Check]:
    """External binaries, and what the MCP layer actually brought up."""
    out: List[Check] = []
    for binary, used_by, required in _BINARIES:
        found = shutil.which(binary)
        if found:
            out.append(Check("Tools", binary, OK, found))
        else:
            out.append(
                Check(
                    "Tools",
                    binary,
                    FAIL if required else WARN,
                    f"not on PATH — {used_by} will not work",
                    f"Install {binary}.",
                )
            )

    tools = getattr(client, "tools", None) if client is not None else None
    if tools:
        out.append(Check("Tools", "MCP tools", OK, f"{len(tools)} registered"))
    elif client is not None:
        out.append(
            Check(
                "Tools",
                "MCP tools",
                FAIL,
                "none registered — the built-in server did not come up",
                "Check the log (LOG_LEVEL=DEBUG) for the server subprocess's error.",
            )
        )

    out.extend(_external_mcp_checks(client))
    return out


def _external_mcp_checks(client: Any) -> List[Check]:
    """Declared vs connected external MCP servers.

    A server that fails to start is skipped with a warning at boot, which scrolls
    away — so the interesting number is declared-minus-connected, and that needs
    both mcp.json and the live client. Counting only what's live would call a
    failed server "not configured".
    """
    declared = _declared_mcp_servers()
    if not declared:
        return []

    # `_members` holds only the servers that actually connected (MultiMCPClient
    # prunes the failures in __enter__), with the built-in one first.
    members = getattr(getattr(client, "mcp_client", None), "_members", None)
    if members is None:
        return [
            Check("Tools", "external MCP servers", INFO,
                  f"{len(declared)} declared in {_short(mcp_config_path())} (not checked)")
        ]

    live = [name for name, _ in members if name != "builtin"]
    missing = [name for name in declared if name not in live]
    if not missing:
        return [Check("Tools", "external MCP servers", OK, f"{len(live)} connected: " + ", ".join(live))]
    return [
        Check(
            "Tools",
            "external MCP servers",
            WARN,
            f"{len(live)}/{len(declared)} connected — not running: {', '.join(missing)}",
            "Check the command in mcp.json (a disabled entry is expected here too).",
        )
    ]


def _declared_mcp_servers() -> List[str]:
    """Server names in mcp.json, tolerantly (the loader owns the real parsing)."""
    path = Path(mcp_config_path())
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {})
        return [
            name
            for name, entry in servers.items()
            if isinstance(entry, dict) and not entry.get("disabled")
        ]
    except (OSError, ValueError, AttributeError):
        return ["(unparsable mcp.json)"]


def _feature_checks() -> List[Check]:
    """Features that are switched ON but missing what they need."""
    out: List[Check] = []
    if config.get("ENABLE_RAG", False) or config.get("ENABLE_EPISODIC_MEMORY", False):
        store = str(config.get("RAG.VECTOR_STORE", "chromadb") or "chromadb").lower()
        module = "faiss" if "faiss" in store else "chromadb"
        out.append(_import_check("Features", f"{module} ({store})", module))
    if config.get("ENABLE_WEB_SEARCH", False) and not (
        config.get("BRAVE_API_KEY") or os.environ.get("BRAVE_API_KEY")
    ):
        out.append(
            Check(
                "Features",
                "web search",
                WARN,
                "ENABLE_WEB_SEARCH is on but no BRAVE_API_KEY is set",
                "Add BRAVE_API_KEY to config.yaml, or turn the feature off in /features.",
            )
        )
    return out


def _import_check(section: str, label: str, module: str) -> Check:
    """Is an optional dependency importable (a feature depends on it)."""
    try:
        __import__(module)
        return Check(section, label, OK, "installed")
    except Exception as e:  # noqa: BLE001 — a broken install raises more than ImportError
        return Check(
            section,
            label,
            FAIL,
            f"cannot be imported ({e})",
            f"`pip install {module}`, or switch the feature off in /features.",
        )


def _state_checks(client: Any) -> List[Check]:
    """The per-session state files: sessions, hooks, memory, steering."""
    out: List[Check] = []

    days = config.get("SESSION_MAX_AGE_DAYS", 30)
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 30
    if days <= 0:
        out.append(
            Check(
                "State",
                "session recording",
                INFO,
                "off (SESSION_MAX_AGE_DAYS: 0) — --resume has nothing to offer",
            )
        )
    else:
        try:
            count = len(list(Path(sessions_dir()).glob("session_*.jsonl")))
        except OSError:
            count = 0
        out.append(
            Check("State", "sessions here", OK, f"{count} recorded, kept {days} days")
        )

    registry = hooks.active()
    if registry.errors:
        out.append(
            Check(
                "State",
                "tool hooks",
                FAIL,
                f"{len(registry.errors)} problem(s) in {_short(hooks_config_path())}",
                "Run /hooks to see them; a bad entry is skipped, not fatal.",
            )
        )
    elif registry.hooks:
        out.append(Check("State", "tool hooks", OK, f"{len(registry.hooks)} loaded"))

    out.extend(_size_checks())
    return out


def _size_checks() -> List[Check]:
    """MEMORY.md and the steering files against their caps.

    Both are re-sent in full every turn and compaction can never reclaim either,
    so being near the cap is a real per-turn cost — and a file already over it is
    being silently truncated, which is worth saying out loud.
    """
    out: List[Check] = []
    try:
        text = MemoryStore().read()
        cap = int(config.get("MEMORY.MAX_CHARS", 2200) or 2200)
        used = len(text)
        # Near the cap matters as much as over it: the store trims silently, so a
        # file sitting at 99% is one fact away from losing one.
        status = WARN if used >= cap * 0.9 else OK
        fix = ""
        if used > cap:
            fix = "Over the cap — the store trims it. Consolidate entries."
        elif status == WARN:
            fix = "Nearly full; the next entry may push an older one out."
        out.append(
            Check(
                "State",
                "MEMORY.md",
                status,
                f"{used} / {cap} chars ({_short(memory_file_path())})",
                fix,
            )
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"/doctor could not size MEMORY.md: {e}")

    try:
        sizes = SteeringStore().sizes()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"/doctor could not size steering files: {e}")
        sizes = []
    for entry in sizes:
        path, chars = _steering_entry(entry)
        if path is None:
            continue
        out.append(
            Check(
                "State",
                "steering",
                INFO,
                f"{_short(path)} — {chars} chars, injected every turn",
            )
        )
    return out


def _steering_entry(entry: Any) -> Tuple[Optional[str], int]:
    """``(path, chars)`` from a ``SteeringStore.sizes()`` row ``(Path, text)``."""
    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        second = entry[1]
        return str(entry[0]), len(second) if isinstance(second, str) else int(second or 0)
    return None, 0


def collect(client: Any = None) -> List[Check]:
    """Run every check. ``client`` may be None (the live-state ones are skipped)."""
    checks: List[Check] = []
    checks.extend(_install_checks())
    checks.extend(_config_checks())
    checks.extend(_provider_checks())
    checks.extend(_tool_checks(client))
    checks.extend(_feature_checks())
    checks.extend(_state_checks(client))
    return checks


def render(checks: List[Check], color: bool = True) -> str:
    """The report: grouped lines, worst-first summary, fixes under the failures."""
    lines: List[str] = []
    problems = sum(1 for c in checks if c.status == FAIL)
    warnings = sum(1 for c in checks if c.status == WARN)

    def paint(text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if color else text

    header = "Doctor"
    if problems:
        header += f" — {problems} problem{'s' if problems != 1 else ''}"
        if warnings:
            header += f", {warnings} warning{'s' if warnings != 1 else ''}"
    elif warnings:
        header += f" — {warnings} warning{'s' if warnings != 1 else ''}"
    else:
        header += " — everything checks out"
    lines.append(paint(header, _BOLD) if color else header)

    section = ""
    for check in checks:
        if check.section != section:
            section = check.section
            lines += ["", paint(f"  {section}", _GRAY) if color else f"  {section}"]
        mark, code = _MARKS.get(check.status, _MARKS[INFO])
        detail = f"  {check.detail}" if check.detail else ""
        lines.append(f"    {paint(mark, code)} {check.name}{detail}")
        if check.fix:
            lines.append(f"      {paint('→ ' + check.fix, _GRAY)}")
    return "\n".join(lines)


def report(client: Any = None) -> str:
    """``collect`` + ``render``, guarding against a check that itself breaks.

    A diagnostic that dies is worse than useless — it becomes the problem being
    diagnosed — so a failure here still prints what was gathered.
    """
    try:
        return render(collect(client))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"/doctor failed: {e}")
        return f"Doctor could not complete its checks: {e}"
