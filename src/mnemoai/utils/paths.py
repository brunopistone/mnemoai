"""Centralized filesystem paths for the assistant.

All persistent state lives under a single app-home directory so it's easy to
find, back up, or relocate:

    ~/.mnemoai/
    ├── config/                             # config.yaml + provider examples
    │   ├── config.yaml                     # user config (installed CLI)
    │   ├── config.yaml*.example            # bundled examples (copied here to read)
    │   └── prompt.yaml                     # application prompts
    ├── mcp/                                # external MCP servers
    │   ├── mcp.json                        # optional, user-created
    │   └── mcp.json.example                # bundled example (copied here to read)
    ├── hooks/                              # user-scriptable tool-call hooks
    │   ├── hooks.json                      # optional, user-created (app home ONLY)
    │   └── hooks.json.example              # bundled example (copied here to read)
    ├── logs/                               # diagnostics (age-swept at startup)
    │   ├── mnemoai.log                     # app log: every traceback lands here
    │   └── mcp.log                         # MCP server subprocess stderr
    ├── plans/plan_<ts>.md                  # approved plan-mode plans
    ├── skills                              # skills folder
    ├── STEERING.md                         # user-authored always-on instructions
    │                                       #   (or CLAUDE.md; STEERING.md wins)
    ├── tasks/                              # background-task output
    └── {profile}/                          # per-user-profile data
        ├── conversations/  todos/  rag_store_*  chunk_cache_*  profile JSON
        ├── {rag,chunk}_session_id_<iid>.txt # per-instance session pointers
        └── models/{model}/                # per-chat-model memory
            ├── episodic_memory/
            └── playbook/

Override the root with ``$MNEMOAI_HOME``. The config file location
can additionally be overridden with ``$MNEMOAI_CONFIG``.
"""

import hashlib
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from mnemoai.utils.logger import logger

DEFAULT_HOME_DIRNAME = ".mnemoai"


def app_home() -> Path:
    """Return the root app-home directory (created), honoring the env override.

    ``$MNEMOAI_HOME`` overrides the default ``~/.mnemoai``.
    """
    env_home = os.environ.get("MNEMOAI_HOME")
    home = Path(env_home).expanduser() if env_home else Path.home() / DEFAULT_HOME_DIRNAME
    home.mkdir(parents=True, exist_ok=True)
    return home


def config_dir() -> Path:
    """Directory holding config.yaml and the bundled config examples (created)."""
    d = app_home() / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mcp_dir() -> Path:
    """Directory holding mcp.json and the bundled mcp example (created)."""
    d = app_home() / "mcp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def hooks_dir() -> Path:
    """Directory holding hooks.json and the bundled hooks example (created)."""
    d = app_home() / "hooks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path() -> Path:
    """Default config.yaml location: ``<app_home>/config/config.yaml`` (not auto-created)."""
    return config_dir() / "config.yaml"


def legacy_config_path() -> Path:
    """Pre-subfolder config location (``<app_home>/config.yaml``), read-only fallback.

    Kept so installs created before the ``config/`` subfolder still load without
    re-running setup. New configs are always written to :func:`config_path`.
    """
    return app_home() / "config.yaml"


def prompts_path() -> Path:
    """Location of the LLM prompts file: ``<app_home>/config/prompts.yaml``.

    All model-facing prompts (system, routing, orchestrator, aggregator, and the
    compaction summary prompts) live here, separate from ``config.yaml`` which
    holds only configuration. Seeded from the bundled template on first run.
    """
    return config_dir() / "prompts.yaml"


def mcp_config_path() -> Path:
    """Location of the external MCP servers config: ``<app_home>/mcp/mcp.json``.

    Holds extra MCP servers to launch alongside mnemoai's built-in server, in
    the same ``{"mcpServers": {...}}`` schema. ``$MNEMOAI_HOME`` moves it with 
    the rest of the app home. Not auto-created.
    """
    return mcp_dir() / "mcp.json"


def legacy_mcp_config_path() -> Path:
    """Pre-subfolder mcp.json location (``<app_home>/mcp.json``), read-only fallback."""
    return app_home() / "mcp.json"


def hooks_config_path() -> Path:
    """Location of the tool-hooks config: ``<app_home>/hooks/hooks.json``.

    **App home only, deliberately** — a hooks file is arbitrary code that runs on
    tool calls, so unlike a per-project ``STEERING.md`` (read-only text) it must
    never arrive with a ``git clone`` and start firing. The same rule already
    applied to global steering: nothing outside the app home silently becomes
    always-on. Not auto-created; presence is the switch.
    """
    return hooks_dir() / "hooks.json"


def _refresh_example(src: Path, dest: Path) -> None:
    """Copy a bundled ``*.example`` to ``dest``, refreshing it when it differs.

    ``*.example`` files are read-only reference (the app never loads them — the
    configurator reads the canonical templates from the package), so unlike the
    live files we keep them in sync with the bundle: a new install gets them, and
    an EXISTING install gets an updated example (e.g. new config keys) on upgrade.
    Only writes when content differs, so it's cheap and idempotent.
    """
    try:
        if dest.exists() and dest.read_text() == src.read_text():
            return
    except OSError:
        pass  # unreadable dest → fall through and overwrite
    shutil.copyfile(src, dest)


# sha256 of every ``SKILL.md`` a PRIOR release shipped for each bundled skill
# (the current bundle is compared at runtime, so it needn't be listed). An
# installed ``SKILL.md`` whose hash is here — meaning a version WE shipped, left
# unmodified by the user — is "pristine", so it is safe to refresh in place on
# upgrade (e.g. to document a new frontmatter key). Any other hash means the user
# edited it, and we never touch it. **Maintenance:** when a bundled skill's
# ``SKILL.md`` changes, append its PREVIOUS shipped hash here so the prior version
# is still recognized as pristine on the next upgrade.
_PRISTINE_BUNDLED_SKILL_HASHES = {
    "skill-creator": {
        "f01fe2c1e1450d4e814041a94806c1d26dd2ac9ca67aa28260a8ee90a29d7338",  # ≤1.2.2
    },
    "steering-creator": {
        "78b29bffef4368f5e065ed833e62b6a0a9e9b2aac9958235078ef5c26d1f5301",  # ≤1.2.2
        "98bf72d8e5a1add8d3f7a640324686900e8a2459e6f4ab7ced05189eace2e82d",  # 1.3.0–1.8.7
    },
    "commit-message": {
        "8c5addd5dfc7fab5adbd9af28b92cc0ce1544af1730185c9196d674afd783409",  # ≤1.2.2
    },
}


# sha256 of every bundled ``prompts.yaml`` a PRIOR release shipped (the current
# bundle is compared at runtime). An installed ``prompts.yaml`` whose hash is here
# — a version WE shipped, unmodified by the user — is "pristine", so it is safe to
# refresh in place on upgrade so prompt improvements reach existing installs. Any
# other hash means the user customized it, and we never touch it. **Maintenance:**
# when the bundled ``prompts.yaml`` changes, append its PREVIOUS shipped hash here.
_PRISTINE_BUNDLED_PROMPTS_HASHES = {
    "0951767d17358af6bfaa2c41769731809e0fdd2115dda2c68743e3e4ec2259e0",  # 0.8.17–1.3.0
    "40467d047e7364e2cdc696e0dc6d935423c7ae3615fae8422ef9f2834659cb2e",  # 1.4.0–1.4.5
    "fe423553bed74ced37b53503e7249fb1981528d8fdda206341782968699f2267",  # 1.5.0
    "4efc9492288fbc386b0e71adfae3482b6698b04990674ffcb92f3d1b53074adf",  # 1.5.1–1.5.2
    "caf01cee0cd0051de383f3650724e6f3d7f30b9b1a3ea5e56a48b5a71c68eff2",  # 1.5.3–1.6.2
    "6ad25a24549ccf27ac6839ec5ed82163e7b50b33b96ac259a02d0456470792a3",  # 1.6.3–1.7.6
    "0e6bd83143f72544979196c3d278f3216aedf6b5e3e6e9b7b0e7af5d3027de8f",  # 1.7.7–1.8.3
    "0490170eda28a2a0f44bda552329cd02d7c250eceefdbad08b693dcfb56ccc29",  # 1.8.4–1.8.7
    "3fc7726cae6df797958af44daeca51b739b488ed5b79348efb4dea7ddc99e922",  # 1.9.0–1.20.0
}


# sha256 of every bundled command file a PRIOR release shipped, keyed by file name
# (the current bundle is compared at runtime). Same contract as the skill hashes
# above: an installed copy whose hash is here is a version WE shipped, so it is
# safe to refresh in place; anything else is the user's file. **Maintenance:** when
# a bundled command changes, append its PREVIOUS shipped hash here. A newly bundled
# command starts with an empty set (nothing superseded yet).
_PRISTINE_BUNDLED_COMMAND_HASHES = {
    "_README.md": set(),
    "explain.md": set(),
}


def _sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes (matches ``shasum -a 256``)."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_if_pristine(src: Path, dest: Path, known: set) -> None:
    """Copy ``src`` over ``dest`` only when the installed copy is *pristine*.

    Pristine means its sha256 is in ``known`` — a version WE shipped, unmodified —
    so the refresh delivers our updates without ever clobbering a user's edits.
    The one place that rule is implemented; the three callers below differ only in
    which file they compare and which hash set they consult.
    """
    if not src.is_file() or not dest.is_file():
        return
    try:
        installed = _sha256(dest)
        if installed == _sha256(src):
            return  # already current — nothing to do
        if installed in known:
            shutil.copyfile(src, dest)  # pristine → safe to refresh
        # else: user-edited → leave untouched
    except OSError:
        pass


def _refresh_pristine_prompts(src: Path, dest: Path) -> None:
    """Refresh the live ``prompts.yaml`` in place IF the installed copy is pristine
    (a version we shipped, unmodified); otherwise leave the user's customized file.

    Unlike ``*.example`` files (always refreshed) and unlike the never-touched
    ``config.yaml``, ``prompts.yaml`` IS loaded as config but is rarely customized —
    so refreshing a pristine copy lets prompt improvements (edits to existing keys,
    which the bundled-fallback loader can't deliver since it only fills MISSING
    keys) reach existing installs on upgrade, without clobbering a user's edits."""
    _refresh_if_pristine(src, dest, _PRISTINE_BUNDLED_PROMPTS_HASHES)


def _refresh_pristine_skill(src_dir: Path, dest_dir: Path) -> None:
    """Refresh a bundled skill's ``SKILL.md`` in place IF the installed copy is
    pristine (a version we shipped, unmodified by the user); otherwise leave it.

    Complements the copy-if-absent seeding: an already-installed bundled skill
    still gets doc/frontmatter updates on upgrade, but a user's own edits are
    never overwritten. Only ``SKILL.md`` is touched, so any extra files the user
    added alongside it are preserved.
    """
    _refresh_if_pristine(
        src_dir / "SKILL.md",
        dest_dir / "SKILL.md",
        _PRISTINE_BUNDLED_SKILL_HASHES.get(dest_dir.name, set()),
    )


def _refresh_pristine_command(src: Path, dest: Path) -> None:
    """Refresh a bundled slash command in place IF the installed copy is pristine.

    A command file is a prompt the USER invokes by name, so the same rule as a
    bundled skill applies: our wording improvements reach existing installs, an
    edited command is the user's own and is never touched.
    """
    _refresh_if_pristine(src, dest, _PRISTINE_BUNDLED_COMMAND_HASHES.get(dest.name, set()))


def seed_example_files() -> None:
    """Copy the package's bundled ``*.example`` templates into the app home.

    Gives users browsable examples right next to their live files:
    ``config/`` gets the ``config.yaml*.example`` templates, ``mcp/`` gets
    ``mcp.json.example`` and ``hooks/`` gets ``hooks.json.example``. The
    ``*.example`` reference files are **refreshed from
    the bundle when they differ** so a new bundled key reaches an EXISTING install
    on upgrade (they're read-only reference, not loaded as config). Bundled example
    skills and slash commands are copied when absent, and an already-installed one
    whose file is still **pristine** (a version we shipped, unmodified) is refreshed
    in place so doc/frontmatter updates also reach existing installs. ``prompts.yaml`` is
    likewise refreshed in place when pristine (so prompt improvements reach
    existing installs). ``config.yaml``/``mcp.json`` and any user-customized
    ``prompts.yaml``/skill are created when absent and otherwise NEVER overwritten.
    """
    pkg_templates = Path(__file__).resolve().parent  # mnemoai/utils/
    try:
        for example in pkg_templates.glob("config.yaml*.example"):
            _refresh_example(example, config_dir() / example.name)
        # prompts.yaml is the live prompts file (not a *.example): seed the actual
        # file so the app has prompts out of the box when absent, and refresh it in
        # place when the installed copy is still PRISTINE (a version we shipped,
        # unmodified) so prompt improvements reach existing installs — a
        # user-customized prompts.yaml is left untouched.
        prompts_template = pkg_templates / "prompts.yaml"
        if prompts_template.is_file():
            dest = prompts_path()
            if not dest.exists():
                shutil.copyfile(prompts_template, dest)
            else:
                _refresh_pristine_prompts(prompts_template, dest)
        mcp_example = pkg_templates / "mcp.json.example"
        if mcp_example.is_file():
            _refresh_example(mcp_example, mcp_dir() / mcp_example.name)
        hooks_example = pkg_templates / "hooks.json.example"
        if hooks_example.is_file():
            _refresh_example(hooks_example, hooks_dir() / hooks_example.name)
        # Seed the bundled example skill(s) into the skills dir so the feature is
        # discoverable out of the box. Per-skill (like the config *.example files
        # above): copy any bundled skill whose directory doesn't exist yet, so a
        # NEW bundled skill also reaches an EXISTING install on upgrade. Never
        # overwrites a user's own skills. Trade-off: a bundled skill the user
        # deleted reappears on upgrade — acceptable for a refreshed example.
        # If the skill already exists AND its SKILL.md is still pristine (a version
        # we shipped, unmodified), refresh just that SKILL.md so doc/frontmatter
        # updates reach existing installs; a user-edited skill is left untouched.
        skills_template_root = pkg_templates / "skills_example"
        if skills_template_root.is_dir():
            dest_root = skills_dir()
            for skill_dir in skills_template_root.iterdir():
                if skill_dir.is_dir():
                    dest = dest_root / skill_dir.name
                    if not dest.exists():
                        shutil.copytree(skill_dir, dest)
                    else:
                        _refresh_pristine_skill(skill_dir, dest)
        # Bundled example slash commands, same per-file rules as the skills above:
        # copied when absent (so a newly bundled command reaches an existing
        # install), refreshed in place only while still pristine.
        commands_template_root = pkg_templates / "commands_example"
        if commands_template_root.is_dir():
            dest_root = commands_dir()
            for cmd_file in commands_template_root.glob("*.md"):
                dest = dest_root / cmd_file.name
                if not dest.exists():
                    shutil.copyfile(cmd_file, dest)
                else:
                    _refresh_pristine_command(cmd_file, dest)
    except OSError:
        # Seeding examples is a convenience; never let it block startup.
        pass


def plans_dir() -> Path:
    """Directory for plan-mode state (created)."""
    d = app_home() / "plans"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Age after which an approved-plan file is swept at startup. Approved plans are
# persisted so they survive compaction and can be re-read shortly after; they're
# not durable artifacts, so old ones are pruned to keep the dir from growing.
PLAN_MAX_AGE_DAYS = 7


def sweep_old_plans(max_age_days: int = PLAN_MAX_AGE_DAYS) -> int:
    """Delete ``plan_*.md`` files older than ``max_age_days``; return the count.

    Best-effort startup housekeeping (0 disables). Only touches ``plan_*.md``
    files in the plans dir, so anything else there is left alone. Also removes a
    stale ``current_plan.json`` left by the retired legacy plan-mode tools.
    """
    d = plans_dir()
    removed = 0
    # Drop the dead legacy state file if present (retired plan_mode.py).
    try:
        legacy = d / "current_plan.json"
        if legacy.is_file():
            legacy.unlink()
    except OSError:
        pass
    if max_age_days <= 0:
        return removed
    cutoff = time.time() - max_age_days * 86400
    try:
        for f in d.glob("plan_*.md"):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


def skills_dir() -> Path:
    """Directory holding agent skills, one ``<name>/SKILL.md`` per skill (created).

    Seeded with a bundled example on first run by :func:`seed_example_files`.
    """
    d = app_home() / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def commands_dir() -> Path:
    """Directory holding user-defined slash commands, one ``<name>.md`` per command
    (created). The FILE NAME is the command (``deploy.md`` → ``/deploy``); the file
    is optional frontmatter (``description``, ``argument_hint``) plus a markdown
    body that becomes the prompt, with ``$ARGUMENTS`` substituted.

    **App home only** — like ``agents/`` and ``skills/``, and unlike a per-project
    ``STEERING.md``: a command is invoked by the user, so a ``git clone`` must not
    be able to redefine what a name they type expands to. Seeded with a bundled
    example on first run by :func:`seed_example_files`.
    """
    d = app_home() / "commands"
    d.mkdir(parents=True, exist_ok=True)
    return d


def agents_dir() -> Path:
    """Directory holding custom sub-agent types, one ``<name>.md`` per agent
    (created). Each file is frontmatter (name, description, tools?, model?) + a
    markdown body used as the sub-agent's system prompt."""
    d = app_home() / "agents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tasks_dir() -> Path:
    """Directory for background-task output (created)."""
    d = app_home() / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    """Directory for app log files (created)."""
    d = app_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mcp_log_path() -> Path:
    """Log file capturing MCP server subprocess stderr, so their noise (e.g.
    ``npm notice`` from an ``npx`` server) stays out of the terminal but remains
    available for debugging a server that fails to start."""
    return logs_dir() / "mcp.log"


def app_log_path() -> Path:
    """Log file capturing the app's own diagnostics, exceptions included.

    The terminal here is a conversation, not a console: a stack trace printed
    into it buries the answer above it and asks the user to debug the app. The
    same trace on disk is exactly what debugging needs — so the console gets one
    line and this file gets everything (see
    :func:`mnemoai.utils.logger.enable_file_logging`).

    Does NOT create the directory (unlike :func:`logs_dir`): ``/doctor`` reports
    this path and must not write anything while doing so.
    """
    return app_home() / "logs" / "mnemoai.log"


# Size cap for the MCP stderr log before it's rotated (bytes). One backup
# generation is kept (``mcp.log.1``), so on-disk use is bounded to ~2x this.
MCP_LOG_MAX_BYTES = 1_000_000

# Size cap + generations for the app log. Rotation bounds ONE noisy run (a
# DEBUG session can write megabytes); expiry below bounds a year of quiet ones.
APP_LOG_MAX_BYTES = 2_000_000
APP_LOG_BACKUPS = 2

# Days a log file is kept. Root-level config key ``LOG_MAX_AGE_DAYS`` overrides
# it; 0 disables the sweep.
LOG_MAX_AGE_DAYS = 7


def sweep_old_logs(max_age_days: int = LOG_MAX_AGE_DAYS) -> int:
    """Delete log files older than ``max_age_days``; return the count.

    Best-effort startup housekeeping (0 disables) over everything under
    ``logs/`` — the app log's rotated generations and the MCP stderr log alike,
    since neither has an owner that expires it. A file being appended to right
    now has a fresh mtime, so a live instance's log is never the one swept.
    """
    if max_age_days <= 0:
        return 0
    root = app_home() / "logs"  # not logs_dir(): a sweep must not create it
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    try:
        for f in root.iterdir():
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


def open_mcp_log():
    """Open the MCP stderr log for appending, rotating it first if it's grown
    past ``MCP_LOG_MAX_BYTES``.

    Simple single-backup rotation (``mcp.log`` -> ``mcp.log.1``, replacing any
    previous backup) keeps the log from growing without bound while preserving
    recent history for debugging. Returns a line-buffered text handle the caller
    owns (must close it). Rotation errors are non-fatal — worst case the log
    keeps appending.
    """
    path = mcp_log_path()
    try:
        if path.exists() and path.stat().st_size >= MCP_LOG_MAX_BYTES:
            backup = path.with_suffix(path.suffix + ".1")
            backup.unlink(missing_ok=True)
            path.rename(backup)
    except OSError:
        pass  # rotation is best-effort; fall through to appending
    return open(path, "a", buffering=1)


def _profile_name() -> str:
    """Resolve the active profile name from config (lazy import to avoid cycles)."""
    from mnemoai.utils.config import config

    return config.get("PROFILE", {}).get("NAME", "default")


def profile_dir(profile: str = None) -> Path:
    """Per-profile data directory (created).

    Args:
        profile: Profile name; resolved from config when omitted.
    """
    name = profile or _profile_name()
    d = app_home() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def instance_id() -> str:
    """A stable id unique to THIS running app instance (process + its MCP child).

    Multiple ``mnemoai`` instances (e.g. one per terminal tab) share the same
    profile dir, so any per-session state they hand to their MCP subprocess via a
    file must be namespaced per-instance or they clobber each other. The id is
    cached in ``$MNEMOAI_INSTANCE_ID`` so it's **inherited by the MCP subprocess**
    (which copies ``os.environ``) — both halves of one instance resolve the same
    pointer file, while a different tab gets a different one. Call this before the
    client copies the env for the subprocess so the child sees the same value.
    """
    iid = os.environ.get("MNEMOAI_INSTANCE_ID")
    if not iid:
        iid = f"{os.getpid()}_{int(time.time() * 1000) % 1_000_000:06d}"
        os.environ["MNEMOAI_INSTANCE_ID"] = iid
    return iid


def rag_session_pointer_path(profile: str = None) -> Path:
    """Per-instance file holding this instance's RAG ``session_id``.

    Namespaced by :func:`instance_id` so concurrent instances (terminal tabs)
    don't overwrite each other's pointer (the multi-tab clobber bug).
    """
    return profile_dir(profile) / f"rag_session_id_{instance_id()}.txt"


def chunk_session_pointer_path(profile: str = None) -> Path:
    """Per-instance file holding this instance's chunk-cache ``session_id``.

    Namespaced by :func:`instance_id`, like :func:`rag_session_pointer_path`.
    """
    return profile_dir(profile) / f"chunk_session_id_{instance_id()}.txt"


# Age after which orphaned RAG/chunk session artifacts are swept at startup.
# Session RAG stores + chunk caches + their pointer files are per-instance
# scratch; an instance cleans up its own on exit, but a crashed/killed instance
# can leave some behind. This bounds that so they don't accumulate — WITHOUT an
# instance deleting another live instance's files (the multi-tab delete-all bug).
# It is the FALLBACK rule: an artifact whose owning process is provably gone is
# reclaimed at once (see :func:`_artifact_is_orphaned`), and age covers only the
# names that can't be attributed to a pid.
RAG_ARTIFACT_MAX_AGE_DAYS = 7

# Upper bound on a plausible OS pid (Linux's raisable ceiling, 2^22; macOS caps
# far lower). A bigger number parsed out of a name is not a pid — some other
# layout's digits — so it must fall back to the age rule, never be treated as a
# dead process.
_MAX_PID = 4_194_304

# Trailing ``_{pid}_{ms}`` of an instance id in an artifact name, before an
# optional extension: `..._34929_928845`, `..._34929_928845.db`.
_ARTIFACT_INSTANCE_RE = re.compile(r"_(\d+)_(\d{4,})(?:\.[A-Za-z0-9]+)?$")


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a running process. Unknown counts as ALIVE.

    Signal 0 only checks; it doesn't deliver. Every error other than "no such
    process" (notably a pid owned by a different user, which raises
    ``PermissionError``) means the process is there or we can't tell — and this
    answer gates a delete, so doubt must always read as alive.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _pid_is_this_app(pid: int) -> bool:
    """Whether ``pid`` looks like another instance of THIS app.

    Liveness alone can't distinguish "that tab is still open" from "that pid was
    recycled by something unrelated months ago", and the two want opposite
    treatment: the first must be protected from the age rule forever, the second
    must not be protected at all. Reading the command line settles it.

    Used only to WIDEN protection — never to justify a delete — so every failure
    (no psutil, a process we may not inspect, a launcher whose command line
    doesn't name us) simply leaves that entry on the plain age rule.
    psutil is imported lazily: paths.py is imported by everything, including the
    MCP server's boot, and this is the one place that needs a process table.
    """
    try:
        import psutil

        return any("mnemoai" in str(part) for part in psutil.Process(pid).cmdline())
    except Exception:
        return False


def _artifact_pid(name: str) -> Optional[int]:
    """The pid embedded in a per-instance artifact name, when one can be read.

    Every one of these files is namespaced by :func:`instance_id` — ``{pid}_{ms}``
    — so the NAME says which process owned it (``chunk_session_id_34929_928845.txt``,
    ``rag_store_bpistone_20260828_140848_34929_928845``). That is what makes "is
    this tied to an app that is still open?" answerable directly, instead of
    waiting out ``RAG_ARTIFACT_MAX_AGE_DAYS`` for the leftovers of a tab someone
    closed without exiting. An unattributable name or an implausible number reads
    as None, i.e. no owner to ask about.
    """
    match = _ARTIFACT_INSTANCE_RE.search(name)
    if not match:
        return None
    pid = int(match.group(1))
    return pid if 0 < pid <= _MAX_PID else None


def _artifact_is_orphaned(name: str) -> bool:
    """Whether a per-instance artifact's owning process is provably gone.

    Deliberately one-directional: True requires a parseable, plausible pid that is
    NOT ours and NOT running, so a live instance's files can never qualify (its
    own pid is alive by definition). Everything ambiguous returns False and stays
    on the age rule.
    """
    if instance_id() in name:  # our own scratch, whatever the pid says
        return False
    pid = _artifact_pid(name)
    return pid is not None and not _pid_alive(pid)


def _artifact_is_in_use(name: str) -> bool:
    """Whether a per-instance artifact belongs to an instance still OPEN.

    The mirror of :func:`_artifact_is_orphaned`, and the reason the age rule can't
    be the whole story: an app left open for weeks stops touching its store, so
    its mtime goes stale while the tab is still using it — and another instance's
    startup sweep would then delete a live session's index. A name that
    identifies a running instance of this app therefore vetoes the age rule
    outright; age only applies to entries with no identifiable owner.

    Requires liveness AND :func:`_pid_is_this_app`, because a pid recycled by
    something unrelated must NOT protect a dead instance's leftovers forever.
    """
    if instance_id() in name:
        return True
    pid = _artifact_pid(name)
    return pid is not None and _pid_alive(pid) and _pid_is_this_app(pid)


def sweep_old_rag_artifacts(
    max_age_days: int = RAG_ARTIFACT_MAX_AGE_DAYS, profile: str = None
) -> int:
    """Delete orphaned RAG/chunk session artifacts at startup.

    Best-effort startup housekeeping (0 disables both rules). Touches only the
    per-session scratch files/dirs (``rag_store_*``, ``chunk_cache_*``,
    ``rag_session_id_*``, ``chunk_session_id_*``), and only when they are
    **orphaned**. Three rules, in order: the owner is provably gone
    (:func:`_artifact_is_orphaned`) → reclaim NOW, since a dead instance will
    never write again; the owner is still open (:func:`_artifact_is_in_use`) →
    keep, however stale, because a live session's index must not vanish under it;
    otherwise fall back to ``max_age_days``, which covers a name no owner can be
    read from. Returns the count removed.
    """
    if max_age_days <= 0:
        return 0
    d = profile_dir(profile)
    cutoff = time.time() - max_age_days * 86400
    prefixes = (
        "rag_store_",
        "chunk_cache_",
        "rag_session_id_",
        "chunk_session_id_",
    )
    removed = 0
    try:
        for entry in d.iterdir():
            if not entry.name.startswith(prefixes):
                continue
            try:
                if not _artifact_is_orphaned(entry.name):
                    if _artifact_is_in_use(entry.name):
                        continue  # a tab that is still open, however idle
                    if entry.stat().st_mtime >= cutoff:
                        continue  # recent, and the owner can't be identified
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink()
                removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


def conversations_dir(profile: str = None) -> Path:
    """Per-profile directory for saved conversations (created).

    ``/save`` writes here and ``/load`` lists from here. Lives under the profile
    dir as ``conversations/`` (see the app-home layout), keeping saved chats out
    of the profile root.

    Args:
        profile: Profile name; resolved from config when omitted.
    """
    d = profile_dir(profile) / "conversations"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Cap on the sanitized cwd dir name (longer paths get truncated + hashed).
_MAX_SANITIZED_CWD = 120

# Age after which a session transcript is swept at startup (30 days):
# resumable sessions are a convenience, not a durable artifact — `/save` is the
# durable, user-curated path and is never swept.
SESSION_MAX_AGE_DAYS = 30


def sanitize_cwd(path) -> str:
    """Make a working-directory path safe to use as a single directory name.

    Sessions are scoped to the directory you launched from, so the cwd becomes
    one flat dir name: everything outside ``[A-Za-z0-9]`` collapses to ``-``.
    Very long paths are truncated and suffixed with a short hash of the FULL
    path, so two deep directories sharing a prefix can't collide.
    """
    raw = str(path or "")
    safe = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-")
    if not safe:
        return "root"
    if len(safe) <= _MAX_SANITIZED_CWD:
        return safe
    digest = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:8]
    return f"{safe[:_MAX_SANITIZED_CWD]}-{digest}"


def sessions_dir(cwd=None, profile: str = None) -> Path:
    """Per-(profile, launch-directory) dir holding resumable session logs (created).

    Sessions are scoped to the directory the app was launched from, so
    ``--resume`` in a project offers that project's sessions and nothing else.
    Distinct from :func:`conversations_dir` — that holds user-curated ``/save``
    files, which are never swept.
    """
    base = cwd if cwd is not None else os.getcwd()
    d = profile_dir(profile) / "sessions" / sanitize_cwd(base)
    d.mkdir(parents=True, exist_ok=True)
    return d


def sweep_old_sessions(
    max_age_days: int = SESSION_MAX_AGE_DAYS, profile: str = None
) -> int:
    """Delete session logs older than ``max_age_days``; return the count.

    Best-effort startup housekeeping (0 disables). Sweeps EVERY project's
    session dir, not just the current one — otherwise a directory you stopped
    working in would keep its logs forever. Empty dirs are removed after.
    Only ``session_*.jsonl`` files are touched.
    """
    if max_age_days <= 0:
        return 0
    root = profile_dir(profile) / "sessions"
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    try:
        for project in root.iterdir():
            if not project.is_dir():
                continue
            for f in project.glob("session_*.jsonl"):
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1
                except OSError:
                    continue
            try:  # prune the dir once its last session ages out
                next(project.iterdir())
            except StopIteration:
                try:
                    project.rmdir()
                except OSError:
                    pass
            except OSError:
                pass
    except OSError:
        pass
    return removed


def memory_file_path(profile: str = None) -> Path:
    """Path to the curated ``MEMORY.md`` (profile-scoped, not auto-created).

    A small, bounded markdown file the agent maintains itself (Hermes-style) and
    that is injected whole into the system prompt at session start. Profile-
    scoped — shared across chat models — since it holds user/environment facts,
    not model-specific learnings (those live under :func:`model_dir`).
    """
    return profile_dir(profile) / "MEMORY.md"


# Accepted names for a user-authored always-on instructions file, in
# per-directory precedence order. Two names are honored so a repo that already
# carries agent instructions under the widely-used ``CLAUDE.md`` needs no second
# file to be picked up; ``STEERING.md`` is this app's own name and wins when a
# directory holds both, which is what lets a project keep the two side by side
# and steer this assistant differently from whatever wrote the other file.
STEERING_FILENAMES = ("STEERING.md", "CLAUDE.md")


def global_steering_path() -> Path:
    """Where a global steering file is WRITTEN: ``<app_home>/STEERING.md``.

    The canonical target for authoring (not auto-created). Reading is broader —
    :func:`steering_files` also accepts the other names in
    :data:`STEERING_FILENAMES` — so use that for discovery, not this.
    """
    return app_home() / "STEERING.md"


def instruction_file_in(directory: Path) -> Optional[Path]:
    """The one instructions file a single directory contributes, or None.

    Applies :data:`STEERING_FILENAMES` in order: the first name that is a
    READABLE file wins and the others are ignored **for that directory only**.
    Shadowing is deliberately per-directory rather than global, so a project
    ``STEERING.md`` doesn't suppress a global ``CLAUDE.md`` and a parent's choice
    doesn't constrain a child's.

    Readability is part of the choice, not an afterthought: a candidate that
    exists but can't be read must fall through to the next name, or an
    unreadable ``STEERING.md`` would shadow a perfectly good ``CLAUDE.md`` beside
    it and the directory would contribute nothing.
    """
    for name in STEERING_FILENAMES:
        f = directory / name
        try:
            # is_file() (never exists()): a DIRECTORY with the accepted name, or a
            # broken symlink, must not be chosen. os.access is a cheap probe that
            # avoids opening the file here.
            if f.is_file() and os.access(f, os.R_OK):
                return f
        except OSError:
            continue  # unreadable parent, bad symlink, etc. — try the next name
    return None


def _walk_boundary_dirs() -> set:
    """Dirs that never contribute an instructions file when there's no ``.git``.

    The home dir and everything above it: a file that high up is not "this
    project's conventions", and treating one as always-on would make an unrelated
    tool's file silently govern every session. Realpath-keyed to match how the
    walk spells its dirs.
    """
    out = set()
    try:
        home = Path.home().resolve()
        for d in [home, *home.parents]:
            out.add(d)
    except Exception:  # no resolvable home: fall back to no boundary
        logger.debug("Could not resolve the home dir for the walk boundary")
    return out


def steering_files(cwd: Path = None) -> list:
    """Discover always-on instruction files in precedence order (low → high).

    User-authored, always-on instructions. Every directory contributes AT MOST
    one file — see :func:`instruction_file_in` for the name precedence within a
    directory. Resolution, broadest to most specific:

      1. ``<app_home>/`` — global/user (applies everywhere)
      2. ``cwd`` walked UP to the project root (the first ancestor containing
         ``.git``, else the filesystem root), collected root-first so a deeper
         (more specific) file is applied last.

    Returns only existing files, de-duplicated by real path, in apply order.
    Tolerant: any resolution error yields what was collected so far rather than
    raising.
    """
    found: list = []
    seen: set = set()

    def _add(f: Path) -> None:
        """Collect ``f`` unless the same real file was already collected."""
        # Keyed on the real path, not the spelling: app_home() is unresolved
        # while the project walk resolves, so a symlinked home would otherwise
        # be injected twice when the walk passes through it.
        key = os.path.normcase(os.path.realpath(f))
        if key not in seen:
            seen.add(key)
            found.append(f)

    # The global tier gets its own guard so an unreadable app home can't cost the
    # project tier (the files are independent; one failing is not the other's
    # problem).
    try:
        g = instruction_file_in(app_home())
        if g is not None:
            _add(g)
    except Exception:
        logger.debug("Could not resolve a global instruction file", exc_info=True)

    try:
        start = Path(cwd).expanduser() if cwd else Path.cwd()
        start = start.resolve()
        # Walk up collecting dirs until a .git root (inclusive) or the fs root.
        # ``.exists()`` deliberately, not is_dir(): a git worktree or submodule
        # writes .git as a FILE, and the walk must still stop there.
        chain = [start] + list(start.parents)
        project_root_idx = None
        for i, d in enumerate(chain):
            try:
                if (d / ".git").exists():
                    project_root_idx = i
                    break
            except OSError:
                continue  # unreadable ancestor: keep looking for the root
        if project_root_idx is not None:
            # An explicit project root wins, even if it IS the home dir — a
            # dotfiles repo checked out at ``$HOME`` is a deliberate choice.
            ancestors = chain[: project_root_idx + 1]
        else:
            # No .git anywhere: bound the walk at the home dir instead of running
            # to the filesystem root. Unbounded, a single ``~/CLAUDE.md`` (or one
            # in ``/``) would silently become always-on instructions for EVERY
            # non-git directory under it — and a home-level instructions file is
            # something other tools already create, so this is likely, not
            # theoretical. The global tier is the app home alone; nothing else
            # gets to be global by accident.
            ancestors = [d for d in chain if d not in _walk_boundary_dirs()]
        # Apply root-first (broadest) → cwd-last (most specific). Per-directory
        # guard: one unreadable ancestor must not abandon the walk, which would
        # silently drop the MOST SPECIFIC files (the deeper ones come last).
        for d in reversed(ancestors):
            try:
                f = instruction_file_in(d)
            except Exception:
                logger.debug("Skipping instruction lookup in %s", d, exc_info=True)
                continue
            if f is not None:
                _add(f)
    except Exception:
        return found
    return found


def sanitize_model_name(name: str) -> str:
    """Make a model id safe to use as a directory name.

    Model ids contain characters that are awkward or illegal in paths
    (``/``, ``:``, spaces, etc.). Collapse anything outside ``[A-Za-z0-9._-]``
    to ``_``.
    """
    if not name:
        return "default"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return safe or "default"


def model_dir(model_name: str, profile: str = None) -> Path:
    """Per-(profile, chat-model) directory for episodic memory + playbook (created).

    Scoping memory by model keeps a store built with one model from
    contaminating another.
    """
    d = profile_dir(profile) / "models" / sanitize_model_name(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d
