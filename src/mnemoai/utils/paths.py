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
    ├── plans/plan_<ts>.md                  # approved plan-mode plans
    ├── skills                              # skills folder
    ├── STEERING.md                         # user-authored always-on instructions
    ├── tasks/                              # background-task output
    └── {profile}/                          # per-user-profile data
        ├── conversations/  todos/  rag_*  chunk_cache_*  profile JSON
        └── models/{model}/                # per-chat-model memory
            ├── episodic_memory/
            └── playbook/

Override the root with ``$MNEMOAI_HOME``. The config file location
can additionally be overridden with ``$MNEMOAI_CONFIG``.
"""

import os
import re
import shutil
import time
from pathlib import Path

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
    "fe423553bed74ced37b53503e7249fb1981528d8fdda206341782968699f2267",  # ≤1.5.0
}


def _sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes (matches ``shasum -a 256``)."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_pristine_prompts(src: Path, dest: Path) -> None:
    """Refresh the live ``prompts.yaml`` in place IF the installed copy is pristine
    (a version we shipped, unmodified); otherwise leave the user's customized file.

    Unlike ``*.example`` files (always refreshed) and unlike the never-touched
    ``config.yaml``, ``prompts.yaml`` IS loaded as config but is rarely customized —
    so refreshing a pristine copy lets prompt improvements (edits to existing keys,
    which the bundled-fallback loader can't deliver since it only fills MISSING
    keys) reach existing installs on upgrade, without clobbering a user's edits."""
    if not src.is_file() or not dest.is_file():
        return
    try:
        installed = _sha256(dest)
        if installed == _sha256(src):
            return  # already current
        if installed in _PRISTINE_BUNDLED_PROMPTS_HASHES:
            shutil.copyfile(src, dest)  # pristine → safe to refresh
        # else: user-customized → leave untouched
    except OSError:
        pass


def _refresh_pristine_skill(src_dir: Path, dest_dir: Path) -> None:
    """Refresh a bundled skill's ``SKILL.md`` in place IF the installed copy is
    pristine (a version we shipped, unmodified by the user); otherwise leave it.

    Complements the copy-if-absent seeding: an already-installed bundled skill
    still gets doc/frontmatter updates on upgrade, but a user's own edits are
    never overwritten. Only ``SKILL.md`` is touched, so any extra files the user
    added alongside it are preserved.
    """
    src_md = src_dir / "SKILL.md"
    dest_md = dest_dir / "SKILL.md"
    if not src_md.is_file() or not dest_md.is_file():
        return
    try:
        installed = _sha256(dest_md)
        if installed == _sha256(src_md):
            return  # already current — nothing to do
        if installed in _PRISTINE_BUNDLED_SKILL_HASHES.get(dest_dir.name, set()):
            shutil.copyfile(src_md, dest_md)  # pristine → safe to refresh
        # else: user-edited → leave untouched
    except OSError:
        pass


def seed_example_files() -> None:
    """Copy the package's bundled ``*.example`` templates into the app home.

    Gives users browsable examples right next to their live files:
    ``config/`` gets the ``config.yaml*.example`` templates and ``mcp/`` gets
    ``mcp.json.example``. The ``*.example`` reference files are **refreshed from
    the bundle when they differ** so a new bundled key reaches an EXISTING install
    on upgrade (they're read-only reference, not loaded as config). Bundled example
    skills are copied when absent, and an already-installed one whose ``SKILL.md``
    is still **pristine** (a version we shipped, unmodified) is refreshed in place
    so doc/frontmatter updates also reach existing installs. ``prompts.yaml`` is
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


# Size cap for the MCP stderr log before it's rotated (bytes). One backup
# generation is kept (``mcp.log.1``), so on-disk use is bounded to ~2x this.
MCP_LOG_MAX_BYTES = 1_000_000


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


def memory_file_path(profile: str = None) -> Path:
    """Path to the curated ``MEMORY.md`` (profile-scoped, not auto-created).

    A small, bounded markdown file the agent maintains itself (Hermes-style) and
    that is injected whole into the system prompt at session start. Profile-
    scoped — shared across chat models — since it holds user/environment facts,
    not model-specific learnings (those live under :func:`model_dir`).
    """
    return profile_dir(profile) / "MEMORY.md"


def global_steering_path() -> Path:
    """The user-level ``STEERING.md`` at the app-home root (not auto-created)."""
    return app_home() / "STEERING.md"


def steering_files(cwd: Path = None) -> list:
    """Discover STEERING.md files in precedence order (low → high priority).

    User-authored, always-on instructions. Resolution, broadest to most
    specific:

      1. ``<app_home>/STEERING.md`` — global/user (applies everywhere)
      2. ``./STEERING.md`` walking from ``cwd`` UP to the project root (the first
         ancestor containing ``.git``, else the filesystem root), collected
         root-first so a deeper (more specific) file is applied last.

    Returns only existing files, de-duplicated, in apply order. Tolerant: any
    resolution error yields an empty list rather than raising.
    """
    found: list = []
    try:
        g = global_steering_path()
        if g.is_file():
            found.append(g)

        start = Path(cwd).expanduser() if cwd else Path.cwd()
        start = start.resolve()
        # Walk up collecting dirs until a .git root (inclusive) or the fs root.
        chain = [start] + list(start.parents)
        project_root_idx = None
        for i, d in enumerate(chain):
            if (d / ".git").exists():
                project_root_idx = i
                break
        ancestors = chain[: project_root_idx + 1] if project_root_idx is not None else chain
        # Apply root-first (broadest) → cwd-last (most specific).
        for d in reversed(ancestors):
            f = d / "STEERING.md"
            if f.is_file() and f not in found:
                found.append(f)
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
