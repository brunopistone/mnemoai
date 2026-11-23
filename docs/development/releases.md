# Versioning and releases

## Stability & Versioning { #stability-versioning }

Mnemo AI follows [Semantic Versioning](https://semver.org/). The **public surface** that versioning protects is:

- **Config keys** in `config.yaml` (the `MODEL_ID` / `VISION_MODEL_ID` / `RAG.EMBED_MODEL_ID` fields, the `ENABLE_*` / `REQUIRE_*` toggles, and the documented section keys).
- **Prompt keys** in `prompts.yaml` (`SYSTEM_PROMPT`, `ROUTING_PROMPT`, `ORCHESTRATOR_PROMPT`, `AGGREGATOR_PROMPT`, `SUMMARY_SYSTEM_PROMPT`, `SUMMARY_TASK_PROMPT`). These live in `prompts.yaml`, not `config.yaml` (keys left in `config.yaml` are ignored with a migration warning).
- **The `mcp.json` schema** for external MCP servers (`mcpServers` with `command` / `args` / `env` / `disabled`).
- **The `hooks.json` schema** for tool hooks (the event names, `matcher` / `type` / `command` / `timeout`, and the exit-code + stdout-JSON contract) — see [Tool hooks](../guides/hooks.md).
- **CLI commands** (`/config`, `/model`, `/params`, `/features`, `/mcp`, `/hooks`, `/memory`, `/skills`, `/plan`, `/auto`, `/compact`, `/clear`, `/save`, `/load`, `/rename`, `/doctor`) and the `mnemoai` console command + its `--no-verbose`, `--auto`, `--resume [SESSION_ID]`, and `--continue` flags.
- **The distribution/import name** (`pip install mnemoai-assistant` → `import mnemoai`).

Pre-1.0.0, minor releases may add features and occasionally adjust these. From **1.0.0** onward, a breaking change to any of the above bumps the **major** version; new backward-compatible features bump the minor; fixes bump the patch. Internal modules (anything under `client/`, `server/`, `models/`, `utils/` not listed above) are **not** part of the public contract and may change between any releases. All changes are recorded in [`CHANGELOG.md`](https://github.com/brunopistone/mnemoai/blob/main/CHANGELOG.md).

## Release Checklist { #release-checklist }

Before tagging a release:

1. **Unit tests + lint pass** (also enforced by CI): `python -m pytest -m "not integration"` and `ruff check --select I .`.
2. **Integration checks** with a live model (`PYTHONPATH=src python -m pytest -m integration`, or manually drive the app and verify). Point the run at a specific provider with `MNEMOAI_CONFIG=/path/to/config.yaml`. Model behavior can vary: investigate failures, retain their evidence, and distinguish application bugs from provider or model behavior. A passing rerun does not erase an earlier failure. Verify that:
   - a greeting / simple Q&A returns a non-empty answer;
   - a tool-backed query runs a tool (e.g. "list files here") and the `[⚙ …]` marker fires;
   - routing handles ordinary tasks directly and useful independent work can be delegated and completed;
   - **plan mode**: `/plan` on → an edit/bash request is blocked; `/plan` off → it proceeds;
   - an external MCP tool (from `mcp.json`) is callable.
3. **Update `CHANGELOG.md`** — move `Unreleased` items under the new version + date.
4. **Bump `version`** in `pyproject.toml`.
5. **Build + validate**: `uv build` then `twine check dist/*`.
6. **Publish the authorized release** using the repository's current Git and release workflow. Verify CI and the GitHub artifacts. A GitHub release and a PyPI upload are separate publication steps; run `twine upload dist/*` only when PyPI publication is part of the release.
