# Testing { #testing }

The test suite uses `pytest` and is split into two tiers under `tests/`:

- **`tests/unit/`** — deterministic checks for logic, provider adapters, file and shell tools, memory, and conversation behavior. They require no live model or provider credentials. Some fixtures use local subprocesses or loopback connections; the complete suite can take several minutes.
- **`tests/integration/`** — end-to-end tests that drive the real agent against a live provider and its built-in MCP tools (routing, tool calls, bash timeout, no silent empty turns). Marked with `@pytest.mark.integration` and **auto-skipped** unless a resolvable runtime config exists (for example `MNEMOAI_CONFIG=/path/to/config.yaml`, `~/.mnemoai/config/config.yaml`, or the checkout fallback `src/mnemoai/utils/config.yaml`) and the configured local provider is reachable.

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run everything (live cases skip when their prerequisites are unavailable)
python -m pytest

# Unit tier only (good for CI and local checks)
python -m pytest tests/unit

# Integration tier only (requires a working provider and runtime configuration)
MNEMOAI_CONFIG=/path/to/config.yaml python -m pytest -m integration

# Run a single file
python -m pytest tests/unit/test_bm25.py
```

When adding new code, keep import-time side effects independent of `config.yaml` so the module stays unit-testable.

**CI.** The `.github/workflows/tests.yml` workflow runs the unit tier and a Ruff
import-sort check on every push and pull request across Python 3.11–3.13.
The integration tier is not run in CI because it needs configured live providers.
Run it locally and record passes, skips, failures, and the selected provider.
See the [release checklist](releases.md#release-checklist).
