"""A disabled tool group must not pay for its dependencies.

`register_tools` gates `describe_image` (→ transformers/torch) and `.rag`
(→ faiss) behind `VISION_MODEL_ID` / `ENABLE_RAG`. Those gates only mean
anything if the IMPORT is inside them: an import creates the module object and
runs its side effects — including OpenMP registration — so importing above the
gate and then declining to register buys nothing. torch and faiss each vendor
their own OpenMP runtime, and a process holding both aborts (`OMP: Error #15`)
the moment faiss searches, which is the failure these gates exist to avoid.

Three import chains defeated the gates while still passing a naive
"is the import inside the if?" reading, so this asserts the OUTCOME
(`sys.modules`) instead:

* `client/__init__` eagerly re-exported `LangGraphClient`, so the server's
  `memory_tool`/`skill_tool` — which import `mnemoai.client.memory.*` for the
  shared stores — pulled the whole client, agent.py, and torch.
* `pdf_reader`/`docx_reader` probed `..rag` at import time just to set a
  `_rag_available` bool, and `fs_read` imports every reader unconditionally.
* `web_crawler` did the same probe.

Each subprocess runs on a synthetic config, so the result does not depend on the
developer's `config.yaml`. These assertions are meaningful whether or not torch
and faiss are installed: an ABSENT module can't be in `sys.modules`, so the
"stays out" direction can't false-pass, and the "is loaded when enabled" case is
skipped when the package is missing rather than asserted vacuously.
"""

import subprocess
import sys
import textwrap

HEAVY = ("torch", "faiss", "transformers")

# A whole config, so nothing here reads the developer's file. VISION_MODEL_ID and
# ENABLE_RAG are substituted per case; every other key falls through to the real
# loader (which is also what makes this work with no config.yaml at all).
PRELUDE = """
import sys, types, json
sys.path.insert(0, "src")

import mnemoai.utils.config as cfgmod
_real_get = cfgmod.config.get
_OVERRIDES = json.loads({overrides!r})
cfgmod.config.get = lambda k, d=None: (
    _OVERRIDES[k] if k in _OVERRIDES else _real_get(k, d)
)

# Keep the vision controller cheap: this suite is about the IMPORT graph, not
# about whether a real provider can be reached.
_mod = types.ModuleType("mnemoai.models.controllers.vision_model_controller")
class VisionModelController:
    def initialize_model(self):
        pass
    def get_model(self):
        return "STUB_VISION_MODEL"
_mod.VisionModelController = VisionModelController
sys.modules["mnemoai.models.controllers.vision_model_controller"] = _mod

from mcp.server.fastmcp import FastMCP
import mnemoai.server.tools as t
"""

BODY = """
t.tool_manager.model_id = _OVERRIDES.get("VISION_MODEL_ID")
t.tool_manager._vision_ready = False
mcp = FastMCP("probe")
t.tool_manager.register_tools(mcp)
import asyncio
names = sorted(x.name for x in asyncio.run(mcp.list_tools()))
loaded = [m for m in {heavy!r} if m in sys.modules]
print(json.dumps({{"heavy": loaded, "tools": names}}))
"""


def _register(vision, rag):
    """Run `register_tools` in a fresh interpreter; return its JSON verdict."""
    import json

    overrides = json.dumps(
        {
            "VISION_MODEL_ID": {"NAME": "stub", "TYPE": "ollama"} if vision else None,
            "ENABLE_RAG": rag,
            # Irrelevant here and each would need a key or a network call.
            "ENABLE_WEB_SEARCH": False,
            "ENABLE_WEB_CRAWL": False,
        }
    )
    script = PRELUDE.format(overrides=overrides) + BODY.format(heavy=list(HEAVY))
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _installed(mod):
    """Is a heavy package importable here at all?"""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {mod}"], capture_output=True, text=True
    )
    return proc.returncode == 0


def test_both_groups_off_loads_neither_runtime():
    """The case that matters: no vision, no RAG, so neither OpenMP runtime."""
    verdict = _register(vision=False, rag=False)
    assert verdict["heavy"] == [], (
        "a disabled group still imported its dependency: "
        f"{verdict['heavy']} — the gate is bypassed by an import above it"
    )


def test_both_groups_off_still_registers_everything_else():
    """Declining the heavy deps must not cost any unconditional tool."""
    verdict = _register(vision=False, rag=False)
    for expected in ("fs_read", "execute_bash", "git_safe", "todo_write", "use_skill"):
        assert expected in verdict["tools"]
    # 23 unconditional + memory + use_skill (both default-on, no heavy deps).
    assert len(verdict["tools"]) == 25, verdict["tools"]


def test_rag_off_keeps_faiss_out_even_though_fs_read_imports_every_reader():
    """`fs_read` pulls in pdf/docx readers, which used to probe `..rag` → faiss."""
    verdict = _register(vision=False, rag=False)
    assert "faiss" not in verdict["heavy"]
    assert "fs_read" in verdict["tools"]


def test_vision_off_keeps_torch_out_even_though_memory_tool_imports_the_client():
    """`memory_tool` imports `mnemoai.client.memory.*`, which used to pull agent.py."""
    verdict = _register(vision=False, rag=False)
    assert "torch" not in verdict["heavy"]
    assert "transformers" not in verdict["heavy"]
    assert "memory" in verdict["tools"]


def test_enabling_rag_registers_its_tools():
    """The gate must still let the group through when it is switched on."""
    verdict = _register(vision=False, rag=True)
    for expected in ("list_documents", "search_in_documents", "clear_documents"):
        assert expected in verdict["tools"]
    if _installed("faiss"):
        assert "faiss" in verdict["heavy"], "RAG on should load its own dependency"


def test_enabling_vision_registers_describe_image():
    """Same for the vision group (the controller itself is stubbed cheap)."""
    verdict = _register(vision=True, rag=False)
    assert "describe_image" in verdict["tools"]


def test_the_client_package_does_not_eagerly_build_the_client():
    """Importing a client SUBMODULE must not drag in the whole client.

    The server does this for the shared MEMORY.md / skills stores. The eager
    `from .client import LangGraphClient` in `client/__init__` meant every such
    import also loaded agent.py and its provider chain.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                sys.path.insert(0, "src")
                import mnemoai.client.memory.memory_store  # noqa: F401
                print([m for m in ("torch", "transformers") if m in sys.modules])
                """
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "[]"


def test_the_lazy_client_export_still_resolves():
    """Laziness must not break `from mnemoai.client import LangGraphClient`."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                sys.path.insert(0, "src")
                from mnemoai.client import LangGraphClient
                print(LangGraphClient.__name__)
                """
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "LangGraphClient"


def test_unknown_client_package_attribute_still_raises():
    """The lazy hook must not turn a typo into something importable."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                sys.path.insert(0, "src")
                import mnemoai.client as c
                try:
                    c.NoSuchThing
                except AttributeError:
                    print("raised")
                """
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "raised"
