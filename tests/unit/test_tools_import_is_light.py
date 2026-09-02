"""Importing the tools package must not initialize the vision model.

``server/tools/__init__`` builds a module-level ``ToolManager``. It used to also
*use* it — binding ``vision_model``/``vision_model_controller`` at import time —
which ran the vision initialization whenever a ``VISION_MODEL_ID`` was
configured, dragging BaseChatModel->transformers->torch into any process that
touched ``mnemoai.server.tools.*``.

That is not merely slow. torch and faiss each vendor their own OpenMP runtime,
and loading both aborts the interpreter outright (``OMP: Error #15``) the moment
faiss runs a search. It killed the pure-logic RAG tests on any interpreter with
torch installed, and the same collision aborts the app's own episodic-memory
search. The import must stay light regardless of what the config says.

**"Lazy" is a property of every entry point, not of one module.** Keeping the
package import light achieved nothing for the real server, which calls
``register_tools`` at module level: the gate deciding whether to register
``describe_image`` asked ``get_vision_model()``, and the tool module bound the
vision names at ITS import — so the build happened at every boot anyway, one or
two frames further down. Both are covered here, and the last test scans the
source so a re-introduced binding fails even in a module these snippets never
import.

**These tests must hold on a machine that has neither torch nor a
``config.yaml``** — i.e. CI, and any fresh clone. Two ways an earlier version of
this file failed to:

* Asserting ``"torch" not in sys.modules`` only proves anything where torch is
  INSTALLED. It isn't a dependency of this project, so in CI that assertion
  passes no matter what the code does. The real property is "the vision
  controller is not CONSTRUCTED", so these stub the controller with a sentinel
  that records its own construction, which needs no heavy packages.
* The initialization is skipped entirely unless ``VISION_MODEL_ID`` is set, so a
  run with no config exercised the trivial path. Each subprocess below injects a
  vision config of its own rather than depending on the developer's.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

REASON = "vision init must stay lazy: torch+faiss in one process aborts (OMP #15)"

# Makes the tools package think vision is configured, and replaces the heavy
# controller with a sentinel that reports when it is constructed. Prepended to
# every snippet so the assertions are about the code, not the environment.
PRELUDE = """
import sys, types
sys.path.insert(0, "src")

import mnemoai.utils.config as cfgmod
_real_get = cfgmod.config.get
cfgmod.config.get = lambda k, d=None: (
    {"NAME": "stub-vision", "TYPE": "ollama"} if k == "VISION_MODEL_ID"
    else _real_get(k, d)
)

_events = []
_mod = types.ModuleType("mnemoai.models.controllers.vision_model_controller")
class VisionModelController:
    def __init__(self):
        _events.append("construct")
    def initialize_model(self):
        _events.append("initialize")
    def get_model(self):
        return "STUB_VISION_MODEL"
_mod.VisionModelController = VisionModelController
sys.modules["mnemoai.models.controllers.vision_model_controller"] = _mod
"""


def _run(body: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter with the checkout's src/ importable."""
    # Dedent the BODY alone: PRELUDE is already flush-left, so dedenting the
    # concatenation finds no common prefix and leaves the body indented.
    return subprocess.run(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def _last_line(proc: subprocess.CompletedProcess) -> str:
    """The snippet's own last line of output.

    Importing the package legitimately prints to STDOUT first when there is no
    runtime ``config.yaml`` — the config loader says so, which is the normal
    state in CI. Comparing the whole buffer failed for that reason alone.
    """
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1].strip()


def test_importing_tools_package_does_not_initialize_vision():
    """The bare package import must not construct the vision controller."""
    proc = _run(
        """
        import mnemoai.server.tools  # noqa: F401
        print(_events)
        """
    )
    assert _last_line(proc) == "[]", REASON


def test_importing_tools_package_does_not_load_torch():
    """The heavy chain must not be pulled in by the bare package import.

    Weaker than the sentinel above wherever torch isn't installed (it then can't
    fail), but it is the only check that covers the whole transitive import
    chain — including a path that reaches torch without going through the
    controller we stub out.
    """
    proc = _run(
        """
        import mnemoai.server.tools  # noqa: F401
        print("torch" in sys.modules or "transformers" in sys.modules)
        """
    )
    assert _last_line(proc) == "False", REASON


def test_registering_tools_does_not_initialize_vision():
    """Registration is the entry point that actually runs in the server.

    This is where the laziness was being undone: the gate deciding whether to
    register ``describe_image`` asked ``get_vision_model()``, so every server boot
    with a ``VISION_MODEL_ID`` built the controller — transformers/torch, ~2.5s —
    to decide whether to skip the import that would have built it. The existing
    ``register_tools`` suite couldn't see it because its stub controller is free
    to construct; this one records construction, and the tool must still be
    registered.
    """
    proc = _run(
        """
        import asyncio
        from mcp.server.fastmcp import FastMCP
        import mnemoai.utils.config as cfgmod
        import mnemoai.server.tools as t

        # Only the vision gate is under test: keep the other optional groups off
        # so this doesn't register (or import) them on a developer's own config.
        _configured = cfgmod.config.get
        cfgmod.config.get = lambda k, d=None: (
            False
            if k in ("ENABLE_RAG", "ENABLE_WEB_CRAWL", "ENABLE_WEB_SEARCH")
            else _configured(k, d)
        )

        mcp = FastMCP("probe")
        t.tool_manager.register_tools(mcp)
        names = sorted(x.name for x in asyncio.run(mcp.list_tools()))
        print([_events, "describe_image" in names])
        """
    )
    assert _last_line(proc) == "[[], True]", REASON


def test_registering_tools_does_not_load_torch():
    """Same gate, checked over the whole transitive import chain.

    Not a weaker copy of the sentinel above: registration imports ~20 tool
    modules, and the stub controller can't see one that reaches torch by another
    route — which is the shape of every gate bypass found so far (``memory_tool``
    → the client package → agent.py → torch). Vacuous where torch isn't
    installed, hence both tests.
    """
    proc = _run(
        """
        from mcp.server.fastmcp import FastMCP
        import mnemoai.utils.config as cfgmod
        import mnemoai.server.tools as t

        _configured = cfgmod.config.get
        cfgmod.config.get = lambda k, d=None: (
            False
            if k in ("ENABLE_RAG", "ENABLE_WEB_CRAWL", "ENABLE_WEB_SEARCH")
            else _configured(k, d)
        )

        t.tool_manager.register_tools(FastMCP("probe"))
        print("torch" in sys.modules or "transformers" in sys.modules)
        """
    )
    assert _last_line(proc) == "False", REASON


def test_vision_is_initialized_on_first_use():
    """Laziness must still yield a working model when something asks for one."""
    proc = _run(
        """
        import mnemoai.server.tools as t
        before = list(_events)
        model = t.vision_model
        print([before, _events, model])
        """
    )
    assert _last_line(proc) == "[[], ['construct', 'initialize'], 'STUB_VISION_MODEL']"


def test_vision_is_initialized_at_most_once():
    """Repeated access must not rebuild the model."""
    proc = _run(
        """
        import mnemoai.server.tools as t
        t.vision_model
        t.vision_model
        t.vision_model_controller
        print(_events)
        """
    )
    assert _last_line(proc) == "['construct', 'initialize']"


def test_importing_describe_image_does_not_initialize_vision():
    """The tool MODULE must import as cheaply as the package.

    It used to bind ``vision_model``/``vision_model_controller`` at its own
    import time, so importing the module built the model — and then held that one
    object for the life of the process. Registration imports this module, so the
    package staying light bought nothing: the server paid the build at boot
    anyway, one frame further down. The names are resolved per call now.
    """
    proc = _run(
        """
        import mnemoai.server.tools.describe_image  # noqa: F401
        print(_events)
        """
    )
    assert _last_line(proc) == "[]", REASON


def test_describe_image_resolves_a_live_model_on_the_call():
    """Deferring the build must still leave the tool a working model.

    The counterpart to the test above: laziness that never resolves would fail
    every call with vision correctly configured, which is exactly the failure the
    old import-time binding existed to avoid.
    """
    proc = _run(
        """
        import json, os, tempfile
        from mnemoai.server.tools.describe_image import register_image_tools

        # Teach the sentinel enough to answer one real call.
        VisionModelController.format_request = lambda self, q, b, e: {"q": q, "ext": e}
        VisionModelController._content_to_text = lambda self, c: c
        class _Reply:
            content = "a red square"
        class _Model:
            def invoke(self, messages):
                return _Reply()
        VisionModelController.get_model = lambda self: _Model()

        captured = {}
        class FakeMCP:
            def tool(self, *a, **k):
                def deco(fn):
                    captured[fn.__name__] = fn
                    return fn
                return deco

        register_image_tools(FakeMCP())
        at_registration = list(_events)

        path = os.path.join(tempfile.mkdtemp(), "x.png")
        with open(path, "w") as f:
            f.write("not really a png; only the extension is checked")
        out = json.loads(captured["describe_image"](path, "what is it?"))
        print([at_registration, _events, out.get("description")])
        """
    )
    assert _last_line(proc) == "[[], ['construct', 'initialize'], 'a red square']"


def test_vision_exports_are_still_reachable():
    """Laziness must not break the names describe_image imports from the package."""
    proc = _run(
        """
        import mnemoai.server.tools as t
        assert hasattr(t, "vision_model")
        assert hasattr(t, "vision_model_controller")
        print("ok")
        """
    )
    assert _last_line(proc) == "ok"


def test_unknown_package_attribute_still_raises():
    """The __getattr__ hook must not swallow genuine typos."""
    proc = _run(
        """
        import mnemoai.server.tools as t
        try:
            t.no_such_name
        except AttributeError:
            print("raised")
        """
    )
    assert _last_line(proc) == "raised"


def test_a_failed_vision_init_stays_loud():
    """A raising init must NOT be remembered as "done".

    ``_vision_ready`` used to be set BEFORE the build, which made a failure both
    permanent and silent: the retry returned None instead of re-raising, so the
    tool held a dead model with nothing logged anywhere. It was worst through the
    old import-time binding (importlib's ``_handle_fromlist`` probes with
    ``hasattr``, which SWALLOWS an AttributeError, so the tool also vanished from
    the registered set) — that binding is gone, but a silent None still fails
    every call, so the flag goes up only after success and each access re-raises.
    """
    proc = _run(
        """
        import mnemoai.server.tools as t

        def boom(self):
            raise AttributeError("provider surface changed")
        VisionModelController.initialize_model = boom

        seen = []
        for _ in range(2):
            try:
                t.tool_manager.vision_model
                seen.append("silent-None")
            except AttributeError:
                seen.append("raised")
        print(seen)
        """
    )
    assert _last_line(proc) == "['raised', 'raised']"


def test_a_failed_init_leaves_no_half_built_controller():
    """A raise must not leave a partly-initialized controller for the next caller."""
    proc = _run(
        """
        import mnemoai.server.tools as t

        def boom(self):
            raise RuntimeError("no credentials")
        VisionModelController.initialize_model = boom

        try:
            t.tool_manager.vision_model
        except RuntimeError:
            pass
        print([t.tool_manager._vision_model, t.tool_manager._vision_model_controller])
        """
    )
    assert _last_line(proc) == "[None, None]"


def test_concurrent_first_use_builds_one_controller():
    """First use is now a tool CALL, and tool calls run on concurrent threads.

    ``ThreadedToolServer`` offloads every sync tool body to a worker thread and the
    SDK dispatches them in parallel, so two ``describe_image`` calls in one wave
    reach the unbuilt model together. Unguarded, each builds its own controller —
    two provider clients, and two copies of whatever the build loads.
    """
    proc = _run(
        """
        import threading, time
        import mnemoai.server.tools as t

        _init = VisionModelController.initialize_model
        def slow(self):
            time.sleep(0.2)  # widen the window another thread can slip into
            _init(self)
        VisionModelController.initialize_model = slow

        together = threading.Barrier(4)
        def hit():
            together.wait()
            t.tool_manager.vision_model

        threads = [threading.Thread(target=hit) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        print(_events)
        """
    )
    assert _last_line(proc) == "['construct', 'initialize']"


def test_no_tool_module_binds_the_vision_exports_at_import():
    """Source guard, because a behavioral test only covers what it imports.

    The eager binding was one line in one tool module, and it silently undid the
    laziness for the whole server. This covers every module under
    ``server/tools/`` — including one added later — at MODULE level only, which is
    the only level that's wrong: inside a tool body is exactly where these names
    belong.
    """
    tools_dir = Path(__file__).resolve().parents[2] / "src/mnemoai/server/tools"
    assert tools_dir.is_dir(), tools_dir
    lazy = {"vision_model", "vision_model_controller"}
    offenders = []
    for path in sorted(tools_dir.rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if lazy & {alias.name for alias in node.names}:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], (
        "these bind the vision exports at import time, which builds the model "
        f"while the server is registering tools: {offenders} — read them off "
        "tool_manager inside the tool body instead"
    )
