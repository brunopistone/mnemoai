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

import subprocess
import sys
import textwrap

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


def test_describe_image_binds_the_real_model_not_a_stale_none():
    """The module-level ``from . import vision_model`` must not capture None.

    ``describe_image`` binds the value at ITS import time, which the lazy
    ``__getattr__`` has to satisfy — if the name resolved before initialization
    the tool would hold None forever and every call would fail, with vision
    correctly configured. ``register_tools`` imports it lazily, which is what
    keeps the package import light while still giving the tool a live model.
    """
    proc = _run(
        """
        import mnemoai.server.tools as t
        import mnemoai.server.tools.describe_image as di
        print([di.vision_model, di.vision_model_controller is not None])
        """
    )
    assert _last_line(proc) == "['STUB_VISION_MODEL', True]"


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
    permanent and silent for the one exception type that matters: ``describe_image``
    binds these names through the package ``__getattr__``, and importlib's
    ``_handle_fromlist`` probes with ``hasattr``, which SWALLOWS an AttributeError.
    The pre-set flag then let the retry return None — binding a dead model forever
    and dropping the tool from the registered set with nothing logged. The flag now
    goes up only after success, so every access re-raises.
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
