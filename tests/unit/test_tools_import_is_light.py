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
search. The subprocess below is the regression: the import must stay light
regardless of what the developer's config says.
"""

import subprocess
import sys
import textwrap

REASON = "vision init must stay lazy: torch+faiss in one process aborts (OMP #15)"


def _run(body: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter with the checkout's src/ importable."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_importing_tools_package_does_not_load_torch():
    """The heavy chain must not be pulled in by the bare package import."""
    proc = _run(
        """
        import sys
        sys.path.insert(0, "src")
        import mnemoai.server.tools  # noqa: F401
        print("torch" in sys.modules)
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", REASON


def test_vision_exports_are_still_reachable():
    """Laziness must not break the names describe_image imports from the package."""
    proc = _run(
        """
        import sys
        sys.path.insert(0, "src")
        import mnemoai.server.tools as t
        # Present as attributes (resolved through the module __getattr__) without
        # having been evaluated at import time.
        assert hasattr(t, "vision_model")
        assert hasattr(t, "vision_model_controller")
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")


def test_unknown_package_attribute_still_raises():
    """The __getattr__ hook must not swallow genuine typos."""
    proc = _run(
        """
        import sys
        sys.path.insert(0, "src")
        import mnemoai.server.tools as t
        try:
            t.no_such_name
        except AttributeError:
            print("raised")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "raised"
