"""Shared pytest fixtures and path setup.

Ensures the repo root is importable so `import utils...`, `import client...`,
and `import server...` work regardless of where pytest is invoked from.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
