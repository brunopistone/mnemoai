"""Shared pytest fixtures and path setup.

Puts the repo's ``src/`` dir on ``sys.path`` so ``import mnemoai``
resolves when running the tests from a checkout (no install step needed).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
