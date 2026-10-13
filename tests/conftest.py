"""Shared pytest fixtures and path setup.

Puts the repo's ``src/`` dir on ``sys.path`` so ``import mnemoai``
resolves when running the tests from a checkout (no install step needed).
"""

import importlib
import os
import socket
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mnemoai.client import hooks  # noqa: E402 — needs the sys.path insert above


@pytest.fixture(autouse=True)
def _no_user_hooks(monkeypatch):
    """Never let a test run the hooks configured on this machine.

    Hooks are shell commands read from the real app home, and every tool call
    fires them — so a developer with a `hooks.json` would have their own hooks
    executed by the suite (and could see tests denied by their own rules). Pin an
    empty snapshot; `tests/unit/test_hooks.py` resets it and supplies its own.
    """
    monkeypatch.setattr(hooks, "_snapshot", hooks.Registry())


@pytest.fixture
def sdk_http_modules(monkeypatch):
    """Mock every installed SDK transport family; never escape to real DNS/network."""
    modules = [importlib.import_module("httpx")]
    try:
        modules.append(importlib.import_module("httpx2"))
    except ModuleNotFoundError:
        pass

    def forbidden(*args, **kwargs):
        pytest.fail("An SDK transport mock was bypassed; real network access is forbidden")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    return modules
