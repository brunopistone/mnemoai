"""Request-local workspace identity shared across the client/MCP boundary."""

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from mnemoai.utils.path_utils import clean_path_syntax

_current = ContextVar("mnemoai_workspace", default=None)
META_KEY = "mnemoai/workspace"


@dataclass
class Workspace:
    cwd: str

    def resolve(self, value):
        path = os.path.expanduser(clean_path_syntax(value))
        return os.path.abspath(os.path.join(self.cwd, path))

    def observe(self, result):
        try:
            data = json.loads(result) if isinstance(result, str) else result
            cwd = data.get("cwd") if isinstance(data, dict) else None
            if isinstance(cwd, str) and os.path.isabs(cwd) and os.path.isdir(cwd):
                self.cwd = cwd
        except (ValueError, OSError):
            pass


def current():
    return _current.get()


def for_agent(agent):
    value = current() or getattr(agent, "_workspace", None)
    return value if isinstance(value, Workspace) else None


@contextmanager
def use(workspace):
    token = _current.set(workspace)
    try:
        yield workspace
    finally:
        _current.reset(token)


@contextmanager
def fork(agent):
    parent = for_agent(agent)
    with use(Workspace(parent.cwd) if parent is not None else None) as workspace:
        yield workspace
