"""Mutable state belonging to one built-in tool server connection."""

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_current = ContextVar("mnemoai_tool_server", default=None)
_cancel = ContextVar("mnemoai_tool_cancel", default=None)
CANCEL_META_KEY = "mnemoai/cancellation"


@dataclass
class ToolServerState:
    initial_cwd: str = field(default_factory=os.getcwd)
    cwd: str | None = None
    read_versions: dict = field(default_factory=dict)
    background_tasks: dict = field(default_factory=dict)
    task_lock: Any = field(default_factory=threading.Lock)
    rag_session: Any = None
    calls: dict = field(default_factory=dict)
    pending: set = field(default_factory=set)


def current():
    return _current.get()


def cancel_event():
    return _cancel.get()


def check_cancelled():
    event = cancel_event()
    if event is not None and event.is_set():
        raise KeyboardInterrupt("Tool request cancelled")


@contextmanager
def request_scope(event=None):
    if event is None and current() is not None:
        event = threading.Event()
    token = _cancel.set(event)
    try:
        yield event
    except BaseException:
        if event is not None:
            event.set()
        raise
    finally:
        _cancel.reset(token)


@contextmanager
def use(state):
    token = _current.set(state)
    try:
        yield state
    finally:
        _current.reset(token)
