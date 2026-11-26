"""Bounded, per-actor detection of repeated failures and unchanged observations."""

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass

from mnemoai.client.agent.plan_policy import is_readonly_bash
from mnemoai.utils.shell_syntax import simple_command_tokens
from mnemoai.utils.tool_results import is_error_result, is_unchanged_result

_READS = {
    "fs_read",
    "glob_search",
    "grep_search",
    "web_search",
    "web_crawl",
    "web_crawler",
    "tavily_extract",
    "tavily_crawl",
    "tavily_map",
}
_POLLING = {
    "get_task_status",
    "get_task_output",
    "list_background_tasks",
    "wait_for_task",
}
REPEAT_LIMIT = 3
RECOVERY_LIMIT = 3


def _waiting(result):
    try:
        data = json.loads(result) if isinstance(result, str) else result
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    active = ("running", "queued", "pending", "starting")
    tasks = data.get("tasks")
    return data.get("status") in active or (
        isinstance(tasks, list)
        and any(
            isinstance(task, dict) and task.get("status") in active for task in tasks
        )
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).digest()


def _arguments(name, args):
    """Ignore display labels/defaults, without reinterpreting shell expansions."""
    result = dict(args)
    if name in {"fs_write", "file_edit"}:
        result.pop("summary", None)
    if name == "fs_read":
        for key, default in {
            "mode": "Line",
            "start_line": 1,
            "end_line": -1,
            "pattern": "",
            "context_lines": 2,
            "depth": 0,
        }.items():
            if result.get(key) == default:
                result.pop(key)
    if name == "file_edit" and result.get("replace_all") is False:
        result.pop("replace_all")
    if name == "execute_bash":
        command = str(result.get("command", ""))
        if not any(char in command for char in "$`*?["):
            tokens = simple_command_tokens(command)
            if tokens is not None:
                result["command"] = tokens
    return result


def _observed(name, result):
    try:
        data = json.loads(result) if isinstance(result, str) else result
    except (ValueError, TypeError):
        return result
    if not isinstance(data, dict):
        return data
    data = {k: v for k, v in data.items() if k not in {"summary", "progress_notice"}}
    if name == "fs_read" and not is_error_result(data) and "content" in data:
        for key in (
            "start_line",
            "end_line",
            "lines_processed",
            "lines_requested",
            "tokens",
            "max_tokens",
            "message",
        ):
            data.pop(key, None)
    return data


def _version(name, args):
    if name not in {"fs_read", "glob_search", "grep_search", "fs_write", "file_edit"}:
        return None
    path = args.get("file_path" if name == "file_edit" else "path")
    if not isinstance(path, str):
        return None
    try:
        info = os.stat(os.path.expanduser(path))
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    except (OSError, ValueError):
        return None


@dataclass
class Observation:
    version: object
    count: int
    revision: int


class ToolProgressGuard:
    def __init__(self):
        self.observations = OrderedDict()
        self.requests = OrderedDict()
        self.refusals = 0
        self.exhausted = False
        self.last_tool = ""
        self.revision = 0

    def repeated(self, name, args):
        if not isinstance(args, dict):
            return False
        previous = self.observations.get(
            self.requests.get(_digest([name, _arguments(name, args)]))
        )
        return (
            previous is not None
            and previous.count >= REPEAT_LIMIT
            and previous.version == _version(name, args)
            and previous.revision == self.revision
        )

    def refuse(self, name):
        self.refusals += 1
        self.exhausted = self.refusals >= RECOVERY_LIMIT
        self.last_tool = name
        return (
            f"Blocked: the same {name} call already returned unchanged output or the same failure "
            f"{REPEAT_LIMIT} times. This call was not executed. Reuse the existing result and "
            "choose a different, relevant next step, correct the arguments, or explain the blocker. "
            "Do not bypass permission or file-staleness checks using another tool."
        )

    def observe(self, name, args, result, outcome="completed"):
        if outcome in {"cancelled", "stalled"} or not isinstance(args, dict):
            return
        failed = outcome not in {"completed", "client_result"} or is_error_result(
            result
        )
        if name in _POLLING and _waiting(result):
            return
        unchanged = is_unchanged_result(result)
        readonly = name in _READS | _POLLING or (
            name == "execute_bash" and is_readonly_bash(str(args.get("command", "")))
        )
        if not failed and not readonly and not unchanged:
            # Opaque tools may have side effects even when they return "OK".
            # Never mistake repeated successful writes/tests for a read loop.
            self.observations.clear()
            self.requests.clear()
            self.refusals = 0
            return
        arguments = _arguments(name, args)
        key, fingerprint, version = (
            _digest([name, arguments]),
            _digest(_observed(name, result)),
            _version(name, args),
        )
        scope = arguments.get("file_path" if name == "file_edit" else "path", arguments)
        observed = _digest([name, scope, fingerprint.hex(), version])
        previous = self.observations.get(observed)
        count = previous.count + 1 if previous else 1
        if count == 1 and not failed and not unchanged:
            self.revision += 1
            self.refusals = 0  # new information permits a fresh verification
        self.observations[observed] = Observation(version, count, self.revision)
        self.observations.move_to_end(observed)
        self.requests[key] = observed
        self.requests.move_to_end(key)
        while len(self.observations) > 64:
            self.observations.popitem(last=False)
        while len(self.requests) > 128:
            self.requests.popitem(last=False)
        if count >= REPEAT_LIMIT:
            self.last_tool = name
            if count > REPEAT_LIMIT:
                self.refusals += 1
                self.exhausted = self.refusals >= RECOVERY_LIMIT
            return (
                "This observation or failure has already been returned repeatedly. "
                "Reuse the available evidence; if the request is satisfied, finish. "
                "Otherwise correct the arguments or choose a relevant different step. "
                "Changing display options or repeating the same read will not advance the task. "
                "Do not bypass permission or file-staleness checks with another tool."
            )

    def reset(self):
        """New user guidance may authorize retrying a previously failed action."""
        self.observations.clear()
        self.requests.clear()
        self.refusals = 0
        self.exhausted = False
        self.last_tool = ""
        self.revision = 0

    def stopped_message(self):
        return (
            f"I stopped a repeated {self.last_tool} loop after attempts to recover did not "
            "produce new information. Completed work and tool results are preserved, but "
            "the task is incomplete. Review the last tool error or provide new information "
            "before retrying."
        )
