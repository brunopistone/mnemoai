"""Bounded observations for report-only review; never execute suggested checks."""

import hashlib
import json
import os
import selectors
import stat
import subprocess
import threading
import time
from pathlib import Path

from mnemoai.client.memory.reflection import redact
from mnemoai.utils.tool_results import exit_status

_FILE_BYTES = 64_000
_DIFF_BYTES = 12_000
_TOOL_LIMIT = 24
_FILE_LIMIT = 12
_PRIVATE_NAMES = {".env", ".pypirc", ".netrc", ".aws", ".ssh", "credentials", "id_rsa", "id_ed25519"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def text(value, limit, *, structured=False):
    remaining, clipped = limit * 4, False

    def bounded(item, depth=0):
        nonlocal remaining, clipped
        if remaining <= 0 or depth > 12:
            clipped = True
            return "[omitted]"
        remaining -= 1
        if isinstance(item, str):
            part = item[:remaining]
            clipped = clipped or len(part) < len(item)
            remaining -= len(part)
            return part
        if isinstance(item, (dict, list)):
            result = {} if isinstance(item, dict) else []
            for key, child in item.items() if isinstance(item, dict) else enumerate(item):
                if remaining <= 0:
                    clipped = True
                    break
                if isinstance(result, dict):
                    result[bounded(str(key))] = bounded(child, depth + 1)
                else:
                    result.append(bounded(child, depth + 1))
            return result
        return item

    value = bounded(value)
    if structured and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            parsed = value
        value = redact(parsed)
    else:
        value = redact(value)
    if not isinstance(value, str):
        value = json.dumps(value, default=str, ensure_ascii=True)
    return value[:limit], clipped or len(value) > limit


def private_path(path):
    return any(part.lower() in _PRIVATE_NAMES or part.lower().startswith(".env.")
               or part.lower().endswith((".pem", ".p12", ".pfx"))
               for part in Path(path).parts)


def file_state(path, root):
    """Read only regular, in-scope files; bound reads and notice concurrent writes."""
    path = Path(os.path.abspath(path))
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("outside the task directory")
    if path.is_symlink() or private_path(path) or private_path(resolved):
        raise ValueError("symlink or credential-like path excluded")
    parts = resolved.relative_to(root).parts
    if not parts:
        raise ValueError("not a regular file")
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory = os.open(root, directory_flags)
        try:
            for part in parts[:-1]:
                child = os.open(part, directory_flags, dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
        finally:
            os.close(directory)
    except FileNotFoundError:
        return {"revision": "missing", "content": "[file is absent]", "truncated": False}
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not a regular file")
        raw = stream.read(_FILE_BYTES + 1)
        after = os.fstat(stream.fileno())
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if stamp(before) != stamp(after) or path.resolve() != resolved:
        raise ValueError("file changed during inspection")
    revision = digest([str(resolved), stamp(after), hashlib.sha256(raw).hexdigest()])
    try:
        content = raw[:_FILE_BYTES].decode("utf-8")
        if "\0" in content:
            raise ValueError("binary file")
    except UnicodeError as exc:
        raise ValueError("non-UTF-8 file") from exc
    preview, clipped = text(content, 4000)
    return {"revision": revision, "content": preview,
            "truncated": clipped or len(raw) > _FILE_BYTES}


def git_state(root):
    """A scoped diff vs HEAD, including staged edits; not a claim of authorship."""
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=1,
        ).decode("ascii").strip()
    except subprocess.CalledProcessError:
        return None
    args = [
        "git", "--no-pager", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
        "-C", str(root), "diff", "--no-ext-diff", "--no-textconv", "--ignore-submodules=all",
        head, "--", ".",
        ":(exclude)**/.env*", ":(exclude)**/*.pem", ":(exclude)**/*.p12",
        ":(exclude)**/*.pfx", ":(exclude)**/credentials", ":(exclude)**/.pypirc",
        ":(exclude)**/.netrc", ":(exclude)**/id_rsa*", ":(exclude)**/id_ed25519*",
        ":(exclude)**/.aws/**", ":(exclude)**/.ssh/**",
    ]
    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1"},
    )
    raw = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 2
            while len(raw) <= _DIFF_BYTES:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise subprocess.TimeoutExpired(args, 2)
                chunk = os.read(proc.stdout.fileno(), min(4096, _DIFF_BYTES + 1 - len(raw)))
                if not chunk:
                    if proc.wait(timeout=max(0.01, remaining)):
                        raise OSError("Workspace diff failed")
                    break
                raw.extend(chunk)
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return {"revision": digest([head, raw.hex()]), "head": head,
            "content": redact(raw[:_DIFF_BYTES].decode("utf-8", "replace")),
            "truncated": len(raw) > _DIFF_BYTES}


class Capture:
    """One parent turn; synchronous worker observations share this bounded sink."""

    def __init__(self, task, root=None):
        self.root = Path(root or os.getcwd()).resolve()
        self.task = task
        self._lock = threading.Lock()
        self._closed = False
        self.tools = []
        self.followups = []
        self.paths = set()
        self._written_paths = set()
        self.gaps = set()
        self.failed_commands = {}
        self.before = self._git()
        self.bindings = {}
        self.after = None
        self.write_serial = 0
        self.shell_serial = 0

    def add_instructions(self, texts):
        with self._lock:
            if not self._closed:
                for value in texts:
                    if len(self.followups) < 8:
                        self.followups.append({"id": f"followup-{len(self.followups) + 1}",
                                               "kind": "user_followup", "text": value})
                    else:
                        self.gaps.add("Additional user instructions exceeded the capture limit.")

    def reopen(self):
        """Open the next checkpoint without erasing already observed tool evidence."""
        with self._lock:
            self._closed = False
            self.bindings = {}
            self.after = None
            self._written_paths = set()
            self.gaps = {gap for gap in self.gaps if not (
                gap.startswith("Evidence ") or gap.startswith("Named-file inspection incomplete")
                or gap in {
                    "Some evidence was omitted to fit the input budget.",
                    "Some current file content was truncated.", "Workspace diff was truncated.",
                    "Workspace diff unavailable.",
                    "A failed shell command has no observed successful rerun of that command.",
                }
            )}

    def _git(self):
        try:
            return git_state(self.root)
        except (OSError, subprocess.SubprocessError):
            self.gaps.add("Workspace diff unavailable.")
            return None

    def record(self, name, args, result, status):
        """Bookkeeping must not affect a tool's result or permission decision."""
        try:
            with self._lock:
                if self._closed:
                    return
                if status == "completed":
                    if name in {"fs_write", "file_edit"}:
                        self.write_serial += 1
                    elif name == "execute_bash":
                        self.shell_serial += 1
                    if name in {"fs_read", "fs_write", "file_edit"} and not (
                        name == "fs_read" and args.get("mode") == "Directory"
                    ):
                        path = args.get("file_path" if name == "file_edit" else "path")
                        if isinstance(path, str) and path:
                            path = os.path.abspath(os.path.expanduser(path))
                            writing = name in {"fs_write", "file_edit"}
                            if writing and path not in self.paths and len(self.paths) >= _FILE_LIMIT:
                                replaceable = self.paths - self._written_paths
                                if replaceable:
                                    self.paths.remove(sorted(replaceable)[-1])
                                    self.gaps.add("Some named-file evidence was omitted to prioritize current changes.")
                            if path in self.paths or len(self.paths) < _FILE_LIMIT:
                                self.paths.add(path)
                                if writing:
                                    self._written_paths.add(path)
                            else:
                                self.gaps.add("Named-file evidence exceeded the capture limit.")
                if len(self.tools) >= _TOOL_LIMIT:
                    self.gaps.add("Tool evidence exceeded the capture limit.")
                    return
                safe_args, safe_result = args, result
                path = args.get("file_path" if name == "file_edit" else "path")
                if name in {"fs_read", "fs_write", "file_edit"} and isinstance(path, str) and private_path(path):
                    safe_args, safe_result = {"path": path}, "[credential-like file content omitted]"
                    self.gaps.add("Credential-like file evidence was omitted.")
                arguments, cut_args = text(safe_args, 1200)
                output, cut_result = text(safe_result, 2400, structured=True)
                self.tools.append({
                    "id": f"tool-{len(self.tools) + 1}", "kind": "tool",
                    "name": name, "arguments": arguments, "result": output, "status": status,
                    "revision": digest([name, arguments, output, status]),
                })
                if cut_args or cut_result:
                    self.gaps.add("Some tool arguments/results were truncated.")
                if name in {"execute_bash", "start_background_task"}:
                    payload = result
                    if isinstance(payload, str):
                        try:
                            payload = json.loads(payload)
                        except ValueError:
                            payload = {}
                    command = str(args.get("command", ""))
                    cwd = payload.get("cwd", str(self.root)) if isinstance(payload, dict) else str(self.root)
                    key = (command, cwd)
                    exit_code = exit_status(payload)
                    if command and (status == "failed" or exit_code not in (None, 0)):
                        self.failed_commands[key] = True
                    elif command and status == "completed" and exit_code == 0:
                        self.failed_commands.pop(key, None)
                if name == "start_background_task" or (
                    name in {"spawn_agent", "resume_agent"} and args.get("run_in_background", True)
                ):
                    self.gaps.add("Background execution is outside this completion checkpoint.")
        except Exception:
            with self._lock:
                self.gaps.add("A tool observation could not be captured.")

    def finish(self, answer, context=()):
        with self._lock:
            self._closed = True
            items = [{"id": "task", "kind": "user_request", "text": self.task},
                     {"id": "answer", "kind": "actor_answer", "text": answer}]
            items.extend(context)
            items.extend(self.followups)
            items.extend(self.tools)
        self.gaps.discard("A failed shell command has no observed successful rerun of that command.")
        if self.failed_commands:
            self.gaps.add("A failed shell command has no observed successful rerun of that command.")
        for i, path in enumerate(sorted(self.paths)):
            try:
                state = file_state(path, self.root)
                self.bindings[path] = state["revision"]
                items.append({"id": f"file-{i + 1}", "kind": "current_file",
                              "path": os.path.relpath(path, self.root), **state})
                if state["truncated"]:
                    self.gaps.add("Some current file content was truncated.")
            except (OSError, ValueError) as exc:
                self.gaps.add(f"Named-file inspection incomplete ({type(exc).__name__}).")
        self.after = self._git()
        if self.after is not None:
            for label, snapshot in (("before", self.before), ("after", self.after)):
                if snapshot is not None:
                    items.append({"id": f"diff-{label}", "kind": "workspace_diff_vs_HEAD",
                                  "phase": label, **snapshot})
                    if snapshot["truncated"]:
                        self.gaps.add("Workspace diff was truncated.")
        return items

    def stale(self):
        try:
            if any(file_state(path, self.root)["revision"] != rev for path, rev in self.bindings.items()):
                return True
            current = git_state(self.root)
            return (current or {}).get("revision") != (self.after or {}).get("revision")
        except (OSError, ValueError, subprocess.SubprocessError):
            return True
