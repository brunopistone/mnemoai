"""Private, one-use conversation handoffs across process restarts."""

import json
import os
import shlex
import tempfile
from pathlib import Path

from mnemoai.utils.paths import app_home

FLAG = "--restart-state"


def recovery_command(path) -> str:
    """A copyable recovery command for the same app home and configuration."""
    command = ["env", f"MNEMOAI_HOME={app_home()}"]
    command.extend(
        f"{key}={os.environ[key]}"
        for key in ("MNEMOAI_CONFIG", "MNEMOAI_PROMPTS")
        if os.environ.get(key)
    )
    return shlex.join(command + ["mnemoai", FLAG, str(path)])


def validate(data: dict) -> dict:
    """Reject incomplete handoffs before changing the running conversation."""
    if not isinstance(data, dict) or data.get("restart_version") != 1:
        raise ValueError("Unsupported restart state")
    if not isinstance(data.get("messages"), list):
        raise ValueError("Restart messages are missing")
    for message in data["messages"]:
        if not isinstance(message, dict) or message.get("role") not in {
            "user",
            "assistant",
        }:
            raise ValueError("Invalid restart message")
        if not isinstance(message.get("content"), (str, list)):
            raise ValueError("Invalid restart message content")
    for key in ("summary", "label", "draft", "cwd"):
        if not isinstance(data.get(key), str):
            raise ValueError(f"Invalid restart {key}")
    for key in ("conversation_path", "session_path"):
        if key not in data or (
            data[key] is not None and not isinstance(data[key], str)
        ):
            raise ValueError(f"Invalid restart {key}")
    if data.get("auto_mode") not in {"off", "edits", "writes", "all"}:
        raise ValueError("Invalid restart approval mode")
    if not isinstance(data.get("plan_mode"), bool):
        raise ValueError("Invalid restart plan mode")
    return data


def write(data: dict) -> Path:
    """Finish and verify a private snapshot before the old process shuts down."""
    text = json.dumps(validate(data), ensure_ascii=False)
    directory = app_home() / "restarts"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="restart-", suffix=".json", dir=directory)
    path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if read(path) != data:
            raise ValueError("Restart state could not be verified")
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def read(path) -> dict:
    """Read a handoff without consuming it until restoration succeeds."""
    path = Path(path)
    directory = app_home() / "restarts"
    if path.parent.resolve() != directory.resolve() or not path.name.startswith(
        "restart-"
    ):
        raise ValueError("Not a restart handoff path")
    with path.open(encoding="utf-8") as stream:
        data = validate(json.load(stream))
    if Path(data["cwd"]).resolve() != Path.cwd().resolve():
        raise ValueError(f"Restart belongs to another directory: {data['cwd']}")
    return data


def argv(original: list[str], path) -> list[str]:
    """Keep launch options, replacing stale resume and approval arguments."""
    result = original[:1]
    index = 1
    while index < len(original):
        arg = original[index]
        if arg in {"--resume", FLAG}:
            index += 1
            if index < len(original) and not original[index].startswith("-"):
                index += 1
            continue
        if arg in {"--continue", "--auto"} or arg.startswith(("--resume=", FLAG + "=")):
            index += 1
            continue
        result.append(arg)
        index += 1
    return result + [FLAG, str(path)]
