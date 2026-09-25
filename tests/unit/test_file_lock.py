"""Real-process contention and abrupt-exit recovery, confined to pytest temp dirs."""

import json
import os
import subprocess
import sys
from pathlib import Path

from mnemoai.utils.atomic_write import atomic_write_json

WORKER = """
import json, os, sys
from mnemoai.utils import atomic_write
from mnemoai.utils.file_lock import file_lock
target, mode = sys.argv[1:]
if mode == "hold":
    with file_lock(target + ".lock"):
        print("LOCKED", flush=True)
        sys.stdin.read(1)
elif mode == "probe":
    try:
        with file_lock(target + ".lock", timeout=0.1):
            pass
    except TimeoutError:
        sys.exit(74)
elif mode == "increment":
    for _ in range(15):
        with file_lock(target + ".lock"):
            with open(target) as stream:
                value = json.load(stream)
            atomic_write.atomic_write_json(target, value + 1)
else:
    original = atomic_write.os.replace
    def crash(src, dst):
        if str(dst) == target:
            if mode == "before":
                os._exit(71)
            original(src, dst)
            os._exit(72)
        return original(src, dst)
    atomic_write.os.replace = crash
    with file_lock(target + ".lock"):
        atomic_write.atomic_write_json(target, 1)
"""


def launch(path, mode):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    return subprocess.Popen(
        [sys.executable, "-c", WORKER, str(path), mode],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def finish(process):
    try:
        _, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    return process.returncode, stderr


def test_independent_writers_do_not_lose_updates(tmp_path):
    path = tmp_path / "counter.json"
    atomic_write_json(str(path), 0)
    processes = [launch(path, "increment") for _ in range(4)]
    try:
        for process in processes:
            code, stderr = finish(process)
            assert code == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert json.loads(path.read_text()) == 60


def test_crashes_preserve_committed_data_and_release_the_lock(tmp_path):
    for point, expected, exit_code in (("before", 0, 71), ("after", 1, 72)):
        path = tmp_path / f"{point}.json"
        atomic_write_json(str(path), 0)
        code, stderr = finish(launch(path, point))
        assert code == exit_code, stderr
        assert json.loads(path.read_text()) == expected
        code, stderr = finish(launch(path, "increment"))
        assert code == 0, stderr
        assert json.loads(path.read_text()) == expected + 15


def test_busy_process_lock_has_a_deadline_and_recovers(tmp_path):
    path = tmp_path / "counter.json"
    atomic_write_json(str(path), 0)
    holder = launch(path, "hold")
    try:
        assert holder.stdout.readline().strip() == "LOCKED"
        code, stderr = finish(launch(path, "probe"))
        assert code == 74, stderr
    finally:
        holder.communicate("x", timeout=30)
    code, stderr = finish(launch(path, "increment"))
    assert code == 0, stderr
    assert json.loads(path.read_text()) == 15
