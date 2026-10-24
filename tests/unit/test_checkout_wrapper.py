"""The legacy checkout launcher must preserve already compiled dependency caches."""

import os
import shutil
import subprocess
from pathlib import Path


def test_wrapper_keeps_bytecode_and_forwards_arguments(tmp_path):
    project = tmp_path / "checkout with spaces"
    scripts = project / "bash" / "system-command-app"
    scripts.mkdir(parents=True)
    wrapper = scripts / "mnemoai-wrapper.sh"
    source = Path(__file__).resolve().parents[2] / "bash" / "system-command-app" / wrapper.name
    shutil.copy2(source, wrapper)
    binaries = project / ".venv" / "bin"
    binaries.mkdir(parents=True)
    (binaries / "activate").write_text(f'export PATH="{binaries}:$PATH"\n')
    python = binaries / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    cached = project / ".venv" / "lib" / "__pycache__" / "preserve.pyc"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"existing bytecode")
    result = subprocess.run(
        ["bash", str(wrapper), "--no-verbose"], text=True, capture_output=True,
        env=os.environ.copy(), timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["-m", "mnemoai", "--no-verbose"]
    assert cached.read_bytes() == b"existing bytecode"
