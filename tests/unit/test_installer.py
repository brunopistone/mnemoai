"""Exercise the POSIX installer without network access or touching real installs."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "install.sh"


@pytest.fixture
def installer(tmp_path):
    commands = tmp_path / "commands"
    commands.mkdir()
    # Only explicit commands are visible; a developer's uv/curl is never invoked.
    for name in ("uname", "getconf", "readlink", "mkdir", "cp", "chmod", "mv", "rm", "mktemp", "tar"):
        source = shutil.which(name)
        if source:
            (commands / name).symlink_to(source)
    env = dict(os.environ, PATH=str(commands))
    env["MNEMOAI_INSTALL_DIR"] = str(tmp_path / "runtime with spaces")
    env["MNEMOAI_BIN_DIR"] = str(tmp_path / "bin with spaces")
    env["INSTALL_TEST_RECORD"] = str(tmp_path / "record.json")
    env["TMPDIR"] = str(tmp_path)

    def executable(name, text):
        path = commands / name
        path.write_text("#!/bin/sh\nset -eu\n" + text)
        path.chmod(0o755)

    def run(*args, shell="/bin/sh"):
        return subprocess.run(
            [shell, str(INSTALLER), *args], env=env,
            text=True, capture_output=True, timeout=15,
        )

    return env, executable, run


def fake_uv(executable):
    import sys

    # Use the interpreter explicitly: the harness intentionally has no Python on PATH.
    executable("uv", f"""exec '{sys.executable}' -c '
import json, os, pathlib, sys
pathlib.Path(os.environ["INSTALL_TEST_RECORD"]).write_text(json.dumps({{
    "args": sys.argv[1:],
    "dirs": {{k: os.environ[k] for k in ("UV_TOOL_DIR", "UV_TOOL_BIN_DIR", "UV_PYTHON_INSTALL_DIR")}},
}}))
if os.environ.get("INSTALL_TEST_FAIL"):
    sys.exit(7)
path = pathlib.Path(os.environ["UV_TOOL_DIR"]) / "mnemoai-assistant/bin/mnemoai"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("#!/bin/sh\\nexit 0\\n")
path.chmod(0o755)
link = pathlib.Path(os.environ["UV_TOOL_BIN_DIR"]) / "mnemoai"
if not link.is_symlink():
    link.symlink_to(path)
' "$@"
""")


@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash", "/bin/dash"])
def test_install_and_rerun_use_private_managed_runtime(installer, shell):
    if not Path(shell).exists():
        pytest.skip(f"{shell} is not installed")
    env, executable, run = installer
    fake_uv(executable)
    for _ in range(2):
        result = run(shell=shell)
        assert result.returncode == 0, result.stderr
    record = json.loads(Path(env["INSTALL_TEST_RECORD"]).read_text())
    assert record["args"] == [
        "--no-config", "tool", "install", "--upgrade", "--compile-bytecode", "--python", "3.12",
        "--managed-python", "mnemoai-assistant",
    ]
    assert record["dirs"]["UV_TOOL_DIR"] == env["MNEMOAI_INSTALL_DIR"] + "/tools"
    assert record["dirs"]["UV_PYTHON_INSTALL_DIR"] == env["MNEMOAI_INSTALL_DIR"] + "/python"
    assert "--force" not in record["args"]
    assert "FRONT of PATH" in result.stdout


def test_explicit_version_and_local_wheel_are_single_arguments(installer, tmp_path):
    env, executable, run = installer
    fake_uv(executable)
    assert run("--version", "1.28.2").returncode == 0
    assert json.loads(Path(env["INSTALL_TEST_RECORD"]).read_text())["args"][-1] == "mnemoai-assistant==1.28.2"
    wheel = tmp_path / "folder with spaces" / "mnemoai_assistant-1.28.2-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.touch()
    assert run("--wheel", str(wheel)).returncode == 0
    assert json.loads(Path(env["INSTALL_TEST_RECORD"]).read_text())["args"][-1] == str(wheel)


@pytest.mark.parametrize("args", [
    ("--version",), ("--version", "--force"), ("--version", "1;touch bad"),
    ("--version", "1.2", "extra"), ("--wheel", "relative.whl"),
    ("--wheel", "/tmp/other_package-1.whl"), ("--unknown",),
])
def test_invalid_options_do_not_install(installer, args):
    env, executable, run = installer
    fake_uv(executable)
    assert run(*args).returncode != 0
    assert not Path(env["INSTALL_TEST_RECORD"]).exists()


def test_help_needs_no_runtime(installer):
    env, _, run = installer
    result = run("--help")
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert not Path(env["MNEMOAI_INSTALL_DIR"]).exists()


def test_unsupported_platform_is_rejected_before_download(installer):
    env, executable, run = installer
    # Replace the symlink, never write through it into the host's uname binary.
    (Path(env["PATH"]) / "uname").unlink()
    executable("uname", "printf 'Windows\\n'\n")
    result = run()
    assert result.returncode != 0
    assert "supported platforms" in result.stderr
    assert not Path(env["MNEMOAI_INSTALL_DIR"]).exists()


def test_failed_install_does_not_claim_success_or_remove_old_command(installer):
    env, executable, run = installer
    fake_uv(executable)
    assert run().returncode == 0
    binary = Path(env["MNEMOAI_BIN_DIR"]) / "mnemoai"
    binary.write_text("previous install")
    env["INSTALL_TEST_FAIL"] = "1"
    result = run()
    assert result.returncode == 7
    assert "Installed:" not in result.stdout
    assert binary.read_text() == "previous install"


@pytest.mark.parametrize("kind", ["file", "symlink", "broken-symlink"])
def test_unrelated_command_is_refused_before_calling_uv(installer, kind):
    env, executable, run = installer
    fake_uv(executable)
    assert run().returncode == 0
    old_command = Path(env["MNEMOAI_BIN_DIR"]) / "mnemoai"
    old_target = old_command.resolve()
    new_bin = old_command.parent.parent / "unrelated"
    new_bin.mkdir()
    target = new_bin / "mnemoai"
    if kind == "file":
        target.write_text("unrelated")
    else:
        target.symlink_to(new_bin / "foreign")
        if kind == "symlink":
            (new_bin / "foreign").write_text("unrelated")
    record = Path(env["INSTALL_TEST_RECORD"])
    before = record.read_bytes()
    env["MNEMOAI_BIN_DIR"] = str(new_bin)
    result = run()
    assert result.returncode != 0
    assert "refusing to replace" in result.stderr
    assert record.read_bytes() == before, "uv must not run before checking ownership"
    assert old_command.resolve() == old_target and old_command.is_file()
    if kind == "file":
        assert target.read_text() == "unrelated"
    else:
        assert target.is_symlink()


def test_existing_command_shadow_is_reported(installer):
    _, executable, run = installer
    fake_uv(executable)
    executable("mnemoai", "exit 0\n")
    result = run()
    assert result.returncode == 0
    assert "PATH still selects the older command" in result.stdout


def test_directory_aliases_still_upgrade_the_owned_command(installer, tmp_path):
    env, executable, run = installer
    fake_uv(executable)
    assert run().returncode == 0
    runtime, bin_dir = Path(env["MNEMOAI_INSTALL_DIR"]), Path(env["MNEMOAI_BIN_DIR"])
    runtime_alias, bin_alias = tmp_path / "runtime-alias", tmp_path / "bin-alias"
    runtime_alias.symlink_to(runtime, target_is_directory=True)
    bin_alias.symlink_to(bin_dir, target_is_directory=True)
    env["MNEMOAI_INSTALL_DIR"], env["MNEMOAI_BIN_DIR"] = str(runtime_alias), str(bin_alias)
    env["PATH"] = str(bin_alias) + os.pathsep + env["PATH"]
    result = run()
    assert result.returncode == 0, result.stderr
    assert "older command" not in result.stdout
    assert "Run mnemoai to start" in result.stdout


@pytest.mark.parametrize("download_ok", [True, False])
def test_bootstrap_rejects_corrupt_or_failed_downloads(installer, download_ok):
    env, executable, run = installer
    executable("curl", "exit 0\n" if download_ok else "exit 22\n")
    executable("shasum", "printf 'badchecksum  archive\\n'\n")
    result = run()
    assert result.returncode != 0
    assert not list(Path(env["MNEMOAI_INSTALL_DIR"]).rglob("uv"))
    if download_ok:
        assert "checksum mismatch" in result.stderr
    assert not list(Path(env["TMPDIR"]).glob("mnemoai-install.*"))


@pytest.mark.parametrize("directory", ["/", "relative", "/usr/bin"])
def test_invalid_install_location_is_rejected(installer, directory):
    env, executable, run = installer
    fake_uv(executable)
    env["MNEMOAI_INSTALL_DIR"] = directory
    assert run().returncode != 0
    assert not Path(env["INSTALL_TEST_RECORD"]).exists()
