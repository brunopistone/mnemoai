"""Exercise the POSIX installer without network access or touching real installs."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "install.sh"
RELEASE_VERSION = "7.8.9"
WHEEL_CONTENT = b"fixture wheel bytes"
RELEASE_DIGEST = hashlib.sha256(WHEEL_CONTENT).hexdigest()


def release_metadata(version=RELEASE_VERSION):
    name = f"mnemoai_assistant-{version}-py3-none-any.whl"
    return {
        "tag_name": f"v{version}", "draft": False, "prerelease": False,
        "assets": [{
            "name": name, "digest": f"sha256:{RELEASE_DIGEST}",
            "browser_download_url": f"https://github.com/brunopistone/mnemoai/releases/download/v{version}/{name}",
        }],
    }


@pytest.fixture
def installer(tmp_path):
    commands = tmp_path / "commands"
    commands.mkdir()
    # Only explicit commands are visible; a developer's uv/curl is never invoked.
    for name in ("uname", "getconf", "readlink", "mkdir", "cp", "chmod", "mv", "rm", "mktemp", "tar", "shasum", "sha256sum"):
        source = shutil.which(name)
        if source:
            (commands / name).symlink_to(source)
    env = dict(os.environ, PATH=str(commands))
    env["MNEMOAI_INSTALL_DIR"] = str(tmp_path / "runtime with spaces")
    env["MNEMOAI_BIN_DIR"] = str(tmp_path / "bin with spaces")
    env["INSTALL_TEST_RECORD"] = str(tmp_path / "record.json")
    env["INSTALL_TEST_RELEASE"] = str(tmp_path / "release.json")
    Path(env["INSTALL_TEST_RELEASE"]).write_text(json.dumps(release_metadata()))
    env["INSTALL_TEST_WHEEL"] = str(tmp_path / "download.whl")
    Path(env["INSTALL_TEST_WHEEL"]).write_bytes(WHEEL_CONTENT)
    env["TMPDIR"] = str(tmp_path)

    def executable(name, text):
        path = commands / name
        if path.is_symlink():
            path.unlink()
        path.write_text("#!/bin/sh\nset -eu\n" + text)
        path.chmod(0o755)

    def run(*args, shell="/bin/sh"):
        return subprocess.run(
            [shell, str(INSTALLER), *args], env=env,
            text=True, capture_output=True, timeout=15,
        )

    executable("curl", """
if [ -n "${INSTALL_TEST_CURL_FAIL:-}" ]; then exit 22; fi
while [ "$#" -gt 0 ]; do
    if [ "$1" = --output ]; then
        case "$2" in
            *.whl) cp "$INSTALL_TEST_WHEEL" "$2" ;;
            *) cp "$INSTALL_TEST_RELEASE" "$2" ;;
        esac
        exit
    fi
    shift
done
exit 1
""")
    return env, executable, run


def fake_uv(executable):
    # Use the interpreter explicitly: the harness intentionally has no Python on PATH.
    executable("uv", f"""exec '{sys.executable}' -c '
import json, os, pathlib, sys
if sys.argv[2] == "run":
    pathlib.Path(os.environ["INSTALL_TEST_RECORD"] + ".python").write_text(json.dumps(sys.argv[1:]))
    os.execv(sys.executable, [sys.executable, *sys.argv[sys.argv.index("python") + 1:]])
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
package = sys.argv[-1]
if "==" in package:
    selected_version = package.split("==", 1)[1]
elif " @ " in package:
    selected_version = package.split("/download/v", 1)[1].split("/", 1)[0]
else:
    selected_version = pathlib.Path(package).name.split("-")[1]
actual = os.environ.get("INSTALL_TEST_ACTUAL_VERSION", selected_version)
metadata_python = path.parent / "python"
metadata_python.write_text("#!/bin/sh\\necho " + actual + "\\n")
metadata_python.chmod(0o755)
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
    assert record["args"][:-1] == [
        "--no-config", "tool", "install", "--upgrade", "--compile-bytecode", "--python", "3.12",
        "--managed-python",
    ]
    wheel = Path(env["MNEMOAI_INSTALL_DIR"]) / "releases" / RELEASE_VERSION / f"mnemoai_assistant-{RELEASE_VERSION}-py3-none-any.whl"
    assert record["args"][-1] == str(wheel)
    assert wheel.read_bytes() == WHEEL_CONTENT
    assert record["dirs"]["UV_TOOL_DIR"] == env["MNEMOAI_INSTALL_DIR"] + "/tools"
    assert record["dirs"]["UV_PYTHON_INSTALL_DIR"] == env["MNEMOAI_INSTALL_DIR"] + "/python"
    assert "--force" not in record["args"]
    assert "FRONT of PATH" in result.stdout
    assert f"Installed MnemoAI {RELEASE_VERSION}:" in result.stdout
    assert not list(Path(env["TMPDIR"]).glob("mnemoai-install.*"))


def test_explicit_version_and_local_wheel_are_single_arguments(installer, tmp_path):
    env, executable, run = installer
    fake_uv(executable)
    assert run("--version", "1.28.2").returncode == 0
    assert not Path(env["INSTALL_TEST_RECORD"] + ".python").exists(), "pins must not consult latest"
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
    assert "Installed MnemoAI" not in result.stdout
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


def test_default_discovers_the_new_release_on_each_run(installer):
    env, executable, run = installer
    fake_uv(executable)
    for version in ("1.28.2", "1.29.0", "2.0.1"):
        Path(env["INSTALL_TEST_RELEASE"]).write_text(json.dumps(release_metadata(version)))
        result = run()
        assert result.returncode == 0, result.stderr
        assert f"Selected MnemoAI {version}" in result.stdout
        assert f"Installed MnemoAI {version}:" in result.stdout
        requirement = json.loads(Path(env["INSTALL_TEST_RECORD"]).read_text())["args"][-1]
        assert f"/releases/{version}/" in requirement
        assert Path(requirement).read_bytes() == WHEEL_CONTENT
    bootstrap = json.loads(Path(env["INSTALL_TEST_RECORD"] + ".python").read_text())
    assert "--no-project" in bootstrap and "--isolated" in bootstrap
    assert "--no-env-file" in bootstrap and "--managed-python" in bootstrap
    assert "-I" in bootstrap
    assert bootstrap[-1].endswith("/release.json"), "parse while uv's temporary interpreter still exists"


@pytest.mark.parametrize("bad", [
    "draft", "prerelease", "bad-tag", "missing-wheel", "duplicate-wheel",
    "wrong-wheel-version", "wrong-url", "missing-digest", "bad-digest", "not-an-object",
])
def test_invalid_release_metadata_is_never_installed(installer, bad):
    env, executable, run = installer
    fake_uv(executable)
    data = release_metadata()
    if bad in ("draft", "prerelease"):
        data[bad] = True
    elif bad == "bad-tag":
        data["tag_name"] = "v7.8.9;touch unwanted"
    elif bad == "missing-wheel":
        data["assets"] = []
    elif bad == "duplicate-wheel":
        data["assets"] *= 2
    elif bad == "wrong-wheel-version":
        data["assets"][0]["name"] = "mnemoai_assistant-1.28.2-py3-none-any.whl"
    elif bad == "wrong-url":
        data["assets"][0]["browser_download_url"] = "https://untrusted.invalid/package.whl"
    elif bad == "missing-digest":
        data["assets"][0].pop("digest")
    elif bad == "bad-digest":
        data["assets"][0]["digest"] = "sha256:not-a-hash"
    elif bad == "not-an-object":
        data = []
    Path(env["INSTALL_TEST_RELEASE"]).write_text(json.dumps(data))
    result = run()
    assert result.returncode != 0
    assert "refusing an unverified install" in result.stderr
    assert not Path(env["INSTALL_TEST_RECORD"]).exists()
    assert not list(Path(env["TMPDIR"]).glob("mnemoai-install.*"))


def test_failed_latest_lookup_does_not_fall_back_to_an_older_package(installer):
    env, executable, run = installer
    fake_uv(executable)
    assert run().returncode == 0
    record = Path(env["INSTALL_TEST_RECORD"])
    record.unlink()
    installed = Path(env["MNEMOAI_BIN_DIR"]) / "mnemoai"
    before = installed.read_bytes()
    env["INSTALL_TEST_CURL_FAIL"] = "1"
    result = run()
    assert result.returncode != 0
    assert "cannot determine the latest release" in result.stderr
    assert not record.exists()
    assert installed.read_bytes() == before


def test_wrong_installed_version_is_not_reported_as_success(installer):
    env, executable, run = installer
    fake_uv(executable)
    env["INSTALL_TEST_ACTUAL_VERSION"] = "1.28.2"
    result = run()
    assert result.returncode != 0
    assert f"expected {RELEASE_VERSION} but installed 1.28.2" in result.stderr
    assert "Installed MnemoAI" not in result.stdout


def test_corrupt_release_download_preserves_app_and_verified_source(installer):
    env, executable, run = installer
    fake_uv(executable)
    assert run().returncode == 0
    record = Path(env["INSTALL_TEST_RECORD"])
    cached = Path(json.loads(record.read_text())["args"][-1])
    record.unlink()
    launcher = Path(env["MNEMOAI_BIN_DIR"]) / "mnemoai"
    before = launcher.read_bytes()
    Path(env["INSTALL_TEST_WHEEL"]).write_bytes(b"corrupt download")
    result = run()
    assert result.returncode != 0
    assert "checksum mismatch" in result.stderr
    assert not record.exists(), "never pass unverified bytes to uv"
    assert cached.read_bytes() == WHEEL_CONTENT
    assert launcher.read_bytes() == before
    assert not list(Path(env["TMPDIR"]).glob("mnemoai-install.*"))


def test_cleanup_targets_cannot_be_injected_through_the_environment(installer, tmp_path):
    env, executable, run = installer
    fake_uv(executable)
    protected = tmp_path / "unrelated"
    protected.mkdir()
    sentinel = protected / "keep.txt"
    sentinel.write_text("keep")
    env["installer_tmp"] = str(protected)
    env["release_stage"] = str(sentinel)
    result = run()
    assert result.returncode == 0, result.stderr
    assert list(protected.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


def test_local_wheel_does_not_require_github_lookup(installer, tmp_path):
    env, executable, run = installer
    fake_uv(executable)
    env["INSTALL_TEST_CURL_FAIL"] = "1"
    wheel = tmp_path / "mnemoai_assistant-3.0.0-py3-none-any.whl"
    wheel.touch()
    result = run("--wheel", str(wheel))
    assert result.returncode == 0, result.stderr
    assert "Installed MnemoAI 3.0.0" in result.stdout
    assert not Path(env["INSTALL_TEST_RECORD"] + ".python").exists()


@pytest.mark.parametrize("bundled", [False, True])
def test_uninstall_commands_in_readme_and_guide_match_installer_layout(installer, bundled):
    import re

    env, executable, _ = installer
    root = INSTALLER.parent
    snippets = []
    for path in (root / "README.md", root / "docs/getting-started/installation.md"):
        snippets.append(next(
            block for block in re.findall(r"```bash\n(.*?)```", path.read_text(), re.S)
            if block.startswith("mnemo_runtime=")
        ))
    assert snippets[0] == snippets[1]
    # Verify the documented command's environment and arguments without deleting anything.
    executable("uv", f"""exec '{sys.executable}' -c '
import json,os,pathlib,sys
pathlib.Path(os.environ["INSTALL_TEST_RECORD"]).write_text(json.dumps({{
    "args":sys.argv[1:], "tools":os.environ["UV_TOOL_DIR"], "bin":os.environ["UV_TOOL_BIN_DIR"]
}}))
' "$@"
""")
    fallback = re.search(r'bootstrap/([\d.]+)/uv', snippets[0]).group(1)
    if bundled:
        target = Path(env["MNEMOAI_INSTALL_DIR"]) / "bootstrap" / fallback / "uv"
        target.parent.mkdir(parents=True)
        (Path(env["PATH"]) / "uv").replace(target)
    result = subprocess.run(["/bin/sh", "-c", snippets[0]], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(Path(env["INSTALL_TEST_RECORD"]).read_text())
    assert record == {
        "args": ["--no-config", "tool", "uninstall", "mnemoai-assistant"],
        "tools": env["MNEMOAI_INSTALL_DIR"] + "/tools", "bin": env["MNEMOAI_BIN_DIR"],
    }
    assert f"bootstrap/{fallback}/uv" in INSTALLER.read_text()


@pytest.mark.parametrize("download_ok", [True, False])
def test_bootstrap_rejects_corrupt_or_failed_downloads(installer, download_ok):
    env, executable, run = installer
    executable("curl", "exit 0\n" if download_ok else "exit 22\n")
    executable("sha256sum", "printf 'badchecksum  archive\\n'\n")
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
