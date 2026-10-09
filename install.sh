#!/bin/sh
# Install/update MnemoAI in a private Python environment. No sudo or shell edits.
set -eu

fail() { printf 'mnemoai installer: %s\n' "$*" >&2; exit 1; }
usage() {
    printf '%s\n' \
        'Usage: sh install.sh [--version VERSION | --wheel /absolute/package.whl]' \
        'Install or update mnemoai-assistant (latest stable by default).' \
        'MNEMOAI_INSTALL_DIR: runtime directory (default ~/.local/share/mnemoai-runtime)' \
        'MNEMOAI_BIN_DIR: command directory (default ~/.local/bin)' \
        'Existing config, memory, sessions and shell profiles are not modified.'
}

main() {
    package=mnemoai-assistant
    case "${1:-}" in
        --help|-h) usage; return ;;
        --version)
            [ "$#" -eq 2 ] || fail '--version needs exactly one version'
            case "$2" in
                ''|*[!0-9a-zA-Z.+-]*|[!0-9]*) fail 'invalid version' ;;
            esac
            package="mnemoai-assistant==$2"
            ;;
        --wheel)
            [ "$#" -eq 2 ] || fail '--wheel needs exactly one absolute wheel path'
            case "$2" in
                /*/mnemoai_assistant-*.whl) [ -f "$2" ] || fail 'wheel does not exist' ;;
                *) fail 'expected an absolute mnemoai_assistant wheel path' ;;
            esac
            package=$2
            ;;
        '') [ "$#" -eq 0 ] || fail 'unexpected arguments' ;;
        *) usage >&2; fail 'unknown option' ;;
    esac

    case "$(uname -s):$(uname -m)" in
        Darwin:arm64|Darwin:aarch64)
            target=aarch64-apple-darwin
            checksum=50487ae565ccd96e499056b4674d438f4c53170202617b4c759defe0c6a1b544 ;;
        Darwin:x86_64)
            target=x86_64-apple-darwin
            checksum=960da44cb4b73685206ddd250b19e0a117fa41095710c1038f081f5cb613efb4 ;;
        Linux:aarch64|Linux:arm64)
            target=aarch64-unknown-linux-gnu
            checksum=6524bd338177ed50d035d39354e12545e993bbeba2ecbddf0480c5b3a81d313f ;;
        Linux:x86_64)
            target=x86_64-unknown-linux-gnu
            checksum=9167d72b3319674b6303c4cbe071854bba13ebdf3d76b1a7cbdc175471fb66d6 ;;
        *) fail 'supported platforms: macOS and glibc Linux, arm64 or x86_64' ;;
    esac
    case "$target" in
        *linux-gnu) getconf GNU_LIBC_VERSION >/dev/null 2>&1 || fail 'Linux requires glibc (musl/Alpine is not supported)' ;;
    esac

    runtime=${MNEMOAI_INSTALL_DIR:-"$HOME/.local/share/mnemoai-runtime"}
    bin_dir=${MNEMOAI_BIN_DIR:-"$HOME/.local/bin"}
    for directory in "$runtime" "$bin_dir"; do
        case "$directory" in
            /|/bin|/usr|/usr/bin|/usr/local|"$HOME") fail 'choose a dedicated installation directory' ;;
            /*) ;;
            *) fail 'installation directories must be absolute paths' ;;
        esac
    done
    mkdir -p "$runtime" "$bin_dir"
    runtime=$(CDPATH='' cd "$runtime" && pwd -P)
    bin_dir=$(CDPATH='' cd "$bin_dir" && pwd -P)
    # uv checks executable conflicts after updating the environment/removing its
    # old launcher. Refuse an unrelated destination BEFORE either can change.
    if [ -e "$bin_dir/mnemoai" ] || [ -L "$bin_dir/mnemoai" ]; then
        if [ ! -L "$bin_dir/mnemoai" ] ||
            [ "$(readlink "$bin_dir/mnemoai")" != "$runtime/tools/mnemoai-assistant/bin/mnemoai" ]; then
            fail "refusing to replace $bin_dir/mnemoai; choose MNEMOAI_BIN_DIR or move the existing command yourself"
        fi
    fi

    # Prefer an existing uv; otherwise fetch a pinned, checksum-verified binary.
    # It remains private to this installer, not another command added to PATH.
    uv_command=$(command -v uv || true)
    if [ -z "$uv_command" ]; then
        uv_command="$runtime/bootstrap/0.12.23/uv"
        if [ ! -x "$uv_command" ]; then
            command -v curl >/dev/null 2>&1 || fail 'curl is required'
            command -v tar >/dev/null 2>&1 || fail 'tar is required'
            if command -v sha256sum >/dev/null 2>&1; then
                hash_command=sha256sum
            elif command -v shasum >/dev/null 2>&1; then
                hash_command=shasum
            else
                fail 'sha256sum or shasum is required to verify the download'
            fi
            # Only this validated mktemp directory is ever removed.
            installer_tmp=$(mktemp -d "${TMPDIR:-/tmp}/mnemoai-install.XXXXXXXX") || fail 'cannot create temporary directory'
            trap 'rm -rf -- "$installer_tmp"' EXIT
            trap 'exit 130' INT
            trap 'exit 143' TERM HUP
            printf 'Downloading the isolated-runtime installer (uv 0.12.23)…\n'
            curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
                --connect-timeout 15 --max-time 180 --retry 2 \
                "https://github.com/astral-sh/uv/releases/download/0.12.23/uv-$target.tar.gz" \
                --output "$installer_tmp/uv.tar.gz"
            if [ "$hash_command" = shasum ]; then
                actual=$("$hash_command" -a 256 "$installer_tmp/uv.tar.gz")
            else
                actual=$("$hash_command" "$installer_tmp/uv.tar.gz")
            fi
            actual=${actual%% *}
            [ "$actual" = "$checksum" ] || fail 'uv checksum mismatch; nothing was installed'
            tar -xzf "$installer_tmp/uv.tar.gz" -C "$installer_tmp" "uv-$target/uv"
            mkdir -p "$runtime/bootstrap/0.12.23"
            # Atomic publication; an interrupted download never becomes executable.
            cp "$installer_tmp/uv-$target/uv" "$runtime/bootstrap/0.12.23/uv.new"
            chmod 755 "$runtime/bootstrap/0.12.23/uv.new"
            mv -f "$runtime/bootstrap/0.12.23/uv.new" "$uv_command"
        fi
    fi

    printf 'Installing MnemoAI with a managed Python 3.12 runtime…\n'
    # Isolate both tool and interpreter storage. Do not import packages from the
    # developer's global Python or write to the user's existing uv tool installs.
    UV_TOOL_DIR="$runtime/tools" UV_TOOL_BIN_DIR="$bin_dir" \
        UV_PYTHON_INSTALL_DIR="$runtime/python" \
        "$uv_command" --no-config tool install --upgrade --compile-bytecode \
        --python 3.12 --managed-python "$package"

    [ -x "$bin_dir/mnemoai" ] || fail 'installation did not produce the mnemoai command'
    printf '\nInstalled: %s/mnemoai\n' "$bin_dir"
    printf 'Your configuration and conversations were left unchanged.\n'
    active=$(command -v mnemoai || true)
    if [ -z "$active" ] || [ ! "$active" -ef "$bin_dir/mnemoai" ]; then
        if [ -n "$active" ]; then
            printf 'Note: your PATH still selects the older command: %s\n' "$active"
        fi
        printf 'Add %s to the FRONT of PATH, or launch the Installed path above.\n' "$bin_dir"
        printf 'Shell profiles were not edited. Re-run this installer to update.\n'
    else
        printf 'Run mnemoai to start. Re-run this installer to update.\n'
    fi
}

# A piped installer does nothing until the whole function has been downloaded.
main "$@"
