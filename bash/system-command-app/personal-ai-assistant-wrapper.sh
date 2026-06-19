#!/bin/bash
# Resolve the real path of the script (follows symlinks)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || realpath "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate venv if exists, otherwise activate conda
if [ -d "$PROJECT_DIR/.venv" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
else
    eval "$(conda shell.bash hook)"
    conda activate personal-ai-assistant
fi

# Remove all __pycache__ directories
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# src layout: put the package's parent (src/) on PYTHONPATH so the package is
# importable from a checkout without installing, then run via module invocation.
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python -m personal_ai_assistant "$@"