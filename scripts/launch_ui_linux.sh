#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  cat >&2 <<EOF
SwiftVR virtual environment was not found at: $VENV_DIR

Install first:
  bash scripts/install_linux.sh --download-checkpoints

Or point VENV_DIR to an existing environment:
  VENV_DIR=/path/to/venv bash scripts/launch_ui_linux.sh --share true
EOF
  exit 1
fi

if ! "$PYTHON_BIN" -c "import gradio" >/dev/null 2>&1; then
  cat >&2 <<EOF
Gradio is not installed in $VENV_DIR.

Repair the environment:
  "$PYTHON_BIN" -m pip install "gradio>=4.44.0" "huggingface_hub>=0.24.0"
  "$PYTHON_BIN" -m pip install --no-deps -e .
EOF
  exit 1
fi

exec "$PYTHON_BIN" app.py "$@"
