#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
HF_REPO_ID="${HF_REPO_ID:-H-oliday/SwiftVR}"
SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
INSTALL_TORCH=true
DOWNLOAD_CHECKPOINTS=false
LAUNCH_UI=false
SHARE="${SWIFTVR_SHARE:-${GRADIO_SHARE:-${Share:-${SHARE:-false}}}}"

usage() {
  cat <<'EOF'
SwiftVR Linux installer

Usage:
  bash scripts/install_linux.sh [options]

Options:
  --venv DIR                  Virtual environment directory. Default: .venv
  --python BIN                Python executable. Default: python3
  --torch-index-url URL       PyTorch wheel index. Default: CUDA 12.4 wheels
  --torch-version VERSION     Torch version. Default: 2.6.0
  --torchvision-version VER   Torchvision version. Default: 0.21.0
  --skip-torch                Do not install torch/torchvision explicitly
  --checkpoint-dir DIR        Checkpoint directory. Default: checkpoints
  --download-checkpoints      Download H-oliday/SwiftVR from Hugging Face
  --hf-repo-id REPO           Hugging Face repo id. Default: H-oliday/SwiftVR
  --launch                    Start the Gradio UI after installation
  --share true|false          Gradio share switch. Default: false
  Share=True                  Also accepted for convenience
  --host HOST                 Gradio host. Default: 0.0.0.0
  --port PORT                 Gradio port. Default: 7860
  -h, --help                  Show this help

Examples:
  bash scripts/install_linux.sh --download-checkpoints
  bash scripts/install_linux.sh --launch --share true
  bash scripts/install_linux.sh --launch Share=True
EOF
}

bool_value() {
  case "${1,,}" in
    1|true|yes|y|on) echo "true" ;;
    0|false|no|n|off) echo "false" ;;
    *) echo "$1" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --torchvision-version)
      TORCHVISION_VERSION="$2"
      shift 2
      ;;
    --skip-torch)
      INSTALL_TORCH=false
      shift
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --download-checkpoints)
      DOWNLOAD_CHECKPOINTS=true
      shift
      ;;
    --hf-repo-id)
      HF_REPO_ID="$2"
      shift 2
      ;;
    --launch)
      LAUNCH_UI=true
      shift
      ;;
    --share)
      SHARE="$(bool_value "$2")"
      shift 2
      ;;
    Share=*|share=*|SHARE=*)
      SHARE="$(bool_value "${1#*=}")"
      shift
      ;;
    --host)
      SERVER_NAME="$2"
      shift 2
      ;;
    --port)
      SERVER_PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -m venv "$VENV_DIR" >/dev/null 2>&1; then
  echo "Could not create a virtual environment with $PYTHON_BIN." >&2
  echo "On Debian/Ubuntu, install python3-venv first: sudo apt-get install python3-venv" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

if [[ "$INSTALL_TORCH" == "true" ]]; then
  if ! python -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$TORCH_INDEX_URL"; then
    cat >&2 <<EOF

Could not install torch==${TORCH_VERSION} / torchvision==${TORCHVISION_VERSION}.
Try overriding the versions or wheel index, for example:
  bash scripts/install_linux.sh --torch-version 2.5.1 --torchvision-version 0.20.1
  bash scripts/install_linux.sh --torch-index-url https://download.pytorch.org/whl/cu121

EOF
    exit 1
  fi
fi

REQ_FILE="$(mktemp)"
trap 'rm -f "$REQ_FILE"' EXIT
grep -Ev '^(torch|torchvision)(==|>=|<=|~=|>|<|$)' requirements.txt > "$REQ_FILE"

python -m pip install -r "$REQ_FILE"
python -m pip install "gradio>=4.44.0" "huggingface_hub>=0.24.0"
python -m pip install --no-deps -e .

if [[ "$DOWNLOAD_CHECKPOINTS" == "true" ]]; then
  python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${HF_REPO_ID}",
    local_dir="${CHECKPOINT_DIR}",
    local_dir_use_symlinks=False,
)
PY
fi

cat <<EOF

SwiftVR UI installation complete.

Activate:
  source "$VENV_DIR/bin/activate"

Launch:
  python app.py --host "$SERVER_NAME" --port "$SERVER_PORT" --share "$SHARE"

EOF

if [[ "$LAUNCH_UI" == "true" ]]; then
  python app.py --host "$SERVER_NAME" --port "$SERVER_PORT" --share "$SHARE"
fi
