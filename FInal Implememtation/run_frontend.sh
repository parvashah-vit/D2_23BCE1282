#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="/home/programmer/PyTorch Project/myenv/bin/python"
PYTHON_BIN="${PYTORCH_PYTHON:-$DEFAULT_PYTHON}"
PORT="${PORT:-8000}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime not found at: $PYTHON_BIN" >&2
  echo "Set PYTORCH_PYTHON to a Python interpreter with torch, torchvision, fastapi, and uvicorn installed." >&2
  exit 1
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
