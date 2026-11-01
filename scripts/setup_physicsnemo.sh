#!/usr/bin/env bash
# Fresh, isolated WSL/Linux x86_64 CUDA12 environment. Does not alter weather-gpu.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_env="$repo_root/.venv/physicsnemo"
if [[ -e "$target_env" ]]; then
  echo "Environment already exists: $target_env; preserve it and use its Python directly."
  exit 1
fi
python3 -m venv "$repo_root/.venv/phase5-bootstrap"
"$repo_root/.venv/phase5-bootstrap/bin/python" -m pip install 'uv==0.13.6'
export UV_PYTHON_INSTALL_DIR="$repo_root/.venv/phase5-python"
export UV_PYTHON_BIN_DIR="$repo_root/.venv/phase5-bin"
export UV_CACHE_DIR="$repo_root/.venv/phase5-cache"
uv_bin="$repo_root/.venv/phase5-bootstrap/bin/uv"
"$uv_bin" python install 3.12.16
"$uv_bin" venv --python 3.12.16 "$target_env"
"$uv_bin" pip install --python "$target_env/bin/python" --torch-backend cu126 -r "$repo_root/requirements-physicsnemo.lock.txt"
"$uv_bin" pip check --python "$target_env/bin/python"
"$target_env/bin/python" -c 'import torch; from physicsnemo.models.fno import FNO; assert torch.cuda.is_available(), "CUDA required"; print("PhysicsNeMo FNO import and CUDA device available:", torch.cuda.get_device_name(0))'
