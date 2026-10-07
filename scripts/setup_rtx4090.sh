#!/usr/bin/env bash
# Run inside the user's rented Linux instance. Installs locked deps; does not train.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "Use this bootstrap inside the Linux x86_64 RTX 4090 instance." >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  echo "Install uv first: python3 -m pip install uv" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is missing; choose a GPU-enabled Vast template." >&2
  exit 1
fi
python3 - <<'PY'
import subprocess
output = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True)
print(output, end='')
versions = [line.rsplit(',', 1)[1].strip() for line in output.strip().splitlines()]
if not versions or any(int(v.split('.')[0]) < 580 for v in versions):
    raise SystemExit('The frozen Linux lock uses CUDA 13; select an NVIDIA driver >=580 host.')
PY
uv sync --frozen --python 3.11
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python scripts/run_rtx4090_pilot.py --check-env
echo "Environment checked. Launch: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_rtx4090_pilot.py"
