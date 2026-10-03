#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/nikki/miniconda3/envs/benchmark_env/bin/python
export LD_LIBRARY_PATH=/home/nikki/miniconda3/envs/benchmark_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$HERE/../.cache/numba"
export MPLCONFIGDIR="$HERE/../.cache/matplotlib"
exec "$PYTHON" -B "$HERE/benchmark_execution.py" "$@"
