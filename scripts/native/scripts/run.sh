#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${BENCHMARK_PYTHON:-python3}"
BENCHMARK_LIBDIR="$("$PYTHON" -c 'import sys; print(sys.prefix + "/lib")')"
export LD_LIBRARY_PATH="$BENCHMARK_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$HERE/../.cache/numba"
export MPLCONFIGDIR="$HERE/../.cache/matplotlib"
exec "$PYTHON" -B "$HERE/benchmark_execution.py" "$@"
