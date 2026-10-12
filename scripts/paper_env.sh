#!/usr/bin/env bash
# Source this file from a launcher; all writable caches stay in this workspace.
PAPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${BENCHMARK_ENV_DIR:-}" ]]; then
    PAPER_ENV="$BENCHMARK_ENV_DIR"
elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-base}" != base && -x "$CONDA_PREFIX/bin/python" ]]; then
    PAPER_ENV="$CONDA_PREFIX"
elif [[ -x /home/nikki/miniconda3/envs/sp2026-paper/bin/python ]]; then
    PAPER_ENV=/home/nikki/miniconda3/envs/sp2026-paper
else
    PAPER_ENV=/home/nikki/miniconda3/envs/benchmark_env
fi
if [[ ! -x "$PAPER_ENV/bin/python" ]]; then
    echo 'Create and activate environments/benchmark.yml, or set BENCHMARK_ENV_DIR.' >&2
    return 1
fi
export BENCHMARK_ENV_DIR="$PAPER_ENV"
export BENCHMARK_PYTHON="$PAPER_ENV/bin/python"
export LD_LIBRARY_PATH="$PAPER_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$PAPER_ENV/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$PAPER_ROOT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$PAPER_ROOT/.cache/numba"
export CELLSIEVE_FONT_DIR="${CELLSIEVE_FONT_DIR:-$PAPER_ROOT/assets/fonts/Helvetica}"
