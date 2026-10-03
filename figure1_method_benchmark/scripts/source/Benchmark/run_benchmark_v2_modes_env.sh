#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${BENCHMARK_ENV:-benchmark_env}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda run -n "${ENV_NAME}" zsh -lc "
  export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}\"
  cd \"${ROOT_DIR}\"
  python Benchmark/scripts/benchmark_gene_panels_v2_modes.py \"\$@\"
" -- "$@"
