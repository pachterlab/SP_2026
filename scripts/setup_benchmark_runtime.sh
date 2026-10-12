#!/usr/bin/env bash
# Run from any working directory, after activating the benchmark environment.
set -euo pipefail
if [[ "${1:-}" == '--help' ]]; then
  echo 'Usage: COMET_BASE_PYTHON=python3.11 bash scripts/setup_benchmark_runtime.sh'
  echo 'Creates the ignored COMET environment and installs its pinned dependencies.'
  exit 0
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
COMET_PYTHON="${COMET_BASE_PYTHON:-python3.11}"
COMET_ENV="$PROJECT_ROOT/scripts/native/.venv_comet"
"$COMET_PYTHON" -m venv "$COMET_ENV"
"$COMET_ENV/bin/python" -m pip install -r environments/comet-requirements.txt
"$COMET_ENV/bin/python" -c 'import xlmhg.mhg_cython; print("Compiled native COMET dependency available")'
