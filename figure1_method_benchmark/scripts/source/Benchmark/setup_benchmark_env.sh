#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-benchmark_env}"
SOURCE_ENV="${2:-ember_env}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env ${ENV_NAME} already exists; skipping clone."
else
  conda create --name "${ENV_NAME}" --clone "${SOURCE_ENV}" -y
fi

conda run -n "${ENV_NAME}" python -m pip install xlmhg gurobipy
conda run -n "${ENV_NAME}" python -m pip install --no-deps \
  -e "${ROOT_DIR}/Benchmark/external/scGeneFit-python" \
  -e "${ROOT_DIR}/Benchmark/external/NSForest"

echo
echo "Ready:"
echo "  conda activate ${ENV_NAME}"
echo "  python Benchmark/scripts/benchmark_gene_panels.py --help"
