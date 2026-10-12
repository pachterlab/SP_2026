#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/paper_env.sh"
exec "$BENCHMARK_PYTHON" -B "$HERE/rerun_figures12.py" "$@"
