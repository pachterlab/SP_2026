#!/usr/bin/env bash
# Keep an already-started Figure 1/2 batch supervised from a detached tmux session.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: supervise_figures12.sh EXISTING_PROCESS_ID BATCH_DIRECTORY' >&2
  exit 2
fi
TASK_PID="$1"
TASK_BATCH="$2"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$HERE/.." && pwd)"
source "$HERE/paper_env.sh"
ENV_DIR="$BENCHMARK_ENV_DIR"
# An internet disconnect does not affect this detached host-side supervisor.
while kill -0 "$TASK_PID" 2>/dev/null; do sleep 15; done
# Let any remaining native children finish before considering a restart.
while pgrep -f "^$ENV_DIR/bin/python.*(benchmark_execution.py|cellsieve.py).*$TASK_BATCH" >/dev/null; do sleep 15; done
if ! "$ENV_DIR/bin/python" -B - "$PROJECT_ROOT" "$TASK_BATCH" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'results/figure1/results/figures12_rerun_status.json'
s=json.loads(p.read_text()) if p.exists() else {}
sys.exit(0 if s.get('batch')==sys.argv[2] and s.get('status')=='complete' else 1)
PY
then
  echo 'Resuming interrupted Figure 1/2 batch.'
  bash "$HERE/rerun_figures12.sh" --alpha 0.9 --batch "$TASK_BATCH"
fi
"$ENV_DIR/bin/python" -B "$HERE/finalize_figures12.py"
"$ENV_DIR/bin/python" -B "$HERE/finalize_figures12.py" --check-only
echo 'Figure 1/2 alpha-0.9 rerun, figure rebuild, documentation and validation complete.'
