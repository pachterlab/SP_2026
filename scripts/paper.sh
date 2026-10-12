#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:---help}" == --help ]]; then
    cat <<'EOF'
Usage: bash scripts/paper.sh COMMAND [arguments]
  check                  Check packages, native runtime and local inputs; no fitting
  render                 Render all seven paper figures into figs/
  verify                 Verify saved main and supplementary evidence
  manuscript             Validate shared central manuscript PDFs
  figure1                Render Figure 1 and its resource supplement
  figure2                Render Figure 2
  figure3                Render Figure 3
  figure1-fit            Fit all Figure 1 benchmark methods in a new native batch
  figures12-fit [args]   Fit CellSieve at alpha 0.9 and rebuild Figures 1–2
  figure3-screen [args]  Fit/resume 20 settings, then regenerate many-to-one scores
  figure3-passes [args]  Reuse pass 1; fit passes 2–3 and evaluate
  sensitivity [args]    Paired tissue alpha sensitivity
  fullinput [args]       Unified tissue/strain workflow (default: saved-result plot)
  go                    Recompute GO enrichment and export LaTeX
  markers [args]        Recompute the 40-gene marker-property audit
Only commands named fit/screen/passes/sensitivity/go/markers run analyses.
EOF
    exit 0
fi
source "$HERE/paper_env.sh"
cd "$PAPER_ROOT"
COMMAND="$1"; shift
case "$COMMAND" in
    check) exec "$BENCHMARK_PYTHON" -B scripts/check_environment.py "$@" ;;
    render) make figures PYTHON="$BENCHMARK_PYTHON"; "$BENCHMARK_PYTHON" -B scripts/collect_figure_evidence.py; exec "$BENCHMARK_PYTHON" -B scripts/sync_manuscript_figures.py ;;
    verify) "$BENCHMARK_PYTHON" -B scripts/verify_workspace.py; exec "$BENCHMARK_PYTHON" -B scripts/verify_supplementary.py ;;
    manuscript) exec "$BENCHMARK_PYTHON" -B scripts/sync_manuscript_figures.py "$@" ;;
    figure1) exec "$BENCHMARK_PYTHON" -B scripts/fig1_method_benchmark.py "$@" ;;
    figure2) exec "$BENCHMARK_PYTHON" -B scripts/fig2_cortex_merfish.py "$@" ;;
    figure3) exec "$BENCHMARK_PYTHON" -B scripts/fig3_annotation_stories.py "$@" ;;
    figure1-fit) "$BENCHMARK_PYTHON" -B scripts/fig1_method_benchmark.py --run-native "$@"; exec "$BENCHMARK_PYTHON" -B scripts/collect_figure_evidence.py ;;
    figures12-fit) exec "$BENCHMARK_PYTHON" -B scripts/rerun_figures12.py --alpha 0.9 "$@" ;;
    figure3-screen) "$BENCHMARK_PYTHON" -B scripts/figure3/zheng_screen/run_screen.py "$@"; exec "$BENCHMARK_PYTHON" -B scripts/figure3/zheng_screen/plot_many_to_one.py ;;
    figure3-passes) exec "$BENCHMARK_PYTHON" -B scripts/figure3/zheng_passes/run_refinement.py "$@" ;;
    sensitivity) exec "$BENCHMARK_PYTHON" -B scripts/subsample_alpha_benchmark.py "$@" ;;
    fullinput) exec "$BENCHMARK_PYTHON" -B scripts/fullinput_benchmarks.py "$@" ;;
    go) "$BENCHMARK_PYTHON" -B scripts/rerun_fig2_go.py "$@"; exec "$BENCHMARK_PYTHON" -B scripts/build_fig2_go.py ;;
    markers) exec "$BENCHMARK_PYTHON" -B scripts/audit_fig2_panel40_markers.py "$@" ;;
    *) echo "Unknown command: $COMMAND. Use --help." >&2; exit 2 ;;
esac
