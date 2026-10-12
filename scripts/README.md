# Paper scripts

Use `bash scripts/paper.sh --help` for the supported entry points. Launchers
share `paper_env.sh`, use the active/dedicated paper Conda environment and keep
all output inside this workspace. The [root README](../README.md) documents the
full order of operations, inputs, packages and reproduction limits.

| Workflow | Main scripts |
|---|---|
| All saved figures | `paper.sh render`, `supplementary_benchmarks.py` |
| Runtime and integrity | `check_environment.py`, `verify_workspace.py`, `verify_supplementary.py` |
| Figure 1 | `fig1_method_benchmark.py`; native engine and adapters in `scripts/native/scripts/` |
| Figure 1–2 update | `rerun_figures12.py`, `finalize_figures12.py` |
| Figure 2 derived tables | `build_fig2_panel40.py`, `build_fig2_violins.py`, `reselect_fig2_panel50.py`, `refit_fig2_merfish.py` |
| Figure 2 render | `fig2_cortex_merfish.py` |
| GO and marker audit | `rerun_fig2_go.py`, `build_fig2_go.py`, `audit_fig2_panel40_markers.py` |
| Figure 3 render | `fig3_annotation_stories.py`; analysis drivers remain beside their Figure 3 experiments |
| Alpha sensitivity | `subsample_alpha_benchmark.py`, `compile_alpha_sensitivity.py` |
| Full-input tissue/strain runs | `fullinput_benchmarks.py`; shared preparation/rendering in `run_8cube_supplements.py` and `supplementary_helpers.py` |
| Manuscript copies | `sync_manuscript_figures.py` |
| Environment installation | `setup_benchmark_runtime.sh` |

Older launcher names are compatibility aliases. `rerun_cellsieve_alpha07.py`
retains an explicit historical alpha override and is not the publication entry
point. `finalize_alpha07_audit.py` is a compatibility helper. Native source
snapshots and historical scripts under `source/` or `provenance/` are evidence,
not user-facing launchers. In particular the original cortex-noise automation
contains external output paths and must not be launched directly.


Paper exports are `figs/Figure1`–`Figure3` and `figs/Supp1`–`Supp4`. All raw
results are under `results/`. `scripts/native/scripts/` contains the native
benchmark engine; `scripts/figure3/` contains the clustering/refinement drivers.
Historical noise scripts in `scripts/figure3/noise_source/` are provenance
snapshots and must not be launched directly. The main launcher sets the runtime
and paths for all active commands.
