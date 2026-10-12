# Workspace boundaries

User instruction: never modify anything in `/home/nikki/covariance_geometry/SEP_2026` or `/home/nikki/celltype_compression_2`, including code, data and outputs. Treat both directories as strictly read-only. Do not run original scientific scripts or shell launchers that write into them. Keep all paper-workspace edits and generated artifacts under `/home/nikki/SP_2026`.

This task was curation only: no new scientific analyses. `make figures` renders saved tables; `make verify` checks file integrity and saved-result consistency. Any future request must preserve the above source-directory boundary. Original copied scripts are provenance snapshots; never silently alter their scientific behavior.

The user subsequently authorized native benchmark repairs and, on 2026-09-29, two real Figure 1 analyses: full cortex without COMET, plus a common stratified subset including COMET. Native-ranked output adapters for NS-Forest and COMET are authorized; replacement selectors and synthetic figure inputs are not. Preserve the original panel grid, evaluation formulas and figure layout. Original CellSieve source, including `/home/nikki/cell_sieve_package`, remains strictly read-only.

Later steering in the same session requests a 2×3 metric/runtime/memory grid for each task, replacing the previous three-panel Figure 1 layout. Save task-titled PDF/SVG/PNG and traceable underlying tables for both real cohorts.

On 2026-10-04 the user explicitly requested deletion of unrelated Figure 1 and Figure 2 results and scripts. Superseded fits, failed alternatives, historical script snapshots and output backups in these two figure folders may be removed. Keep the current alpha-0.7 figures, required comparator/L1 proof files, input cohorts, three reference checks, and the active benchmark/runtime scripts. The cleanup inventory is `archive_notes/figure1_figure2_cleanup.json`; preserve the protected external source-directory boundary above.

# Current paper workflow (12 October 2026)

The current paper has Figures 1–3; Figure 4 is retired. Figures 1–2 use alpha
0.9. Full-input supplements use one selected alpha per cohort in
`results/supplementary/fullinput_benchmarks/`; historical mixed-alpha runs remain local
under `results/supplementary/local_archive/`. Never infer active figures from historical
planning documents in `archive_notes/legacy_docs/`.

Use `scripts/paper.sh` and the `sp2026-paper` Conda environment. The root README
and `figure_manifest.md` are the active reproduction guides. Ordinary rendering
and verification must not fit models. Keep native scientific source snapshots
unchanged, preserve result evidence locally and Git-ignore raw artifacts. Place
new caches under the workspace and metadata under `metadata/`. The alpha audit
is `metadata/cell_sieve_alpha_audit.csv`. Original source-directory protections
above remain in force, especially `/home/nikki/covariance_geometry/SEP_2026`.

# Central layout

Publication files are only `figs/Figure1`–`Figure3` and `figs/Supp1`–`Supp4`.
All active executable analysis code is under `scripts/`, including native and
Figure 3 drivers. All numerical evidence is under `results/`. Do not create
per-analysis figure or script folders. Auxiliary plots go in `figs/diagnostics/`.
The manuscript references the central PDFs directly. Historical paths are
resolved through `metadata/layout_migration.json`; do not rewrite scientific
measurements or native provenance simply to rename directories.
