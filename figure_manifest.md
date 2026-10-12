# Paper figures

All publication figures are exported directly to `figs/` as PDF, SVG and PNG.
Active code is in `scripts/`; all numerical evidence is under `results/`.

| Figure | Contents | Results | Render command |
|---|---|---|---|
| Figure1 | Full cortex and shared-subset benchmark; alpha 0.9 | `results/figure1/` | `bash scripts/paper.sh figure1` |
| Figure2 | Cortex 40-gene panel and 50-gene MERFISH analysis; alpha 0.9 | `results/figure2/` | `bash scripts/paper.sh figure2` |
| Figure3 | Cortex-noise recovery and Zhengmix4eq refinement; alpha 0.7 | `results/figure3/` | `bash scripts/paper.sh figure3` |
| Supp1 | Cortex conditional CV and measured resources | `results/figure1/` | Same as Figure1 |
| Supp2 | Nine tissues, one selected alpha per cohort | `results/supplementary/fullinput_benchmarks/` | `bash scripts/paper.sh fullinput --stage plot` |
| Supp3 | Eight cortex strains, alpha 0.9 | Same as Supp2 | Same as Supp2 |
| Supp4 | 900 paired alpha-sensitivity runs | `results/supplementary/alpha_subsampling/` | `bash scripts/supplementary_benchmarks.sh` |

Supp2/Supp3 PDFs retain the six metric pages. Their PNG/SVG exports show the
first (refit macro-F1) page; other page exports are in `figs/diagnostics/`.
`bash scripts/paper.sh render` regenerates every paper figure without fitting.
GO tables and marker-property figures are supporting outputs; marker plots and
per-tissue sensitivity plots are in `figs/diagnostics/`.

The manuscript references the same central PDFs. Figure 4 is retired. Historical
recorded paths are resolved through `metadata/layout_migration.json`; the native
execution evidence is preserved, with no new fits during organization.
