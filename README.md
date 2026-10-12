# Cell Sieve: analyses and figures

Code and figures for the Cell Sieve paper, including method benchmarks,
cortex and MERFISH analyses, annotation refinement, and tissue/strain comparisons.
The manuscript is maintained separately.

## Find the figures

Publication figures are in `figs/`, with PDF, SVG and PNG exports.
Use the PDFs for the manuscript or Overleaf, and PNGs for quick previews.

| Figure | Contents |
|---|---|
| `Figure1` | Cortex method benchmark |
| `Figure2` | Cortex gene panel and MERFISH evaluation |
| `Figure3` | Annotation refinement |
| `Supp1` | Figure 1 conditional cross-validation and resource measurements |
| `Supp2` | Nine B6J tissues at selected alpha values |
| `Supp3` | Eight cortex strains at alpha 0.9 |
| `Supp4` | Paired alpha sensitivity across tissues |

Supp2 and Supp3 PDFs each contain six metric pages. Their PNG and SVG exports
show the first page, which reports refit macro-F1. Additional plots are in
`figs/diagnostics/`.

See [figure_manifest.md](figure_manifest.md) for the result files behind each panel.

## Set up the environment

The supported environment is Linux x86-64 with Conda. To install:

```bash
cd /path/to/SP_2026
conda env create -f environments/benchmark.yml
conda activate sp2026-paper
bash scripts/setup_benchmark_runtime.sh
bash scripts/paper.sh check
```

For an existing installation, activate `sp2026-paper` and run the check command.
The [environment guide](environments/README.md) describes package versions and
platform requirements. COMET uses a separate NumPy-1 environment created by the
setup script in `scripts/native/.venv_comet/`.

Model fitting requires a valid Gurobi license. Set `GRB_LICENSE_FILE` if your
license is outside Gurobi's default search locations. Figure rendering requires
Helvetica fonts in `assets/fonts/Helvetica/`, or a font directory specified with
`CELLSIEVE_FONT_DIR`.

Commands use the activated Conda environment unless `BENCHMARK_ENV_DIR` is set.
To select an environment explicitly, for example in a tmux session:

```bash
export BENCHMARK_ENV_DIR=/path/to/conda/envs/sp2026-paper
```

## Prepare the data and saved results

Git includes code, environment specifications, documentation and publication
figures. Expression matrices, fitted models, predictions and solver logs are
stored locally and must be transferred separately when using a new clone.
A public download URL is not yet available.

Use these inventories to locate the required data:

- [Input data guide](h5ads/README.md) and [data locations](h5ads/data_locations.json)
- [Input checksums](archive_notes/self_contained_inputs.json)
- [Figure 1 results](results/figure1/README.md)
- [Figure 2 results](results/figure2/README.md)
- [Figure 3 results](results/figure3/README.md)
- [Supplementary results](results/supplementary/README.md)

Keep relative paths and symlinks when transferring data and results. The
full-input supplements require 16 prepared cohorts, available through
`results/supplementary/fullinput_benchmarks/h5ads/`. All H5AD payloads are stored
in `h5ads/`; the analysis folders contain relative links. See
[h5ads/README.md](h5ads/README.md) for the data inventory and archive command.

## Regenerate figures from saved results

After setting up the environment and restoring the data/result bundle, run:

```bash
bash scripts/paper.sh check
bash scripts/paper.sh render
bash scripts/paper.sh verify
```

`render` rebuilds the main and supplementary figures, GO table and marker-audit
plot from completed results. It also checks the seven publication PDFs and
records their hashes in `results/manuscript/figure_sources.json`.
`verify` checks saved results, predictions, metrics and input hashes.
Neither command fits models. These commands can take several minutes because
they read large local files.

To render individual figures:

```bash
bash scripts/paper.sh figure1
bash scripts/paper.sh figure2
bash scripts/paper.sh figure3
bash scripts/paper.sh fullinput --stage plot
```

`make figures` and `make verify` are also available after activating the environment.
The `bash scripts/paper.sh manuscript` command checks the publication PDFs and
records their hashes; it does not require manuscript source files or compile TeX.

## Rerun analyses

The following commands fit models or recompute analyses. Save a copy of completed
results if you need to retain the published figures and recorded timings.
Solver results under time limits, runtime and memory use can vary by machine.

### Figures 1 and 2

```bash
# Inspect inputs and environment without fitting.
bash scripts/paper.sh figures12-fit --dry-run

# Fit all methods on full cortex and the shared 2,500-cell subset.
bash scripts/paper.sh figure1-fit

# Fit CellSieve at alpha 0.9 using saved comparators, then rebuild Figure 2.
bash scripts/paper.sh figures12-fit
```

`figures12-fit` writes a timestamped CellSieve batch. Resume a compatible batch
with `--batch <printed-batch-path>`. After a new all-method benchmark, run
`figures12-fit` to rebuild Figure 2 dependencies with the saved comparator bundle.

Full cortex uses 86,427 cells and 5,083 genes. The 50-gene MERFISH-overlap panel
uses a separate 556-gene universe. The supplementary cortex input has 24,170
genes and is not interchangeable with these inputs. Fits use alpha 0.9, seed 42,
a gap target of 0.10 and four requested workers, with time limits of 600 seconds
for full cortex/overlap and 300 seconds for the subset.

The published full-cortex L1 points come from five historical independent fits;
a fresh ranking-based run will not reproduce their timings, and the historical
input has no checksum. The original MERFISH evaluation recipe is unavailable;
the current five-fold refit is documented in the
[Figure 2 guide](results/figure2/README.md). See the
[native benchmark protocol](results/figure1/native/PROTOCOL.md) for full settings.

```bash
# Recompute GO enrichment against the frozen whole-mouse genome.
bash scripts/paper.sh go

# Recompute specificity, HVG/DE categories and panel recovery.
bash scripts/paper.sh markers
```

GO requires the frozen ontology and mouse annotations, including `sources.json`,
in `results/figure2/resources/go/`. Outputs include a LaTeX table.

### Figure 3

```bash
# Fit the 20 initializations, skipping completed fits.
bash scripts/paper.sh figure3-screen --resume

# Reuse the first pass and fit passes 2–3.
bash scripts/paper.sh figure3-passes

# Render the figure.
bash scripts/paper.sh figure3
```

These analyses require the frozen Zhengmix4eq counts, reference labels and
initial partitions. Fitting uses 60 genes, alpha 0.7, a 300-second solver limit
and correction margin greater than 0.30. The follow-up uses 30 PCs, 10 neighbors
and resolution 0.6. Reference labels are used for evaluation only.

Cortex panels A–B use the saved 80-gene, alpha-0.7 noise experiment. Its saved
flags and centroid distances support figure reproduction, but the complete
original fitting code is unavailable. A new fit should be treated as a new
experiment. See the [Figure 3 guide](results/figure3/README.md) for details.
The historical `poison_expt.py` is a provenance copy with external output paths
and should not be used to rerun the analysis.

### Tissue and strain supplements

Use new output directories to retain the completed publication results:

```bash
bash scripts/paper.sh sensitivity --tissue all --data-dir /path/to/annotated \
  --inputs-dir results/supplementary/new_fullinput --out-dir results/supplementary/new_sensitivity
bash scripts/paper.sh fullinput --stage run --data-dir /path/to/annotated \
  --sensitivity-dir results/supplementary/new_sensitivity --out-dir results/supplementary/new_fullinput
```

Each cohort uses the alpha with the highest mean sensitivity score: 0.7 for
Adrenal, Diencephalon/Pituitary and Kidney; 0.5 for Heart; and 0.9 for the other
five tissues and all cortex strains. The 16 unique cohorts produce 255 displayed
method/budget rows because B6J cortex appears in both comparisons.

The sensitivity figure shows means and standard deviations, with stars marking
empirical maxima. BH-corrected paired tests do not establish a unique significant
winner. See the [supplementary guide](results/supplementary/README.md).

For long runs, use tmux on the licensed computer so the job continues after
disconnecting.

## Supplemental manuscript tables

`supplemental_data/` contains CellSieve result tables and selected-gene lists for
Figures 1–3 and Supp1–Supp4, grouped as seven tables (S1–S7). Gene lists
are included alongside run results. The GO table supports the main-text analysis. Start with `supplemental_data/README.md` and
`table_index.csv` for descriptions, source files and checksums.

Regenerate these tables from saved results without fitting models:

```bash
python scripts/prepare_supplemental_data.py
```

Upload this folder separately with the manuscript. `results/`, `h5ads/` and
`supplemental_data/` are excluded from Git.

## Repository layout

| Directory | Contents |
|---|---|
| `figs/` | Publication figures and diagnostic plots |
| `scripts/` | Rendering, fitting and verification commands |
| `results/` | Numerical results, models and analysis-specific documentation |
| `h5ads/` | All H5AD data files and file inventory |
| `supplemental_data/` | Manuscript supplemental tables and gene lists |
| `environments/` | Conda recipe and pinned Python dependencies |
| `external/` | Dependency source snapshots used by the analyses |
| `assets/` | Rendering assets, including local fonts |
| `metadata/` | Current inventories, checks and alpha audit |
| `archive_notes/` | Historical documentation and provenance |
| `logs/` | Local execution logs |

Large data files, fitted results, diagnostics, installed environments and caches
are Git-ignored. Historical paths can be looked up in
`metadata/layout_migration.json`.
