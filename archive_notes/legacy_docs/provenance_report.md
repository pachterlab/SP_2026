# CellSieve provenance report


Audit scope: source working tree recursively inventoried before copying; authored scripts/notebooks read statically, table/report consistency checked, relevant Git history inspected. Source remained read-only. No scientific models, PCA, enrichment, benchmarks or annotation analyses were run.

Each copied file has an original absolute path, revision (working tree or full Git commit), destination and SHA-256 in `archive_notes/copy_manifest.csv`. `source_inventory.csv` records every non-Git source file, including vendor files, models, logs and notebooks. Large H5AD/pickle/cache files are indexed, not copied. Verification means existing-file consistency, not methodological endorsement.


## Figure/panel: 1 — B–C

Existing seven-method cortex panel-size comparison

### Source

/home/nikki/celltype_compression_2/Benchmark/results/CortexHippocampus_v2_coarse_posonly_full_cv/benchmark_summary_all.csv

### Script

Original: Benchmark/scripts/benchmark_gene_panels_v2_modes.py + cell_sieve_v4.py (historical run version not fully pinned)

Copied: `figure1_method_benchmark/scripts/source/Benchmark/scripts/benchmark_gene_panels_v2_modes.py`

### Inputs

CortexHippocampus/B6J_CortexHippocampus_subset.h5ad; 86,427 cells, 23 celltype labels; no shared_gene_universe file in this run; all-gene status inferred, not proven by an invocation log

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `figure1_method_benchmark/results/benchmark`

### Status

EXISTING BUT VERSION AMBIGUOUS

### Notes

35 successful summary rows match per-run classification reports, CV metrics and gene counts. Combined table refreshed May 20 points CellSieve rows into a v4 directory despite its v2/coarse/posonly parent name. Remaining methods retain older outputs. Full-data panel selection precedes 3-fold classifier CV; this is conditional selected-panel CV, not nested feature-selection validation. accuracy/macro_f1 are full-data refit metrics. COMET fallback provenance is not logged; successful rows do not by themselves prove original COMET ranking rather than the optional Wilcoxon fallback. Display that uncertainty. Current defaults and script timestamp are not a recovered historical invocation.



## Figure/panel: 1 — A

Method/workflow and versioned implementation

### Source

/home/nikki/celltype_compression_2/cell_sieve_v4.py; Benchmark/scripts/benchmark_gene_panels_v2_modes.py

### Script

Original: cell_sieve_v4.py

Copied: `figure1_method_benchmark/scripts/source/cell_sieve_v4.py`

### Inputs

--adata-path and --celltype-column; normalize_total 10,000 then log1p; prior subsample at most 15,000 cells; seed default 42

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `figure1_method_benchmark/scripts/source`

### Status

VERIFIED EXISTING

### Notes

Current implementation uses SGDClassifier(loss=log_loss, penalty=elasticnet) as the multiclass one-vs-rest prior, then localized L2 LogisticRegression and Gurobi exact-cardinality MILP. Do not describe the elastic-net prior as a multinomial solver or each panel as a certified global minimum. alpha CLI is L1 ratio, distinct from SGD alpha=0.001. Current file differs from Git HEAD by annotation options and sweep defaults. Historical HEAD retained as evidence, not asserted identical to every May/June run. Output directories v5/v6 do not imply separate v5/v6 scripts; none exist.



## Figure/panel: 1 — D

Later cortex alpha/panel sweep and redundant candidate markers

### Source

/home/nikki/celltype_compression_2/cell_sieve_v6_output/CortexHippocampus

### Script

Original: cell_sieve_v4.py; redundant_solutions.py

Copied: `figure1_method_benchmark/scripts/source/redundant_solutions.py`

### Inputs

B6J cortex/hippocampus; 23 cell types; alphas 0.1/0.3/0.5/0.7/0.9; panel sizes 10/20/40/80/160

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `figure1_method_benchmark/results/cortex_v6`

### Status

EXISTING BUT VERSION AMBIGUOUS

### Notes

Later than main comparator benchmark; preserved separately. selected_genes, coefficient matrices, reports and summary files retained. Pearson-correlated substitutes are candidates, not demonstrated interchangeable predictive panels. Threshold invocation not logged. Standalone accuracy/global_macro_f1 are in-sample; cv_macro_f1 uses a preselected panel and globally fit scaler.



## Figure/panel: 1 — D alternative

Newest June 29 cortex output comparison

### Source

/home/nikki/celltype_compression_2/test_output

### Script

Original: No matching producer found; cell_sieve_package directory is empty

Copied: `NOT FOUND`

### Inputs

B6J cortex/hippocampus; alpha 0.7; panels 5/10/20/40/80

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `figure1_method_benchmark/results/june_test_version_ambiguous`

### Status

EXISTING BUT VERSION AMBIGUOUS

### Notes

Newest timestamps do not establish authority. Schema drops mode/restriction and renames alpha; at 10 genes accuracy 0.6642368704 versus v6 0.2843787242. No script/log establishes the change. Do not substitute into the earlier comparator plot.



## Figure/panel: 1 — E supporting

Dedicated L1/L2 marker-ranking control

### Source

/home/nikki/celltype_compression_2/Benchmark/results/L1_L2_comp/CortexHippocampus

### Script

Original: Benchmark/scripts/benchmark_l1_l2_comp.py

Copied: `figure1_method_benchmark/scripts/source/Benchmark/scripts/benchmark_l1_l2_comp.py`

### Inputs

B6J cortex; 64,820 train / 21,607 test; 20 genes; exact hvg_count and celltype_column in summary

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `figure1_method_benchmark/results/l1_l2_holdout`

### Status

VERIFIED EXISTING

### Notes

Separate holdout experiment, not an additional line in the full-data/conditional-CV plot. L1/L2 refer to ranking models; downstream evaluation is shared logistic regression. Later L1_L2_comp_full has 86,427 train and test cells and is retained as an explicitly in-sample alternative, not a replacement.



## Figure/panel: 2 — A–B

Existing marker identities, expression, coefficients and GO interpretation

### Source

/home/nikki/celltype_compression_2/cell_sieve_v5_output/CortexHippocampus/panel_size_46_ElasticNet_Unconstrained_alpha_0.90

### Script

Original: Panel selector: cell_sieve_v4.py lineage; GO/heatmap/violin producer scripts NOT FOUND in working tree or Git path history

Copied: `figure1_method_benchmark/scripts/source/cell_sieve_v4.py (shared); scripts/fig2_cortex_merfish.py (new table-only renderer)`

### Inputs

B6J cortex/hippocampus; 23 cell types; 46-gene alpha-0.9 unrestricted panel

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `figure2_cortex_merfish/results/biology_panel46`

### Status

EXISTING BUT INCOMPLETE

### Notes

Existing GO tables and already-computed expression/z-score/coefficients retained without re-running enrichment or expression summaries. These are 46 genes, not the different overlap-constrained MERFISH 50-gene panel. Producer/invocation and GO background/version metadata are incomplete.



## Figure/panel: 2 — C–F

50-gene MERFISH spatial classification validation

### Source

/home/nikki/celltype_compression_2/cell_sieve_v5_output/merfish_validation/CortexHippocampus_v4_overlap_panel40_alpha0.90/panel_size_50_ElasticNet_Unconstrained_alpha_0.90/merfish_spatial_prediction

### Script

Original: MERFISH fitting/evaluation producer NOT FOUND; no false attribution to the older Git plot_spatial_preds.py

Copied: `scripts/fig2_cortex_merfish.py (new saved-table renderer only)`

### Inputs

50 genes from MERFISH overlap-constrained B6J panel; metadata label_col=class, min_cells=10, n_splits=5; 8,891 evaluated of 8,899 total, 9 evaluated of 13 total classes; exact original spatial H5AD path not recoverable

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `figure2_cortex_merfish/results/merfish_panel50_cv`

### Status

EXISTING BUT INCOMPLETE

### Notes

accuracy=0.8176808008098077, macro_f1=0.7385569256899186 verified both in JSON/CSV and directly against saved true/predicted labels. All 50 selected genes match overlap file. Numerical result is VERIFIED EXISTING; end-to-end provenance is incomplete. Five-fold metadata supports within-MERFISH evaluation; this is not proof of a frozen scRNA-to-MERFISH classifier transfer. Obtain producer code before making that claim. Later 50-gene train_all_test_all is explicitly in-sample (83.2004%/78.8171%); not preferred merely because newer. Older 40-gene and 46-gene/17-overlap versions retained only as compact metric evidence.



## Figure/panel: 3 — A–D planned

Kidney 10% label-noise benchmark

### Source

No kidney poisoned dataset, truth key, flagged output or recovery summary found in the recursively audited source

### Script

Original: Generic building blocks exist: poison_injector.py, cell_sieve_v4.py, poison_expt_grader.py; current poison_expt.py targets Cortex

Copied: `figure3_annotation/noise_benchmark/scripts/source`

### Inputs

Planned kidney dataset/version and labels require specification before future work

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `NONE`

### Status

PLANNED / NOT YET IMPLEMENTED

### Notes

No verified kidney ~72% result. No kidney injected-error confusion matrix, corrected-label output, or PCA validation found. Current generic correction code is a single pass, not an implemented kidney experiment. No new analysis was run.



## Figure/panel: 3 reference only — Existing groundwork

Traceable cortex 10% noise and PCA-centroid validation

### Source

/home/nikki/celltype_compression_2/poison_expt_alphas_output/Cortex_POISONED_010/panel_size_80_ElasticNet_Unconstrained_alpha_0.70; poison_expt_alphas_output/targeted_metrics_summary.csv

### Script

Original: poison_expt.py; poison_injector.py; cell_sieve_v4.py; poison_expt_grader.py

Copied: `figure3_annotation/noise_benchmark/scripts/source/poison_expt.py`

### Inputs

86,427 cortex cells; 8,642 injected labels (floor of 10%); 80 genes; alpha 0.7; celltype_poisoned; five-fold classifier OOF probabilities, margin >0.30 in current source

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `figure3_annotation/noise_benchmark/results/cortex_reference`

### Status

VERIFIED EXISTING

### Notes

Saved counts cross-checked against truth key and flagged labels: caught 6,293, missed 2,349, restored 5,972, wrong corrections 321, clean flagged 1,251. 72.8187919463% is catch/injected; 94.8990942317% is restored/caught; restored/all injected is 69.1043739875%. PCA median-centroid result: 620/1,251 nearer predicted centroid (49.5603517186%); proximity does not establish ground-truth misannotation. OOF only refers to final classifier, since feature selection/scaling used all cells. Older 40-gene sweep files use different settings and are excluded. Existing alluvial labels human_error_caught/collateral are interpretations, not established truth.



## Figure/panel: 3 — E groundwork

Existing one-pass kidney annotation transfer

### Source

/home/nikki/celltype_compression_2/auto_milp_output/Kidney/chen_transfer_results; Git 5f51ded:transfer_labels.py

### Script

Original: transfer_labels.py recovered from Git 5f51ded (deleted in afb6679)

Copied: `figure3_annotation/iterative_annotation/existing_scripts/transfer_labels.py`

### Inputs

Kidney/B6J_Kidney_subset.h5ad (89,940 training cells, 17 labels); Kidney/chen_kidney_w12.h5ad (33,129 query cells, 18 labels); 28 selected_genes_negative_allowed genes match coefficients; query gene_symbols and cell_type per historical example

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `figure3_annotation/iterative_annotation/existing_results/kidney_transfer`

### Status

VERIFIED EXISTING

### Notes

Saved overlap summary: nine shared classes, 25,375 overlap-truth cells, accuracy 0.8681773399014778. This is limited to mapped overlapping labels, not all query cells. Fuzzy string label mapping threshold defaults to 0.75. Small joblib retained but never deserialized; expression data and query-labeled H5AD remain external. Historical script is a provenance recovery, not a new method.



## Figure/panel: 3 — F planned

Iterative automatic kidney annotation/refinement

### Source

No multi-round kidney workflow, stopping rule, iteration trace or iteration results found

### Script

Original: NOT FOUND; one-pass transfer and generic one-pass misannotation correction are only groundwork

Copied: `figure3_annotation/iterative_annotation/existing_scripts/transfer_labels.py (groundwork only)`

### Inputs

Not specified

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `NONE`

### Status

PLANNED / NOT YET IMPLEMENTED

### Notes

Not implemented or run in this cleanup. Repeated panel/alpha sweeps are not annotation iterations.



## Figure/panel: Supplement — S1–S2

Existing cross-tissue multi-method comparisons

### Source

/home/nikki/celltype_compression_2/Benchmark/results/cell_sieve_v4_alpha_custom_8_tissues_all_methods

### Script

Original: Benchmark/scripts/benchmark_gene_panels_v2_modes.py; cell_sieve_v4.py

Copied: `figure1_method_benchmark/scripts/source/Benchmark/scripts/benchmark_gene_panels_v2_modes.py (shared)`

### Inputs

B6J Adrenal alpha .9, Gastrocnemius .7, GonadsFemale .5, GonadsMale .9, Kidney .9, Liver .9; shared 3,000-gene universes retained; panels normally 10/20/40/80/160

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `supplementary/all_tissue_benchmarks/results/multimethod`

### Status

EXISTING BUT INCOMPLETE

### Notes

Directory says 8 tissues, but only six tissue runs exist here. 181 successful rows match underlying reports; 30 COMET failure rows are retained as failures, never zeros. Gastrocnemius has an additional successful panel-size row; preserve it rather than deduplicate by guess. Some method actual_panel_size can differ from requested; renderer uses actual. Cortex benchmark did not record the same 3,000-gene filter, so these are not guaranteed matched all-tissue settings. No complete newer multi-method Heart/Diencephalon run found.



## Figure/panel: Supplement — S3 available tables

Available single-method tissue alpha/panel sweeps

### Source

/home/nikki/celltype_compression_2/cell_sieve_v6_output (Gastrocnemius/Heart/Liver); cell_sieve_v5_output (Adrenal/DiencephalonPituitary/GonadsFemale/GonadsMale/Kidney)

### Script

Original: cell_sieve_v4.py lineage; exact invocations not archived

Copied: `figure1_method_benchmark/scripts/source/cell_sieve_v4.py (shared)`

### Inputs

B6J tissue datasets with celltype labels; explicit source dataset/alpha/size in each summary

### Outputs

Original files: see Source and per-file copy manifest.

Organized: `supplementary/all_tissue_benchmarks/results/standalone`

### Status

EXISTING BUT VERSION AMBIGUOUS

### Notes

Retain summaries and selected genes only. v5 and v6 have different size grids; do not treat them as a single controlled comparison. B-cell ground truth is not an 8cube tissue and is excluded from the paper supplement.



## Additional audit evidence

Read-only HDF5 metadata for the principal cortex, kidney, Chen W12 and MERFISH-overlap inputs is recorded in `archive_notes/input_h5ad_metadata.json`; expression matrices were not loaded or transformed. `verification_report.json` records numerical consistency checks and a pre-existing COMET vendor encoding issue. Original COMET source was retained unchanged rather than silently repaired.

Input metadata confirms 86,427 × 5,083 for the B6J cortex subset, 89,940 × 8,389 for B6J kidney, 33,129 × 32,034 for Chen W12, and 86,427 × 556 for the MERFISH-overlap training subset. These are stored input shapes, not newly filtered data. See [figure_plan.md](figure_plan.md) for assembly choices and explicit placeholders.

Newly rendered figure provenance (renderer hashes, saved-input scope and output hashes): `archive_notes/generated_artifact_manifest.json`. The 50 selected MERFISH genes were also checked against the saved overlap table; all 50 are present.

## Figure 2 MERFISH refit — 2026-09-30

The user explicitly authorized refitting the fixed 50-gene model on MERFISH and checking historical scores. `scripts/refit_fig2_merfish.py` uses `/home/nikki/HCR/MERFISH/Merfish_cortex.h5ad`; all 8,891 eligible cell IDs, class labels, coordinates, and their order match the historical table. The explicit balanced logistic-regression reconstruction uses fold-specific scaling and five shuffled stratified folds (seed 42). Pooled CV accuracy is 0.8175683275222134 and macro-F1 is 0.7317982496972286, versus historical 0.8176808008098077 and 0.7385569256899186. This is a documented new fit, not an exact reproduction of the unavailable original MERFISH recipe. Historical files remain intact. New coefficients, scaling, intercepts, held-out predictions, comparisons, validation checks, and hashed provenance are in `figure2_cortex_merfish/results/merfish_panel50_coefficients`. The final figure uses this new fit for E and G; D uses the original 8cube 50-gene coefficients. See the Figure 2 README for the complete current panel map.
