"""Derive Figure 2 A expression summaries from the real Figure 1 CellSieve panel.

The gene selection and classifier coefficients are saved native package outputs.
This script only summarizes the original cortex expression matrix for those genes.
"""
from pathlib import Path
import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from figure_style import ROOT


DATA = ROOT / 'h5ads/B6J_CortexHippocampus_subset.h5ad'
POINTER = ROOT / 'figure1_method_benchmark/results/native_active_batch.json'
DEST = ROOT / 'figure2_cortex_merfish/results/biology_panel40_fig1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    pointer = json.loads(POINTER.read_text())
    if pointer['status'] != 'complete_with_package_cellsieve':
        raise ValueError('Figure 1 package CellSieve run is incomplete')
    native = Path(pointer['cellsieve_batch']) / (
        'full_600/cell_sieve/B6J_CortexHippocampus_subset/'
        'cell_sieve_package_runs/Package_0.1.0/panel_size_40/panel_size_40_alpha_0.90'
    )
    genes_path = native / 'selected_genes.txt'
    coefficients_path = native / 'model_coefficients.csv'
    genes = genes_path.read_text().splitlines()
    coefficients = pd.read_csv(coefficients_path, index_col=0)
    if len(genes) != 40 or len(set(genes)) != 40 or coefficients.shape != (23, 40):
        raise ValueError('Expected the native full-cortex 40-gene panel and 23-class coefficients')
    if coefficients.columns.tolist() != genes:
        raise ValueError('Native coefficient columns do not match selected genes')
    metrics = pd.read_csv(native / 'summary_metrics.csv')
    if len(metrics) != 1 or int(metrics.iloc[0]['actual_panel_size']) != 40:
        raise ValueError('Native run summary does not confirm a 40-gene panel')

    adata = ad.read_h5ad(DATA, backed='r')
    if adata.shape != (86427, 5083) or any(gene not in adata.var_names for gene in genes):
        raise ValueError('Unexpected cortex matrix or missing selected gene')
    expression = adata[:, genes].to_memory().X
    if sparse.issparse(expression):
        expression = expression.toarray()
    labels = adata.obs['celltype'].astype(str).to_numpy()
    means = pd.DataFrame(expression, columns=genes).groupby(labels).mean().loc[coefficients.index]
    zscores = (means - means.mean(axis=0)) / means.std(axis=0, ddof=0)
    if not np.isfinite(zscores.to_numpy()).all():
        raise ValueError('Non-finite expression z-score')

    # Check the same expression calculation against the previously saved table.
    old = pd.read_csv(
        ROOT / 'figure2_cortex_merfish/results/biology_panel46/heatmaps/'
        'mean_expression_by_celltype.csv', index_col=0
    )
    shared = [gene for gene in genes if gene in old.columns]
    if not shared or not np.allclose(means[shared], old.loc[means.index, shared], atol=1e-10):
        raise ValueError('Cortex expression calculation differs from saved real-data table')

    heatmaps = DEST / 'heatmaps'
    heatmaps.mkdir(parents=True, exist_ok=True)
    (DEST / 'selected_genes.txt').write_text('\n'.join(genes) + '\n')
    means.to_csv(heatmaps / 'mean_expression_by_celltype.csv')
    zscores.to_csv(heatmaps / 'mean_expression_by_celltype_gene_zscore.csv')
    coefficients.to_csv(heatmaps / 'model_coefficients_by_celltype.csv')
    provenance = {
        'selection': 'Native CellSieve package, Figure 1 full-cortex 40-gene run',
        'selected_genes_source': str(genes_path),
        'selected_genes_sha256': sha256(genes_path),
        'coefficients_source': str(coefficients_path),
        'coefficients_sha256': sha256(coefficients_path),
        'expression_source': str(DATA),
        'expression_source_sha256': sha256(DATA),
        'n_cells': adata.n_obs,
        'n_celltypes': len(means),
        'n_genes': len(genes),
        'expression_summary': 'Arithmetic mean of AnnData X per celltype; population z-score across celltypes for each gene',
        'matched_existing_mean_genes': len(shared),
    }
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Wrote Figure 2 A/B data: {len(genes)} native genes, {len(means)} cell types, {adata.n_obs:,} cells')


if __name__ == '__main__':
    main()
