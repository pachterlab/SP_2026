"""Figure 4 exploratory quantities from the saved B6J cortex object.

Calls the unmodified local EMBER implementation for Psi/Zeta/Psi_block. No
minimum-gene equation, depth correction, or replacement entropy formula.
"""
from pathlib import Path
import json
import hashlib
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
H5AD = ROOT / 'h5ads/Figure4_B6J_CortexHippocampus_namefiltered.h5ad'
RESULTS = ROOT / 'figure4_theoretical_minimum/results'
EMBER_SOURCE = ROOT / 'external/ember_package/src'
sys.path.insert(0, str(EMBER_SOURCE))
from ember_py.generate_entropy_metrics import generate_entropy_metrics
import ember_py.generate_entropy_metrics as ember_module


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(H5AD)
    if a.n_obs != 86427 or a.n_vars != 24170:
        raise ValueError('Unexpected B6J, name-filtered matrix')
    if a.obs['Genotype'].astype(str).ne('B6J').any() or a.obs['celltype'].astype(str).eq('low quality').any():
        raise ValueError('Non-B6J or low-quality cells found')
    X = a.X.tocsr() if sparse.issparse(a.X) else sparse.csr_matrix(a.X)
    counts = a.layers['cellbender_counts'].tocsr()
    if X.shape != counts.shape:
        raise ValueError('Expression and counts matrices differ in shape')
    labels = a.obs['celltype'].astype(str).to_numpy()
    classes, codes = np.unique(labels, return_inverse=True)
    n = len(labels)
    group_counts = np.bincount(codes)
    pi = group_counts / n
    hp = float(-(pi * np.log2(pi)).sum())
    depth_all = a.obs['total_umi_all_source_genes'].to_numpy(dtype=float)
    depth_raw = a.obs['total_umi_raw_layer_all_source_genes'].to_numpy(dtype=float)
    depth_retained = a.obs['total_umi_retained_genes'].to_numpy(dtype=float)
    if not np.allclose(depth_retained, np.asarray(counts.sum(axis=1)).ravel()):
        raise ValueError('Saved UMI depth differs from retained matrix')

    groups = pd.DataFrame({
        'celltype': classes,
        'n_cells': group_counts,
        'pi_k': pi,
        'mean_umi_depth_raw_layer_all_source_genes': [depth_raw[codes == k].mean() for k in range(len(classes))],
        'median_umi_depth_raw_layer_all_source_genes': [np.median(depth_raw[codes == k]) for k in range(len(classes))],
        'mean_umi_depth_all_source_genes': [depth_all[codes == k].mean() for k in range(len(classes))],
        'median_umi_depth_all_source_genes': [np.median(depth_all[codes == k]) for k in range(len(classes))],
        'mean_umi_depth_retained_genes': [depth_retained[codes == k].mean() for k in range(len(classes))],
        'median_umi_depth_retained_genes': [np.median(depth_retained[codes == k]) for k in range(len(classes))],
    })
    groups.to_csv(RESULTS / 'celltype_abundance_depth.csv', index=False)
    a.obs[['Genotype', 'celltype', 'Mouse_Tissue_ID', 'Sex', 'source_row',
           'total_umi_raw_layer_all_source_genes', 'total_umi_all_source_genes', 'total_umi_retained_genes']].to_csv(
        RESULTS / 'b6j_cell_depth.csv', index_label='cellID'
    )
    print(f'N={n:,} G={a.n_vars:,} K={len(classes)} H_P={hp:.6f} bits; '
          f'mean/median full UMI depth={depth_all.mean():.1f}/{np.median(depth_all):.1f}', flush=True)

    print('Running local EMBER generate_entropy_metrics on the complete saved B6J matrix...', flush=True)
    psi, psi_block, zeta = generate_entropy_metrics(
        ad.AnnData(X=X, obs=pd.DataFrame({'celltype': pd.Categorical(labels)},
                                         index=a.obs.index.copy()),
                   var=pd.DataFrame(index=a.var_names.copy())),
        'celltype'
    )
    print(f'EMBER completed in {time.time()-started:.0f}s', flush=True)
    ember = pd.DataFrame({'gene_name': a.var_names, 'Psi': psi, 'Zeta': zeta})
    ember.to_csv(RESULTS / 'ember_gene_metrics.csv', index=False)
    psi_block.index.name = 'gene_name'
    psi_block.to_csv(RESULTS / 'ember_psi_block.csv')

    # Empirical detection and abundance use the same saved B6J count matrix.
    detected = counts > 0
    detected_count = np.asarray(detected.sum(axis=0)).ravel()
    gene_umi = np.asarray(counts.sum(axis=0)).ravel()
    expression_mean = np.asarray(X.mean(axis=0)).ravel()
    dgk = pd.DataFrame(index=a.var_names, columns=classes, dtype=float)
    for k, name in enumerate(classes):
        dgk[name] = np.asarray(detected[codes == k].sum(axis=0)).ravel() / group_counts[k]
    dgk.index.name = 'gene_name'
    dgk.to_csv(RESULTS / 'd_gk_by_celltype.csv')
    dg = detected_count / n
    if not np.allclose(dg, dgk.to_numpy() @ pi, atol=1e-12):
        raise ValueError('Marginal detection does not equal abundance-weighted block detection')
    genes = ember.assign(
        d_g=dg,
        detected_cells=detected_count,
        total_umi=gene_umi,
        mean_umi_per_cell=gene_umi/n,
        mean_log_normalized_expression=expression_mean,
    )
    genes['valid_psi'] = np.isfinite(genes['Psi']) & genes['Psi'].between(0, 1)
    genes.to_csv(RESULTS / 'gene_level_metrics.csv', index=False)

    summary = {
        'N': n, 'G': a.n_vars, 'K': len(classes),
        'H_P_bits': hp,
        'mean_umi_depth_raw_layer_all_source_genes': float(depth_raw.mean()),
        'median_umi_depth_raw_layer_all_source_genes': float(np.median(depth_raw)),
        'mean_umi_depth_all_source_genes': float(depth_all.mean()),
        'median_umi_depth_all_source_genes': float(np.median(depth_all)),
        'mean_umi_depth_retained_genes': float(depth_retained.mean()),
        'median_umi_depth_retained_genes': float(np.median(depth_retained)),
        'valid_Psi_gene_count': int(genes.valid_psi.sum()),
        'invalid_or_undefined_Psi_gene_count': int((~genes.valid_psi).sum()),
        'Psi_median_valid': float(genes.loc[genes.valid_psi, 'Psi'].median()),
        'd_g_median': float(genes.d_g.median()),
        'ember_implementation': str(Path(ember_module.__file__).resolve()),
        'python_executable': sys.executable,
        'ember_source_sha256': hashlib.sha256(Path(ember_module.__file__).read_bytes()).hexdigest(),
        'ember_package_version': '0.1.2',
        'ember_call': 'ember_py.generate_entropy_metrics.generate_entropy_metrics(complete B6J AnnData, "celltype")',
        'no_minimum_gene_equation': True,
        'cumulative_analysis': 'Calculated after native 100-draw Psi p-value filtering; no theoretical bound',
    }
    (RESULTS / 'summary_statistics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Analysis finished in {time.time()-started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
