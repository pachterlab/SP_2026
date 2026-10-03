"""Join native 100-draw Psi with its native 1,000-permutation p-values."""
from pathlib import Path
import json

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'figure4_theoretical_minimum/results'
NATIVE = RESULTS / 'native_ember_sex100_p1000'
H5AD = ROOT / 'h5ads/Figure4_B6J_CortexHippocampus_namefiltered.h5ad'


def main():
    full = pd.read_csv(RESULTS / 'gene_level_metrics.csv').rename(columns={
        'Psi': 'Psi_full_B6J', 'Zeta': 'Zeta_full_B6J',
        'valid_psi': 'valid_Psi_full_B6J'
    })
    sampled = pd.read_csv(NATIVE / 'entropy_metrics_celltype.csv').rename(columns={
        'Unnamed: 0': 'gene_name',
        'Psi_mean_celltype': 'Psi',
        'Psi_std_celltype': 'Psi_std_100_draws',
        'Psi_valid_counts_celltype': 'Psi_valid_draws',
        'Zeta_mean_celltype': 'Zeta_100_draws',
    })
    pvalues = pd.read_csv(NATIVE / 'pvals_entropy_metrics_celltype.csv').rename(
        columns={
            'Psi p-value': 'Psi_p_value',
            'Psi q-value': 'Psi_q_value',
            'Zeta p-value': 'Zeta_p_value',
            'Zeta q-value': 'Zeta_q_value',
        }
    )
    if full.gene_name.duplicated().any() or sampled.gene_name.duplicated().any() or pvalues.gene_name.duplicated().any():
        raise ValueError('Duplicate genes in full or native EMBER tables')
    if set(full.gene_name) != set(sampled.gene_name) or not set(pvalues.gene_name).issubset(set(full.gene_name)):
        raise ValueError('Native 100-draw gene set differs from complete B6J gene set')
    native = sampled.set_index('gene_name').loc[pvalues.gene_name, 'Psi'].to_numpy()
    if not np.allclose(native, pvalues.Psi.to_numpy(), atol=1e-12, rtol=0):
        raise ValueError('Native p-values do not correspond to the native 100-draw Psi')
    table = full.merge(sampled, on='gene_name', how='left', validate='one_to_one').merge(
        pvalues[['gene_name', 'Psi_p_value', 'Psi_q_value', 'Zeta_p_value', 'Zeta_q_value']],
        on='gene_name', how='left', validate='one_to_one'
    )
    table['Psi_p_le_0_05'] = table.Psi.notna() & table.Psi_p_value.le(.05)
    table.to_csv(RESULTS / 'gene_level_metrics_with_pvalues.csv', index=False)
    significant = table.loc[table.Psi_p_le_0_05].sort_values(
        ['Psi', 'gene_name'], ascending=[False, True]
    )
    significant.to_csv(RESULTS / 'significant_gene_metrics_psi_order.csv', index=False)
    if significant.empty:
        raise ValueError('No genes pass native 100-draw Psi p<=0.05')

    a = ad.read_h5ad(H5AD)
    if a.n_obs != 86427 or a.n_vars != len(table) or a.obs['Genotype'].astype(str).ne('B6J').any():
        raise ValueError('Cumulative data are not the complete saved B6J matrix')
    max_genes = min(500, len(significant))
    indices = a.var_names.get_indexer(significant.gene_name.iloc[:max_genes])
    if (indices < 0).any():
        raise ValueError('Significant gene absent from B6J object')
    detected = (a.layers['cellbender_counts'][:, indices] > 0).toarray()
    cumulative = np.maximum.accumulate(detected, axis=1)
    classes, codes = np.unique(a.obs['celltype'].astype(str).to_numpy(), return_inverse=True)
    curve = pd.DataFrame({
        'genes_in_psi_order': np.arange(1, max_genes + 1),
        'cumulative_fraction_cells_with_detection': cumulative.mean(axis=0),
        'least_observed_celltype_fraction': np.min([
            cumulative[codes == k].mean(axis=0) for k in range(len(classes))
        ], axis=0),
        'last_added_gene': significant.gene_name.iloc[:max_genes].to_numpy(),
        'last_added_Psi': significant.Psi.iloc[:max_genes].to_numpy(),
        'last_added_Psi_p_value': significant.Psi_p_value.iloc[:max_genes].to_numpy(),
    })
    curve.to_csv(RESULTS / 'descriptive_cumulative_detection_significant_top500.csv', index=False)
    summary_path = RESULTS / 'summary_statistics.json'
    summary = json.loads(summary_path.read_text())
    summary['native_Psi_p_le_0_05_genes'] = len(significant)
    summary['native_Psi_p_value_threshold'] = .05
    summary['native_ember_sampling_draws'] = 100
    summary['native_ember_pvalue_permutations'] = 1000
    summary['native_ember_sample_id_col'] = 'Mouse_Tissue_ID'
    summary['native_ember_category_col'] = 'Sex'
    summary['p_value_filter'] = 'Uncorrected native 100-draw EMBER Psi p-value <= 0.05; q-value saved separately'
    summary['cumulative_analysis'] = (
        'Descriptive detection among first 1..500 native p<=0.05 genes ordered by '
        'native 100-draw Psi; full B6J cells used to evaluate coverage; not a theoretical bound'
    )
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{len(significant):,} native-EMBER 100-draw Psi p<=0.05 genes; '
          f'descriptive curve through {max_genes} genes', flush=True)


if __name__ == '__main__':
    main()
