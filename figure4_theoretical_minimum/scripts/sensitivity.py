"""Sensitivity of the conditional additive envelope to observed cells and UMI thinning."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'figure4_theoretical_minimum/results'
H5AD = ROOT / 'h5ads/Figure4_B6J_CortexHippocampus_namefiltered.h5ad'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from derive_capacity import gains


def group_detection(matrix, codes, n_groups):
    n = matrix.shape[0]
    indicator = sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(n_groups, n))
    counts = np.bincount(codes, minlength=n_groups)
    return (indicator @ matrix).toarray() / counts[:, None], counts


def crossing(dgk, counts):
    pi = counts / counts.sum()
    hp = float(-(pi * np.log2(pi)).sum())
    _, _, _, gain = gains(dgk.T, pi)
    ordered = np.sort(gain)[::-1]
    pos = int(np.searchsorted(np.cumsum(ordered), hp) + 1)
    return hp, pos if pos <= len(ordered) else None, float(ordered[0])


def main():
    selected = pd.read_csv(RESULTS / 'binary_detection_gene_capacity.csv', usecols=['gene_name'])
    summary = json.loads((RESULTS / 'summary_statistics.json').read_text())
    a = ad.read_h5ad(H5AD)
    if a.n_obs != 86427 or a.obs.Genotype.astype(str).ne('B6J').any():
        raise ValueError('Expected saved B6J working object')
    indices = a.var_names.get_indexer(selected.gene_name)
    if (indices < 0).any():
        raise ValueError('Significant gene missing from saved B6J object')
    counts = a.layers['cellbender_counts'][:, indices].tocsr()
    labels = a.obs.celltype.astype(str).to_numpy()
    classes, codes = np.unique(labels, return_inverse=True)
    if len(classes) != 23:
        raise ValueError('Unexpected cell-type count')
    sample_depth = summary['mean_umi_depth_all_source_genes']
    depth_rows = []
    for q in (1.0, .75, .5, .25, .1, .05):
        probabilities = counts.copy()
        if q == 1:
            probabilities.data = np.ones_like(probabilities.data, dtype=float)
        else:
            probabilities.data = -np.expm1(np.log1p(-q) * probabilities.data.astype(float))
        dgk, group_sizes = group_detection(probabilities, codes, len(classes))
        hp, m, highest = crossing(dgk, group_sizes)
        depth_rows.append({'retained_molecule_fraction_q': q,
                           'expected_full_gene_umi_depth_per_cell': q * sample_depth,
                           'H_P_bits': hp, 'additive_envelope_crossing': m,
                           'highest_single_gene_entropy_reduction_bits': highest,
                           'candidate_genes': len(selected)})
        print('depth', q, m, flush=True)
        del probabilities
    pd.DataFrame(depth_rows).to_csv(RESULTS / 'depth_thinning_sensitivity.csv', index=False)

    detected = counts.copy()
    detected.data = np.ones_like(detected.data, dtype=float)
    baseline_dgk, baseline_n = group_detection(detected, codes, len(classes))
    saved = pd.read_csv(RESULTS / 'd_gk_by_celltype.csv', index_col=0)
    if not np.allclose(baseline_dgk, saved.loc[selected.gene_name, classes].to_numpy().T, atol=1e-12):
        raise ValueError('Saved observability does not reproduce from working count object')
    rng = np.random.default_rng(20260930)
    by_class = [np.flatnonzero(codes == k) for k in range(len(classes))]
    cell_rows = []
    for fraction in (.1, .25, .5, 1.0):
        repeats = 1 if fraction == 1 else 8
        for replicate in range(repeats):
            if fraction == 1:
                row_ids = np.arange(len(codes))
            else:
                row_ids = np.sort(np.concatenate([
                    rng.choice(ids, size=max(1, round(fraction * len(ids))), replace=False)
                    for ids in by_class
                ]))
            dgk, group_sizes = group_detection(detected[row_ids], codes[row_ids], len(classes))
            hp, m, highest = crossing(dgk, group_sizes)
            cell_rows.append({'cell_fraction': fraction, 'replicate': replicate,
                              'N': len(row_ids), 'minimum_celltype_n': int(group_sizes.min()),
                              'H_P_bits': hp, 'additive_envelope_crossing': m,
                              'highest_single_gene_entropy_reduction_bits': highest,
                              'sampling': 'stratified without replacement from B6J cells'})
        print('cells', fraction, 'repeats', repeats, flush=True)
    pd.DataFrame(cell_rows).to_csv(RESULTS / 'cell_number_sensitivity.csv', index=False)


if __name__ == '__main__':
    main()
