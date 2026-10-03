"""Entropy bookkeeping for observed binary gene detection; no Psi surrogate."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'figure4_theoretical_minimum/results'


def binary_entropy(p):
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    out[mask] = -p[mask] * np.log2(p[mask]) - (1 - p[mask]) * np.log2(1 - p[mask])
    return out


def gains(dgk, pi):
    marginal = dgk @ pi
    h_detection = binary_entropy(marginal)
    h_given_class = binary_entropy(dgk) @ pi
    gain = h_detection - h_given_class
    if gain.min() < -1e-10:
        raise ValueError('Negative entropy reduction beyond roundoff')
    return marginal, h_detection, h_given_class, np.maximum(gain, 0)


def envelope(dgk, pi, names, psi, pvalues, threshold_bits):
    marginal, h_det, h_cond, gain = gains(dgk, pi)
    table = pd.DataFrame({'gene_name': names, 'Psi_native_100_draw': psi,
                          'Psi_p_value': pvalues, 'd_g': marginal,
                          'H_detection_bits': h_det,
                          'H_detection_given_celltype_bits': h_cond,
                          'single_gene_entropy_reduction_bits': gain})
    table = table.sort_values(['single_gene_entropy_reduction_bits', 'gene_name'],
                              ascending=[False, True]).reset_index(drop=True)
    table['rank'] = np.arange(1, len(table) + 1)
    table['additive_envelope_bits'] = table.single_gene_entropy_reduction_bits.cumsum()
    crossing = int(np.searchsorted(table.additive_envelope_bits.to_numpy(), threshold_bits) + 1)
    return table, crossing if crossing <= len(table) else None


def main():
    groups = pd.read_csv(RESULTS / 'celltype_abundance_depth.csv')
    block = pd.read_csv(RESULTS / 'd_gk_by_celltype.csv')
    metrics = pd.read_csv(RESULTS / 'gene_level_metrics_with_pvalues.csv',
                          usecols=['gene_name', 'Psi', 'Psi_p_value', 'Psi_p_le_0_05'])
    selected = metrics.loc[metrics.Psi_p_le_0_05].copy()
    if len(selected) != 9004 or selected.Psi_p_value.gt(.05).any():
        raise ValueError('Expected native EMBER Psi p<=0.05 candidate set')
    block = block.set_index('gene_name').loc[selected.gene_name, groups.celltype]
    pi = groups.pi_k.to_numpy(dtype=float)
    hp = float(-(pi * np.log2(pi)).sum())
    table, m_envelope = envelope(block.to_numpy(dtype=float), pi,
                                  selected.gene_name.to_numpy(), selected.Psi.to_numpy(),
                                  selected.Psi_p_value.to_numpy(), hp)
    table.to_csv(RESULTS / 'binary_detection_gene_capacity.csv', index=False)
    table.loc[table['rank'] <= 500].to_csv(RESULTS / 'capacity_envelope_top500.csv', index=False)
    uniform_pi = np.full(len(pi), 1 / len(pi))
    uniform_hp = math.log2(len(pi))
    _, uniform_crossing = envelope(block.to_numpy(dtype=float), uniform_pi,
                                    selected.gene_name.to_numpy(), selected.Psi.to_numpy(),
                                    selected.Psi_p_value.to_numpy(), uniform_hp)
    sensitivity = pd.DataFrame([
        {'prior': 'observed', 'H_P_bits': hp, 'additive_envelope_crossing': m_envelope,
         'ideal_fixed_binary_coding_floor': math.ceil(math.log2(len(pi)))},
        {'prior': 'equal_celltype_weight', 'H_P_bits': uniform_hp,
         'additive_envelope_crossing': uniform_crossing,
         'ideal_fixed_binary_coding_floor': math.ceil(math.log2(len(pi)))},
    ])
    sensitivity.to_csv(RESULTS / 'imbalance_sensitivity.csv', index=False)
    result = {
        'N': int(groups.n_cells.sum()), 'G': int(json.loads((RESULTS / 'summary_statistics.json').read_text())['G']),
        'K': len(pi), 'H_P_bits': hp, 'candidate_genes_ember_psi_p_le_0_05': len(selected),
        'additive_envelope_crossing_genes': m_envelope,
        'ideal_fixed_binary_coding_floor_genes': math.ceil(math.log2(len(pi))),
        'claimed_minimum_gene_count': None,
        'reason': 'The 4-gene additive crossing assumes conditional independence and is below the 5-gene fixed binary coding floor. Observed marginals and EMBER Psi do not identify multi-gene capacity or a rigorous attainable minimum.',
    }
    (RESULTS / 'capacity_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
