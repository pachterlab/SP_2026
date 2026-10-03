"""Run native local EMBER: 100 Sex-balanced samples and 1,000 permutations."""
from pathlib import Path
import hashlib
import json
import sys

import h5py
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
H5AD = ROOT / 'h5ads/Figure4_B6J_EMBER_permutation_input.h5ad'
RESULTS = ROOT / 'figure4_theoretical_minimum/results'
NATIVE = RESULTS / 'native_ember_sex100_p1000'
LOCAL_SOURCE = ROOT / 'external/ember_package/src'
sys.path.insert(0, str(LOCAL_SOURCE))
from ember_py.light_ember import light_ember

NUM_DRAWS = 100
N_ITERATIONS = 1000
N_CPUS = 8
SEED = 42


def main():
    with h5py.File(H5AD, 'r') as f:
        if tuple(f['X'].attrs['shape']) != (86427, 24170):
            raise ValueError('Wrong B6J EMBER input shape')
        needed = {'Genotype', 'celltype', 'Sex', 'Mouse_Tissue_ID'}
        if not needed.issubset(f['obs'].keys()):
            raise ValueError('Missing actual B6J sampling metadata')
        genotype = f['obs']['Genotype']
        cats = [value.decode() for value in genotype['categories'][:]]
        codes = genotype['codes'][:]
        if {cats[i] for i in set(codes)} != {'B6J'}:
            raise ValueError('Non-B6J cells in EMBER input')
        sex = f['obs']['Sex']
        sexcats = [value.decode() for value in sex['categories'][:]]
        if {sexcats[i] for i in set(sex['codes'][:])} != {'Female', 'Male'}:
            raise ValueError('Unexpected Sex categories')
    NATIVE.mkdir(parents=True, exist_ok=True)
    print('Native local EMBER: 100 Sex-balanced samples; 1,000 p-value permutations; '
          'sample_id_col=Mouse_Tissue_ID; category_col=Sex; partition=celltype.', flush=True)
    light_ember(
        h5ad_dir=str(H5AD),
        partition_label='celltype',
        save_dir=str(NATIVE),
        sampling=True,
        sample_id_col='Mouse_Tissue_ID',
        category_col='Sex',
        condition_col=None,
        num_draws=NUM_DRAWS,
        save_draws=False,
        seed=SEED,
        partition_pvals=True,
        block_pvals=False,
        n_pval_iterations=N_ITERATIONS,
        n_cpus=N_CPUS,
    )
    entropy = pd.read_csv(NATIVE / 'entropy_metrics_celltype.csv', index_col=0)
    pvals = pd.read_csv(NATIVE / 'pvals_entropy_metrics_celltype.csv', index_col=0)
    if len(entropy) != 24170 or not pvals.index.is_unique:
        raise ValueError('Incomplete native EMBER result')
    if not set(pvals.index).issubset(set(entropy.index)):
        raise ValueError('Native p-value genes absent from 100-sample result')
    selected = pvals.loc[pvals['Psi p-value'].le(.05)]
    selected.to_csv(RESULTS / 'psi_p_le_0_05_genes.csv')
    source_files = [
        LOCAL_SOURCE / 'ember_py/light_ember.py',
        LOCAL_SOURCE / 'ember_py/generate_entropy_metrics.py',
        LOCAL_SOURCE / 'ember_py/generate_pvals.py',
        LOCAL_SOURCE / 'ember_py/sample_replicates.py',
    ]
    metadata = {
        'input': str(H5AD),
        'python_executable': sys.executable,
        'input_cells': 86427,
        'input_genes': 24170,
        'input_genotype': 'B6J only',
        'partition_label': 'celltype',
        'sample_id_col': 'Mouse_Tissue_ID',
        'category_col': 'Sex',
        'condition_col': None,
        'num_draws': NUM_DRAWS,
        'n_pval_iterations': N_ITERATIONS,
        'n_cpus': N_CPUS,
        'seed_for_balanced_draws': SEED,
        'filter': 'native Psi p-value <= 0.05 (uncorrected, per user request)',
        'n_native_tested_genes': len(pvals),
        'n_selected_genes': len(selected),
        'p_value_resolution': 1/(N_ITERATIONS + 1),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'note': 'Native 100-draw Psi and matching native 1000-permutation p-values are used for significance; full-cohort Psi is saved separately.',
    }
    (RESULTS / 'pvalue_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Native tested genes {len(pvals):,}; Psi p <= 0.05: {len(selected):,}', flush=True)


if __name__ == '__main__':
    main()
