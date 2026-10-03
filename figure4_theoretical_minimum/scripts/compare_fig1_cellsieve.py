"""Collect actual Figure 1 CellSieve panels for the Figure 4 comparison."""
from pathlib import Path
import hashlib
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'figure4_theoretical_minimum/results'
SOURCE = ROOT / 'figure1_method_benchmark/results/full_cortex_package_l1/benchmark_summary_all.csv'


def main():
    metrics = pd.read_csv(RESULTS / 'gene_level_metrics_with_pvalues.csv',
                          usecols=['gene_name', 'Psi_p_le_0_05'])
    retained = set(metrics.gene_name)
    significant = set(metrics.loc[metrics.Psi_p_le_0_05, 'gene_name'])
    summary = json.loads((RESULTS / 'summary_statistics.json').read_text())
    benchmark = pd.read_csv(SOURCE)
    benchmark = benchmark.loc[(benchmark.method == 'cell_sieve') &
                              (benchmark.status == 'ok') &
                              (benchmark.dataset == 'B6J_CortexHippocampus_subset')]
    if set(benchmark.actual_panel_size) != {10, 20, 40, 80, 160}:
        raise ValueError('Expected five actual full-cortex CellSieve panels')
    rows = []
    panel_genes = []
    for row in benchmark.itertuples(index=False):
        path = Path(row.output_dir) / 'selected_genes.txt'
        genes = [gene.strip() for gene in path.read_text().splitlines() if gene.strip()]
        size = int(row.actual_panel_size)
        if len(genes) != size or len(set(genes)) != size:
            raise ValueError(f'Unexpected selected gene count or duplicates in {path}')
        for gene in genes:
            panel_genes.append({'panel_size': size, 'gene_name': gene,
                                'retained_in_figure4': gene in retained,
                                'ember_psi_p_le_0_05': gene in significant})
        rows.append({'panel_size': size, 'actual_selected_genes': len(genes),
                     'macro_f1': float(row.macro_f1),
                     'cv_macro_f1': float(row.cv_macro_f1),
                     'selected_genes_in_figure4_universe': len(set(genes) & retained),
                     'selected_genes_ember_psi_p_le_0_05': len(set(genes) & significant),
                     'selected_genes_file': str(path),
                     'selected_genes_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'figure1_dataset': row.dataset,
                     'figure1_result_origin': row.result_origin})
    out = pd.DataFrame(rows).sort_values('panel_size')
    out.to_csv(RESULTS / 'figure1_cellsieve_panel_comparison.csv', index=False)
    threshold = out.loc[out.macro_f1 >= .9, 'panel_size']
    limit = {'cell_types': int(summary['K']),
             'partition_entropy_bits': float(summary['H_P_bits']),
             'readout_assumption': 'one binary detected/not-detected value per gene',
             'task_assumption': 'error-free fixed-panel identification of every annotated cell type',
             'calculation': 'ceil(log2(K))',
             'ideal_binary_coding_floor_genes': math.ceil(math.log2(int(summary['K']))),
             'first_tested_cellsieve_panel_macro_f1_ge_0_9': int(threshold.min()) if len(threshold) else None,
             'comparison_caveat': 'Necessary ideal binary coding floor only, not an attainable minimum, a bound on CellSieve continuous-expression panels, or a prediction of macro-F1. The first tested panel above 0.9 is an empirical threshold among five tested sizes, not an optimized minimum.'}
    (RESULTS / 'binary_coding_limit_vs_cellsieve.json').write_text(json.dumps(limit, indent=2) + '\n')
    pd.DataFrame(panel_genes).to_csv(RESULTS / 'figure1_cellsieve_selected_genes_join.csv', index=False)
    provenance = {
        'figure1_summary': str(SOURCE),
        'figure1_summary_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'figure4_N': summary['N'], 'figure4_K': summary['K'],
        'figure4_gene_universe_G': summary['G'],
        'note': 'Figure 1 used a 5,083-gene source object; Figure 4 used 24,170 retained genes from the all-gene source. Cell count and cell-type annotation are compared, but these are not identical feature universes.',
    }
    (RESULTS / 'figure1_cellsieve_comparison_provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n')
    print(out[['panel_size', 'macro_f1', 'selected_genes_ember_psi_p_le_0_05']].to_string(index=False))


if __name__ == '__main__':
    main()
