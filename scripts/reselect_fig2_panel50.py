"""Select the native 50-gene panel at the active Figure 1 alpha in the cortex/MERFISH overlap.

Reconstruct the unavailable historical 556-gene overlap object from the preserved
real inputs, retaining cortex gene order, expression and all 86,427 cells.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import csv
import os
import shutil
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
from workspace_paths import ROOT, CELLSIEVE
from build_fig2_panel40 import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha', type=float, help='Explicit selection alpha; default is the active Figure 1 alpha')
    args = parser.parse_args()
    pointer = json.loads((ROOT / 'results/figure1/results/latest_cellsieve_batch.json').read_text())
    if pointer.get('status') != 'complete':
        raise ValueError('Complete Figure 1 CellSieve first')
    alpha = float(pointer['alpha']) if args.alpha is None else args.alpha
    if not 0 <= alpha <= 1:
        parser.error('--alpha must be between 0 and 1')
    base = ROOT / 'results/figure2/results'
    saved_provenance = base / 'panel50/provenance.json'
    if saved_provenance.is_file():
        saved = json.loads(saved_provenance.read_text())
        genes_file = base / 'panel50/selected_genes.txt'
        metrics = pd.read_csv(base / 'panel50/summary_metrics.csv')
        if (saved.get('cell_sieve_alpha') == alpha
                and saved.get('cellsieve_sha256') == sha256(CELLSIEVE)
                and saved.get('selected_genes_sha256') == sha256(genes_file)
                and len(genes_file.read_text().splitlines()) == 50
                and np.isclose(metrics.iloc[0]['alpha_complexity'], alpha)):
            print(f'Reusing validated native alpha-{alpha:g} overlap panel', flush=True)
            return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = base / (f'panel50_alpha{alpha:.2f}_' + stamp)
    run.mkdir()
    cortex_path = ROOT / 'h5ads/B6J_CortexHippocampus_subset.h5ad'
    merfish_path = ROOT / 'h5ads/Merfish_cortex.h5ad'
    cortex = ad.read_h5ad(cortex_path, backed='r')
    merfish = ad.read_h5ad(merfish_path, backed='r')
    merfish_genes = set(merfish.var_names)
    genes = [g for g in cortex.var_names if g in merfish_genes]
    if cortex.shape != (86427, 5083) or len(genes) != 556 or len(set(genes)) != 556:
        raise ValueError('Unexpected real cortex/MERFISH gene universe')
    original_panel = (base / 'panel50/selected_genes.txt').read_text().splitlines()
    if not set(original_panel).issubset(genes):
        raise ValueError('Historical selected genes missing from reconstructed overlap')
    overlap = cortex[:, genes].to_memory()
    cortex.file.close()
    merfish.file.close()
    source = run / 'B6J_CortexHippocampus_subset_MERFISH_overlap_genes.h5ad'
    overlap.write_h5ad(source)
    del overlap
    command = [sys.executable, '-B', str(CELLSIEVE), '--adata-path', str(source),
        '--out-dir', str(run / 'native'), '--alpha', str(alpha), '--panel-size', '50',
        '--n-workers', '4', '--random-seed', '42', '--time-limit', '600', '--mip-gap', '0.10']
    provenance = {'cell_sieve_alpha': alpha, 'command': command,
        'cellsieve_source': str(CELLSIEVE), 'cellsieve_sha256': sha256(CELLSIEVE),
        'cortex_source': str(cortex_path), 'cortex_sha256': sha256(cortex_path),
        'merfish_source': str(merfish_path), 'merfish_sha256': sha256(merfish_path),
        'overlap_shape': [86427, 556], 'overlap_genes': genes,
        'expression': 'Unmodified cortex AnnData X; no additional normalization',
        'reconstruction': 'Historical overlap h5ad unavailable. Intersection of preserved cortex and MERFISH gene names, in cortex order; shape matches recorded historical metadata; historical 50 genes verified present.'}
    (run / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    with (run / 'native.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    output = run / f'native/panel_size_50_alpha_{alpha:.2f}'
    selected = (output / 'selected_genes.txt').read_text().splitlines()
    metrics = pd.read_csv(output / 'summary_metrics.csv')
    coefficients = pd.read_csv(output / 'model_coefficients.csv', index_col=0)
    if (len(selected) != 50 or len(set(selected)) != 50 or not set(selected).issubset(genes)
            or coefficients.shape != (23, 50) or coefficients.columns.tolist() != selected
            or not np.isclose(metrics.iloc[0]['alpha_complexity'], alpha)):
        raise ValueError('Native overlap panel failed alpha/shape validation')
    target = base / 'panel50'
    backup = run / 'previous_panel50'
    shutil.copytree(target, backup)
    # Keep checksum-preserved historical copies addressable after replacement.
    manifest_path = ROOT / 'archive_notes/copy_manifest.csv'
    with manifest_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        copied = list(reader)
    relocated = []
    for row in copied:
        if row['destination_path'].startswith('results/figure2/results/panel50/'):
            previous = row['destination_path']
            preserved = backup / Path(previous).name
            if sha256(preserved) != row['sha256']:
                raise ValueError('Historical copied-file checksum mismatch: ' + previous)
            row['destination_path'] = str(preserved.relative_to(ROOT))
            relocated.append({'historical_destination': previous,
                              'preserved_destination': row['destination_path'], 'sha256': row['sha256']})
    if relocated:
        with manifest_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(copied)
        (run / 'historical_copy_relocations.json').write_text(json.dumps(relocated, indent=2) + '\n')
    for file in output.iterdir():
        if file.is_file():
            shutil.copy2(file, target / file.name)
    provenance.update(selected_genes_source=str(output / 'selected_genes.txt'),
                      selected_genes_sha256=sha256(output / 'selected_genes.txt'))
    (target / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Validated new 50-gene cortex/MERFISH overlap panel at alpha {alpha:g}', flush=True)


if __name__ == '__main__':
    main()
