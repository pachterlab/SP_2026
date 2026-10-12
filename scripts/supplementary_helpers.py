#!/usr/bin/env python3
"""Shared preparation, provenance checks and resume helpers for results/supplementary runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import run_8cube_supplements as shared

ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)
BASE = shared.ROOT / 'results/supplementary/fullinput_benchmarks'


def inside_workspace(path):
    path = Path(path).resolve()
    if not path.is_relative_to(shared.ROOT):
        raise ValueError('Outputs must stay inside /home/nikki/SP_2026')
    path.mkdir(parents=True, exist_ok=True)
    return path


def engine_module():
    sys.path.insert(0, str(shared.NATIVE))
    import benchmark_execution
    return benchmark_execution


def settings(engine, subset, output, alpha, panels, workers, time_limit):
    saved = sys.argv
    sys.argv = ['benchmark_execution.py', '--adata-path', str(subset)]
    try:
        args = engine.parse_args()
    finally:
        sys.argv = saved
    args.out_dir = str(output)
    args.methods = 'cell_sieve'
    args.panel_sizes = ','.join(map(str, panels))
    args.max_genes = 0
    args.no_normalize = True
    args.n_workers = workers
    args.cell_sieve_time_limit = time_limit
    args.cell_sieve_alpha = alpha
    args.cv_splits = 3
    args.random_seed = 42
    args.resume = True
    return args




def get_subset(base, tissue, strain, explicit=None):
    subset = Path(explicit).resolve() if explicit else (base / 'h5ads' / f'{tissue}_{strain}.h5ad').resolve()
    metadata = json.loads(subset.with_suffix('.json').read_text())
    if not metadata.get('all_namefiltered_genes') or metadata.get('max_genes') != 0:
        raise ValueError(f'All name-filtered genes required; variance-filtered subset rejected: {subset}')
    if metadata['strain'] != strain or metadata['benchmark_sha256'] != shared.digest(subset):
        raise ValueError(f'Subset identity failed: {subset}')
    import h5py
    with h5py.File(subset, 'r') as handle:
        from anndata.io import read_elem
        obs = read_elem(handle['obs'])
        if set(obs.Genotype.astype(str)) != {strain}:
            raise ValueError(f'Unexpected strain labels in {subset}')
    return subset, metadata['benchmark_sha256']


def prepare_full_input(base, data_dir, tissue, strain, destination):
    """Reuse a validated prepared cohort, or extract it from the source atlas."""
    import os
    import shutil
    destination = Path(destination)
    existing = Path(base) / 'h5ads' / f'{tissue}_{strain}.h5ad'
    if destination.is_file():
        get_subset(base, tissue, strain, destination)
        return
    if existing.is_file():
        _, checksum = get_subset(base, tissue, strain, existing)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(existing, destination)
        except OSError:
            shutil.copyfile(existing, destination)
        metadata = json.loads(existing.with_suffix('.json').read_text())
        metadata.update(benchmark_h5ad=str(destination), all_namefiltered_genes_h5ad=str(destination),
                        benchmark_sha256=checksum)
        shared.atomic_json(destination.with_suffix('.json'), metadata)
    else:
        shared.prepare(Path(data_dir) / f'{tissue}_annotated.h5ad', strain, destination, 0, 10)




def locked_config(output, value):
    p = output / 'configuration.json'
    if p.exists():
        previous = json.loads(p.read_text())
        # Execution script hashes remain historical; scientific settings must match.
        migration = json.loads((shared.ROOT / 'metadata/layout_migration.json').read_text())
        equivalents = {v['after']:v['before'] for v in migration['script_hashes'].values() if 'after' in v}
        def normalized(config):
            return {k:equivalents.get(v,v) if isinstance(v,str) else v for k,v in config.items() if k != 'script_sha256'}
        if normalized(previous) != normalized(value):
            raise ValueError(f'Configuration changed; use a fresh output directory: {output}')
        return
    shared.atomic_json(p, value)



