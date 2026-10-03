"""Extract the exact B6J, non-low-quality, name-filtered cortex AnnData.

Reads the original large H5AD in bounded CSR row chunks. No entropy, HVG,
abundance, or expression-based gene selection is applied.
"""
from pathlib import Path
import json
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'h5ads/upstream/CortexHippocampus_annotated.h5ad'
OUT = ROOT / 'h5ads/Figure4_B6J_CortexHippocampus_namefiltered.h5ad'
EMBER_OUT = ROOT / 'h5ads/Figure4_B6J_EMBER_permutation_input.h5ad'
RESULTS = ROOT / 'figure4_theoretical_minimum/results'
EXCLUDED_PREFIXES = ('Gm', 'mt-', 'Rps', 'Rpl', 'ENSMUSG')


def categorical(f, name):
    group = f['obs'][name]
    categories = np.array([x.decode() for x in group['categories'][:]])
    return categories, group['codes'][:]


def csr_rows(group, first, last, n_genes):
    ptr = group['indptr'][first:last + 1]
    start, end = int(ptr[0]), int(ptr[-1])
    return sparse.csr_matrix(
        (group['data'][start:end], group['indices'][start:end], ptr - start),
        shape=(last - first, n_genes),
    )


def main():
    if not SOURCE.exists() and OUT.is_file() and EMBER_OUT.is_file():
        metadata=json.loads((RESULTS/'subset_provenance.json').read_text())
        for path in (OUT,EMBER_OUT):
            with h5py.File(path,'r') as f:
                assert tuple(f['X'].attrs['shape'])==(metadata['N'],metadata['G']),str(path)
        print('Using preserved B6J analysis inputs; upstream atlas is not needed.',flush=True)
        return
    t0 = time.time()
    with h5py.File(SOURCE, 'r') as f:
        genotype, gcodes = categorical(f, 'Genotype')
        celltypes, ccodes = categorical(f, 'celltype')
        samples, scodes = categorical(f, 'Mouse_Tissue_ID')
        sexes, sexcodes = categorical(f, 'Sex')
        assert 'B6J' in genotype and 'low quality' in celltypes
        b6j = gcodes == np.where(genotype == 'B6J')[0][0]
        keep = b6j & (ccodes != np.where(celltypes == 'low quality')[0][0])
        row_ids = np.flatnonzero(keep)
        genes = np.array([x.decode() for x in f['var']['gene_name'][:]])
        gene_mask = np.array([
            not (gene.startswith(EXCLUDED_PREFIXES) or gene.endswith('Rik'))
            for gene in genes
        ])
        gene_ids = np.flatnonzero(gene_mask)
        if not len(row_ids) or not len(gene_ids) or len(set(genes[gene_mask])) != len(gene_ids):
            raise ValueError('Empty subset or duplicate retained genes')
        print(f'B6J {int(b6j.sum()):,}; after low-quality removal {len(row_ids):,}; '
              f'genes after name rule {len(gene_ids):,}/{len(genes):,}', flush=True)

        # Keep only the original normalized X and its matching observed UMI counts.
        x_parts, count_parts, depth_all_parts, depth_raw_parts = [], [], [], []
        chunk_rows = 2500
        for first in range(0, len(gcodes), chunk_rows):
            last = min(first + chunk_rows, len(gcodes))
            local = np.flatnonzero(keep[first:last])
            if not len(local):
                continue
            x = csr_rows(f['X'], first, last, len(genes))[local, :][:, gene_ids].tocsr()
            all_counts = csr_rows(f['layers']['cellbender_counts'], first, last, len(genes))[local, :]
            depth_all_parts.append(np.asarray(all_counts.sum(axis=1)).ravel())
            raw_counts = csr_rows(f['layers']['raw_counts'], first, last, len(genes))[local, :]
            depth_raw_parts.append(np.asarray(raw_counts.sum(axis=1)).ravel())
            counts = all_counts[:, gene_ids].tocsr()
            x_parts.append(x)
            count_parts.append(counts)
            if (first // chunk_rows) % 25 == 0:
                print(f'Processed source rows {last:,}/{len(gcodes):,}; kept blocks {len(x_parts)}; '
                      f'elapsed {time.time()-t0:.0f}s', flush=True)
        X = sparse.vstack(x_parts, format='csr')
        counts = sparse.vstack(count_parts, format='csr')
        depth_all = np.concatenate(depth_all_parts)
        depth_raw = np.concatenate(depth_raw_parts)
        del x_parts, count_parts, depth_all_parts, depth_raw_parts
        if X.shape != counts.shape or X.shape != (len(row_ids), len(gene_ids)):
            raise ValueError('Extracted matrix shape mismatch')
        if X.nnz != counts.nnz or not np.array_equal(X.indptr, counts.indptr) or not np.array_equal(X.indices, counts.indices):
            raise ValueError('Normalized expression and UMI count sparsity differ')
        cell_ids = np.array([x.decode() for x in f['obs']['cellID'][:]], dtype=object)[row_ids]
        obs = pd.DataFrame({
            'Genotype': pd.Categorical(['B6J'] * len(row_ids)),
            'celltype': pd.Categorical(celltypes[ccodes[row_ids]]),
            'Mouse_Tissue_ID': pd.Categorical(samples[scodes[row_ids]]),
            'Sex': pd.Categorical(sexes[sexcodes[row_ids]]),
            'source_row': row_ids,
            'total_umi_all_source_genes': depth_all,
            'total_umi_raw_layer_all_source_genes': depth_raw,
            'total_umi_retained_genes': np.asarray(counts.sum(axis=1)).ravel(),
        }, index=pd.Index(cell_ids, name='cellID'))
        var = pd.DataFrame({'source_gene_column': gene_ids}, index=pd.Index(genes[gene_mask], name='gene_name'))
        adata = ad.AnnData(X=X, obs=obs, var=var, layers={'cellbender_counts': counts})
        if adata.obs['Genotype'].astype(str).nunique() != 1 or adata.obs['celltype'].astype(str).eq('low quality').any():
            raise ValueError('Unexpected cells in saved B6J object')
        OUT.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(OUT)
        ember_input = ad.AnnData(X=X, obs=obs[['Genotype', 'celltype', 'Mouse_Tissue_ID', 'Sex']].copy(), var=var[[]].copy())
        ember_input.write_h5ad(EMBER_OUT)
        print(f'Wrote lean native EMBER input {EMBER_OUT} ({EMBER_OUT.stat().st_size/1e9:.2f} GB)', flush=True)
        print(f'Wrote {OUT} ({OUT.stat().st_size/1e9:.2f} GB) in {time.time()-t0:.0f}s', flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / 'subset_provenance.json').write_text(json.dumps({
            'source': str(SOURCE),
            'source_size_bytes': SOURCE.stat().st_size,
            'source_mtime_ns': SOURCE.stat().st_mtime_ns,
            'saved_h5ad': str(OUT),
            'native_ember_permutation_h5ad': str(EMBER_OUT),
            'genotype_column': 'Genotype',
            'genotype_value': 'B6J',
            'celltype_column': 'celltype',
            'sex_category_column': 'Sex',
            'sample_id_column': 'Mouse_Tissue_ID',
            'removed_celltype': 'low quality',
            'gene_filter': 'Exclude names starting Gm, mt-, Rps, Rpl, ENSMUSG or ending Rik',
            'expression_matrix': 'original X (log1p normalized)',
            'umi_matrix': 'original layers/cellbender_counts',
            'observed_depth': 'Sum of original cellbender_counts across all 56,953 source genes, before gene-name filtering',
            'raw_observed_depth': 'Sum of original raw_counts layer across all 56,953 source genes',
            'N': adata.n_obs,
            'G': adata.n_vars,
            'K': adata.obs['celltype'].nunique(),
            'no_other_gene_filter': True,
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()
