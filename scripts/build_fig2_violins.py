"""Save measured per-cell expression for two genes selected in Figure 2 B."""
import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from build_fig2_panel40 import DATA, DEST, sha256


def main():
    genes = ['Npas3', 'Flt1']
    selected = (DEST / 'selected_genes.txt').read_text().splitlines()
    if not set(genes).issubset(selected):
        raise ValueError('Violin genes must be selected in panel B')
    adata = ad.read_h5ad(DATA, backed='r')
    expression = adata[:, genes].to_memory().X
    if sparse.issparse(expression):
        expression = expression.toarray()
    frame = pd.DataFrame(expression, columns=genes, index=adata.obs_names)
    frame.index.name = 'cell_id'
    frame.insert(0, 'celltype', adata.obs['celltype'].astype(str).to_numpy())
    adata.file.close()
    if not np.isfinite(expression).all():
        raise ValueError('Non-finite expression')
    means = pd.read_csv(DEST / 'heatmaps/mean_expression_by_celltype.csv', index_col=0)
    actual = frame.groupby('celltype')[genes].mean().loc[means.index]
    if not np.allclose(actual, means[genes], atol=1e-10):
        raise ValueError('Per-cell expression must reproduce panel A source means')
    dest = DEST / 'violins'
    dest.mkdir(exist_ok=True)
    frame.to_csv(dest / 'expression_by_cell.csv.gz')
    provenance = {
        'expression_source': str(DATA), 'expression_source_sha256': sha256(DATA),
        'genes': genes, 'n_cells': len(frame), 'n_celltypes': frame.celltype.nunique(),
        'values': 'Original AnnData X, no additional transformation or subsampling',
        'validation': 'Cell-type means reproduce saved Figure 2 A means (atol=1e-10)',
    }
    (dest / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Saved measured expression for {len(frame):,} cells: {genes}')


if __name__ == '__main__':
    main()
