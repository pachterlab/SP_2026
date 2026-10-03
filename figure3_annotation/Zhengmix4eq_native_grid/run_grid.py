#!/usr/bin/env python3
"""Fixed Scanpy clustering grid; generate every partition before reference evaluation.

Run with benchmark_env Python. No CellSieve, parameter selection or model fitting
against reference labels. Both phases are independently reproducible.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['MPLBACKEND'] = 'Agg'
BASE = Path(__file__).resolve().parent
DATA = BASE.parent / 'Zhengmix4eq_single_pass_grid_60genes/inputs'
os.environ['MPLCONFIGDIR'] = str(BASE/'.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR'] = str(BASE/'.cache/numba')
CONFIG = dict(seed=42, n_pcs=[20,30,50], n_neighbors=[10,20,30],
              resolutions=[0.2,0.4,0.6,0.8,1.0], n_hvg=2000,
              min_gene_cells=3, normalize_target=10000, scale_clip=10,
              leiden_flavor='leidenalg', leiden_iterations=-1,
              neighbor_transformer='sklearn', parameter_selection=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(name, obj):
    (BASE/name).write_text(json.dumps(obj,indent=2)+'\n')


def fit():
    """Expression-only phase: deliberately never opens the reference-label file."""
    import numpy as np
    import pandas as pd
    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    from scipy.io import mmread
    from threadpoolctl import threadpool_limits
    start = time.monotonic()
    write_json('frozen_grid.json',CONFIG)
    versions={name:importlib.metadata.version(name) for name in
              ['scanpy','anndata','numpy','scipy','pandas','scikit-learn','igraph','leidenalg']}
    write_json('software_versions.json',versions)
    ids=(DATA/'cell_ids.txt').read_text().splitlines()
    genes=pd.read_csv(DATA/'genes.csv').gene.astype(str).to_numpy()
    a=ad.AnnData(sparse.csr_matrix(mmread(DATA/'counts.mtx').T,dtype=np.float32),
                obs=pd.DataFrame(index=pd.Index(ids,name='cell_id')),
                var=pd.DataFrame(index=pd.Index(genes,name='gene')))
    a.var_names_make_unique()
    assert a.n_obs==len(set(ids)) and (np.asarray(a.X.sum(axis=1)).ravel()>0).all()
    with threadpool_limits(limits=4):
        sc.pp.filter_genes(a,min_cells=CONFIG['min_gene_cells'])
        sc.pp.normalize_total(a,target_sum=CONFIG['normalize_target'])
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a,n_top_genes=CONFIG['n_hvg'],flavor='seurat')
        h=a[:,a.var.highly_variable].copy()
        sc.pp.scale(h,max_value=CONFIG['scale_clip'])
        sc.tl.pca(h,n_comps=max(CONFIG['n_pcs']),svd_solver='arpack',random_state=CONFIG['seed'])
        # Only the expression embedding is needed for the graph/partition grid.
        graph_data=ad.AnnData(obs=pd.DataFrame(index=a.obs_names.copy()))
        graph_data.obsm['X_pca']=h.obsm['X_pca'].copy()
        partitions=[]; settings=[]
        for pcs,neighbors in itertools.product(CONFIG['n_pcs'],CONFIG['n_neighbors']):
            sc.pp.neighbors(graph_data,n_neighbors=neighbors,n_pcs=pcs,use_rep='X_pca',
                            random_state=CONFIG['seed'],transformer=CONFIG['neighbor_transformer'])
            for resolution in CONFIG['resolutions']:
                sc.tl.leiden(graph_data,resolution=resolution,random_state=CONFIG['seed'],
                             flavor=CONFIG['leiden_flavor'],n_iterations=CONFIG['leiden_iterations'],
                             key_added='partition')
                partitions.append(graph_data.obs.partition.astype(int).to_numpy())
                settings.append(dict(n_pcs=pcs,n_neighbors=neighbors,resolution=resolution))
            print('Frozen partitions for %d PCs, %d neighbors (%d / 45)'%
                  (pcs,neighbors,len(partitions)),flush=True)
    assert len(settings)==45
    np.savez_compressed(BASE/'frozen_partitions.npz',partitions=np.stack(partitions),
                        cell_ids=np.asarray(ids),settings_json=np.asarray(json.dumps(settings)))
    write_json('fit_complete.json',dict(
        n_cells=a.n_obs,n_genes_after_filter=a.n_vars,n_hvg=h.n_vars,
        n_partitions=len(settings),reference_labels_opened=False,
        elapsed_seconds=time.monotonic()-start,
        partitions_sha256=sha256(BASE/'frozen_partitions.npz'),
        input_hashes={name:sha256(DATA/name) for name in ['counts.mtx','genes.csv','cell_ids.txt']},
    ))
    print('All 45 expression-only partitions frozen.',flush=True)


def evaluate():
    """Read truth only after the complete fixed grid is frozen. Select no setting."""
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(BASE.parents[1]/'scripts'))
    from figure_style import set_style
    set_style()
    before=sha256(BASE/'frozen_partitions.npz')
    complete=json.loads((BASE/'fit_complete.json').read_text())
    assert complete['n_partitions']==45 and before==complete['partitions_sha256']
    assert json.loads((BASE/'frozen_grid.json').read_text())==CONFIG
    frozen=np.load(BASE/'frozen_partitions.npz',allow_pickle=False)
    ids=frozen['cell_ids'];partitions=frozen['partitions']
    settings=json.loads(str(frozen['settings_json']))
    reference=pd.read_csv(DATA/'hidden_ground_truth.csv',index_col='cell_id')
    assert reference.index.is_unique and set(reference.index)==set(ids)
    y=reference.loc[ids,'ground_truth']
    assert not y.isna().any()
    y=y.astype(str).to_numpy()
    rows=[]
    for setting,labels in zip(settings,partitions):
        table=pd.crosstab(pd.Series(y),pd.Series(labels))
        row,col=linear_sum_assignment(-table.to_numpy())
        n_correct=int(table.to_numpy()[row,col].sum())
        rows.append(dict(setting,n_cells=len(y),n_clusters=len(np.unique(labels)),
            ari=float(adjusted_rand_score(y,labels)),nmi=float(normalized_mutual_info_score(y,labels)),
            accuracy=n_correct/len(y),matched_cells=n_correct,
            many_to_one_accuracy=float(table.max(axis=0).sum()/len(y))))
    result=pd.DataFrame(rows)
    result.to_csv(BASE/'all_settings_results.csv',index=False)
    assert len(result)==45 and not result.duplicated(['n_pcs','n_neighbors','resolution']).any()
    assert (result.accuracy<=result.many_to_one_accuracy+1e-12).all()
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    pairs=list(itertools.product(CONFIG['n_pcs'],CONFIG['n_neighbors']))
    for ax,(column,title) in zip(axes.flat,[('ari','ARI'),('nmi','NMI'),('accuracy','One-to-one matched accuracy'),('n_clusters','Number of clusters')]):
        table=result.pivot(index=['n_pcs','n_neighbors'],columns='resolution',values=column).reindex(pairs)
        values=table.to_numpy()
        if column=='n_clusters':im=ax.imshow(values,aspect='auto',cmap='viridis')
        else:im=ax.imshow(values,aspect='auto',cmap='Blues',vmin=0,vmax=1)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value=values[i,j]
                text=str(int(value)) if column=='n_clusters' else ('%.1f%%'%(100*value) if column=='accuracy' else '%.3f'%value)
                rgba=im.cmap(im.norm(value)); brightness=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]
                ax.text(j,i,text,ha='center',va='center',fontsize=8,color='black' if brightness>.55 else 'white')
        ax.set(title=title,xlabel='Leiden resolution',ylabel='PCs / neighbors',
               xticks=range(5),xticklabels=CONFIG['resolutions'],
               yticks=range(9),yticklabels=['%d / %d'%pair for pair in pairs])
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Zhengmix4eq: native Scanpy expression-only grid\n45 predeclared settings; FACS labels used only after all partitions were frozen',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(BASE/('grid_summary.'+ext),dpi=250,bbox_inches='tight')
    plt.close(fig)
    summary=dict(n_settings=45,n_cells=len(y),fit_seconds=complete['elapsed_seconds'],
        accuracy_min=float(result.accuracy.min()),accuracy_max=float(result.accuracy.max()),
        ari_min=float(result.ari.min()),ari_max=float(result.ari.max()),
        nmi_min=float(result.nmi.min()),nmi_max=float(result.nmi.max()),
        n_clusters_min=int(result.n_clusters.min()),n_clusters_max=int(result.n_clusters.max()),
        settings_at_least_90_percent_matched_accuracy=int((result.accuracy>=.9).sum()),
        parameter_selection_performed=False)
    write_json('summary.json',summary)
    lines=['# Zhengmix4eq native Scanpy grid','',
        'All 45 settings were generated before opening the hidden FACS labels. No setting is selected or promoted for a downstream model.', '',
        'Preprocessing: all cells retained, genes detected in ≥3 cells, normalization to 10,000 counts, log1p, 2,000 Seurat-flavor HVGs, scaling clipped at 10, PCA (ARPACK). PCA is computed once; the first 20/30/50 PCs feed exact sklearn kNN graphs. Native Scanpy Leiden uses leidenalg until convergence, fixed seed 42. Nine graphs are reused across five resolutions. No CellSieve is run.','',
        'Accuracy uses an evaluation-only maximum-count, one-to-one Hungarian mapping separately for each frozen partition. Unmatched clusters count as errors. The CSV also reports many-to-one accuracy as a secondary purity measure, which can be inflated by overclustering.','',
        'Fit time: %.1f seconds. Accuracy range: %.2f–%.2f%%; ARI range: %.4f–%.4f. %d / 45 settings reached ≥90%% one-to-one agreement. These are descriptive evaluation results, not a parameter-selection rule.'%(complete['elapsed_seconds'],100*result.accuracy.min(),100*result.accuracy.max(),result.ari.min(),result.ari.max(),summary['settings_at_least_90_percent_matched_accuracy']), '',
        '[All settings CSV](all_settings_results.csv) · [Summary plot](grid_summary.png) · [PDF](grid_summary.pdf)','',
        '| PCs | Neighbors | Resolution | Clusters | ARI | NMI | Matched accuracy |',
        '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| {n_pcs} | {n_neighbors} | {resolution:.1f} | {n_clusters} | {ari:.4f} | {nmi:.4f} | {accuracy:.2%} |'.format(**r))
    lines.extend(['','## Reproduce','','```sh',
        'LD_LIBRARY_PATH=/home/nikki/miniconda3/envs/benchmark_env/lib /home/nikki/miniconda3/envs/benchmark_env/bin/python '+str(BASE/'run_grid.py'),
        '```','','Use `--evaluate-only` to regenerate the CSV, table and plots from frozen partitions without fitting. Source hashes and software versions are recorded.'])
    (BASE/'README.md').write_text('\n'.join(lines)+'\n')
    assert before==sha256(BASE/'frozen_partitions.npz')
    write_json('validation.json',dict(all_45_settings_present=True,partitions_unchanged_by_evaluation=True,
        all_cells_aligned_to_truth=True,script_sha256=sha256(__file__),
        hidden_truth_sha256=sha256(DATA/'hidden_ground_truth.csv')))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['fit','evaluate'])
    parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args()
    if args.stage=='fit':fit();return
    if args.stage=='evaluate':evaluate();return
    stages=['evaluate'] if args.evaluate_only else ['fit','evaluate']
    for stage in stages:
        with (BASE/(stage+'.log')).open('w') as stream:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--stage',stage],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('Saved results: '+str(BASE/'all_settings_results.csv'))


if __name__=='__main__':
    main()
