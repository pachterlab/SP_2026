"""Native 30-gene repeat with frozen injected labels; evaluation runs separately."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import os, sys, json, hashlib, subprocess
BASE=PROJECT_ROOT/'results/figure3/cellsieve30_300s'
OLD=BASE.parent/'noise_benchmark/results/cortex_reference'
SOURCE=PROJECT_ROOT / 'h5ads/B6J_CortexHippocampus_subset_POISONED_010.h5ad'
PACKAGE=PROJECT_ROOT / 'external/cell_sieve_package/cellsieve.py'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='4'
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(BASE/'.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR']=str(BASE/'.cache/numba')
os.environ['GRB_LICENSE_FILE']='/home/nikki/gurobi.lic'
import numpy as np, pandas as pd, anndata as ad

def evaluate():
    import scanpy as sc
    from threadpoolctl import threadpool_limits
    out=BASE/'cortex_pca';out.mkdir(exist_ok=True)
    truth=pd.read_csv(OLD/'truth_key.csv').set_index('cell_barcode',verify_integrity=True)
    flags=pd.read_csv(OLD/'panel80_alpha070/flagged_misannotations.csv').set_index('cell_barcode',verify_integrity=True)
    assert (flags.annotated_label==truth.loc[flags.index,'poisoned_label']).all()
    injected=truth.is_poisoned.astype(str).str.lower().eq('true')
    flagged=truth.index.isin(flags.index)
    final=truth.poisoned_label.copy();final.loc[flags.index]=flags.predicted_label
    after=final.eq(truth.true_label);before=truth.poisoned_label.eq(truth.true_label)
    counts=dict(total_cells=len(truth),poisoned=int(injected.sum()),clean=int((~injected).sum()),poison_flagged=int((injected&flagged).sum()),poison_missed=int((injected&~flagged).sum()),clean_flagged=int((~injected&flagged).sum()),clean_unflagged=int((~injected&~flagged).sum()),recovered=int((injected&after).sum()),wrong_fix=int((injected&flagged&~after).sum()),accuracy_before=float(before.mean()),accuracy_after=float(after.mean()))
    pd.DataFrame([counts]).to_csv(out/'targeted_alluvial_counts.csv',index=False)
    truth.to_csv(out/'truth_key.csv')
    print('Recomputing PCA from original raw counts.',flush=True)
    source=ad.read_h5ad(SOURCE,backed='r')
    raw=source.layers['raw_counts']
    if hasattr(raw,'to_memory'):raw=raw.to_memory()
    a=ad.AnnData(raw.copy(),obs=pd.DataFrame(index=source.obs_names),var=source.var.copy())
    source.file.close();del source,raw
    with threadpool_limits(limits=4):
        sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a);sc.pp.scale(a,max_value=10);sc.tl.pca(a,svd_solver='arpack',random_state=0)
    coords=a.obsm['X_pca'];ratio=a.uns['pca']['variance_ratio']
    labels=truth.true_label.to_numpy();reference=(~injected&~flagged).to_numpy()
    types=sorted(truth.true_label.unique())
    centers=np.vstack([np.median(coords[reference&(labels==t)],axis=0) for t in types])
    reference_counts={t:int((reference&(labels==t)).sum()) for t in types}
    assert min(reference_counts.values())>=20
    positions=np.flatnonzero((~injected&flagged).to_numpy());rows=[]
    for i in positions:
        barcode=truth.index[i];original=labels[i];predicted=flags.loc[barcode,'predicted_label']
        d0=float(np.linalg.norm(coords[i]-centers[types.index(original)]));d1=float(np.linalg.norm(coords[i]-centers[types.index(predicted)]))
        rows.append(dict(cell_barcode=barcode,annotated_label=original,predicted_label=predicted,checked=True,annotated_centroid_distance=d0,predicted_centroid_distance=d1,distance_delta_annotated_minus_predicted=d0-d1,predicted_is_closer=d1<d0))
    pd.DataFrame(rows).to_csv(out/'clean_false_positive_centroid_check.csv',index=False)
    np.savez_compressed(out/'pca_visualization.npz',coords=coords,variance_ratio=ratio,cell_ids=truth.index.to_numpy(dtype=str),reference_labels=labels.astype(str),reference_mask=reference,clean_reassigned_positions=positions,centroids=centers,centroid_labels=np.array(types,dtype=str))
    saved=pd.read_csv(OLD/'clean_false_positive_centroid_check.csv').set_index('cell_barcode')
    recomputed=pd.DataFrame(rows).set_index('cell_barcode')
    delta_error=float(np.max(np.abs(saved.loc[recomputed.index,'distance_delta_annotated_minus_predicted']-recomputed.distance_delta_annotated_minus_predicted)))
    print('Maximum difference from saved PCA distance advantages:',delta_error,flush=True)
    assert np.allclose(saved.loc[recomputed.index,'distance_delta_annotated_minus_predicted'],recomputed.distance_delta_annotated_minus_predicted,atol=1e-4)
    (out/'pca_method.json').write_text(json.dumps(dict(source_path=str(SOURCE),normalization=10000,log1p=True,scale_clip=10,svd_solver='arpack',random_state=0,n_pcs=coords.shape[1],use_highly_variable=True,n_highly_variable_genes=int(a.var.highly_variable.sum()),centroid_method='coordinate-wise median',reference_counts=reference_counts,max_difference_from_original_distance_delta=delta_error),indent=2)+'\n')
    print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':evaluate()
