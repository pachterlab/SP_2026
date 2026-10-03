#!/usr/bin/env python3
"""Frozen native Zhengmix4eq partition -> three native CellSieve passes (reuse the frozen first pass).

Run without arguments with benchmark_env Python. Each fitting round executes in
its own process with no reference labels in its input. Evaluation is a separate
process and its outputs are never consumed by fitting.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'inputs'
FROZEN = BASE.parent / 'Zhengmix4eq_native_grid'
PACKAGE = BASE.parents[1] / 'external/cell_sieve_package/cellsieve.py'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(BASE / '.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR'] = str(BASE / '.cache/numba')
os.environ.setdefault('GRB_LICENSE_FILE', '/home/nikki/gurobi.lic')
CONFIG = dict(rounds=3, panel_size=60, time_limit_seconds=300, alpha=0.7,
              correction_margin=0.30, mip_gap=0.10, workers=4, seed=42,
              initializer=dict(n_pcs=30,n_neighbors=10,resolution=0.6))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    """Copy frozen baseline and pass 1; never refit or inspect reference labels."""
    source=BASE.parent/'Zhengmix4eq_single_pass_grid_60genes'
    case=source/'cases/pcs30_knn10_res0.6'
    for name in ['inputs','h5ad','results','figures','logs','native','provenance']:
        (BASE/name).mkdir(parents=True,exist_ok=True)
    assert digest(PACKAGE)==digest(source/'provenance/cellsieve.py')
    shutil.copy2(PACKAGE,BASE/'provenance/cellsieve.py')
    shutil.copy2(source/'inputs/hidden_ground_truth.csv',DATA/'hidden_ground_truth.csv')
    for r in [0,1]:
        shutil.copy2(case/f'h5ad/round{r}_expression_only.h5ad',BASE/f'h5ad/round{r}_expression_only.h5ad')
    shutil.copytree(case/'native/round1',BASE/'native/round1',dirs_exist_ok=True)
    shutil.copy2(source/'logs/fit_pcs30_knn10_res0.6.log',BASE/'logs/reused_round1.log')
    save_json(BASE/'config.json',CONFIG)
    save_json(BASE/'provenance/reused_pass.json',dict(source=str(case),
        initialization='30 PCs / 10 neighbors / resolution 0.6',
        selection='Same 30-PC / 10-neighbor / resolution-0.6 condition selected in the original 40-gene screen; held fixed for this 60-gene repeat.',
        input_hashes={str(case/f'h5ad/round{r}_expression_only.h5ad'):digest(case/f'h5ad/round{r}_expression_only.h5ad') for r in [0,1]},
        package_sha256=digest(PACKAGE),script_sha256=digest(__file__)))


def run_round(round_no):
    """Call actual package main(), with passive observers saving native outputs."""
    import importlib.util
    import numpy as np
    import pandas as pd
    import anndata as ad
    from threadpoolctl import threadpool_limits
    start=time.monotonic()
    previous=BASE/f'h5ad/round{round_no-1}_expression_only.h5ad'
    a=ad.read_h5ad(previous)
    assert 'ground_truth' not in a.obs
    out=BASE/f'native/round{round_no}'
    out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('native_cellsieve',PACKAGE)
    native=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    # The observer calls the original function exactly once and returns its
    # result unchanged. It only preserves probabilities omitted by native CLI.
    original_predict=native.cross_val_predict
    def record_predictions(*args,**kwargs):
        probabilities=original_predict(*args,**kwargs)
        np.savez_compressed(out/'oof_predictions.npz',probabilities=probabilities,
                            classes=np.asarray(sorted(a.obs.celltype.astype(str).unique())),
                            cell_ids=a.obs_names.to_numpy(dtype=str))
        return probabilities
    native.cross_val_predict=record_predictions
    original_pairs=native.select_mrmr_constraints
    def record_constraints(*args,**kwargs):
        pairs=original_pairs(*args,**kwargs)
        pairs.to_csv(out/'native_constraints.csv',index=False)
        return pairs
    native.select_mrmr_constraints=record_constraints
    argv=['cellsieve','--adata-path',str(previous),'--out-dir',str(out),
          '--celltype-column','celltype','--panel-size','60','--time-limit','300',
          '--alpha','0.7','--mip-gap','0.10','--min-correction-margin','0.30',
          '--random-seed','42','--n-workers','4','--correct-misannotations']
    save_json(out/'argv.json',argv)
    sys.argv=argv
    with threadpool_limits(limits=4):
        native.main()
    panel=out/'panel_size_60_alpha_0.70'
    cleaned=panel/f'CLEANED_{previous.name}'
    if not cleaned.exists():
        raise RuntimeError('Native CellSieve did not produce the required corrected H5AD; inspect round log.')
    clean=ad.read_h5ad(cleaned)
    assert clean.obs_names.equals(a.obs_names)
    updated=clean.obs.celltype_corrected.astype(str).to_numpy()
    pred=np.load(out/'oof_predictions.npz',allow_pickle=False)
    assert np.array_equal(pred['cell_ids'],a.obs_names)
    classes=pred['classes']; probs=pred['probabilities']
    label_idx=pd.Index(classes).get_indexer(a.obs.celltype.astype(str))
    prediction=classes[probs.argmax(axis=1)]
    margin=probs.max(axis=1)-probs[np.arange(a.n_obs),label_idx]
    expected=np.where((prediction!=a.obs.celltype.astype(str).to_numpy())&(margin>.30),prediction,a.obs.celltype.astype(str))
    assert np.array_equal(updated,expected)
    a.obs[f'round{round_no}_annotation']=updated
    a.obs[f'round{round_no}_prediction']=prediction
    a.obs[f'round{round_no}_confidence']=probs.max(axis=1)
    a.obs[f'round{round_no}_margin']=margin
    a.obsm[f'round{round_no}_probabilities']=probs
    a.uns[f'round{round_no}_classes']=classes
    genes=(panel/'selected_genes.txt').read_text().splitlines()
    assert len(genes)==60 and len(set(genes))==60
    a.uns[f'round{round_no}_panel']=genes
    a.obs['celltype']=updated
    a.write_h5ad(BASE/f'h5ad/round{round_no}_expression_only.h5ad',compression='gzip')
    save_json(out/'completed.json',dict(round=round_no,elapsed_seconds=time.monotonic()-start,
        cells_changed=int((updated!=clean.obs.celltype.astype(str).to_numpy()).sum()),
        self_comparison_constraints=int((pd.read_csv(out/'native_constraints.csv').true_label==pd.read_csv(out/'native_constraints.csv').wrong_label).sum()),
        source_sha256=digest(PACKAGE),reference_labels_used=False))
    print(f'ROUND {round_no} FROZEN',flush=True)


def evaluate(round_no):
    """Reference-only evaluator; its outputs are never inputs to CellSieve."""
    import numpy as np
    import pandas as pd
    import anndata as ad
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,f1_score
    path=BASE/f'h5ad/round{round_no}_expression_only.h5ad'
    before=digest(path)
    a=ad.read_h5ad(path)
    reference=pd.read_csv(DATA/'hidden_ground_truth.csv',index_col='cell_id')
    assert reference.index.is_unique and set(reference.index)==set(a.obs_names)
    y=reference.loc[a.obs_names,'ground_truth'].astype(str).to_numpy()
    def map_labels(labels):
        table=pd.crosstab(pd.Series(labels),pd.Series(y)).sort_index(axis=1)
        return table.idxmax(axis=1).to_dict()
    initial=a.obs.round0_annotation.astype(str).to_numpy()
    mapping=map_labels(initial)
    save_json(BASE/'results/round0_reference_mapping.json',mapping)
    fixed_map=lambda labels:np.asarray([mapping.get(v,'UNMATCHED') for v in labels])
    labels=a.obs[f'round{round_no}_annotation'].astype(str).to_numpy()
    previous=a.obs[f'round{max(0,round_no-1)}_annotation'].astype(str).to_numpy()
    was_correct=fixed_map(previous)==y
    is_correct=fixed_map(labels)==y
    optimal_mapping=map_labels(labels)
    optimal=np.asarray([optimal_mapping.get(v,'UNMATCHED') for v in labels])
    changed=labels!=previous
    row=dict(round=round_no,accuracy=float(np.mean(optimal==y)),
             fixed_mapping_accuracy=float(is_correct.mean()),
             macro_f1=float(f1_score(y,optimal,labels=sorted(set(y)),average="macro",zero_division=0)),
             ari=float(adjusted_rand_score(y,labels)),nmi=float(normalized_mutual_info_score(y,labels)),
             cells_changed=int(changed.sum()),corrected_errors=int((~was_correct&is_correct).sum()),
             introduced_errors=int((was_correct&~is_correct).sum()),
             wrong_to_wrong=int((changed&~was_correct&~is_correct).sum()),
             correct_to_correct=int((changed&was_correct&is_correct).sum()),
             n_clusters=len(np.unique(labels)),mapping_unchanged=optimal_mapping==mapping)
    assert row['corrected_errors']-row['introduced_errors']==int(is_correct.sum()-was_correct.sum())
    assert row['cells_changed']==row['corrected_errors']+row['introduced_errors']+row['wrong_to_wrong']+row['correct_to_correct']
    save_json(BASE/f'results/round{round_no}_metrics.json',row)
    rows=[json.loads((BASE/f'results/round{i}_metrics.json').read_text()) for i in range(round_no+1)]
    pd.DataFrame(rows).to_csv(BASE/'results/metrics.csv',index=False)
    pd.DataFrame(dict(cell_id=a.obs_names,ground_truth=y,previous_annotation=previous,
                     annotation=labels,mapped_annotation=fixed_map(labels),changed=changed,
                     corrected=~was_correct&is_correct,introduced=was_correct&~is_correct)).to_csv(
                     BASE/f'results/round{round_no}_cell_evaluation.csv',index=False)
    assert digest(path)==before
    if round_no==3:
        a.obs['ground_truth']=y
        a.obs['final_annotation']=labels
        a.uns['reference_mapping_json']=json.dumps(mapping)
        a.write_h5ad(BASE/'h5ad/final_with_evaluation.h5ad',compression='gzip')
    print(json.dumps(row),flush=True)


def report():
    import pandas as pd
    m=pd.read_csv(BASE/'results/metrics.csv')
    assert m['round'].tolist()==list(range(4))
    assert digest(PACKAGE)==digest(BASE/'provenance/cellsieve.py')
    save_json(BASE/'results/summary.json',dict(initial_accuracy=float(m.accuracy.iloc[0]),
        final_accuracy=float(m.accuracy.iloc[-1]),initial_macro_f1=float(m.macro_f1.iloc[0]),
        final_macro_f1=float(m.macro_f1.iloc[-1]),net_correct_cells=int(m.corrected_errors.sum()-m.introduced_errors.sum()),
        all_accuracy_steps_positive=bool((m.accuracy.diff().dropna()>0).all()),
        all_macro_f1_steps_positive=bool((m.macro_f1.diff().dropna()>0).all()),
        all_reference_mappings_unchanged=bool(m.mapping_unchanged.all()),native_source_unchanged=True))
    (BASE/'README.md').write_text("""# Three-pass repeat of the original follow-up condition

Zhengmix4eq, frozen 30-PC / 10-neighbor / resolution-0.6 partition. The condition was selected in the original 40-gene screen and held fixed for this repeat; therefore this is exploratory reference-informed selection, not an independent benchmark. Ground truth remains absent from CellSieve inputs, panel fitting, updates, and stopping. Exactly three passes are evaluated regardless of outcomes. The first pass and baseline are reused byte-for-byte from the screen; two additional native passes are run.

Each pass uses the unchanged local CellSieve package, 60 genes, alpha 0.7, seed 42, 300-second MILP limit, MIP gap 0.10, four workers, native OOF predictions and correction margin >0.30. Native global panel selection and scaling precede classifier CV. No UMAP is computed or shown.

Accuracy and macro-F1 use each completed partition's independent majority-reference mapping, allowing multiple clusters per reference type. Error transitions use the frozen initial majority mapping. Both accuracy variants and mapping stability are saved in results/metrics.csv. The final H5AD includes truth for evaluation only and is not fitted. Original one-pass artifacts and all new native outputs are preserved.

Reproduce with benchmark_env Python and LD_LIBRARY_PATH set to its lib directory: scripts/run_refinement.py. The script copies the saved first pass, runs passes 2–3, evaluates all completed passes, and writes metrics and H5ADs. The main paper Figure 3 renderer reads these outputs without fitting.
""")
    print(m.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','round','evaluate','report'])
    parser.add_argument('--round',type=int,choices=range(4),default=0)
    args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='round':run_round(args.round);return
    if args.stage=='evaluate':evaluate(args.round);return
    if args.stage=='report':report();return
    (BASE/'logs').mkdir(parents=True,exist_ok=True)
    def launch(stage,r=0):
        print(f'Starting {stage}, round {r}',flush=True)
        with (BASE/f'logs/{stage}_{r}.log').open('w') as log:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--stage',stage,'--round',str(r)],
                           stdout=log,stderr=subprocess.STDOUT,check=True,cwd=BASE)
        print(f'Finished {stage}, round {r}',flush=True)
    launch('prepare');launch('evaluate');launch('evaluate',1)
    for r in range(2,4):
        launch('round',r);launch('evaluate',r)
    launch('report')


if __name__=='__main__':main()
