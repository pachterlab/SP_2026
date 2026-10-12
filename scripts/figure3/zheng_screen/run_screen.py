#!/usr/bin/env python3
"""Twenty frozen Zhengmix4eq partitions -> one native CellSieve pass each.

Run without arguments with benchmark_env Python. Each fitting round executes in
its own process with no reference labels in its input. Evaluation is a separate
process and its outputs are never consumed by fitting.
"""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
(PROJECT_ROOT / 'figs/diagnostics/zheng_screen').mkdir(parents=True, exist_ok=True)
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
ROOT = PROJECT_ROOT / 'results/figure3/Zhengmix4eq_single_pass_grid_60genes'
BASE = ROOT
DATA = BASE / 'inputs'
FROZEN = BASE.parent / 'Zhengmix4eq_native_grid'
PACKAGE = PROJECT_ROOT / 'external/cell_sieve_package/cellsieve.py'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(BASE / '.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR'] = str(BASE / '.cache/numba')
os.environ.setdefault('GRB_LICENSE_FILE', '/home/nikki/gurobi.lic')
import itertools
SETTINGS=[dict(n_pcs=p,n_neighbors=n,resolution=r) for p,n,r in
          itertools.product([20,30],[10,30],[0.2,0.4,0.6,0.8,1.0])]
CONFIG=dict(settings=SETTINGS,passes=1,panel_size=60,time_limit_seconds=300,
            alpha=0.7,correction_margin=0.30,mip_gap=0.10,workers_per_case=4,
            concurrent_cases=6,seed=42,ground_truth_parameter_selection=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


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


def case_name(setting):
    return 'pcs{n_pcs}_knn{n_neighbors}_res{resolution:.1f}'.format(**setting)


def prepare():
    """Reuse byte-identical frozen expression and provisional labels from prior screen."""
    previous=ROOT.parent/'Zhengmix4eq_single_pass_grid'
    if not previous.exists():
        previous=ROOT
    for name in ['inputs','h5ad','results','logs','cases','provenance']:
        (ROOT/name).mkdir(exist_ok=True)
    save_json(ROOT/'config.json',CONFIG)
    if previous!=ROOT:
        shutil.copytree(previous/'inputs',DATA,dirs_exist_ok=True)
        shutil.copy2(previous/'h5ad/expression_only.h5ad',ROOT/'h5ad/expression_only.h5ad')
    shutil.copy2(PACKAGE,ROOT/'provenance/cellsieve.py')
    for setting in SETTINGS:
        name=case_name(setting);case=ROOT/'cases'/name
        (case/'h5ad').mkdir(parents=True,exist_ok=True)
        source=previous/'cases'/name/'h5ad/round0_expression_only.h5ad'
        if source!=case/'h5ad/round0_expression_only.h5ad':
            shutil.copy2(source,case/'h5ad/round0_expression_only.h5ad')
        assert digest(source)==digest(case/'h5ad/round0_expression_only.h5ad')
        save_json(case/'initializer.json',setting)
    save_json(ROOT/'provenance/source_hashes.json',{str(p):digest(p) for p in [PACKAGE,Path(__file__),previous/'h5ad/expression_only.h5ad']})
    save_json(ROOT/'provenance/initialization.json',dict(source=str(previous),frozen_initializations_reused=True,reference_labels_used_for_fitting=False))
    print('Reused all 20 frozen initializations.',flush=True)


def fit_case(index):
    global BASE
    BASE=ROOT/'cases'/case_name(SETTINGS[index])
    assert digest(PACKAGE)==digest(ROOT/'provenance/cellsieve.py')
    run_round(1)


def evaluate():
    """Evaluate only after ALL 20 single-pass outputs exist; never feed back."""
    import numpy as np
    import pandas as pd
    import anndata as ad
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,f1_score
    for setting in SETTINGS:
        assert (ROOT/'cases'/case_name(setting)/'native/round1/completed.json').exists()
    a=ad.read_h5ad(ROOT/'h5ad/expression_only.h5ad')
    reference=pd.read_csv(DATA/'hidden_ground_truth.csv',index_col='cell_id')
    assert reference.index.is_unique and set(reference.index)==set(a.obs_names)
    y=reference.loc[a.obs_names,'ground_truth'].astype(str).to_numpy()
    a.obs['ground_truth']=y
    def mapping(labels):
        tab=pd.crosstab(pd.Series(y),pd.Series(labels))
        r,c=linear_sum_assignment(-tab.to_numpy())
        return {str(tab.columns[j]):str(tab.index[i]) for i,j in zip(r,c)}
    def apply(labels,m):return np.asarray([m.get(v,'UNMATCHED') for v in labels])
    rows=[];hashes={}
    for setting in SETTINGS:
        name=case_name(setting);case=ROOT/'cases'/name
        path=case/'h5ad/round1_expression_only.h5ad'
        hashes[name]=digest(path)
        b=ad.read_h5ad(path)
        assert b.obs_names.equals(a.obs_names) and 'ground_truth' not in b.obs
        initial=b.obs.round0_annotation.astype(str).to_numpy()
        final=b.obs.round1_annotation.astype(str).to_numpy()
        before_mapping=mapping(initial);after_mapping=mapping(final)
        pred0=apply(initial,before_mapping);pred1=apply(final,after_mapping)
        fixed1=apply(final,before_mapping)
        correct0=pred0==y;correct1=fixed1==y;changed=initial!=final
        completed=json.loads((case/'native/round1/completed.json').read_text())
        panel=list(b.uns['round1_panel'])
        row=dict(setting,case=name,n_cells=len(y),
            initial_accuracy=float(correct0.mean()),final_accuracy=float(np.mean(pred1==y)),
            final_fixed_mapping_accuracy=float(correct1.mean()),
            initial_ari=float(adjusted_rand_score(y,initial)),final_ari=float(adjusted_rand_score(y,final)),
            initial_nmi=float(normalized_mutual_info_score(y,initial)),final_nmi=float(normalized_mutual_info_score(y,final)),
            initial_macro_f1=float(f1_score(y,pred0,labels=np.unique(y),average='macro',zero_division=0)),
            final_macro_f1=float(f1_score(y,pred1,labels=np.unique(y),average='macro',zero_division=0)),
            initial_groups=len(np.unique(initial)),final_groups=len(np.unique(final)),
            cells_changed=int(changed.sum()),corrected_errors=int((~correct0&correct1).sum()),
            introduced_errors=int((correct0&~correct1).sum()),wrong_to_wrong=int((changed&~correct0&~correct1).sum()),
            mapping_unchanged=before_mapping==after_mapping,panel_size=len(panel),
            fit_seconds=completed['elapsed_seconds'],self_comparison_constraints=completed['self_comparison_constraints'])
        import re
        solver_log=(ROOT/f'logs/fit_{name}.log').read_text()
        gaps=re.findall(r'Best objective .*?gap ([0-9.eE+\-]+)%',solver_log)
        row['solver_gap_percent']=float(gaps[-1]) if gaps else None
        statuses=re.findall(r'Time limit reached|Optimal solution found',solver_log)
        row['solver_status']=('time_limit' if statuses[-1]=='Time limit reached' else 'gap_tolerance_met') if statuses else 'other; see log'
        row['accuracy_change_pp']=100*(row['final_accuracy']-row['initial_accuracy'])
        row['ari_change']=row['final_ari']-row['initial_ari']
        row['nmi_change']=row['final_nmi']-row['initial_nmi']
        row['net_correct_cells']=row['corrected_errors']-row['introduced_errors']
        assert row['cells_changed']==row['corrected_errors']+row['introduced_errors']+row['wrong_to_wrong']
        assert row['net_correct_cells']==int(correct1.sum()-correct0.sum())
        assert row['panel_size']==60
        save_json(case/'reference_evaluation.json',row)
        save_json(case/'reference_mappings.json',dict(initial=before_mapping,final=after_mapping))
        pd.DataFrame(dict(cell_id=a.obs_names,ground_truth=y,initial_annotation=initial,final_annotation=final,
            changed=changed,corrected=~correct0&correct1,introduced=correct0&~correct1)).to_csv(case/'cell_evaluation.csv',index=False)
        for column in ['round0_annotation','round1_annotation','round1_prediction','round1_confidence','round1_margin']:
            a.obs[name+'__'+column]=b.obs[column].to_numpy()
        a.obsm[name+'__probabilities']=b.obsm['round1_probabilities']
        a.uns[name+'__classes']=b.uns['round1_classes']
        a.uns[name+'__panel']=panel
        assert digest(path)==hashes[name]
        rows.append(row)
    table=pd.DataFrame(rows)
    assert len(table)==20 and not table.duplicated(['n_pcs','n_neighbors','resolution']).any()
    table.to_csv(ROOT/'results/all_settings_results.csv',index=False)
    a.write_h5ad(ROOT/'h5ad/all_cases_with_evaluation.h5ad',compression='gzip')
    save_json(ROOT/'results/validation.json',dict(n_cases=20,n_cells=a.n_obs,all_cases_single_pass=True,
        native_source_unchanged=digest(PACKAGE)==digest(ROOT/'provenance/cellsieve.py'),
        evaluation_did_not_change_model_outputs=True,output_sha256=hashes,
        all_panels_size_60=True,all_mappings_unchanged=bool(table.mapping_unchanged.all())))
    print(table[['case','initial_accuracy','final_accuracy','accuracy_change_pp','corrected_errors','introduced_errors']].to_string(index=False),flush=True)


def report():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    sys.path.insert(0,str(PROJECT_ROOT/'scripts'))
    from figure_style import set_style
    set_style()
    table=pd.read_csv(ROOT/'results/all_settings_results.csv')
    pairs=list(itertools.product([20,30],[10,30]))
    fig,axes=plt.subplots(3,4,figsize=(13,8),layout='constrained')
    for col,(pcs,neighbors) in enumerate(pairs):
        t=table.query('n_pcs==@pcs and n_neighbors==@neighbors').sort_values('resolution')
        for row,(metric,label,scale) in enumerate([('accuracy','Matched accuracy (%)',100),('ari','ARI',1),('nmi','NMI',1)]):
            ax=axes[row,col]
            ax.plot(t.resolution,scale*t['initial_'+metric],'o--',color='#767676',ms=4,label='Initial Leiden')
            ax.plot(t.resolution,scale*t['final_'+metric],'o-',color='#266a9e',ms=4,label='One CellSieve pass')
            ax.set(xlabel='Leiden resolution',ylabel=label,xticks=[.2,.4,.6,.8,1.0])
            if row==0:ax.set_title(f'{pcs} PCs / {neighbors} neighbors')
            ax.text(-.14,1.04,chr(65+row*4+col),transform=ax.transAxes,weight='bold',fontsize=10)
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',color='.9',lw=.5)
            if row==0 and col==0:ax.legend(frameon=False,fontsize=7)
    for row,metric,scale in [(0,'accuracy',100),(1,'ari',1),(2,'nmi',1)]:
        lo=scale*table[['initial_'+metric,'final_'+metric]].min().min()
        hi=scale*table[['initial_'+metric,'final_'+metric]].max().max()
        pad=max(.025*scale,.10*(hi-lo))
        for ax in axes[row]:ax.set_ylim(max(0,lo-pad),min(scale,hi+pad))
    improved=int((table.accuracy_change_pp>1e-10).sum())
    worsened=int((table.accuracy_change_pp< -1e-10).sum())
    tied=20-improved-worsened
    fig.suptitle(f'Zhengmix4eq: one CellSieve pass across 20 initializations\nAccuracy: {improved} improved, {tied} unchanged, {worsened} worsened | 60 genes; 300 s solver limit',fontsize=12)
    for ext in ['png','pdf','svg']:fig.savefig(PROJECT_ROOT/f'figs/diagnostics/zheng_screen/Figure3_single_pass_grid.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    metrics=[('accuracy_change_pp','Accuracy change (pp)'),('ari_change','ARI change'),('nmi_change','NMI change'),
             ('corrected_errors','Errors corrected'),('introduced_errors','Errors introduced'),('net_correct_cells','Net correct cells (fixed mapping)')]
    for ax,(metric,title) in zip(axes.flat,metrics):
        p=table.pivot(index=['n_pcs','n_neighbors'],columns='resolution',values=metric).reindex(pairs)
        values=p.to_numpy();signed=metric not in ['corrected_errors','introduced_errors']
        bound=max(float(np.abs(values).max()),1e-6)
        im=ax.imshow(values,aspect='auto',cmap='RdBu' if signed else 'Blues',vmin=-bound if signed else 0,vmax=bound)
        for i in range(4):
            for j in range(5):
                v=values[i,j]
                label=f'{v:+.2f}' if metric=='accuracy_change_pp' else (f'{v:+.3f}' if metric.endswith('_change') else str(int(v)))
                rgba=im.cmap(im.norm(v));bright=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]
                ax.text(j,i,label,ha='center',va='center',fontsize=8,color='black' if bright>.55 else 'white')
        ax.set(title=title,xticks=range(5),xticklabels=[.2,.4,.6,.8,1.0],yticks=range(4),
               yticklabels=[f'{p} / {n}' for p,n in pairs],xlabel='Leiden resolution',ylabel='PCs / neighbors')
        fig.colorbar(im,ax=ax,shrink=.8)
    for ext in ['png','pdf']:fig.savefig(PROJECT_ROOT/f'figs/diagnostics/zheng_screen/changes_and_errors.{ext}',dpi=250,bbox_inches='tight')
    plt.close(fig)
    by_resolution=table.groupby('resolution').agg(cases=('case','count'),
        accuracy_improved=('accuracy_change_pp',lambda x:int((x>1e-10).sum())),
        accuracy_worsened=('accuracy_change_pp',lambda x:int((x< -1e-10).sum())),
        min_accuracy_change_pp=('accuracy_change_pp','min'),max_accuracy_change_pp=('accuracy_change_pp','max'))
    by_resolution.to_csv(ROOT/'results/resolution_summary.csv')
    summary=dict(cases=20,accuracy_improved=improved,accuracy_unchanged=tied,accuracy_worsened=worsened,
        ari_improved=int((table.ari_change>1e-10).sum()),nmi_improved=int((table.nmi_change>1e-10).sum()),
        accuracy_change_pp_min=float(table.accuracy_change_pp.min()),accuracy_change_pp_max=float(table.accuracy_change_pp.max()),
        all_settings_improve_accuracy=improved==20,all_mappings_unchanged=bool(table.mapping_unchanged.all()),
        solver_time_limit_cases=int((table.solver_status=='time_limit').sum()))
    save_json(ROOT/'results/summary.json',summary)
    lines=['# Zhengmix4eq: single-pass CellSieve across 20 settings','',
        f'Accuracy improved in {improved}/20 cases, was unchanged in {tied}, and worsened in {worsened}. All settings are retained.', '',
        'Predeclared grid: PCs 20/30 × neighbors 10/30 × resolution 0.2/0.4/0.6/0.8/1.0. These are the exact corresponding partitions from the earlier frozen native Scanpy grid. The 20 cases are not ranked, filtered, or retuned using reference scores. The grid is exploratory because the earlier 45-setting evaluation was already seen.', '',
        'Preprocessing: raw counts, genes in at least three cells, normalization to 10,000 and log1p. Original initialization: 2,000 Seurat HVGs, scaling clip 10, PCA50 (ARPACK), first requested PCs, exact sklearn neighbors, native Leiden (leidenalg) to convergence, seed 42. CellSieve uses all retained normalized genes, not just HVGs. Frozen labels are reused exactly. PCA is saved; no UMAP is computed or plotted.', '',
        'Each case starts independently from its initial Leiden labels and calls the unmodified local CellSieve main() exactly once: 60 genes, alpha 0.7, seed 42, four workers, MIP gap 0.10, 300-second MILP limit, native five-fold predictions and probability-margin threshold >0.30. Six cases run concurrently. Passive wrappers save native probabilities and constraints unchanged. No later refinement rounds are run.', '',
        'All 20 fitting processes finish before evaluation reads FACS labels. Primary accuracy uses a separate optimal one-to-one Hungarian mapping for each initial/final partition; unmatched groups count wrong. Error transitions use each case\'s frozen initial mapping. The CSV reports both final accuracies and whether mappings changed. Macro-F1 averages over the true reference classes with unmatched predictions counted as errors. ARI and NMI compare partitions without mapping. The 20 settings use the same cells, so they are sensitivity analyses, not independent replicates.', '',
        'Native limitations: panel selection and scaling precede classifier cross-validation, so native OOF predictions are not fully nested validation. Native constraint sampling can include self-comparisons with five or fewer groups; this behavior is preserved and the constraints/counts are saved. The 300-second limit bounds only the solver, not total runtime. Time-limited solver incumbents can vary between runs/machines.', '',
        '[Main figure](figures/Figure3_single_pass_grid.png) · [Changes and errors](figures/changes_and_errors.png) · [All settings CSV](results/all_settings_results.csv) · [Resolution summary](results/resolution_summary.csv)', '',
        '| PCs | Neighbors | Resolution | Initial accuracy | After one pass | Change (pp) | Corrected | Introduced | Initial/final K |',
        '|---|---|---|---|---|---|---|---|---|']
    for r in table.itertuples():lines.append(f'| {r.n_pcs} | {r.n_neighbors} | {r.resolution:.1f} | {r.initial_accuracy:.2%} | {r.final_accuracy:.2%} | {r.accuracy_change_pp:+.3f} | {r.corrected_errors} | {r.introduced_errors} | {r.initial_groups}/{r.final_groups} |')
    lines+=['','## Reproduce','','```sh',
        'LD_LIBRARY_PATH=/home/nikki/miniconda3/envs/benchmark_env/lib /home/nikki/miniconda3/envs/benchmark_env/bin/python '+str(Path(__file__).resolve()),
        '```','','Use `--stage report` to redraw saved results, `--stage evaluate` to reevaluate frozen predictions, or `--resume` to finish only cases missing completed outputs. Each case has its initial and refined truth-free H5AD, native selected genes/coefficients/flags/probabilities/constraints, and evaluated cell transitions. `h5ad/all_cases_with_evaluation.h5ad` combines normalized expression, counts, PCA, hidden truth, all initial/final annotations and predictions/confidences/panels. This evaluation H5AD is never used for fitting. Source hashes, versions, config and logs are saved.']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    from concurrent.futures import ThreadPoolExecutor
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','fit','evaluate','report'])
    parser.add_argument('--case',type=int,choices=range(20))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='fit':fit_case(args.case);return
    if args.stage=='evaluate':evaluate();return
    if args.stage=='report':report();return
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    def launch(stage,index=None):
        suffix='' if index is None else '_'+case_name(SETTINGS[index])
        argv=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage',stage]
        if index is not None:argv+=['--case',str(index)]
        with (ROOT/f'logs/{stage}{suffix}.log').open('w') as stream:
            subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        print('Completed '+stage+suffix,flush=True)
    if not args.resume:launch('prepare')
    else:assert json.loads((ROOT/'config.json').read_text())==CONFIG
    indices=[i for i,s in enumerate(SETTINGS) if not args.resume or not (ROOT/'cases'/case_name(s)/'native/round1/completed.json').exists()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda i:launch('fit',i),indices))
    launch('evaluate');launch('report')


if __name__=='__main__':main()
