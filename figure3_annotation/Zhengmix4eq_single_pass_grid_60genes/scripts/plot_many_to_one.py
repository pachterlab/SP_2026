#!/usr/bin/env python3
"""Evaluation-only many-to-one mapping and alluvial plots for all 20 saved cases."""
from pathlib import Path
import hashlib
import json
import os
import sys

BASE=Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(BASE/'.cache/matplotlib')
sys.path.insert(0,str(BASE.parents[1]/'scripts'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch,Rectangle
from sklearn.metrics import accuracy_score,f1_score
from figure_style import set_style

PAIRS=[(20,10),(20,30),(30,10),(30,30)]
RESOLUTIONS=[.2,.4,.6,.8,1.0]
DISPLAY={'b.cells':'B','cd14.monocytes':'Mono','naive.cytotoxic':'CD8','regulatory.t':'Treg'}
COLORS={'b.cells':'#3977aa','cd14.monocytes':'#df9b37','naive.cytotoxic':'#539b70','regulatory.t':'#946bb0'}


def map_clusters(labels,truth):
    """Independent per-stage majority mapping; sorted reference names break ties."""
    table=pd.crosstab(labels,truth).sort_index(axis=1)
    return table.idxmax(axis=1).to_dict()


def alluvial(ax,frame,m0,m1,title,headers=False):
    """Initial cluster -> reference population -> refined cluster, widths = cells."""
    truths=sorted(frame.ground_truth.unique())
    stages=['initial_annotation','ground_truth','final_annotation']
    orders=[sorted(m0,key=lambda c:(truths.index(m0[c]),int(c))),truths,
            sorted(m1,key=lambda c:(truths.index(m1[c]),int(c)))]
    n=len(frame);max_nodes=max(map(len,orders));gap=.019
    # Common scale across all stages: one cell has exactly the same ribbon height.
    unit=(.90-gap*(max_nodes-1))/n
    positions=[]
    for stage,order in zip(stages,orders):
        counts=frame[stage].value_counts();total=n*unit+gap*(len(order)-1)
        y=(1-total)/2;pos={}
        for node in order:
            h=int(counts[node])*unit;pos[node]=(y,h);y+=h+gap
        positions.append(pos)
    xs=[.08,.48,.88];width=.035
    for step in range(2):
        left,right=stages[step:step+2]
        counts=pd.crosstab(frame[left],frame[right])
        left_offset={node:positions[step][node][0] for node in orders[step]}
        right_offset={node:positions[step+1][node][0] for node in orders[step+1]}
        for l in orders[step]:
            for r in orders[step+1]:
                count=int(counts.loc[l,r]) if l in counts.index and r in counts.columns else 0
                if not count:continue
                h=count*unit;y0=left_offset[l];y1=right_offset[r]
                x0=xs[step]+width;x1=xs[step+1];mid=(x0+x1)/2
                vertices=[(x0,y0),(mid,y0),(mid,y1),(x1,y1),(x1,y1+h),
                          (mid,y1+h),(mid,y0+h),(x0,y0+h),(x0,y0)]
                codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,
                       MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
                truth=r if step==0 else l
                ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor=COLORS[truth],alpha=.4,edgecolor='none'))
                left_offset[l]+=h;right_offset[r]+=h
    for stage_index,order in enumerate(orders):
        for node in order:
            y,h=positions[stage_index][node]
            truth=node if stage_index==1 else (m0 if stage_index==0 else m1)[node]
            ax.add_patch(Rectangle((xs[stage_index],y),width,h,facecolor=COLORS[truth],edgecolor='white',lw=.25))
            label=DISPLAY[node] if stage_index==1 else 'c'+node
            if stage_index==0:x=xs[0]-.012;ha='right'
            else:x=xs[stage_index]+width+.008;ha='left'
            ax.text(x,y+h/2,label,fontsize=6.5,ha=ha,va='center',color='black',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.1) if stage_index==1 else None)
    ax.text(-.02,1.03,f'Resolution {title:.1f}',fontsize=7,transform=ax.transAxes,ha='left')
    if headers:
        for x,name in zip(xs,['Initial','FACS','Refined']):ax.text(x,1.38,name,fontsize=8,ha='left')
    ax.set(xlim=(-.03,1.02),ylim=(-.025,1.06));ax.axis('off')


def main():
    set_style()
    old=pd.read_csv(BASE/'results/all_settings_results.csv')
    frames={};maps={};rows=[];mapping_rows=[];flow_rows=[];hashes={}
    for setting in old.itertuples():
        p=BASE/'cases'/setting.case/'cell_evaluation.csv'
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        d=pd.read_csv(p,dtype={'cell_id':str,'initial_annotation':str,'final_annotation':str,'ground_truth':str})
        assert d.cell_id.is_unique and len(d)==setting.n_cells and not d.isna().any().any()
        truth=d.ground_truth
        m0=map_clusters(d.initial_annotation,truth);m1=map_clusters(d.final_annotation,truth)
        pred0=d.initial_annotation.map(m0);pred1=d.final_annotation.map(m1)
        fixed1=d.final_annotation.map(m0)
        assert not pred0.isna().any() and not pred1.isna().any() and not fixed1.isna().any()
        r=dict(case=setting.case,n_pcs=setting.n_pcs,n_neighbors=setting.n_neighbors,resolution=setting.resolution,
               n_cells=len(d),initial_groups=len(m0),final_groups=len(m1),
               initial_accuracy=accuracy_score(truth,pred0),final_accuracy=accuracy_score(truth,pred1),
               initial_macro_f1=f1_score(truth,pred0,labels=sorted(truth.unique()),average='macro',zero_division=0),
               final_macro_f1=f1_score(truth,pred1,labels=sorted(truth.unique()),average='macro',zero_division=0),
               final_fixed_mapping_accuracy=accuracy_score(truth,fixed1),mapping_unchanged=m0==m1,
               cells_changed=int((d.initial_annotation!=d.final_annotation).sum()),
               corrected_fixed_mapping=int(((pred0!=truth)&(fixed1==truth)).sum()),
               introduced_fixed_mapping=int(((pred0==truth)&(fixed1!=truth)).sum()))
        r['accuracy_change_pp']=100*(r['final_accuracy']-r['initial_accuracy'])
        r['macro_f1_change']=r['final_macro_f1']-r['initial_macro_f1']
        # Unconstrained majority accuracy cannot be lower than one-to-one accuracy.
        assert r['initial_accuracy']+1e-12>=setting.initial_accuracy
        assert r['final_accuracy']+1e-12>=setting.final_accuracy
        rows.append(r);frames[setting.case]=d;maps[setting.case]=(m0,m1)
        for stage,m in [('initial_annotation',m0),('final_annotation',m1)]:
            for cluster,reference in m.items():
                subset=d[d[stage]==cluster]
                mapping_rows.append(dict(case=setting.case,stage=stage,cluster=cluster,mapped_reference=reference,
                    cluster_cells=len(subset),majority_cells=int((subset.ground_truth==reference).sum())))
        for left,right in [('initial_annotation','ground_truth'),('ground_truth','final_annotation')]:
            flows=d.groupby([left,right]).size()
            assert int(flows.sum())==len(d)
            for (source,target),count in flows.items():
                flow_rows.append(dict(case=setting.case,source_stage=left,target_stage=right,source=source,target=target,cells=int(count)))
    t=pd.DataFrame(rows);assert len(t)==20
    t.to_csv(BASE/'results/many_to_one_metrics.csv',index=False)
    pd.DataFrame(mapping_rows).to_csv(BASE/'results/many_to_one_cluster_mappings.csv',index=False)
    pd.DataFrame(flow_rows).to_csv(BASE/'results/alluvial_flows.csv',index=False)
    fig=plt.figure(figsize=(14,13),layout='constrained')
    grid=fig.add_gridspec(3,4,height_ratios=[1,1,3.1])
    for col,(pcs,neighbors) in enumerate(PAIRS):
        sub=t.query('n_pcs==@pcs and n_neighbors==@neighbors').sort_values('resolution')
        for row,metric,label in [(0,'accuracy','Many-to-one accuracy (%)'),(1,'macro_f1','Macro-F1')]:
            ax=fig.add_subplot(grid[row,col]);scale=100 if row==0 else 1
            ax.plot(sub.resolution,scale*sub['initial_'+metric],'o--',color='#767676',ms=4,label='Initial Leiden')
            ax.plot(sub.resolution,scale*sub['final_'+metric],'o-',color='#266a9e',ms=4,label='One CellSieve pass')
            ax.set(xlabel='Leiden resolution',ylabel=label,xticks=RESOLUTIONS)
            lo=scale*t[['initial_'+metric,'final_'+metric]].min().min()
            hi=scale*t[['initial_'+metric,'final_'+metric]].max().max()
            pad=max(.018*scale,.10*(hi-lo))
            ax.set_ylim(max(0,lo-pad),min(scale,hi+pad))
            if row==0:ax.set_title(f'{pcs} PCs / {neighbors} neighbors')
            if row==0 and col==0:ax.legend(frameon=False,fontsize=7,loc='lower right')
            ax.text(-.15,1.04,chr(65+row*4+col),transform=ax.transAxes,weight='bold',fontsize=11)
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='.9',lw=.5)
        bottom=grid[2,col].subgridspec(5,1,hspace=.22)
        for j,res in enumerate(RESOLUTIONS):
            ax=fig.add_subplot(bottom[j,0]);case=sub[sub.resolution==res].case.item()
            alluvial(ax,frames[case],*maps[case],res,headers=j==0)
            if j==0:ax.text(-.15,1.2,chr(73+col),transform=ax.transAxes,weight='bold',fontsize=11)
    fig.suptitle('Zhengmix4eq: one CellSieve pass across 20 initializations\nIndependent many-to-one cluster mapping for evaluation; 60 genes; K inferred from expression',fontsize=13)
    fig.supxlabel('Alluvials: initial clusters / FACS populations / refined clusters. Ribbon width = cell count; color = FACS type.\nCluster bars use their majority-reference color. Each resolution shows all 3,994 cells.',fontsize=9)
    for ext in ['png','pdf','svg']:fig.savefig(BASE/f'figures/Figure3_many_to_one.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    summary=dict(cases=20,accuracy_improved=int((t.accuracy_change_pp>1e-10).sum()),
        accuracy_unchanged=int((t.accuracy_change_pp.abs()<=1e-10).sum()),accuracy_worsened=int((t.accuracy_change_pp< -1e-10).sum()),
        macro_f1_improved=int((t.macro_f1_change>1e-10).sum()),macro_f1_worsened=int((t.macro_f1_change< -1e-10).sum()),
        mappings_unchanged=int(t.mapping_unchanged.sum()),models_refitted=False)
    (BASE/'results/many_to_one_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (BASE/'results/many_to_one_provenance.json').write_text(json.dumps(dict(input_sha256=hashes,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),all_flow_counts_conserved=True),indent=2)+'\n')
    (BASE/'figures/Figure3_many_to_one.md').write_text('''# Many-to-one evaluation and alluvial mapping

Two metric rows show accuracy and macro-F1 across five resolutions for each PC/neighbor pair. The bottom row has five alluvials per column, one per resolution, so all 20 cases are shown without selecting a representative result. Each alluvial shows initial clusters, FACS populations, and refined clusters, with cell-count-proportional ribbons. Cluster nodes are ordered and colored by their majority reference assignment; ribbons retain the actual FACS type, including minority cells. B = B cells, Mono = CD14 monocytes, CD8 = naive cytotoxic T cells, Treg = regulatory T cells. This is two pairwise contingency diagrams joined at FACS populations, not a direct initial-to-final cell trajectory. Exact flow counts and mapping tables are saved.

Each cluster maps to its most frequent reference population independently before and after refinement. Multiple clusters can map to one reference type. Ties are resolved by sorted reference name. Macro-F1 averages equally over all four reference populations; missing predicted types receive zero F1. Both accuracy and macro-F1 are calculated over individual cells after mapping. Ground truth is used only in this evaluation; no models, panels or cell assignments are changed. The metrics CSV also records final accuracy using the fixed initial majority mapping and whether the mapping changed.

Many-to-one accuracy is cluster purity: it deliberately does not penalize splitting one true population across several clusters, and can increase with overclustering. These results assess reference agreement after mapping, not recovery of the correct group count or held-out classifier accuracy. Preserve the earlier one-to-one CSV as a separate analysis; do not mix its scores with these.

Reproduce with:

```sh
LD_LIBRARY_PATH=/home/nikki/miniconda3/envs/benchmark_env/lib /home/nikki/miniconda3/envs/benchmark_env/bin/python /home/nikki/SP_2026/figure3_annotation/Zhengmix4eq_single_pass_grid/scripts/plot_many_to_one.py
```
''')
    print(json.dumps(summary,indent=2));print(t[['n_pcs','n_neighbors','resolution','initial_accuracy','final_accuracy','initial_macro_f1','final_macro_f1']].to_string(index=False))

if __name__=='__main__':main()
