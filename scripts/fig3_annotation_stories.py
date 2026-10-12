#!/usr/bin/env python3
"""Render Figure 3's three experiments from saved cell-level/metric outputs only."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'results/figure3/.cache/matplotlib'))
import numpy as np
import pandas as pd
from figure_style import ROOT,plt,set_style,save
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch,Rectangle,ConnectionPatch
from matplotlib.lines import Line2D

BASE=ROOT/'results/figure3'
PCA=BASE/'cellsieve30_300s/cortex_pca/pca_visualization.npz'
# Cortex outcomes use green/red; Zhengmix settings use blue/orange/purple/gold.
RECOVERED,ERROR,MISSED='#56A582','#C43C4E','#B4B8BD'
SCREEN=BASE/'Zhengmix4eq_single_pass_grid_60genes'
ITER=BASE/'Zhengmix4eq_three_pass_60genes'
NOISE=BASE/'noise_benchmark/results/cortex_reference'
COHORT='#59636E'
TITLE_SIZE=18
PANEL_TITLE_SIZE=14


def panel(ax,letter,title):
    ax.text(-.10,1.07,letter,transform=ax.transAxes,weight='bold',fontsize=TITLE_SIZE,va='bottom')
    ax.set_title(title,loc='left',fontsize=PANEL_TITLE_SIZE,fontweight='bold',pad=14,
                 linespacing=1.15)


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='.90',lw=.6)


def noise_alluvial(ax,counts,before,after):
    """One count-proportional diagram: full cohort -> noise status -> outcomes."""
    total=counts['total_cells'];unit=.55/total;width=.022
    assert counts['poisoned']+counts['clean']==total
    outcomes=[('Restored to reference',counts['recovered'],RECOVERED,'poisoned'),
              ('Changed to another wrong label',counts['wrong_fix'],ERROR,'poisoned'),
              ('Missed injected error',counts['poison_missed'],ERROR,'poisoned'),
              ('Changed away from reference',counts['clean_flagged'],ERROR,'clean'),
              ('Unchanged',counts['clean_unflagged'],RECOVERED,'clean')]
    assert sum(x[1] for x in outcomes)==total
    def ribbon(x0,y0,x1,y1,h,color):
        mid=(x0+x1)/2
        verts=[(x0,y0),(mid,y0),(mid,y1),(x1,y1),(x1,y1+h),
               (mid,y1+h),(mid,y0+h),(x0,y0+h),(x0,y0)]
        codes=[PlotPath.MOVETO,*([PlotPath.CURVE4]*3),PlotPath.LINETO,
               *([PlotPath.CURVE4]*3),PlotPath.CLOSEPOLY]
        ax.add_patch(PathPatch(PlotPath(verts,codes),facecolor=color,alpha=.43,edgecolor='none'))
    xs=[.04,.38,.70];source_bottom=.20
    ax.add_patch(Rectangle((xs[0],source_bottom),width,.55,color=COHORT,lw=0))
    middle={};top=.88
    for cohort in ['poisoned','clean']:
        h=counts[cohort]*unit;middle[cohort]=top-h;top-=h+.11
    source_top=source_bottom+.55
    for cohort,color,label in [('poisoned',ERROR,'Injected-error cells'),('clean',RECOVERED,'Originally clean cells')]:
        h=counts[cohort]*unit;source_top-=h
        ribbon(xs[0]+width,source_top,xs[1],middle[cohort],h,color)
        ax.add_patch(Rectangle((xs[1],middle[cohort]),width,h,color=color,lw=0))
        ax.text(xs[1]-.018,middle[cohort]+h/2,f'{label}\n{counts[cohort]:,}',ha='right',va='center',fontsize=12,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
    cursors={k:y+counts[k]*unit for k,y in middle.items()}
    target_top=1.00
    changed_anchor=None
    for row,(label,count,color,cohort) in enumerate(outcomes):
        h=count*unit;y=target_top-h;cursors[cohort]-=h
        ribbon(xs[1]+width,cursors[cohort],xs[2],y,h,color)
        ax.add_patch(Rectangle((xs[2],y),width,h,color=color,lw=0))
        text_y=y+h/2
        if cohort=='clean' and label=='Changed away from reference':changed_anchor=(xs[2]+width,text_y)
        denominator='injected' if cohort=='poisoned' else 'clean'
        ax.annotate(f'{label}\n{count:,}',
                    xy=(xs[2]+width,y+h/2),xytext=(.76,text_y),fontsize=12,ha='left',va='center',
                    arrowprops=dict(arrowstyle='-',color='.55',lw=.6))
        target_top=y-.20
    ax.text(xs[0]-.018,source_bottom+.55/2,f'All cells\n{total:,}',ha='right',va='center',fontsize=14)
    ax.set(xlim=(-.06,1.08),ylim=(-.36,1.12));ax.axis('off')
    return changed_anchor


def main():
    set_style()
    plt.rcParams.update({'font.size':14,'axes.labelsize':14,'axes.titlesize':18,
                         'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
                         'mathtext.fontset':'custom','mathtext.rm':'Helvetica',
                         'mathtext.it':'Helvetica:italic','mathtext.bf':'Helvetica:bold',
                         'mathtext.sf':'Helvetica','mathtext.tt':'Helvetica',
                         'mathtext.cal':'Helvetica','mathtext.fallback':None,
                         'lines.linewidth':2.25})
    truth_path=NOISE/'truth_key.csv';flags_path=NOISE/'panel80_alpha070/flagged_misannotations.csv'
    truth=pd.read_csv(truth_path).set_index('cell_barcode',verify_integrity=True)
    flags=pd.read_csv(flags_path).set_index('cell_barcode',verify_integrity=True)
    assert flags.index.isin(truth.index).all()
    assert (flags.annotated_label==truth.loc[flags.index,'poisoned_label']).all()
    injected=truth.is_poisoned.astype(str).str.lower().eq('true')
    flagged=truth.index.isin(flags.index)
    final=truth.poisoned_label.copy();final.loc[flags.index]=flags.predicted_label
    before=truth.poisoned_label.eq(truth.true_label);after=final.eq(truth.true_label)
    assert (injected==~before).all()
    counts=dict(total_cells=len(truth),poisoned=int(injected.sum()),clean=int((~injected).sum()),
                poison_flagged=int((injected&flagged).sum()),poison_missed=int((injected&~flagged).sum()),
                clean_flagged=int((~injected&flagged).sum()),clean_unflagged=int((~injected&~flagged).sum()),
                recovered=int((injected&after).sum()),wrong_fix=int((injected&flagged&~after).sum()))
    recorded=pd.read_csv(NOISE/'targeted_alluvial_counts.csv').iloc[0]
    assert all(int(recorded[k])==v for k,v in counts.items())
    centroid_path=NOISE/'clean_false_positive_centroid_check.csv'
    centroid=pd.read_csv(centroid_path).set_index('cell_barcode',verify_integrity=True)
    expected=truth.index[(~injected)&flagged]
    assert set(centroid.index)==set(expected)
    assert centroid.checked.astype(str).str.lower().eq('true').all()
    assert (centroid.annotated_label==truth.loc[centroid.index,'true_label']).all()
    assert (centroid.predicted_label==flags.loc[centroid.index,'predicted_label']).all()
    distance_delta=centroid.annotated_centroid_distance-centroid.predicted_centroid_distance
    assert np.isfinite(distance_delta).all()
    assert np.allclose(distance_delta,centroid.distance_delta_annotated_minus_predicted)
    assert np.array_equal(distance_delta.gt(0),centroid.predicted_is_closer.astype(str).str.lower().eq('true'))
    closer=int(distance_delta.gt(0).sum());original_closer=int(distance_delta.lt(0).sum())
    centroid_summary=dict(checked=len(centroid),closer_to_assigned=closer,closer_to_original=original_closer,
        ties=int(distance_delta.eq(0).sum()),fraction_closer_to_assigned=closer/len(centroid),centroid_method='coordinate-wise median',
        reference_cells='Originally unperturbed and unflagged cells; at least 20 per reference type',
        metric='Euclidean distance in saved PCA space')
    grid=pd.read_csv(SCREEN/'results/many_to_one_metrics.csv')
    run=pd.read_csv(ITER/'results/metrics.csv')
    # Display the user-requested first three passes; retain the full source run.
    run=run.loc[run['round'].between(0,3)].copy()
    assert len(grid)==20 and not grid.duplicated(['n_pcs','n_neighbors','resolution']).any()
    assert run['round'].tolist()==list(range(4)) and run.mapping_unchanged.all()
    assert run.n_clusters.nunique()==1
    followup=grid.loc[grid.case=='pcs30_knn10_res0.6'].iloc[0]
    assert np.isclose(run.accuracy.iloc[0],followup.initial_accuracy)
    assert np.isclose(run.accuracy.iloc[1],followup.final_accuracy)
    assert np.isclose(run.macro_f1.iloc[1],followup.final_macro_f1)
    fig=plt.figure(figsize=(16,9))
    screen_axes=[fig.add_axes([x,.085,.215,.37]) for x in [.055,.325]]
    iterative_ax=fig.add_axes([.75,.085,.225,.37])
    noise_ax=fig.add_axes([.055,.575,.64,.34])
    changed_anchor=noise_alluvial(noise_ax,counts,float(before.mean()),float(after.mean()))
    for text in noise_ax.texts:text.set_fontsize(14)
    panel(noise_ax,'A','Cell Sieve celltype recovery after injected noise (B6J cortex)')
    noise_ax.texts[-1].set_position((-.035,1.07))
    centroid_ax=fig.add_axes([.75,.675,.225,.24])
    # Show the paired difference across all PCA dimensions directly.
    bound=float(np.max(np.abs(distance_delta)))*1.05
    edges=np.linspace(-bound,bound,41)
    values,edges=np.histogram(distance_delta,bins=edges)
    centers=(edges[:-1]+edges[1:])/2
    assert values.sum()==len(centroid)
    assert values[centers>0].sum()==closer
    colors=np.where(centers>0,RECOVERED,'#767676')
    centroid_ax.bar(centers,values,width=np.diff(edges)*.95,color=colors,edgecolor='none')
    centroid_ax.axvline(0,color='.3',lw=1,ls='--')
    centroid_ax.set(xlim=(-bound,bound),xlabel=r'$d_{\mathrm{original}} - d_{\mathrm{assigned}}$',ylabel='Reassigned cells')
    centroid_ax.set_ylim(0,max(values)*1.15)
    centroid_ax.set_xticks([-2,-1,0,1,2])
    clean(centroid_ax);panel(centroid_ax,'B','Proximity of reassigned cells\nto PCA centroids (B6J cortex)')
    # Place the short arrow after the rendered first line of the branch label.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    branch_label=next(t for t in noise_ax.texts if t.get_text().startswith('Changed away from reference'))
    bounds=branch_label.get_window_extent(renderer)
    line_width=renderer.get_text_width_height_descent('Changed away from reference',
        branch_label.get_fontproperties(),ismath=False)[0]
    point=fig.dpi/72
    label_x=branch_label.get_transform().transform(branch_label.get_position())[0]
    arrow_x=label_x+line_width+5*point
    arrow_y=bounds.y1-.55*branch_label.get_fontsize()*point
    start=fig.transFigure.inverted().transform((arrow_x,arrow_y))
    end=fig.transFigure.inverted().transform((arrow_x+22*point,arrow_y))
    connector=ConnectionPatch(xyA=start,coordsA=fig.transFigure,
        xyB=end,coordsB=fig.transFigure,arrowstyle='-|>',
        mutation_scale=13,lw=1.8,color=ERROR,clip_on=False,shrinkA=0,shrinkB=0)
    fig.add_artist(connector)
    noise_ax.add_patch(Rectangle((changed_anchor[0]-.03,changed_anchor[1]-.045),.055,.09,
        fill=False,edgecolor=ERROR,lw=1.5,zorder=5))

    pairs=[(20,10),(20,30),(30,10),(30,30)]
    colors=['#266A9E','#CA7425','#9467A4','#B79A20'];markers=['o','s','^','D']
    for ax,metric,title,letter in zip(screen_axes,['accuracy','macro_f1'],
        ['Accuracy before and after refinement\n(Zhengmix4eq)',
         'Macro-F1 before and after refinement\n(Zhengmix4eq)'],'CD'):
        factor=1
        scores=grid[['initial_'+metric,'final_'+metric]].to_numpy()
        lo=float(scores.min());hi=float(scores.max());pad=.08*(hi-lo)
        for (pcs,neighbors),color,marker,offset in zip(pairs,colors,markers,[-.045,-.015,.015,.045]):
            t=grid.query('n_pcs==@pcs and n_neighbors==@neighbors').sort_values('resolution')
            x=t.resolution+offset
            original=factor*t['initial_'+metric];refined=factor*t['final_'+metric]
            ax.plot(np.vstack([x-.006,x+.006]),np.vstack([original,refined]),color=color,lw=1.6,alpha=.75)
            ax.plot(x-.006,original,marker,ms=6,color=color,markerfacecolor='white',markeredgewidth=1.5,linestyle='none')
            ax.plot(x+.006,refined,marker,ms=6,color=color,linestyle='none',label=f'{pcs} PCs / {neighbors} neighbors')
        ax.set(xlabel='Leiden resolution',ylabel='Accuracy' if metric=='accuracy' else 'Macro-F1',
               xticks=[.2,.4,.6,.8,1.0],xlim=(.12,1.08))
        ax.set_ylim(factor*(lo-pad),factor*(hi+pad))
        clean(ax);panel(ax,letter,title)
    iterative_ax.plot(run['round'],run.macro_f1,color=colors[2],marker=markers[2],
                      linestyle='-',ms=9,label='30 PCs / 10 neighbors\nLeiden resolution 0.6')
    scores=run.macro_f1.to_numpy()
    pad=max(.001,.23*(scores.max()-scores.min()))
    iterative_ax.set(xticks=range(4),xlim=(-.2,3.2),xlabel='CellSieve pass',
                     ylabel='Macro-F1',ylim=(scores.min()-pad,scores.max()+pad))
    for r in run.itertuples():
        iterative_ax.annotate(f'{r.macro_f1:.4f}',(r.round,r.macro_f1),xytext=(0,12),
                              textcoords='offset points',ha='center',fontsize=14)
    clean(iterative_ax);panel(iterative_ax,'E','Macro-F1 after iterative CellSieve\npasses (60-gene model)\n(Zhengmix4eq)')
    iterative_ax.legend(loc='lower right',frameon=False,fontsize=12)
    handles,labels=screen_axes[0].get_legend_handles_labels()
    handles.extend([Line2D([],[],marker='o',color='.3',markerfacecolor='white',linestyle='none',markersize=6),
                    Line2D([],[],marker='o',color='.3',linestyle='none',markersize=6)])
    labels.extend(['Original','Refined'])
    screen_axes[1].legend(handles,labels,loc='center left',bbox_to_anchor=(1.035,.5),
                     ncol=1,frameon=False,fontsize=12,borderaxespad=0,labelspacing=1.1,
                     handlelength=1.8,handletextpad=.6)
    save(fig,'results/figure3/exports','Figure3')
    results=BASE/'results';results.mkdir(exist_ok=True)
    pd.DataFrame(dict(bin_left=edges[:-1],bin_right=edges[1:],cell_count=values,closer_center=np.where(centers>0,'assigned','original'))).to_csv(results/'figure3_centroid_histogram.csv',index=False)
    grid.to_csv(results/'figure3_unsupervised_screen.csv',index=False)
    run.to_csv(results/'figure3_three_pass_metrics.csv',index=False)
    pd.DataFrame([dict(counts,accuracy_before=before.mean(),accuracy_after=after.mean())]).to_csv(results/'figure3_noise_counts.csv',index=False)
    centroid.assign(distance_advantage=distance_delta).to_csv(results/'figure3_centroid_distances.csv')
    (results/'figure3_centroid_summary.json').write_text(json.dumps(centroid_summary,indent=2)+'\n')
    paths=[SCREEN/'config.json',ITER/'config.json',PCA,BASE/'cellsieve30_300s/cortex_pca/pca_method.json',truth_path,flags_path,centroid_path,ROOT/'scripts/figure3/noise_source/poison_expt.py',NOISE/'targeted_alluvial_counts.csv',SCREEN/'results/many_to_one_metrics.csv',ITER/'results/metrics.csv']
    (results/'figure3_provenance.json').write_text(json.dumps(dict(models_refitted_by_renderer=False,umap_used=False,displayed_passes=[0,1,2,3],
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        panel_size_CD_EF=60,solver_time_limit_seconds=300,followup_condition='30 PCs / 10 neighbors / resolution 0.6; held fixed from original 40-gene screen'),indent=2)+'\n')
    summary=dict(centroid_closer_to_assigned=closer,centroid_checked=len(centroid),noise_recovered=counts['recovered'],screen_accuracy_improved=int((grid.final_accuracy>grid.initial_accuracy).sum()),
        screen_macro_f1_improved=int((grid.final_macro_f1>grid.initial_macro_f1).sum()),
        iterative_initial_accuracy=float(run.accuracy.iloc[0]),iterative_final_accuracy=float(run.accuracy.iloc[-1]),
        iterative_initial_macro_f1=float(run.macro_f1.iloc[0]),iterative_final_macro_f1=float(run.macro_f1.iloc[-1]),
        accuracy_improves_every_pass=bool((run.accuracy.diff().dropna()>0).all()))
    (results/'figure3_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
