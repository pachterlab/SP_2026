"""Presentation only; follows SEP_2026 typography/layout without scientific code."""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[1]

def set_style():
    fontdir = Path(os.environ.get('CELLSIEVE_FONT_DIR', str(ROOT / 'assets/fonts/Helvetica')))
    if fontdir.is_dir():
        for p in sorted(fontdir.iterdir()):
            if p.suffix.lower() in ('.ttf','.otf','.ttc'):
                font_manager.fontManager.addfont(str(p))
    family = None
    for candidate in ('Helvetica', 'Helvetica LT Std'):
        try:
            p = font_manager.findfont(font_manager.FontProperties(family=[candidate]), fallback_to_default=False)
            if font_manager.FontProperties(fname=p).get_name() == candidate:
                family = candidate
                break
        except ValueError:
            pass
    if family is None:
        raise RuntimeError('Helvetica required. Set CELLSIEVE_FONT_DIR to a directory with Helvetica TTF/OTF files.')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[family], 'font.size':8,
        'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,
        'ytick.labelsize':7,'axes.linewidth':0.7,'lines.linewidth':1.2,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':'tight'})
    return family

def label(ax, text):
    ax.text(-.12,1.05,text,transform=ax.transAxes,weight='bold',fontsize=11,va='bottom')

def save(fig, directory, stem):
    directory=ROOT/'figs' if stem in {'Figure1','Figure2','Figure3','Supp1','Supp2','Supp3','Supp4'} else ROOT/'figs/diagnostics'
    directory.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png'):
        fig.savefig(directory/(stem+'.'+ext),dpi=350)
    plt.close(fig)

METHODS={'cell_sieve':('CellSieve v4','#266a9e'),'scgenefit':('scGeneFit','#ca7425'),
    'comet':('COMET*','#56a582'),'nsforest':('NS-Forest','#9467a4'),
    'l1_regularization':('L1 ranking','#bd5366'),'logistic_regression_de':('L2 ranking','#767676')}

def benchmark_curves(ax, table, metric):
    """Plot only successful saved values; never fill failures or average versions."""
    for method,(name,color) in METHODS.items():
        d=table[(table.method==method)&(table.status=='ok')].sort_values('actual_panel_size')
        if len(d):
            ax.plot(d.actual_panel_size,d[metric],'-o',color=color,label=name,ms=3)
    ax.set(xlabel='Selected genes (actual panel size)',ylabel={'cv_accuracy':'Conditional CV accuracy','cv_macro_f1':'Conditional CV macro-F1'}[metric],ylim=(0,1.02))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='.9',linewidth=.5)
