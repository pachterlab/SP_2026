"""Render successful rows from six extant tissue comparisons without filling gaps."""
import pandas as pd
from figure_style import ROOT,plt,set_style,label,save,benchmark_curves,METHODS

def main():
    set_style();tables=sorted((ROOT/'supplementary/all_tissue_benchmarks/results/multimethod').glob('*/benchmark_summary_all.csv'))
    for metric,stem in [('cv_macro_f1','Supplement_multitissue_macro_F1'),('cv_accuracy','Supplement_multitissue_accuracy')]:
        fig,axes=plt.subplots(2,3,figsize=(11,6.4))
        for ax,path,letter in zip(axes.flat,tables,'ABCDEF'):
            df=pd.read_csv(path);benchmark_curves(ax,df,metric)
            ax.set_title(path.parent.name.replace('_alpha_',' · alpha '));label(ax,letter)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.07),ncol=5,frameon=False)
        fig.subplots_adjust(left=.08,right=.98,bottom=.24,top=.92,hspace=.5,wspace=.4)
        fig.text(.08,.015,'Existing 3,000-gene-universe runs; conditional classifier CV after full-data panel selection.\nCOMET failed in all six tissue runs and is absent from these curves. Heart / Diencephalon multi-method results are missing.',fontsize=8)
        save(fig,'supplementary/all_tissue_benchmarks/figures',stem)
if __name__=='__main__':main()
