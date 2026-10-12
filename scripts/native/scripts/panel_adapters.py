"""Native ranked-output panel adapters authorized by the user; no replacement selectors."""
from pathlib import Path
import json,sys,subprocess
import numpy as np
import pandas as pd
from native_io import BASE,CODE,RUNTIME,environment


def combine_native_ranks(rows,path):
    rows=rows.sort_values(['native_rank','tie_score','gene','cluster'],kind='stable')
    rows.to_csv(path/'native_rank_evidence.csv',index=False)
    ranking=rows.drop_duplicates('gene',keep='first').reset_index(drop=True)
    ranking.insert(0,'global_panel_rank',np.arange(1,len(ranking)+1))
    ranking.to_csv(path/'native_global_ranking.csv',index=False)
    return ranking.gene.astype(str).tolist()


def nsforest_ranking(adata,column,workers,path,max_panel=160):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    cached=path/'native_global_ranking.csv'
    if cached.exists():return pd.read_csv(cached).gene.astype(str).tolist()
    sys.path.insert(0,str(RUNTIME/'vendor/NSForest'));import nsforest as ns
    data=adata.copy();data.obs[column]=data.obs[column].astype('category')
    data=ns.pp.prep_medians(data,column);data=ns.pp.prep_binary_scores(data,column)
    result=ns.nsforesting.NSForest(data,column,n_jobs=workers,n_top_genes=max_panel,n_binary_genes=max_panel,
        save=True,save_supplementary=True,output_folder=str(path)+'/',outputfilename_prefix='native_nsforest')
    supp=pd.read_csv(path/'native_nsforest_supplementary.csv')
    rows=pd.DataFrame({'cluster':supp.clusterName.astype(str),'gene':supp.binary_genes.astype(str),
        'native_score':supp.binary_score,'native_rf_importance':supp.rf_feature_importance})
    rows['native_rank']=rows.groupby('cluster',sort=False).cumcount()+1;rows['tie_score']=-rows.native_score
    if not set(rows.gene)<=set(adata.var_names):raise ValueError('Unexpected NS-Forest gene')
    rank=combine_native_ranks(rows,path)
    (path/'adapter.json').write_text(json.dumps({'engine':'Authors NSForest API','native_module':ns.__file__,
        'n_top_genes':max_panel,'n_binary_genes':max_panel,'n_genes_eval':6,'n_trees':1000,
        'expected_classes':sorted(adata.obs[column].astype(str).unique().tolist()),'returned_classes':sorted(rows.cluster.unique().tolist()),
        'rule':'Best native within-class binary rank; ties by native binary score descending then gene name. Deduplicate; take first k. No external filler.',
        'interpretation':'Native-ranking budget adaptation; native minimal-combination outputs retained separately','available_ranked_genes':len(rank)},indent=2)+'\n')
    return rank


def comet_ranking(adata,labels,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    cached=path/'native_global_ranking.csv'
    if cached.exists():return pd.read_csv(cached).gene.astype(str).tolist()
    if adata.n_obs>65536:raise ValueError('Native COMET cell-count limit exceeded')
    ids=[f'G{i:05d}' for i in range(adata.n_vars)];lookup=dict(zip(ids,adata.var_names.astype(str)))
    cells=[f'C{i:06d}' for i in range(adata.n_obs)];classes=[f'T{int(x):03d}' for x in labels]
    matrix=adata.X.toarray() if hasattr(adata.X,'toarray') else np.asarray(adata.X)
    pd.DataFrame(matrix.T,index=ids,columns=cells).to_csv(path/'markers.tsv',sep='\t')
    pd.DataFrame({'cell':cells,'cluster':classes}).to_csv(path/'clusters.tsv',sep='\t',header=False,index=False)
    pd.DataFrame({'native_gene':ids,'original_gene':list(lookup.values())}).to_csv(path/'gene_mapping.csv',index=False)
    env=environment(BASE/'.venv_comet/bin/python')
    subprocess.run([sys.executable,'-B',str(CODE/'run_comet_native.py'),'--markers',str(path/'markers.tsv'),
        '--clusters',str(path/'clusters.tsv'),'--out',str(path/'cli')],check=True,env=env)
    evidence=[]
    for cls in sorted(set(classes)):
        files=list((path/'cli/native').rglob(f'cluster_{cls}_singleton_all_ranked.csv'))
        if len(files)!=1:raise ValueError('Missing native COMET ranking '+cls)
        for row in pd.read_csv(files[0]).itertuples():
            gene=str(row.gene_1);symbol=gene.removesuffix('_negation')
            if symbol not in lookup:raise ValueError('Unknown native COMET gene')
            evidence.append({'cluster':cls,'gene':lookup[symbol],'native_gene':gene,'native_rank':int(row.rank),
                'native_score':float(row.mHG_stat),'tie_score':float(row.mHG_stat),'source_table':str(files[0])})
    rank=combine_native_ranks(pd.DataFrame(evidence),path)
    (path/'adapter.json').write_text(json.dumps({'engine':'Authors COMET CLI','ranking':'Native singleton_all_ranked final rank',
        'rule':'Best native class rank; ties by native mHG_stat ascending then gene name. Negations map to measured genes with polarity retained in evidence. Take first k.',
        'native_options':{'K':2,'Abbrev':'2','Trim':2000,'Down':False},'available_ranked_genes':len(rank)},indent=2)+'\n')
    return rank


def take_panel(ranking,k):
    if len(ranking)<k:raise ValueError(f'Native ranking has {len(ranking)} genes; cannot supply {k}; no filler permitted')
    return ranking[:k]
