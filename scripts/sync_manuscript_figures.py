"""Validate the manuscript's shared central figures and record their hashes."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=[]
    for name in ['Figure1','Figure2','Figure3','Supp1','Supp2','Supp3','Supp4']:
        path=ROOT/'figs'/f'{name}.pdf'
        if not path.is_file():raise FileNotFoundError(path)
        manifest.append(dict(figure=name,source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    out=ROOT/'results/manuscript';out.mkdir(parents=True,exist_ok=True)
    (out/'figure_sources.json').write_text(json.dumps({'files':manifest,'policy':'Manuscript references figs/ directly; no duplicate exports'},indent=2)+'\n')
    print('Validated seven central manuscript figure PDFs; no copies created.')

if __name__=='__main__':main()
