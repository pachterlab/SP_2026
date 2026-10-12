"""Move local H5AD payloads into h5ads/, retaining working relative links."""
from pathlib import Path
import csv
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'h5ads'
records=[]
for source in sorted(ROOT.rglob('*.h5ad')):
    if source.is_symlink() or source.is_relative_to(DATA):
        continue
    target=DATA/'analysis_data'/'consolidated'/source.relative_to(ROOT)
    assert not target.exists(), target
    before=source.stat()
    target.parent.mkdir(parents=True,exist_ok=True)
    source.rename(target)
    source.symlink_to(os.path.relpath(target,source.parent))
    after=source.stat()
    assert (before.st_dev,before.st_ino,before.st_size)==(after.st_dev,after.st_ino,after.st_size)
    records.append({'original_path':str(source.relative_to(ROOT)),
                    'h5ad_path':str(target.relative_to(ROOT)), 'bytes':after.st_size})
path=ROOT/'metadata/h5ad_consolidation.csv'
if records:
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
all_files=list(ROOT.rglob('*.h5ad'))
assert all(p.exists() for p in all_files)
assert all(p.is_symlink() or p.is_relative_to(DATA) for p in all_files)
assert all(p.resolve().is_relative_to(DATA) for p in all_files)
with (DATA/'file_inventory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['file','bytes']);w.writeheader()
    for p in sorted(DATA.rglob('*.h5ad')):
        if not p.is_symlink():w.writerow({'file':str(p.relative_to(DATA)),'bytes':p.stat().st_size})
print(f'Moved {len(records)} files; all {len(all_files)} H5AD paths resolve inside h5ads/.')
