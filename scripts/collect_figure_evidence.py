"""Centralize historical/native image artifacts without changing any fit output data."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'metadata/layout_migration.json'
    manifest=json.loads(path.read_text())
    moves=manifest['relocations']
    records=manifest.setdefault('collected_image_evidence',[])
    for source in sorted((ROOT/'results').rglob('*')):
        if not source.is_file() or source.suffix.lower() not in ('.png','.svg','.pdf'):
            continue
        if source.name=='methods_fig.pdf':
            destination=ROOT/'assets/methods.pdf'
        else:
            destination=ROOT/'figs/diagnostics/evidence'/source.relative_to(ROOT/'results')
        original=str(source.relative_to(ROOT))
        target=str(destination.relative_to(ROOT))
        checksum=hashlib.sha256(source.read_bytes()).hexdigest()
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=checksum:
                raise ValueError(f'Conflicting existing figure evidence: {destination}')
            source.unlink()
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(source,destination)
        # Both the current layout and the original recorded paths resolve directly.
        for old,prefix in list(moves.items()):
            if original==prefix or original.startswith(prefix+'/'):
                moves[old+original[len(prefix):]]=target
        moves[original]=target
        records.append(dict(source=original,destination=target,sha256=checksum))
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Centralized {len(records)} preserved image artifacts in figs/diagnostics/.')


if __name__=='__main__':main()
