"""Resolve preserved historical paths to checksum-verified workspace copies."""
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLSIEVE = ROOT / 'external/cell_sieve_package/cellsieve.py'


@lru_cache(maxsize=1)
def preserved_paths():
    manifest = ROOT / 'archive_notes/self_contained_inputs.json'
    return {r['source']: ROOT / r['local'] for r in json.loads(manifest.read_text())['files']}


def local_path(value):
    """Retain original provenance strings while reading their local copies."""
    path = Path(value)
    name = str(path)
    mapped = preserved_paths().get(name)
    if mapped is not None:
        return mapped.resolve()
    old_root = '/home/nikki/SP_2026'
    if name == old_root or name.startswith(old_root + '/'):
        return (ROOT / path.relative_to(old_root)).resolve()
    if name.startswith('/home/nikki/celltype_compression_2/'):
        raise FileNotFoundError('Historical input has no preserved local copy: ' + name)
    return path.resolve()
