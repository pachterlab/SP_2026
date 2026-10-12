"""Recompute panel40 GO enrichment from frozen, local GO Consortium inputs."""
import gzip
import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / 'results/figure2/resources/go'
PANEL = ROOT / 'results/figure2/results/biology_panel40_fig1/selected_genes.txt'
DEST = PANEL.parent / 'go_analysis/whole_mouse_genome'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ontology(path):
    terms, aliases, record = {}, {}, {}
    version = None
    def finish():
        if record.get('id', '').startswith('GO:') and not record.get('obsolete'):
            terms[record['id']] = record.copy()
            for alias in record.get('aliases', []):
                aliases[alias] = record['id']
    for line in path.read_text().splitlines():
        if line.startswith('data-version: '):
            version = line.split(': ', 1)[1]
        if line.startswith('['):
            finish()
            record = {}
        elif ': ' in line:
            key, value = line.split(': ', 1)
            if key in ('id', 'name', 'namespace'):
                record[key] = value
            elif key == 'alt_id':
                record.setdefault('aliases', []).append(value)
            elif key == 'is_obsolete' and value == 'true':
                record['obsolete'] = True
            elif key == 'is_a':
                record.setdefault('parents', []).append(value.split()[0])
            elif key == 'relationship' and value.startswith('part_of '):
                record.setdefault('parents', []).append(value.split()[1])
    finish()
    return terms, aliases, version


def main():
    manifest = json.loads((RESOURCE / 'sources.json').read_text())
    for name, entry in manifest['files'].items():
        if sha(RESOURCE / name) != entry['sha256']:
            raise ValueError(f'Frozen GO input hash mismatch: {name}')
    terms, aliases, ontology_version = read_ontology(RESOURCE / 'go-basic.obo')
    @lru_cache(None)
    def ancestors(term):
        return frozenset([term]).union(*(ancestors(aliases.get(p, p))
            for p in terms[term].get('parents', []) if aliases.get(p, p) in terms))

    gene_terms, symbols, synonyms = defaultdict(set), {}, defaultdict(set)
    rejected = defaultdict(int)
    headers = []
    with gzip.open(RESOURCE / 'mgi.gaf.gz', 'rt') as handle:
        for line in handle:
            if line.startswith('!'):
                headers.append(line.strip())
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 17:
                raise ValueError('Malformed GAF annotation')
            db, gid, symbol, qualifier, term, _, evidence = fields[:7]
            if fields[12] != 'taxon:10090':
                rejected['non_mouse'] += 1
                continue
            if 'NOT' in qualifier.split('|') or evidence == 'ND':
                rejected['NOT_or_ND'] += 1
                continue
            term = aliases.get(term, term)
            if term not in terms:
                rejected['obsolete_or_missing_term'] += 1
                continue
            key = gid if gid.startswith(db + ':') else f'{db}:{gid}'
            symbols[key] = symbol
            synonyms[symbol].add(key)
            for alias in fields[10].split('|'):
                if alias:
                    synonyms[alias].add(key)
            gene_terms[key].update(ancestors(term))
    canonical = defaultdict(set)
    for gid, symbol in symbols.items():
        canonical[symbol].add(gid)
    genes = PANEL.read_text().splitlines()
    if len(genes) != 40 or len(set(genes)) != 40:
        raise ValueError('Expected exactly 40 unique selected genes')
    mappings, selected = [], set()
    for gene in genes:
        matches = canonical.get(gene) or synonyms.get(gene, set())
        if len(matches) > 1:
            raise ValueError(f'Ambiguous gene mapping: {gene}: {matches}')
        gid = next(iter(matches), None)
        mappings.append({'selected_gene': gene, 'gene_id': gid,
                         'canonical_symbol': symbols.get(gid),
                         'mapping': 'exact' if canonical.get(gene) else 'synonym' if gid else 'unannotated'})
        if gid:
            selected.add(gid)
    if len(selected) != sum(m['gene_id'] is not None for m in mappings):
        raise ValueError('Multiple panel symbols map to the same gene')
    term_genes = defaultdict(set)
    for gid, annotations in gene_terms.items():
        for term in annotations:
            term_genes[term].add(gid)
    # Test every non-root term with mouse annotations, including zero panel hits.
    roots = {'GO:0008150', 'GO:0003674', 'GO:0005575'}
    rows = []
    gene_labels = {m['gene_id']: m['selected_gene'] for m in mappings if m['gene_id']}
    N, n = len(gene_terms), len(selected)
    for term, population in term_genes.items():
        if term in roots:
            continue
        hit_ids = selected & population
        k, K = len(hit_ids), len(population)
        rows.append({'go_id': term, 'term': terms[term]['name'],
                     'namespace': terms[term]['namespace'],
                     'panel_hits': k, 'panel_size': n,
                     'background_hits': K, 'background_size': N,
                     'fold_enrichment': (k/n)/(K/N),
                     'p_value': float(hypergeom.sf(k-1, N, K, n)),
                     'genes': ';'.join(sorted(gene_labels[g] for g in hit_ids))})
    results = pd.DataFrame(rows).sort_values(['p_value', 'go_id']).reset_index(drop=True)
    m = len(results)
    adjusted = results.p_value.to_numpy() * m / np.arange(1, m+1)
    results['fdr_bh'] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    DEST.mkdir(parents=True, exist_ok=True)
    results.to_csv(DEST / 'all_go_results.csv', index=False)
    pd.DataFrame(mappings).to_csv(DEST / 'panel_gene_mapping.csv', index=False)
    (DEST / 'background_gene_ids.txt').write_text('\n'.join(sorted(gene_terms))+'\n')
    pd.DataFrame([{'gene_id': gid, 'symbol': symbols[gid], 'go_id': term}
                  for gid in sorted(selected) for term in sorted(gene_terms[gid])]).to_csv(
                      DEST / 'panel_go_annotations.csv', index=False)
    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'cell_sieve_alpha': json.loads((PANEL.parent / 'provenance.json').read_text())['cell_sieve_alpha'],
        'selected_genes_sha256': sha(PANEL), 'selected_gene_count': len(genes),
        'annotated_panel_size': n, 'unannotated_genes': [r['selected_gene'] for r in mappings if r['gene_id'] is None],
        'background_size': N, 'tested_terms': m, 'ontology_version': ontology_version,
        'background': 'All mouse MGI gene IDs with at least one usable annotation across BP/MF/CC',
        'test': 'One-sided hypergeometric survival function P(X >= observed overlap)',
        'multiple_testing': 'Benjamini-Hochberg across all tested BP/MF/CC terms, including zero-hit terms',
        'annotations': 'Mouse only; exclude NOT and ND; propagate is_a and part_of; retain other evidence codes',
        'table_terms': ['GO:0007399', 'GO:0048731', 'GO:0048856', 'GO:0007275', 'GO:0032502'],
        'term_selection': 'Same five developmental terms as previous manuscript table; not selected by new P values',
        'sources': manifest, 'gaf_header': headers, 'excluded_annotation_rows': dict(rejected),
        'script_sha256': sha(Path(__file__)),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
        'artifacts': {name: sha(DEST / name) for name in ['all_go_results.csv', 'panel_gene_mapping.csv', 'background_gene_ids.txt', 'panel_go_annotations.csv']}}
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'{N:,} background genes; {n}/40 annotated panel genes; {m:,} terms tested')
    print('Unannotated:', provenance['unannotated_genes'])
    print(results[results.go_id.isin(provenance['table_terms'])][['go_id', 'panel_hits', 'p_value', 'fdr_bh']].to_string(index=False))


if __name__ == '__main__':
    main()
