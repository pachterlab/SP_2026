"""Compatibility audit entry point for the legacy alpha-0.7 Figure 1/2 run."""
import json
from workspace_paths import ROOT
from finalize_figures12 import main

if __name__ == '__main__':
    pointer = json.loads((ROOT / 'results/figure1/results/latest_cellsieve_batch.json').read_text())
    if pointer.get('alpha') != 0.7:
        raise SystemExit('The active run is not alpha 0.7. Use finalize_figures12.py instead.')
    main()
