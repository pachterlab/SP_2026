"""Compatibility entry point: rerun Figures 1 and 2 at the legacy alpha 0.7.

For a different alpha use rerun_figures12.sh --alpha VALUE.
"""
import sys
from rerun_figures12 import main

if __name__ == '__main__':
    main(['--alpha', '0.7', *sys.argv[1:]])
