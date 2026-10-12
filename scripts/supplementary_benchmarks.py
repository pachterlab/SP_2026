"""Render the current alpha sensitivity and selected-alpha results/supplementary results."""
from compile_alpha_sensitivity import main as render_sensitivity
from fullinput_benchmarks import render_saved
from figure_style import ROOT


def main():
    render_sensitivity()
    render_saved(ROOT / 'results/supplementary/fullinput_benchmarks')


if __name__ == '__main__':
    main()
