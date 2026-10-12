# Paper runtime

`benchmark.yml` creates `sp2026-paper` with Python 3.11.13, Ghostscript and the
47 pinned Python distributions needed by the current paper workflows, including
transitive requirements. Pins were collected from the original `benchmark_env`;
`paper-requirements.lock.txt` gives the same Python graph. The dedicated local
`sp2026-paper` environment was cloned from the original execution environment to
retain its numerical libraries, then checked against these pins. The original
environment is unchanged. Extra inherited packages are not paper dependencies
and are not included in the portable YAML.

```bash
conda env create -f environments/benchmark.yml
conda activate sp2026-paper
bash scripts/setup_benchmark_runtime.sh
bash scripts/paper.sh check
```

The environment check verifies exact pins, imports, Helvetica, Ghostscript,
compiled XL-mHG and required local input/result files. It writes
`metadata/environment_check.json`; it does not solve a Gurobi model or fit data.
The recorded local Conda package inventory is `paper-conda-packages.json`.

COMET has its own NumPy 1.23.5 runtime because compiled XL-mHG and the preserved
COMET implementation are incompatible with the main NumPy 2 runtime. Its exact
requirements are `comet-requirements.txt`; the native Linux x86-64 Python 3.11
wheel is in `scripts/native/vendor/`. Other platforms need a
compatible compiled build of that same implementation. No statistical fallback
is substituted. Do not install COMET's dependencies into the main environment.

A Gurobi license is required for fitting and is supplied separately. Ghostscript
rasterizes the Figure 1 methods PDF. The original system Ghostscript is 9.50;
a newly solved YAML may supply a newer version, affecting rasterized artwork
without changing scientific results. Helvetica is loaded from local assets or
`CELLSIEVE_FONT_DIR`; font binaries are not in Git.

Set `BENCHMARK_ENV_DIR` to override launcher selection. Activating the paper
environment also works. Child processes inherit its library path and use only
workspace caches. TeX/Overleaf is separate and is needed only to compile the
manuscript, not to generate the analysis figures.
