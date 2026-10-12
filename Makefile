PYTHON ?= python
NATIVE_PYTHON ?= $(PYTHON)
FIGURE12_ALPHA ?= 0.9
export PYTHONDONTWRITEBYTECODE := 1
export MPLBACKEND := Agg
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export NUMBA_CACHE_DIR := $(CURDIR)/.cache/numba

.PHONY: help check verify verify-supplementary figures figure1 figure2 figure3 supplementary manuscript
help:
	@echo 'make check | figures | verify | manuscript: check runtime, render saved results, validate evidence, sync manuscript PDFs'
	@echo 'Explicit fitting: figure1-native | figures12-rerun | figure2-refit | figure3-screen | figure3-passes'
check:
	$(PYTHON) -B scripts/check_environment.py
verify:
	$(PYTHON) -B scripts/verify_workspace.py
	$(PYTHON) -B scripts/verify_supplementary.py
verify-supplementary:
	$(PYTHON) -B scripts/verify_supplementary.py
figures: figure1 figure2 supplementary figure3 figure2-go figure2-markers-render
figure1:
	$(PYTHON) -B scripts/fig1_method_benchmark.py
figure2:
	$(PYTHON) -B scripts/fig2_cortex_merfish.py
figure3:
	$(PYTHON) -B scripts/fig3_annotation_stories.py
supplementary:
	$(PYTHON) -B scripts/supplementary_benchmarks.py
manuscript:
	$(PYTHON) -B scripts/sync_manuscript_figures.py

# Scientific execution is explicit; ordinary rendering never fits models.
.PHONY: figure1-native figures12-rerun figure2-panel40 figure2-refit figure2-go figure2-go-whole-genome figure2-go-rerun figure2-markers figure2-markers-render figure3-screen figure3-passes
figure1-native:
	$(NATIVE_PYTHON) -B scripts/fig1_method_benchmark.py --run-native
figures12-rerun:
	bash scripts/rerun_figures12.sh --alpha $(FIGURE12_ALPHA)
figure2-panel40:
	$(PYTHON) -B scripts/build_fig2_panel40.py
	$(PYTHON) -B scripts/build_fig2_violins.py
	$(PYTHON) -B scripts/fig2_cortex_merfish.py
figure2-refit:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) -B scripts/refit_fig2_merfish.py --force
	$(PYTHON) -B scripts/fig2_cortex_merfish.py
figure2-go figure2-go-whole-genome:
	$(PYTHON) -B scripts/build_fig2_go.py
figure2-go-rerun:
	$(PYTHON) -B scripts/rerun_fig2_go.py
	$(PYTHON) -B scripts/build_fig2_go.py
figure2-markers-render:
	$(PYTHON) -B scripts/audit_fig2_panel40_markers.py --render-only
figure2-markers:
	$(PYTHON) -B scripts/audit_fig2_panel40_markers.py
figure3-screen:
	$(PYTHON) -B scripts/figure3/zheng_screen/run_screen.py --resume
	$(PYTHON) -B scripts/figure3/zheng_screen/plot_many_to_one.py
figure3-passes:
	$(PYTHON) -B scripts/figure3/zheng_passes/run_refinement.py
