# Repository Cleanup Summary

## Purpose

This note records the final cleanup pass performed before report writing. The goal was to keep only the files and outputs that support the final coursework workflow:

- rerunning the final experiment pipeline
- regenerating report tables
- regenerating final report figures
- understanding the evidence used in the paper

## Final Core Workflow

### Main experiment pipeline

- `run_experiment.py`
- `src/data.py`
- `src/preprocess.py`
- `src/representation.py`
- `src/metrics.py`
- `src/evaluator.py`
- `src/baseline.py`
- `src/runner.py`
- `src/optimizers/ga.py`
- `src/optimizers/pso.py`
- `src/optimizers/sa.py`

### Configs kept for reproducibility

- `config/experiment.yaml`
- `config/experiment_local_ga.yaml`
- `config/experiment_local_pso.yaml`
- `config/experiment_local_sa.yaml`
- `config/experiment_robustness_b30.yaml`
- `config/experiment_robustness_b30_seed1.yaml`
- `config/experiment_robustness_b30_seed2.yaml`
- `config/ga_tuning_base_b20.yaml`
- `config/ga_tuning_mut002_b20.yaml`
- `config/ga_tuning_pop10_b20.yaml`
- `config/ga_tuning_pop15_b20.yaml`

### Report-generation scripts kept

- `scripts/generate_report_tables.py`
- `scripts/make_additional_plots.py`

### Final report outputs kept

- `docs/generated/summary_main.csv`
- `docs/generated/summary_robustness.csv`
- `docs/generated/paper_table_main.md`
- `docs/generated/paper_table_robustness.md`
- `results/figures/*_revised.png`
- `results/figures/revision_notes.md`

## Removed

### Clearly obsolete or duplicated tracked files

- `docs/figures/`
  - old notebook-generated figure copies duplicated by the final `results/figures/*_revised.png`
- `docs/generated_test/`
  - obsolete duplicate test export folder not used in the final workflow
- `config/ga_tuning_mut010_b20.yaml`
  - abandoned tuning config; no corresponding run outputs or documented final use

### Clearly obsolete local/generated artifacts

- `results/figures/additional_plots_summary.md`
  - superseded by the final report docs and `revision_notes.md`
- all `all_runs_incremental.csv`
  - intermediate append-mode checkpoints, not needed once final `all_runs.csv` exists
- all per-run `results/**/plots/`
  - old run-local diagnostic plots replaced by the final report figures
- empty legacy folders:
  - `results/raw/`
  - `results/plots/`
  - `results/convergence/`
- local cache folders:
  - `__pycache__/`
  - `scripts/__pycache__/`
- local notebook-only plotting files not part of the final workflow:
  - `notebooks/02_report_figures.ipynb`
  - `notebooks/02_report_figures.executed.ipynb`
  - `notebooks/analysis_plots.ipynb`
- local large zip archives not needed for the final report workflow:
  - `GeneratedLabelledFlows.zip`
  - `MachineLearningCSV.zip`

## Kept Intentionally Despite Looking Optional

- `notebooks/00_colab_run.ipynb`
- `notebooks/01_colab_single_notebook.ipynb`
  - kept because they remain part of the documented reproducibility path for Colab users

- `results/ga_tuning/`
- `docs/ga_tuning_plan.md`
- `docs/ga_tuning_results.md`
  - kept because the lightweight GA tuning process is referenced in the report-supporting documentation

- `results/robustness/b30_seed1/` and `results/robustness/b30_seed2/`
  - kept because the final robustness figures and aggregated tables depend on those outputs directly

- `Assessment Sheet COMP2024 Coursework Spring 2026.pdf`
  - not part of the tracked repo workflow, but intentionally left in place locally because it is still used as the marking authority during report writing

## Validation After Cleanup

The following lightweight checks were kept feasible after cleanup:

- Python source and scripts remain importable
- final raw result paths referenced by the README still exist
- final figure generation still points to valid robustness outputs
- final revised figures remain under `results/figures/`

## Remaining Ambiguity

- The Colab notebooks are probably not needed for the final report-only workflow, but they were kept because they are still documented reproducibility routes.
- The GA tuning artifacts may not be needed in the final submission package, but they were kept because they support the written discussion of tuning decisions.
