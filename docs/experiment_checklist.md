# Experiment Checklist

Use this checklist before and after every optimizer run.

## Pre-run

- Confirm dataset files exist:
  - `dataset/UNSW_NB15_training-set.csv`
  - `dataset/UNSW_NB15_testing-set.csv`
- Confirm the intended config file is selected:
  - `config/experiment_local_ga.yaml`
  - `config/experiment_local_pso.yaml`
  - `config/experiment_local_sa.yaml`
- Confirm `seed=0`
- Confirm `evaluations_B=50` unless this run is a deliberate follow-up adjustment
- Confirm only one optimizer is enabled in `run.enabled_algorithms`
- Confirm `output.results_dir` points to a unique run folder under `results/`
- Confirm tuning decisions will be based primarily on validation metrics and convergence, not repeated test chasing

## Run completion

- `summary.csv` exists
- `raw/all_runs.csv` exists
- `raw/all_runs_incremental.csv` exists
- `convergence/` contains a CSV for the active optimizer
- `notes.txt` exists

## Post-run review

- Record the main purpose of the run in `notes.txt`
- Record the runtime and any memory issues in `notes.txt`
- Copy the key metrics into the experiment overview table:
  - `val_best_score`
  - `test_recall`
  - `test_fpr`
  - `test_f1`
  - `test_selected_features`
  - `test_runtime_sec`
- Inspect convergence:
  - If still improving near the end, consider `evaluations_B: 80`
  - If flat early, do not increase budget first; inspect fitness or search space
- Decide the next action and write it in `notes.txt`

## Stop conditions

- Stop if the run already gives a defensible baseline-vs-optimizer comparison
- Stop if increasing `evaluations_B` would make the run unreasonably heavy
- Stop if repeated changes are being driven by test-set performance rather than validation behavior
- Stop if runtime or memory instability becomes a bigger risk than expected performance gain
