# Reproducibility Settings

Project root:
- `C:\Users\kazHP\OneDrive - University of Nottingham Malaysia\Year2\Winter\Artificial_Intelligence_Methods\Course_Work`

Environment:
- Python version observed in this environment: `Python 3.14.2`
- Dependencies from `requirements.txt`:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `pyyaml`
  - `tqdm`

Main comparison result sources:
- `results/ga/b120_seed0/summary.csv`
- `results/pso/b120_seed0/summary.csv`
- `results/sa/b120_seed0/summary.csv`
- `results/vns/b120_seed0/summary.csv`
- `results/tabu/b120_seed0/summary.csv`

Robustness result sources:
- `docs/generated/summary_robustness.csv` for `RF`, `GA`, `PSO`, `SA`
- `results/vns/robustness_b30_seeds012/summary.csv`
- `results/tabu/robustness_b30_seeds012/summary.csv`

Core execution entry point:
- `run_experiment.py`

Core modules used by the final pipeline:
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
- `src/optimizers/vns.py`
- `src/optimizers/tabu.py`

Dataset settings (from `config/experiment.yaml`):
- training file: `dataset/UNSW_NB15_training-set.csv`
- test file: `dataset/UNSW_NB15_testing-set.csv`
- target column: `label`
- dropped columns: `id`, `attack_cat`
- task type: binary classification
- positive label: `1`
- negative label: `0`

Split protocol:
- outer protocol uses the provided train/test files
- inner validation split: `val_size = 0.2`
- `train_test_split(..., random_state=seed, stratify=full_y)` is used for the train-inner / val-inner split in `src/runner.py`
- the same split is used for baseline and all optimisers under the same seed

Preprocessing settings:
- categorical columns: `proto`, `service`, `state`
- numeric imputation: median
- categorical imputation: most frequent
- one-hot handle unknown: ignore
- preprocessing is fit on the training partition only and applied to validation/test via transform

Representation settings:
- solution mode: `grouped_original_features`
- minimum selected original features: `k_min = 8`
- solution vector combines:
  - grouped original feature-selection genes
  - six Random Forest hyperparameter genes
- decoding logic lives in `src/representation.py`

Main local B=120 search-space settings (from `config/experiment_local_*_b120.yaml` files):
- `n_estimators: [100, 300]`
- `max_depth: [6, 20]`
- `min_samples_split: [2, 20]`
- `min_samples_leaf: [1, 8]`
- `max_features: ["sqrt", "log2"]`
- `class_weight: [null, "balanced"]`

Full experiment search-space settings (from `config/experiment.yaml`):
- `n_estimators: [100, 600]`
- `max_depth: [6, 30, null]`
- `min_samples_split: [2, 20]`
- `min_samples_leaf: [1, 8]`
- `max_features: ["sqrt", "log2"]`
- `class_weight: [null, "balanced"]`

Fitness settings (from `config/experiment.yaml` and `src/evaluator.py`):
- primary metric: recall
- `alpha_fpr = 0.05`
- `lambda_fpr = 20.0`
- `lambda_feat = 0.2`
- implemented fitness:
  - `fitness = recall - lambda_fpr * max(0, fpr - alpha_fpr) - lambda_feat * (k / d)`
  - where `k` is the number of selected original features and `d` is the total number of original features

Fairness / accounting settings:
- same dataset split per seed across all methods
- same preprocessing path across all methods
- same Random Forest search-space policy across compared methods within the same experiment family
- same evaluation budget accounting within a run
- evaluator cache hits still consume evaluation units in the current design
- baseline uses all original features and fixed default RF parameters

Baseline definition from code (`src/baseline.py`):
- `n_estimators = 100`
- `max_depth = None`
- `min_samples_split = 2`
- `min_samples_leaf = 1`
- `max_features = "sqrt"`
- `class_weight = None`
- all original grouped features are used

Checkpoint / recovery notes:
- best-solution checkpoints are written before final test
- `run_status.json` is used to record run stage
- final test can be rerun from checkpoint if needed

Ambiguities / notes:
- `ILS` is implemented in the repository but does not currently have a final `B=120` main result in the agreed result set.
- `GA`'s logged `test_score_mean` appears inconsistent with the current fitness implementation and should not be used as a trusted headline metric.
