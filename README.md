# COMP2024 Coursework (Spring 2026)

Public project workspace for the COMP2024 Artificial Intelligence Methods coursework.

## Project Scope

This project studies IDS feature selection and hyperparameter optimization using metaheuristic algorithms, with a comparison against non-metaheuristic baselines.

## Structure

- `src/` - source code for preprocessing, training, and optimization
- `config/` - configuration files
- `notebooks/` - exploratory notebooks
- `results/` - experiment outputs and figures
- `run_experiment.py` - main entry point

## File Guide

- `run_experiment.py` - CLI entry point. Loads config and starts full experiment run.
- `requirements.txt` - minimal Python dependency list used in local/Colab setups.
- `config/experiment.yaml` - experiment settings (dataset paths, budget, seeds, model search space, output paths).
- `notebooks/00_colab_run.ipynb` - Colab-first execution notebook (clone, dataset copy, smoke test, full run).
- `notebooks/analysis_plots.ipynb` - post-run analysis notebook for quick result inspection.
- `src/__init__.py` - package marker for `src` modules.
- `src/data.py` - loads UNSW-NB15 train/test CSVs and resolves target/features safely.
- `src/preprocess.py` - leakage-safe preprocessing (fit on training fold only) and one-hot group mapping.
- `src/representation.py` - solution encoding/decoding (feature mask + RF hyperparameters), k-min enforcement.
- `src/metrics.py` - binary classification metrics (Accuracy, Precision, Recall, F1, FPR, feature count, runtime).
- `src/evaluator.py` - objective evaluation on validation, final one-shot test evaluation, per-(algorithm, seed) cache.
- `src/baseline.py` - baseline RandomForest with default hyperparameters and all original features.
- `src/runner.py` - orchestration for splits, optimizers, fairness budget handling, and result artifact writing.
- `src/optimizers/__init__.py` - optimizer package marker.
- `src/optimizers/ga.py` - Genetic Algorithm optimizer under fixed evaluation budget.
- `src/optimizers/pso.py` - Binary/continuous PSO optimizer under fixed evaluation budget.
- `src/optimizers/sa.py` - Simulated Annealing optimizer under fixed evaluation budget.

## Dataset

This repository does not include raw dataset files.

Expected local files:

- `dataset/UNSW_NB15_training-set.csv`
- `dataset/UNSW_NB15_testing-set.csv`

## Quick Start

1. Create a Python environment.
2. Install required packages.
3. Place dataset CSV files under `dataset/`.
4. Run:

```bash
python run_experiment.py --config config/experiment.yaml
```

## Colab Start

In Colab, shell commands use `!` prefix.  
So these are equivalent in intent:

- Local terminal: `python run_experiment.py --config config/experiment.yaml`
- Colab cell: `!python run_experiment.py --config config/experiment.yaml`

Recommended Colab flow:

```python
# Clone
REPO_URL = "https://github.com/Pepper161/comp2024-coursework-spring-2026.git"
!git clone "$REPO_URL"
%cd comp2024-coursework-spring-2026

# If Course_Work exists as subfolder, enter it
import os
if os.path.isdir("Course_Work"):
    os.chdir("Course_Work")
print("cwd:", os.getcwd())

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Place datasets in dataset/ (copy from Drive)
!mkdir -p dataset
!cp "/content/drive/MyDrive/<path>/UNSW_NB15_training-set.csv" dataset/
!cp "/content/drive/MyDrive/<path>/UNSW_NB15_testing-set.csv" dataset/
!ls -lh dataset

# Smoke test
!python run_experiment.py --config config/experiment.yaml --budget-override 30 --max-seeds 1 --skip-plots

# Full run
!python run_experiment.py --config config/experiment.yaml
```

## Notes

- This repository is for coursework development and reproducible experimentation.
- Keep sensitive or very large files out of version control.
