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
- `config/experiment_local_ga.yaml` - lightweight local profile that runs only GA with `seed=0`.
- `config/experiment_local_pso.yaml` - lightweight local profile that runs only PSO with `seed=0`.
- `config/experiment_local_sa.yaml` - lightweight local profile that runs only SA with `seed=0`.
- `config/experiment_local_tabu.yaml` - lightweight local smoke profile that runs only Tabu Search with `B=10`, `seed=0`.
- `config/experiment_local_ils.yaml` - lightweight local smoke profile that runs only Iterated Local Search with `B=10`, `seed=0`.
- `config/experiment_local_vns.yaml` - lightweight local smoke profile that runs only Variable Neighborhood Search with `B=10`, `seed=0`.
- `config/experiment_local_ga_b120.yaml` - main-run local profile that runs only GA with `B=120`, `seed=0`.
- `config/experiment_local_pso_b120.yaml` - main-run local profile that runs only PSO with `B=120`, `seed=0`.
- `config/experiment_local_sa_b120.yaml` - main-run local profile that runs only SA with `B=120`, `seed=0`.
- `config/experiment_local_tabu_b120.yaml` - main-run local profile that runs only Tabu Search with `B=120`, `seed=0`.
- `config/experiment_local_ils_b120.yaml` - main-run local profile that runs only Iterated Local Search with `B=120`, `seed=0`.
- `config/experiment_local_vns_b120.yaml` - main-run local profile that runs only Variable Neighborhood Search with `B=120`, `seed=0`.
- `config/experiment_robustness_b30.yaml` - lightweight multi-seed robustness profile (`B=30`, `seeds=[0,1,2]`) for GA/PSO/SA.
- `notebooks/00_colab_run.ipynb` - Colab-first execution notebook (clone, dataset copy, smoke test, full run).
- `notebooks/01_colab_single_notebook.ipynb` - generic single-notebook Colab execution mirror for upload-and-run workflows.
- `notebooks/02_colab_single_ga.ipynb` - single-file Colab notebook that runs only GA at `B=50`, `seed=0`.
- `notebooks/03_colab_single_pso.ipynb` - single-file Colab notebook that runs only PSO at `B=50`, `seed=0`.
- `notebooks/04_colab_single_sa.ipynb` - single-file Colab notebook that runs only SA at `B=50`, `seed=0`.
- `scripts/generate_report_tables.py` - converts raw CSV outputs into paper-ready CSV/Markdown summary tables.
- `scripts/make_additional_plots.py` - generates the final revised report figures under `results/figures/`.
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
- `src/optimizers/tabu.py` - Tabu Search optimizer under fixed evaluation budget.
- `src/optimizers/ils.py` - Iterated Local Search optimizer under fixed evaluation budget.
- `src/optimizers/vns.py` - Variable Neighborhood Search optimizer under fixed evaluation budget.

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

## Local Single-Optimizer Runs

For a more stable local workflow, run one optimizer at a time with the lightweight configs:

```bash
python run_experiment.py --config config/experiment_local_ga.yaml
python run_experiment.py --config config/experiment_local_pso.yaml
python run_experiment.py --config config/experiment_local_sa.yaml
```

These profiles use:

- `seed=0`
- `evaluations_B=50`
- reduced RF search ranges
- nested output folders under `results/` so each optimizer run is preserved independently

Current local output layout:

- `results/ga/b50_seed0`
- `results/pso/b50_seed0`
- `results/sa/b50_seed0`

Each run folder contains:

- `run_config.yaml` - exact config used for that run
- `seed_list.txt` - seed list used for that run
- `notes.txt` - editable run log template for recording purpose, changes, runtime issues, and conclusions
- `raw/`, `convergence/`, `plots/` - generated experiment artifacts
- `best_solutions/` - saved feature selections and best hyperparameters per algorithm/seed

## Team Run Assignments

Use the same repository state and dataset files for all team runs.

Common setup for every member:

```bash
git clone https://github.com/Pepper161/comp2024-coursework-spring-2026.git
cd comp2024-coursework-spring-2026/Course_Work
```

Required dataset files:

- `dataset/UNSW_NB15_training-set.csv`
- `dataset/UNSW_NB15_testing-set.csv`

### Umer - VNS (`B=120`, `seed=0`)

Create a branch, run VNS, then commit the generated results:

```bash
git checkout -b umer-vns-b120
python run_experiment.py --config config/experiment_local_vns_b120.yaml
git add results/vns/b120_seed0
git commit -m "Add VNS B120 seed0 results"
git push origin umer-vns-b120
```

Then open a Pull Request from `umer-vns-b120`.

### Gobran - TS (`B=120`, `seed=0`)

Create a branch, run Tabu Search, then commit the generated results:

```bash
git checkout -b gobran-tabu-b120
python run_experiment.py --config config/experiment_local_tabu_b120.yaml
git add results/tabu/b120_seed0
git commit -m "Add Tabu Search B120 seed0 results"
git push origin gobran-tabu-b120
```

Then open a Pull Request from `gobran-tabu-b120`.

### SayedA - ILS (`B=120`, `seed=0`)

Create a branch, run Iterated Local Search, then commit the generated results:

```bash
git checkout -b sayeda-ils-b120
python run_experiment.py --config config/experiment_local_ils_b120.yaml
git add results/ils/b120_seed0
git commit -m "Add ILS B120 seed0 results"
git push origin sayeda-ils-b120
```

Then open a Pull Request from `sayeda-ils-b120`.

### If final test fails after optimization

Do not rerun the whole optimizer immediately. First check whether the checkpoint exists:

- `results/<method>/b120_seed0/best_solutions/<algorithm>_seed_0.json`

If it exists, rerun only the final test:

```bash
python scripts/run_final_test_from_checkpoint.py --project-root . --config <config-path> --checkpoint <checkpoint-json> --seed 0
```

Examples:

```bash
python scripts/run_final_test_from_checkpoint.py --project-root . --config config/experiment_local_vns_b120.yaml --checkpoint results/vns/b120_seed0/best_solutions/vns_seed_0.json --seed 0
python scripts/run_final_test_from_checkpoint.py --project-root . --config config/experiment_local_tabu_b120.yaml --checkpoint results/tabu/b120_seed0/best_solutions/tabu_seed_0.json --seed 0
python scripts/run_final_test_from_checkpoint.py --project-root . --config config/experiment_local_ils_b120.yaml --checkpoint results/ils/b120_seed0/best_solutions/ils_seed_0.json --seed 0
```

After the final test rerun succeeds, commit the updated `results/<method>/b120_seed0` folder and open the PR.

If the checkpoint has `test_metrics` but `raw/all_runs.csv` or `summary.csv` is still missing, finalize the normal result artifacts before committing:

```bash
python scripts/finalize_run_from_checkpoint.py --project-root . --config <config-path> --results-dir <results-dir> --checkpoint <checkpoint-json> --seed 0
```

If the optimizer completed but the final CSV artifacts are still missing after the recovery steps above, do not rerun the whole `B=120` experiment immediately. Message the team lead first so the checkpoint can be reviewed before spending another long run.

## Robustness Check

Use the lightweight robustness profile only as supporting evidence, not as a replacement for the main `B=50, seed=0` result:

```bash
python run_experiment.py --config config/experiment_robustness_b30.yaml
```

This profile keeps runtime practical while adding:

- repeated seeds: `0,1,2`
- `optimization_wall_time_sec`
- `total_run_wall_time_sec`
- richer convergence logs with best validation recall/FPR/feature count

## Paper Table Generation

After experiments complete, generate paper-ready tables from raw CSV outputs:

```bash
python scripts/generate_report_tables.py --main results/ga/b50_seed0/raw/all_runs.csv results/pso/b50_seed0/raw/all_runs.csv results/sa/b50_seed0/raw/all_runs.csv --robustness results/robustness/b30_seeds012/raw/seed_0_results.csv results/robustness/b30_seed1/raw/all_runs.csv results/robustness/b30_seed2/raw/all_runs.csv --output-dir docs/generated
```

This creates:

- `docs/generated/summary_main.csv`
- `docs/generated/summary_robustness.csv`
- `docs/generated/paper_table_main.md`
- `docs/generated/paper_table_robustness.md`

## Final Report Figures

Generate the final revised report figures with:

```bash
python scripts/make_additional_plots.py
```

This creates:

- `results/figures/tradeoff_scatter_revised.png`
- `results/figures/distribution_test_f1_revised.png`
- `results/figures/distribution_test_fpr_revised.png`
- `results/figures/distribution_selected_features_revised.png`
- `results/figures/distribution_runtime_revised.png`
- `results/figures/feature_selection_frequency_revised.png`
- `results/figures/convergence_summary_across_seeds_revised.png`
- `results/figures/revision_notes.md`

## Colab Start

There are two supported Colab workflows:

- `notebooks/00_colab_run.ipynb` - use this when you want to clone the repository and run the modular codebase.
- `notebooks/01_colab_single_notebook.ipynb` - use this when you want a generic single uploaded notebook without cross-file imports or `git clone`.
- `notebooks/02_colab_single_ga.ipynb`, `notebooks/03_colab_single_pso.ipynb`, `notebooks/04_colab_single_sa.ipynb` - use these when you want one uploaded notebook per optimizer for separate `B=50`, `seed=0` runs.

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

Single-file Colab flow for separate optimizer runs:

1. Upload one of these notebooks to Colab:
   - `notebooks/02_colab_single_ga.ipynb`
   - `notebooks/03_colab_single_pso.ipynb`
   - `notebooks/04_colab_single_sa.ipynb`
2. Upload the two CSV files into a `dataset/` folder in the Colab working directory.
3. Run the notebook from top to bottom.
4. Keep `RUN_SMOKE_TEST = False` for the final `B=50`, `seed=0` run.

Each notebook writes to a separate output path:

- `results_colab/ga/b50_seed0`
- `results_colab/pso/b50_seed0`
- `results_colab/sa/b50_seed0`

## Notes

- This repository is for coursework development and reproducible experimentation.
- Keep sensitive or very large files out of version control.
