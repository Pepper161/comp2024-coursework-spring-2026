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
python run_experiment.py
```

## Notes

- This repository is for coursework development and reproducible experimentation.
- Keep sensitive or very large files out of version control.
