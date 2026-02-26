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
