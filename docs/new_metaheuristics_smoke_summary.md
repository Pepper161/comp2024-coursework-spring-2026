# New Metaheuristics Smoke Summary

## Scope

This file records the initial integration and smoke-test status for the new metaheuristic optimizers:

- Tabu Search (`tabu`)
- Iterated Local Search (`ils`)
- Variable Neighborhood Search (`vns`)

All smoke tests used the existing IDS workflow with:

- `evaluations_B = 10`
- `seed = 0`
- the same preprocessing, representation, evaluator, fitness, and result logging path as the existing `ga` / `pso` / `sa` pipeline

## Files Added

- `src/optimizers/tabu.py`
- `src/optimizers/ils.py`
- `src/optimizers/vns.py`
- `config/experiment_local_tabu.yaml`
- `config/experiment_local_ils.yaml`
- `config/experiment_local_vns.yaml`
- `docs/new_metaheuristics_smoke_summary.md`

## Files Changed

- `src/runner.py`
- `config/experiment.yaml`
- `README.md`

## Smoke Tests

### Tabu Search

- Status: **passed**
- Command: `python run_experiment.py --config config/experiment_local_tabu.yaml`
- Output path: `results/tabu/b10_seed0`
- Evaluations used: `10`
- Best validation score: `-1.144395`
- Test F1: `0.870243`
- Test FPR: `0.360730`
- Selected features: `26`
- Optimization wall time (s): `579.469`

### Iterated Local Search

- Status: **passed**
- Command: `python run_experiment.py --config config/experiment_local_ils.yaml`
- Output path: `results/ils/b10_seed0`
- Evaluations used: `10`
- Best validation score: `-1.452097`
- Test F1: `0.865909`
- Test FPR: `0.353297`
- Selected features: `28`
- Optimization wall time (s): `384.256`

### Variable Neighborhood Search

- Status: **passed**
- Command: `python run_experiment.py --config config/experiment_local_vns.yaml`
- Output path: `results/vns/b10_seed0`
- Evaluations used: `10`
- Best validation score: `0.879721`
- Test F1: `0.930783`
- Test FPR: `0.150459`
- Selected features: `15`
- Optimization wall time (s): `405.659`

## Initial Design Assumptions

- **Tabu Search** uses a single current solution, a small sampled neighborhood, a tabu list over recent accepted solution signatures, and an aspiration rule that allows tabu solutions if they improve the global best.
- **ILS** uses a simple bounded local-search phase, followed by a perturbation step when no local improvement is found.
- **VNS** uses increasing neighborhood sizes and resets to the smallest neighborhood when improvement is found.
- All three methods reuse the shared evaluator, so cache hits still consume budget exactly as they do for the existing methods.
- These are initial practical implementations intended to fit the current codebase cleanly and support honest low-budget testing before any larger-budget comparison.

## Remaining Limitations Before Larger Runs

- The smoke tests confirm integration and output correctness, not competitive final performance.
- `B = 10` is too small for stable method comparison on this IDS task.
- Tabu Search and ILS produced weak smoke-test metrics relative to the baseline, so they likely need parameter tuning before any meaningful comparison at `B = 30` or `B = 120`.
- VNS produced the strongest smoke-test result of the three new methods, but this is still only a single low-budget run and should not be over-interpreted.
- No Colab single-file notebooks have been created yet for `tabu`, `ils`, or `vns`; only the modular local experiment path is integrated at this stage.

## Follow-up B=30 Check

After the initial smoke tests, all three methods were also run once at:

- `evaluations_B = 30`
- `seed = 0`

using:

- `config/experiment_local_tabu_b30.yaml`
- `config/experiment_local_ils_b30.yaml`
- `config/experiment_local_vns_b30.yaml`

### B=30 outcomes

- **Tabu Search**
  - Output path: `results/tabu/b30_seed0`
  - Validation best score: `-0.058399`
  - Test F1: `0.893477`
  - Test FPR: `0.277514`
  - Selected features: `26`
  - Optimization wall time: `1890.544 s`

- **Iterated Local Search**
  - Output path: `results/ils/b30_seed0`
  - Validation best score: `0.516493`
  - Test F1: `0.912898`
  - Test FPR: `0.187838`
  - Selected features: `27`
  - Optimization wall time: `1964.111 s`

- **Variable Neighborhood Search**
  - Output path: `results/vns/b30_seed0`
  - Validation best score: `0.888170`
  - Test F1: `0.926896`
  - Test FPR: `0.161784`
  - Selected features: `14`
  - Optimization wall time: `895.539 s`

### Practical interpretation

- `TS` still completes honestly through the shared pipeline, but its current initial configuration is not competitive.
- `ILS` improves substantially over its `B=10` smoke-test behavior and becomes worth keeping for further tuning.
- `VNS` is the strongest of the three initial new methods at `B=30`, with the best validation score, the best test F1, the lowest test FPR, and the smallest feature subset.
- Before any `B=120` or report-facing comparison, `TS` and `ILS` should be tuned, while `VNS` is the most reasonable candidate to scale first.
