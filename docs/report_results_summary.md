# Report Results Summary

This file consolidates the experiment outputs currently intended for the coursework paper.

## Primary reported setting

- Dataset: `UNSW_NB15_training-set.csv` / `UNSW_NB15_testing-set.csv`
- Task: binary classification (`label`)
- Base model: Random Forest
- Feature selection: grouped original features
- `k_min=8`
- Seed: `0`
- Main comparison budget: `evaluations_B=50`
- Optimizers compared: `GA`, `PSO`, `SA`
- Baseline: default Random Forest with all features

## Main comparison table

The baseline metrics are identical across the three `B=50` runs; only runtime varies slightly because the baseline was re-evaluated in separate CLI runs. The baseline row below uses the values recorded in `results/ga/b50_seed0/raw/all_runs.csv`.

| Method | Val Best Score | Val Recall | Val FPR | Test Accuracy | Test Precision | Test Recall | Test F1 | Test FPR | Selected Features | Test Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline RF | - | - | - | 0.8718 | 0.8188 | 0.9852 | 0.8943 | 0.2670 | 42 | 36.0577 |
| GA | 0.8658 | 0.9467 | 0.0485 | 0.9151 | 0.8826 | 0.9754 | 0.9267 | 0.1589 | 17 | 30.5208 |
| PSO | 0.8678 | 0.9535 | 0.0497 | 0.9111 | 0.8746 | 0.9788 | 0.9238 | 0.1719 | 18 | 29.4528 |
| SA | 0.8634 | 0.9491 | 0.0445 | 0.9061 | 0.8701 | 0.9750 | 0.9196 | 0.1784 | 18 | 48.2189 |

## Main interpretation points

- All three metaheuristics improved the baseline on test accuracy, test F1, test FPR, and number of selected features.
- The baseline achieved the highest recall, but at the cost of a much higher false positive rate and use of all 42 features.
- GA gave the best overall trade-off on the test set:
  - lowest test FPR among `GA/PSO/SA`
  - highest test F1
  - fewest selected features
  - moderate runtime
- PSO achieved the highest validation best score and the highest test recall among the optimizers, but its test FPR and feature count were slightly worse than GA.
- SA improved over baseline but was the slowest optimizer and delivered the weakest overall test trade-off among the three metaheuristics.

## Suggested wording for the paper

- Baseline vs optimized models:
  - "All three metaheuristic methods substantially reduced the false positive rate relative to the default Random Forest baseline while also reducing the selected feature set from 42 original features to 17-18."
- Best method:
  - "Among the three optimizers, GA provided the best overall trade-off between validation quality, test F1-score, false positive rate, and feature reduction."
- Trade-off statement:
  - "The default baseline preserved the highest recall, but this came with a significantly higher false positive rate. The optimized methods accepted a small recall reduction in exchange for improved F1-score, lower false positives, and leaner feature subsets."

## Supporting GA tuning evidence

Lightweight GA tuning was run at `seed=0` and `evaluations_B=20` to search cheaply for promising directions before any promotion to `B=50`.

| Run ID | Change | Val Best Score | Selected Features | Wall Runtime (s) | Decision |
|---|---|---:|---:|---:|---|
| `ga_b20_base` | Baseline lightweight GA config | 0.8387 | 24 | 874.0 | Keep as reference only |
| `ga_b20_mut002` | `mutation_rate: 0.05 -> 0.02` | 0.8387 | 24 | 925.2 | Reject |
| `ga_b20_pop10` | `pop_size: 25 -> 10` | 0.7925 | 18 | 1035.5 | Reject |
| `ga_b20_pop15` | `pop_size: 25 -> 15` | 0.8387 | 24 | 911.3 | Keep as weak reference, not a candidate |

## Tuning conclusion

- No lightweight `B=20` GA run was strong enough to justify promotion to a new `B=50` confirmation run.
- The existing `results/ga/b50_seed0` run should remain the main reported GA result.
- An important implementation insight is that, with `B=20`, any `GA` run using `pop_size >= 20` spends the full budget evaluating the initial population, so mutation or crossover changes do not become active.

## Source files

- `results/ga/b50_seed0/raw/all_runs.csv`
- `results/pso/b50_seed0/raw/all_runs.csv`
- `results/sa/b50_seed0/raw/all_runs.csv`
- `docs/ga_tuning_results.md`
