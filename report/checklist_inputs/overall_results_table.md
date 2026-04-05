# Overall Results Table

Source files used:
- `results/ga/b120_seed0/summary.csv`
- `results/pso/b120_seed0/summary.csv`
- `results/sa/b120_seed0/summary.csv`
- `results/vns/b120_seed0/summary.csv`
- `results/tabu/b120_seed0/summary.csv`

Selection rule:
- This table uses the strongest available **main-comparison** setting: `B=120`, `seed=0`.
- `RF` is taken from the baseline row duplicated inside each method summary and treated as the common benchmark row.
- `total_run_wall_time_sec` is used as the runtime column because it is uniformly available across methods, while optimisation-only runtime is discussed separately in the paper text.

| Method | Role | Budget / Seed | Accuracy | Precision | Recall | F1 | FPR | Selected Features | Total Runtime (s) | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RF | Baseline | B=120 / seed=0 | 0.8718 | 0.8188 | 0.9852 | 0.8943 | 0.2670 | 42 | 55.55 | `results/ga/b120_seed0/summary.csv` baseline row |
| GA | Primary | B=120 / seed=0 | 0.9080 | 0.8766 | 0.9694 | 0.9206 | 0.1672 | 17 | 5489.78 | `results/ga/b120_seed0/summary.csv` |
| PSO | Primary | B=120 / seed=0 | 0.8934 | 0.8740 | 0.9421 | 0.9068 | 0.1664 | 17 | 3797.22 | `results/pso/b120_seed0/summary.csv` |
| SA | Primary | B=120 / seed=0 | 0.9078 | 0.8736 | 0.9733 | 0.9208 | 0.1725 | 17 | 5751.24 | `results/sa/b120_seed0/summary.csv` |
| Tabu Search | Secondary | B=120 / seed=0 | 0.9137 | 0.8782 | 0.9790 | 0.9259 | 0.1664 | 19 | 3961.71 | `results/tabu/b120_seed0/summary.csv` |
| VNS | Secondary | B=120 / seed=0 | 0.9128 | 0.8783 | 0.9769 | 0.9250 | 0.1658 | 10 | 4868.60 | `results/vns/b120_seed0/summary.csv` |

Best-overall working note:
- **Best overall by balanced judgement:** `VNS`.
- Reason: it combines near-best F1 with the lowest feature count and the lowest FPR among the top-performing methods.
- This claim is strengthened, not weakened, by the lightweight robustness evidence at `B=30, seeds=0,1,2`, where `VNS` also maintains the strongest balance among the non-baseline methods.
- **Best overall by raw F1 only:** `Tabu Search`.
- This distinction should be preserved in the paper so that the final conclusion is not reduced to a single metric.

Cautions:
- `GA`'s logged `test_score_mean` in `results/ga/b120_seed0/summary.csv` does not match the current fitness formula in code and should not be used for narrative comparison.
- This table is a **single-seed main comparison**. Stability claims should be qualified using the separate robustness table.
- The robustness evidence now supports the wording that `Tabu Search` is a strong single-run secondary method, while `VNS` is the stronger secondary method once stability and low-FPR behaviour are considered together.
