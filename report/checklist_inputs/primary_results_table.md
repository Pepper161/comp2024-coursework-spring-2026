# Primary Results Table

Source files used:
- `results/ga/b120_seed0/summary.csv`
- `results/pso/b120_seed0/summary.csv`
- `results/sa/b120_seed0/summary.csv`

Selection rule:
- This table contains the agreed **primary methods only**: `RF`, `GA`, `PSO`, and `SA`.
- It uses the strongest available main-comparison setting: `B=120`, `seed=0`.

| Method | Role | Budget / Seed | Accuracy | Precision | Recall | F1 | FPR | Selected Features | Optimisation Time (s) | Total Runtime (s) | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RF | Baseline | B=120 / seed=0 | 0.8718 | 0.8188 | 0.9852 | 0.8943 | 0.2670 | 42 | 0.00 | 55.55 | `results/ga/b120_seed0/summary.csv` baseline row |
| GA | Primary | B=120 / seed=0 | 0.9080 | 0.8766 | 0.9694 | 0.9206 | 0.1672 | 17 | 5417.40 | 5489.78 | `results/ga/b120_seed0/summary.csv` |
| PSO | Primary | B=120 / seed=0 | 0.8934 | 0.8740 | 0.9421 | 0.9068 | 0.1664 | 17 | 3770.89 | 3797.22 | `results/pso/b120_seed0/summary.csv` |
| SA | Primary | B=120 / seed=0 | 0.9078 | 0.8736 | 0.9733 | 0.9208 | 0.1725 | 17 | 5695.27 | 5751.24 | `results/sa/b120_seed0/summary.csv` |

Best-primary working note:
- **Best primary by balanced judgement:** `GA`.
- Reason: `GA` gives the lowest FPR among the primary methods while maintaining near-top F1 and the same compact 17-feature subset as PSO and SA.
- **Best primary by raw F1 only:** `SA`.
- This distinction should be stated explicitly in the paper so the primary-method conclusion remains metric-aware.

Cautions:
- `PSO` is the fastest primary optimiser in the main comparison, but this should not be treated as an automatic win because its F1 is clearly lower than both `GA` and `SA`.
- `GA`'s `test_score_mean` should not be used in the paper because it appears inconsistent with the current fitness function implementation.
