# Robustness Results Table

## Purpose

Paper-ready source table for lightweight robustness evidence across the compared metaheuristics.

## Included Methods

- `GA`
- `PSO`
- `SA`
- `Tabu Search`
- `VNS`

## Source Files

- `docs/generated/summary_robustness.csv` for `GA`, `PSO`, `SA`
- `results/tabu/robustness_b30_seeds012/summary.csv`
- `results/vns/robustness_b30_seeds012/summary.csv`

## Selection Rule

- This table uses the lightweight robustness setting `B=30`, `seeds=0,1,2`.
- It is intended to qualify stability and consistency, not to replace the main `B=120, seed=0` comparison.
- `RF` is omitted from the main body version of this table because the purpose here is to compare robustness across the optimised methods; if needed, the baseline row can be added in an appendix version.

## Table

| Method | Budget / Seeds | Test Recall Mean | Test F1 Mean | Test FPR Mean | Selected Features Mean | Optimisation Time Mean (s) | Notes | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| GA | B=30 / seeds 0,1,2 | 0.9515 | 0.9112 | 0.1678 | 23.00 | 1166.64 | Best-primary main-run method, but not the strongest primary method in lightweight robustness. | `docs/generated/summary_robustness.csv` |
| PSO | B=30 / seeds 0,1,2 | 0.9511 | 0.9134 | 0.1608 | 18.67 | 1130.81 | Strongest primary method in lightweight robustness by combined `F1`/`FPR`/feature balance. | `docs/generated/summary_robustness.csv` |
| SA | B=30 / seeds 0,1,2 | 0.9752 | 0.9132 | 0.1978 | 22.00 | 1263.21 | High recall, but materially worse `FPR` than other primary methods. | `docs/generated/summary_robustness.csv` |
| Tabu Search | B=30 / seeds 0,1,2 | 0.9786 | 0.9106 | 0.2098 | 24.00 | 1438.84 | Strong single-run method, but weaker robustness profile than `VNS` due to higher `FPR` and larger subsets. | `results/tabu/robustness_b30_seeds012/summary.csv` |
| VNS | B=30 / seeds 0,1,2 | 0.9704 | 0.9234 | 0.1610 | 19.33 | 1247.57 | Strongest overall robustness profile by balanced judgment across the optimised methods. | `results/vns/robustness_b30_seeds012/summary.csv` |

## Working Interpretation

- **Best overall robustness method by balanced judgement:** `VNS`
- **Strongest primary robustness method by balanced judgement:** `PSO`
- **Highest robustness recall:** `Tabu Search`
- **Highest robustness F1:** `VNS`

## Tuned Tabu Follow-Up

Follow-up source:
- `results/tabu/robustness_tuned_neigh5_b30_seeds012/summary.csv`

Follow-up setting:
- `Tabu Search` with `tabu_tenure = 5`, `neighborhood_size = 5`
- same `B = 30`, `seeds = 0,1,2` framework as the original Tabu robustness run

Follow-up summary:
- tuned Tabu mean `F1`: `0.9109`
- original Tabu mean `F1`: `0.9106`
- tuned Tabu mean `FPR`: `0.2082`
- original Tabu mean `FPR`: `0.2098`
- tuned Tabu mean selected features: `23.33`
- original Tabu mean selected features: `24.00`
- tuned Tabu mean optimisation time: `1608.18 s`
- original Tabu mean optimisation time: `1438.84 s`

Interpretation:
- The tuned setting is marginally better than the original fixed-setting Tabu robustness profile on `F1`, `FPR`, and feature count.
- The improvement is small, and the runtime is worse.
- This strengthens the claim that Tabu’s light-budget behaviour is somewhat configuration-sensitive, but it does **not** overturn the broader robustness interpretation.
- `VNS` remains clearly stronger than both original and tuned Tabu by balanced judgment at the lightweight robustness setting.

## Cautions

- This is a lighter-budget setting than the main comparison and must not be treated as equivalent evidence.
- The table should be used to qualify stability and consistency, not to overturn the main-comparison design.
- `GA` main-run `test_score` inconsistency remains unresolved and is not relevant to this robustness table because the table relies on directly interpretable metrics only.
