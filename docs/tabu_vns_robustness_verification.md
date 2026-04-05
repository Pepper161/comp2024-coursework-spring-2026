# Tabu and VNS Robustness Verification

Scope:
- Verify lightweight robustness evidence for `Tabu Search` and `VNS`
- Target setting: `B=30`, `seeds = [0, 1, 2]`

Verified config files:
- `config/experiment_robustness_tabu_b30.yaml`
- `config/experiment_robustness_vns_b30.yaml`

Verified output folders:
- `results/tabu/robustness_b30_seeds012`
- `results/vns/robustness_b30_seeds012`

Verified artifacts present for both methods:
- `raw/all_runs.csv`
- `raw/all_runs_incremental.csv`
- `raw/seed_0_results.csv`
- `raw/seed_1_results.csv`
- `raw/seed_2_results.csv`
- `summary.csv`
- per-seed convergence logs under `convergence/`
- per-seed checkpoints under `best_solutions/`

Comparability judgement:
- These runs are directly comparable to the existing `GA / PSO / SA` robustness evidence in methodological terms.
- They use the same dataset files, the same preprocessing settings, the same grouped-feature representation, the same fitness function, the same validation protocol, the same seed list `[0,1,2]`, and the same budget `B=30`.
- They also use the same narrowed local Random Forest search space used by the lightweight robustness path.

Important note:
- `GA / PSO / SA` robustness is currently summarised in `docs/generated/summary_robustness.csv`, while `Tabu Search` and `VNS` are summarised separately in their own `summary.csv` files.
- For paper tables or plots, combine these sources explicitly rather than assuming a single pre-merged robustness CSV already exists.

Top-line observations:
- `VNS` robustness is strong and consistent enough to support discussion as a serious overall contender.
- `Tabu Search` robustness exists and is directly comparable, but it is less stable than `VNS` and does not overturn the current balanced-judgement preference for `VNS`.
