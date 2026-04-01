# Results Overview

Keep one row per completed run.

| Run ID | Optimizer | Config | Seed | Eval B | Val Best Score | Test Recall | Test FPR | Test F1 | Selected Features | Runtime (s) | What Changed | Decision Next |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ga_b50_s0 | GA | `config/experiment_local_ga.yaml` | 0 | 50 | 0.8658 | 0.9754 | 0.1589 | 0.9267 | 17 | 1951.7 | Initial GA run under local single-optimizer workflow | Keep `B=50`; run PSO next |
| pso_b50_s0 | PSO | `config/experiment_local_pso.yaml` | 0 | 50 | 0.8678 | 0.9788 | 0.1719 | 0.9238 | 18 | 1903.3 | Initial PSO run under local single-optimizer workflow | Keep `B=50`; run SA next |
| sa_b50_s0 | SA | `config/experiment_local_sa.yaml` | 0 | 50 | 0.8634 | 0.9750 | 0.1784 | 0.9196 | 18 | 2438.2 | Initial SA run under local single-optimizer workflow | First fair comparison complete; do not raise budget automatically |

## Notes

- Compare optimizers primarily on validation behavior first.
- Use test metrics for reporting and final discussion, not repeated opportunistic tuning.
- If a follow-up run changes one parameter, add a new row rather than editing the old row.
- At `B=50`, all three optimizers reduced FPR and feature count substantially relative to baseline, with only a small recall drop.
- On this first comparison, GA produced the strongest overall test trade-off among the three optimizers.
