# Results Overview Template

Keep one row per completed run.

| Run ID | Optimizer | Config | Seed | Eval B | Val Best Score | Test Recall | Test FPR | Test F1 | Selected Features | Runtime (s) | What Changed | Decision Next |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ga_b50_s0 | GA | `config/experiment_local_ga.yaml` | 0 | 50 |  |  |  |  |  |  | Initial GA run |  |
| pso_b50_s0 | PSO | `config/experiment_local_pso.yaml` | 0 | 50 |  |  |  |  |  |  | Initial PSO run |  |
| sa_b50_s0 | SA | `config/experiment_local_sa.yaml` | 0 | 50 |  |  |  |  |  |  | Initial SA run |  |

## Notes

- Compare optimizers primarily on validation behavior first.
- Use test metrics for reporting and final discussion, not repeated opportunistic tuning.
- If a follow-up run changes one parameter, add a new row rather than editing the old row.
