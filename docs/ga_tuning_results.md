# GA Tuning Results

Lightweight GA tuning was run at `seed=0` and `evaluations_B=20` to cheaply test directional changes before considering any heavier confirmation run.

| Run ID | Change | Val Best Score | Selected Features | Wall Runtime (s) | Convergence Comment | Decision |
|---|---|---:|---:|---:|---|---|
| `ga_b20_base` | Baseline lightweight GA config | 0.8387 | 24 | 874.0 | Best score found early; no late improvement | Keep as reference only |
| `ga_b20_mut002` | `mutation_rate: 0.05 -> 0.02` | 0.8387 | 24 | 925.2 | Identical to base because `pop_size=25` exhausted the budget before offspring were created | Reject |
| `ga_b20_pop10` | `pop_size: 25 -> 10` | 0.7925 | 18 | 1035.5 | Offspring were created, but none beat the best initial individual | Reject |
| `ga_b20_pop15` | `pop_size: 25 -> 15` | 0.8387 | 24 | 911.3 | Best score reached after offspring started, but still matched the base score | Keep as weak reference, not a candidate |

## Decision

- No `B=20` candidate met the promotion rule for `B=50`.
- The lightweight search produced 3 consecutive runs without meaningful improvement after the initial baseline.
- Stop the cheap GA tuning here and keep the existing `results/ga/b50_seed0` run as the main GA result.

## Important implementation insight

- At `B=20`, any GA run with `pop_size >= 20` spends the entire budget evaluating the initial population.
- Under that condition, changing `mutation_rate` or `crossover_rate` has no practical effect because offspring are never produced.
- If GA tuning is resumed later, only settings that allow offspring within the budget should be explored.
