# GA Lightweight Tuning Plan

## Goal

Use cheap GA runs at `evaluations_B=20` to identify promising directions before promoting any candidate to `evaluations_B=50`.

## Fixed conditions

- optimizer: `ga`
- `seed=0`
- dataset and preprocessing unchanged
- one factor changed per run
- decisions driven primarily by:
  - validation best score
  - convergence behavior
  - selected feature count
  - runtime

## Promotion rule

Promote a config from `B=20` to `B=50` only if it is clearly promising:

- better validation score, or
- similar validation but fewer selected features, or
- similar validation with faster convergence/runtime

## Stop rule

Stop automatically if:

- 3 consecutive `B=20` runs show no meaningful validation improvement, or
- gains are too small to justify more runs

## Meaningful improvement heuristic

Treat a run as promising if any of the following is true:

- `val_best_score` improves by at least `0.005`
- `val_best_score` is within `0.003` of the current best and selected features are lower
- `val_best_score` is within `0.003` of the current best and runtime is noticeably lower

## Planned run order

1. `ga_b20_base` - current GA settings, lower budget
2. `ga_b20_mut002` - mutation rate only, `0.05 -> 0.02`
3. `ga_b20_pop10` - population size only, `25 -> 10`
4. `ga_b20_pop15` - population size only, `25 -> 15`

If none of these runs is promising enough, stop and keep the existing `ga_b50_s0` result as the main GA result.
