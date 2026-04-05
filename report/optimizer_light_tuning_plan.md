# Optimizer Light Tuning Plan

## Purpose

This note defines a limited follow-up tuning study for optimizer-specific control parameters. The goal is not to redesign the coursework as a full optimizer-tuning paper. The goal is only to check whether some of the gap between literature expectations and observed results is explained by using fixed practical defaults for the optimizers themselves.

## Rationale

The current comparison already controls the shared outer framework:

- same dataset
- same preprocessing
- same grouped-feature representation
- same RF search space
- same evaluator logic
- same budget accounting

However, the optimizer-specific parameters were not tuned in a method-sensitive way. This means the observed ranking may reflect a mixture of:

- genuine method suitability
- interaction with the current search space
- untuned optimizer control parameters

For this reason, a small follow-up tuning pass is justified.

## Scope

This tuning study is intentionally narrow.

Included methods:

- `PSO`
- `SA`
- `Tabu Search`

Excluded for now:

- `GA`
- `VNS`

Reason for exclusion:

- `GA` already performs strongly in the primary set.
- `VNS` already performs strongly as the overall balanced winner under the current evaluation setting.
- The immediate need is to test whether weaker-than-expected methods improve materially under small parameter adjustments.

## Study Design

### Phase 1: Single-seed screening

Use a lightweight screening configuration:

- budget: `B = 30`
- seed: `0`

For each method, vary only one or two influential control parameters. Keep the number of candidate settings small. The purpose is to identify whether the method shows obvious sensitivity, not to find a globally optimal configuration.

### Phase 2: Small robustness check

If a screened configuration is clearly better than the current default for that method, validate it under:

- budget: `B = 30`
- seeds: `0, 1, 2`

This phase is only for checking whether the apparent gain survives a minimal robustness check.

### Phase 3: Promotion rule

Only if the tuned setting remains clearly better after the lightweight robustness check should it be considered for:

- possible replacement of the original fixed-setting version in a follow-up comparison
- or a short sensitivity-analysis subsection in the paper

Do not move directly to large-scale `B = 120` tuning unless the small-budget evidence is clearly positive.

## Parameter Priorities

### PSO

Primary controls to test:

- `w`
- `c1 / c2` balance

Suggested lightweight grid:

- `w`: current default, lower inertia candidate
- `c1/c2`: balanced, cognition-heavy, social-heavy

Interpretation target:

- does PSO improve its `F1` without losing its already strong low-`FPR` profile?

### SA

Primary controls to test:

- `T0`
- `alpha`

Suggested lightweight grid:

- `T0`: lower and higher than current default
- `alpha`: faster cooling and slower cooling

Interpretation target:

- can SA retain its strong raw `F1` tendency while reducing instability in `FPR`?

### Tabu Search

Primary controls to test:

- `tabu_tenure`
- `neighborhood_size`

Suggested lightweight grid:

- shorter vs longer tabu tenure
- smaller vs larger neighborhood scan

Interpretation target:

- can Tabu preserve its strong single-run behaviour while improving robustness and reducing `FPR` drift?

## What Not To Do

The following are out of scope for this follow-up:

- full hyperparameter sweeps for every optimizer
- large multi-seed tuning campaigns
- `B = 120` tuning for all candidate settings
- simultaneous tuning of many optimizer-specific parameters
- reframing the coursework as an optimizer benchmarking paper

## Decision Rules

The tuned setting for a method should be considered meaningfully improved only if it shows a better overall balance in terms of:

- `F1`
- `FPR`
- selected feature count
- runtime

Raw `F1` alone is not sufficient.

Priority of interpretation:

1. avoid materially worse `FPR`
2. preserve or improve `F1`
3. avoid bloated feature subsets
4. runtime is secondary but still relevant

## Paper Integration

If this study is run, the safest way to present it is:

- original main comparison remains the fixed-settings comparison
- limited optimizer-specific tuning is reported separately as a follow-up sensitivity analysis

Suggested wording:

> Initial comparisons used fixed practical settings for each optimizer. Because some methods underperformed relative to expectations from prior literature, a limited method-specific tuning pass was conducted. Only a small number of influential optimizer control parameters were adjusted under lightweight budgets in order to keep the study within coursework scope.

## Immediate Next Step

Run a lightweight tuning screen for:

1. `PSO`
2. `SA`
3. `Tabu Search`

using `B = 30`, `seed = 0`, and a very small number of optimizer-specific settings per method.
