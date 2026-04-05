# Core Definitions

## Task definition
Binary intrusion-detection classification on UNSW-NB15, where the system jointly searches for a compact subset of grouped original features and a Random Forest hyperparameter setting that improves detection quality while controlling false positives.

## Baseline definition
The benchmark is a default Random Forest that uses all original grouped features with fixed default hyperparameters (`n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"`, `class_weight=None`) and no metaheuristic search.

## Search representation
A candidate solution combines grouped original feature-selection decisions with six Random Forest hyperparameter genes; decoding is handled by `src/representation.py`, and at least `k_min = 8` original features must remain selected.

## Fitness function
The implemented objective in `src/evaluator.py` is:
`fitness = recall - lambda_fpr * max(0, fpr - alpha_fpr) - lambda_feat * (k / d)`
with `alpha_fpr = 0.05`, `lambda_fpr = 20.0`, and `lambda_feat = 0.2`, so the search explicitly rewards recall while penalising excessive false positives and unnecessarily large feature subsets.

## Fairness policy
All compared methods use the same data split for a given seed, the same preprocessing pipeline, the same evaluator logic, the same feature-selection representation, the same Random Forest search-space policy within a comparison family, and the same evaluation-budget accounting rules.

## Preprocessing summary
The pipeline drops `id` and `attack_cat`, imputes numeric features with the median, imputes categorical features with the most frequent value, one-hot encodes `proto`, `service`, and `state` with `handle_unknown="ignore"`, and fits preprocessing only on the training partition before transforming validation and test data.

## Best overall vs best primary
- **Best overall** should be judged over `RF`, `GA`, `PSO`, `SA`, `Tabu Search`, and `VNS`.
- **Best primary** should be judged only over the agreed primary methods: `GA`, `PSO`, and `SA`, with `RF` retained as the benchmark.
- Current working interpretation from the agreed result set:
  - **Best overall by balanced judgement:** `VNS`
  - **Best overall by raw F1 only:** `Tabu Search`
  - **Best primary by balanced judgement:** `GA`
  - **Best primary by raw F1 only:** `SA`

## Metric-consistency note
The logged `GA` main-run `test_score_mean` does not appear to match the current fitness function implementation and should not be used as a trusted headline result; the paper should rely on directly interpretable metrics such as F1, FPR, recall, selected features, and runtime.
