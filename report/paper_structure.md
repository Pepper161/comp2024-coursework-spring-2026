# Revised Paper Structure

## Working Principle

Show the empirical results for all implemented methods, but keep the primary interpretive focus narrower and literature-defensible.

## Scope Statement

This study reports overall empirical results for all implemented methods, namely `RF`, `GA`, `PSO`, `SA`, `Tabu Search`, and `VNS`. However, the primary comparative discussion focuses on `RF`, `GA`, `PSO`, and `SA` because these methods are supported more directly by recent IDS optimization literature, while `Tabu Search` and `VNS` are treated as additional comparators that help interpret the search landscape.

## 1. Introduction

### 1.1 Background and Motivation

- Explain why high-dimensional features and false alarms are major problems in IDS.
- Introduce `UNSW-NB15` as the main benchmark and explain why it is appropriate.
- Motivate both feature selection and hyperparameter optimization.

### 1.2 Research Objective

- Compare how much improvement multiple metaheuristics can deliver over an `RF` baseline.
- State explicitly that the study evaluates not only predictive performance, but also `FPR`, number of selected features, and computational cost.

### 1.3 Contributions

- A reproducible comparison on `UNSW-NB15`
- An overall empirical comparison across all implemented methods
- A structured interpretation that separates primary comparison from additional comparison
- A discussion of security trade-offs rather than accuracy-only gains

## 2. Related Work and Study Positioning

### 2.1 IDS on UNSW-NB15 and Conventional Baselines

- Justify `RF` as the baseline.
- Summarize the most common evaluation metrics used on `UNSW-NB15`.
- Note that `RF` is a natural conventional baseline in `UNSW-NB15` IDS studies and that `RF` importance is also used in feature-selection pipelines.

### 2.2 Metaheuristic Optimization for IDS

- `GA`
- `PSO`
- `SA`

This section should make clear that `GA` and `PSO` have stronger direct IDS support, while `SA` still has usable recent IDS-adjacent evidence.

### 2.3 Additional Local-Search Comparators

- `Tabu Search`
- `VNS`

Frame these carefully:

- not invalid
- still relevant as additional comparators
- weaker direct IDS literature support than the primary methods

### 2.4 Positioning of This Study

- Primary focus: `RF`, `GA`, `PSO`, `SA`
- Additional comparison: `Tabu Search`, `VNS`
- State clearly that the results section will still report all implemented methods

## 3. Problem Formulation and Experimental Protocol

### 3.1 Task Definition

- Binary IDS on `UNSW-NB15`
- Objective: improve detection quality while reducing `FPR` and unnecessary features

### 3.2 Data and Preprocessing

- Train/test split
- Leakage-safe preprocessing
- Categorical handling
- One-hot grouping
- Duplicate handling if applicable

Make clear that preprocessing choices matter for recent `UNSW-NB15` feature-selection studies.

### 3.3 Baseline Model

- Default `Random Forest`
- All features
- Default or fixed baseline hyperparameters

### 3.4 Search Representation

- Feature mask
- `RF` hyperparameters
- Joint search space

### 3.5 Fitness / Objective Function

- Define what is optimized during search
- Clarify whether validation `F1` is primary
- Explain how `FPR` and feature penalties are incorporated

### 3.6 Evaluation Criteria

- `Accuracy`
- `Precision`
- `Recall`
- `F1`
- `FPR`
- Number of selected features
- Runtime

These should be tied both to the coursework rubric and to common `UNSW-NB15` reporting practice.

### 3.7 Reproducibility and Fairness Policy

- Same budget
- Same seed policy
- Same baseline learner
- Same preprocessing
- Same evaluation split

## 4. Algorithms

### 4.1 Primary Algorithms

- `GA`
- `PSO`
- `SA`

For each algorithm include:

- short concept
- representation or move behavior
- why it was selected
- expected search behavior

### 4.2 Additional Algorithms

- `Tabu Search`
- `VNS`

For each algorithm include:

- short concept
- why it was included
- why it is treated as additional rather than primary

Important:

- "additional" must not imply "hidden"
- these methods are still part of the empirical study
- the distinction is interpretive, not suppressive

## 5. Results

### 5.1 Overall Results for All Methods

Show one total comparison table covering:

- `RF`
- `GA`
- `PSO`
- `SA`
- `Tabu Search`
- `VNS`

This is the most important structural safeguard because it prevents the paper from looking selective or defensive.

### 5.2 Primary Comparison Focus

Focus interpretation on:

- `RF`
- `GA`
- `PSO`
- `SA`

Use this subsection for the main comparison figures and tables.

### 5.3 Trade-off and Security Implications

- `F1` versus `FPR`
- feature reduction versus detection quality
- runtime trade-offs
- security meaning of false positives and missed attacks

This section should directly address the coursework requirement for security trade-offs.

### 5.4 Additional Observations on Tabu and VNS

- Were they competitive?
- Did they outperform any primary methods?
- If yes, why were they still not treated as primary?

This section should explain positioning, not hide outcomes.

## 6. Discussion and Threats to Validity

### 6.1 Interpretation of Primary Results

- Did `GA`, `PSO`, and `SA` behave as expected from the literature?
- Was the improvement over the `RF` baseline meaningful?

### 6.2 Interpreting Additional Methods

- Explain strong or weak outcomes for `Tabu Search` and `VNS`
- Discuss mismatch between empirical strength and literature support if needed

### 6.3 Threats to Validity

- Single dataset
- Limited seeds
- Limited budget
- Uneven literature support across methods
- No broader family such as `GWO`

This section is better placed here than under Experimental Setup because it belongs to interpretation, not procedure.

## 7. Conclusion

### 7.1 Best-Performing Method Overall

- State the empirically best method across all implemented methods

### 7.2 Best-Performing Primary Method

- State the best method within the primary comparison set

### 7.3 Practical Takeaway

- Was optimization worth it relative to the `RF` baseline?
- What is the operational meaning of the observed trade-offs?

### 7.4 Future Work

- Broader datasets
- More seeds
- Stronger additional optimizers
- Multi-objective extension

## Appendix

- Detailed parameter settings
- Convergence curves
- Extra tables
- Extended plots

## Why This Structure Is Better

### 1. It Matches the Rubric More Directly

- detailed algorithm description
- baseline-inclusive comparison
- most effective method
- security trade-offs

### 2. It Separates Literature Strength from Result Strength

- `GA` and `PSO` have stronger direct IDS support
- `SA` is still supportable
- `Tabu Search` and `VNS` remain visible without being overstated

### 3. It Avoids the Appearance of Selective Reporting

- all methods are shown first
- primary focus is an interpretive choice, not a filtering trick

## Key Sentence to Reuse in the Paper

This study reports overall empirical results for all implemented methods, namely RF, GA, PSO, SA, Tabu Search, and VNS. However, the primary comparative discussion focuses on RF, GA, PSO, and SA because these methods are supported more directly by recent IDS optimization literature, while Tabu Search and VNS are treated as additional comparators that help interpret the search landscape.
