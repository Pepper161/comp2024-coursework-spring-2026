# IDS Feature Selection and Hyperparameter Optimisation Using Metaheuristic Algorithms Versus a Default Random Forest Baseline

> Template note: the official conference-paper `.docx` template referenced in the assessment sheet is not present in the cleaned repository. This draft is therefore written in markdown with sectioning that can be transferred into the official template without changing the intended paper structure.

## Abstract

This coursework investigates feature selection and hyperparameter optimisation for intrusion detection on the UNSW-NB15 dataset using three metaheuristic algorithms: Genetic Algorithm (GA), Particle Swarm Optimisation (PSO), and Simulated Annealing (SA). A default Random Forest using all available features is used as the non-metaheuristic baseline required by the assessment. The optimisation problem jointly searches for a grouped original-feature subset and Random Forest hyperparameters while preserving a leakage-safe evaluation protocol. The primary comparison uses a budget of 50 objective evaluations at seed 0, while a lighter robustness check uses a budget of 30 objective evaluations across seeds 0, 1, and 2. In the main comparison, all three metaheuristics substantially reduced false positives and selected feature count relative to the baseline while maintaining high recall. GA produced the strongest overall single-run trade-off, achieving the best test F1-score (0.9267), the lowest test false positive rate among the metaheuristics (0.1589), and the smallest selected feature set (17 features). In the lower-budget repeated-seed robustness check, PSO appeared more stable, with the lowest mean test false positive rate (0.1608 ± 0.0054), the smallest mean selected feature count (18.67 ± 1.53), and the fastest mean optimisation time (1130.81 ± 60.46 s). SA improved over the baseline but was less attractive in stability and computational cost. The evidence therefore supports a conditional conclusion rather than a universal one: GA is the strongest method in the main comparison, while PSO appears more robust under the lighter repeated-seed setting.

## 1. Introduction

Intrusion Detection Systems (IDSs) analyse network traffic or system activity to identify malicious behaviour, policy violations, or unauthorised access attempts. In practice, IDS datasets often contain many features, some of which are redundant or weakly informative. This creates two linked optimisation questions: which features should be retained, and which classifier hyperparameters should be chosen. For IDS deployment, this is not only a predictive-accuracy problem. A useful IDS must also control false positives, because excessive false alarms reduce analyst trust and increase operational burden.

The assessment sheet defines this coursework as an IDS feature-selection and hyperparameter-optimisation task and requires at least three metaheuristics, one baseline comparison, and explicit discussion of trade-offs between detection accuracy, false positives, and number of selected features. This project follows that brief directly. The fixed base classifier is Random Forest, and the candidate search methods are GA, PSO, and SA. These three methods were chosen because they represent meaningfully different optimisation behaviours while still fitting the assessment brief: GA performs population-based recombination and mutation, PSO performs cooperative swarm search using personal and global best positions, and SA performs single-solution stochastic search with occasional uphill moves to escape local minima. Together they provide a useful comparison between evolutionary, swarm-based, and trajectory-based metaheuristic optimisation.

The core research question is therefore:

> Can metaheuristic optimisation improve an IDS built on Random Forest by reducing false positives and feature count while maintaining strong detection performance, relative to a default baseline?

To answer this question, the project compares a default Random Forest baseline against GA-, PSO-, and SA-optimised feature subset and hyperparameter configurations on the UNSW-NB15 dataset in a binary classification setting.

## 2. Problem Formulation

The coursework treats IDS feature selection and hyperparameter tuning as a joint optimisation problem. Each candidate solution contains:

1. a binary mask over the original input features, and
2. a set of Random Forest hyperparameters.

The implementation uses grouped original-feature selection rather than independent one-hot-column selection. This matters because categorical variables such as `proto`, `service`, and `state` are encoded into multiple one-hot columns. Selecting a categorical feature therefore enables all of its encoded columns as a group. This keeps the representation interpretable and avoids unstable partial selection of one-hot fragments.

The optimisation objective used in the repository is:

`fitness = Recall - lambda_fpr * max(0, FPR - alpha) - lambda_feat * (k / D)`

where:

- `Recall` is the validation recall,
- `FPR` is the validation false positive rate,
- `alpha = 0.05` is the FPR tolerance threshold,
- `lambda_fpr = 20.0` is the penalty weight for exceeding that threshold,
- `lambda_feat = 0.2` is the feature-count penalty,
- `k` is the number of selected original features,
- `D` is the total number of original features after dropping the target and leakage-prone columns.

This formulation is appropriate for IDS because it favours attack detection while explicitly penalising false positives and unnecessarily large feature subsets. The project therefore does not optimise only for raw classification performance. It optimises for a more operationally relevant balance.

## 3. Dataset and Preprocessing

The project uses the UNSW-NB15 dataset, which is explicitly listed in the assessment sheet as an acceptable alternative dataset. The target variable is `label`, treated as a binary classification task (`0 = normal`, `1 = attack`). The repository loads:

- `dataset/UNSW_NB15_training-set.csv`
- `dataset/UNSW_NB15_testing-set.csv`

Before preprocessing, the implementation removes:

- `label` from the predictors, because it is the target;
- `id`, because it is not a meaningful predictive feature;
- `attack_cat`, because it would leak attack-category information into the binary classifier.

The preprocessing pipeline is leakage-safe and follows the boundary required for honest evaluation:

- numeric features are imputed with the median;
- categorical features (`proto`, `service`, `state`) are imputed with the most frequent value and one-hot encoded with `handle_unknown="ignore"`;
- the preprocessing transform is fit only on the relevant training data for each stage.

During optimisation, the training CSV is split into `train_inner` and `val_inner` using stratified sampling with `val_size = 0.2`. Preprocessing is fit on `train_inner` only, then applied to `val_inner`. The testing CSV is never used during optimisation. For final one-shot test evaluation, preprocessing is re-fit on the full training CSV and then applied to the held-out test CSV. This protocol prevents test leakage while still using all training data for the final reported model.

## 4. Methods

### 4.1 Baseline

The benchmark model is a default Random Forest using all original features and default hyperparameters. This satisfies the assessment requirement for at least one non-metaheuristic comparison method using the default feature set and default hyperparameter values. The baseline is intentionally simple. Its role is to show what performance is obtained without any feature selection or metaheuristic search.

### 4.2 Shared Search Space

All three metaheuristics optimise the same Random Forest search space in the final reported runs:

- `n_estimators`: 100 to 300
- `max_depth`: 6 to 20
- `min_samples_split`: 2 to 20
- `min_samples_leaf`: 1 to 8
- `max_features`: `{sqrt, log2}`
- `class_weight`: `{None, balanced}`

The grouped feature mask also enforces `k_min = 8`, preventing trivial near-empty solutions.

### 4.3 Genetic Algorithm

The GA implementation is a steady-state genetic algorithm. It evaluates an initial population, then repeatedly:

1. selects parents by tournament selection,
2. applies one-point crossover for feature genes,
3. mixes hyperparameter genes from the parents,
4. mutates feature bits and perturbs continuous genes,
5. replaces the current worst individual.

GA was chosen because it is naturally suited to mixed discrete-continuous search spaces. Feature masks are binary, while hyperparameters are decoded from continuous gene values. GA can therefore explore both parts of the representation within a unified chromosome.

### 4.4 Particle Swarm Optimisation

The PSO implementation uses a hybrid binary/continuous update. Each particle keeps:

- its current position,
- its personal best position,
- the swarm global best position.

Hyperparameter genes are updated continuously, while feature genes are updated through a sigmoid-transformed velocity and sampled as binary activations. PSO was chosen because it provides a contrast with GA: it is still population-based, but it does not rely on recombination. Instead, it uses social and cognitive movement towards promising regions of the search space.

### 4.5 Simulated Annealing

The SA implementation is a single-solution search process. At each step it:

1. proposes a neighbour by flipping a small number of feature bits and perturbing hyperparameter genes,
2. accepts improvements directly,
3. occasionally accepts worse solutions with a probability controlled by temperature.

The temperature decreases geometrically. SA was chosen because it represents a different optimisation style from GA and PSO: instead of maintaining a population or swarm, it relies on local stochastic exploration with a controlled probability of escaping local minima.

## 5. Experimental Setup and Reproducibility

The final evidence in this repository is intentionally split into two levels.

### 5.1 Main Comparison

The main comparison uses:

- `evaluations_B = 50`
- `seed = 0`
- separate single-optimizer runs for GA, PSO, and SA
- the same Random Forest search space for all three methods

This is the strongest single-run comparison available in the repository and is treated as the primary result set.

### 5.2 Robustness Check

The supporting robustness check uses:

- `evaluations_B = 30`
- `seeds = 0, 1, 2`
- the same preprocessing logic and optimisation objective
- repeated runs for GA, PSO, and SA

This robustness stage is lighter and therefore not directly equivalent to the main comparison. It is used to test stability and computational-cost patterns, not to replace the primary result.

### 5.3 Fairness Controls

The repository implements several fairness and reproducibility controls:

- all methods use the same evaluation budget `B` within a given comparison;
- cache hits still consume an evaluation unit, so no method gets extra search for free;
- the optimiser cache is isolated per `(algorithm, seed)`;
- `n_jobs` is fixed to `1` for Random Forest;
- the model `random_state` is derived from the current seed;
- the inner split is stratified;
- run configuration and seed list are saved to each result folder.

These design choices matter because metaheuristic comparisons are otherwise easy to distort through inconsistent budgets, inconsistent randomness, or accidental data leakage.

## 6. Results

### 6.1 Main Comparison: `B = 50`, `seed = 0`

Table 1 summarises the main comparison. It is based on the final raw outputs under:

- `results/ga/b50_seed0/raw/all_runs.csv`
- `results/pso/b50_seed0/raw/all_runs.csv`
- `results/sa/b50_seed0/raw/all_runs.csv`

| Method | Eval B | Test Accuracy | Test Precision | Test Recall | Test F1 | Test FPR | Features | Optimisation Time (s) | Final Test Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline RF | 0 | 0.8718 | 0.8188 | 0.9852 | 0.8943 | 0.2670 | 42 | 0.0000 | 40.4638 |
| GA | 50 | 0.9151 | 0.8826 | 0.9754 | 0.9267 | 0.1589 | 17 | 1901.7177 | 30.0006 |
| PSO | 50 | 0.9111 | 0.8746 | 0.9788 | 0.9238 | 0.1719 | 18 | - | 29.4528 |
| SA | 50 | 0.9061 | 0.8701 | 0.9750 | 0.9196 | 0.1784 | 18 | - | 48.2189 |

![Trade-off scatter for robustness runs](../results/figures/tradeoff_scatter_revised.png)

The baseline achieves the highest recall (`0.9852`), but it also produces the highest false positive rate (`0.2670`) and uses all `42` original features. This establishes the central IDS trade-off in the project: preserving extremely high recall is possible, but it may come with an operationally unattractive false-alarm burden.

All three metaheuristics improve on the baseline in test accuracy, test F1, test FPR, and selected feature count. The practical meaning of this is important. The optimised models do not merely fit slightly better. They also reduce the feature set by more than half while substantially lowering false positives. This is a meaningful result for IDS because a smaller feature set can reduce data-processing cost and model complexity, while lower FPR reduces analyst fatigue.

Among the three optimisers, GA provides the strongest overall single-run trade-off:

- best test F1 (`0.9267`);
- lowest test FPR among the optimisers (`0.1589`);
- smallest selected feature set (`17`);
- strong accuracy (`0.9151`).

PSO performs very competitively. Its validation best score is the highest in the main comparison (`0.8678`), and its test recall (`0.9788`) is slightly higher than GA. However, it keeps one extra feature and yields a higher test FPR than GA. SA still improves on the baseline, but its test F1, test FPR, and feature compactness are weaker than GA, and its final test evaluation time is noticeably larger.

On the evidence available in the main comparison, GA is therefore the strongest candidate. However, that conclusion should be framed carefully because this comparison is based on a single seed.

### 6.2 Robustness Check: `B = 30`, `seeds = 0,1,2`

Table 2 summarises the lighter repeated-seed robustness runs.

| Method | Runs | Val Best Score | Test Recall | Test FPR | Test F1 | Features | Optimisation Time (s) | Total Run Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_rf_default | 3 | - | 0.9854 ± 0.0002 | 0.2675 ± 0.0004 | 0.8943 ± 0.0000 | 42.00 ± 0.00 | 0.00 ± 0.00 | 43.32 ± 2.88 |
| ga | 3 | 0.7998 ± 0.0350 | 0.9515 ± 0.0191 | 0.1678 ± 0.0124 | 0.9112 ± 0.0084 | 23.00 ± 1.73 | 1166.64 ± 44.74 | 1219.45 ± 38.67 |
| pso | 3 | 0.7461 ± 0.0657 | 0.9511 ± 0.0162 | 0.1608 ± 0.0054 | 0.9134 ± 0.0085 | 18.67 ± 1.53 | 1130.81 ± 60.46 | 1184.36 ± 52.16 |
| sa | 3 | 0.4610 ± 0.6024 | 0.9752 ± 0.0159 | 0.1978 ± 0.0711 | 0.9132 ± 0.0165 | 22.00 ± 4.36 | 1263.21 ± 231.15 | 1320.98 ± 221.50 |

![Distribution of test FPR across robustness runs](../results/figures/distribution_test_fpr_revised.png)

![Distribution of optimisation time across robustness runs](../results/figures/distribution_runtime_revised.png)

The robustness evidence is deliberately more modest. Only three seeds are available, and the budget is smaller than in the main comparison. Therefore, these results should be interpreted as supporting evidence rather than definitive proof of stability. Even with that caution, some patterns are clear.

First, the baseline remains very high-recall but consistently high-FPR. Second, both GA and PSO continue to reduce FPR substantially relative to the baseline. Third, SA becomes the least stable method in this lower-budget repeated setting, with much larger standard deviation in both validation score and test FPR. This matters because an IDS optimisation method is only attractive if it performs consistently, not only occasionally.

The robustness results also improve the computational-cost discussion. In this repeated-seed setting:

- PSO has the lowest mean optimisation time (`1130.81 ± 60.46 s`);
- GA is close behind (`1166.64 ± 44.74 s`);
- SA is slower and much more variable (`1263.21 ± 231.15 s`).

This makes PSO particularly attractive when optimisation cost is considered explicitly. By contrast, the main comparison alone might have encouraged a stronger GA preference. The two result layers therefore tell a more nuanced story: GA is the strongest main-run method, but PSO looks more stable and computationally efficient in the lighter repeated-seed setting.

### 6.3 Feature Selection Content

Feature selection is not only about reducing the total number of variables. The report should also ask which features are repeatedly retained by the optimisers.

![Feature selection frequency across robustness runs](../results/figures/feature_selection_frequency_revised.png)

The feature-frequency heatmap shows that some features are selected consistently across more than one method. For example, `dmean` and `sbytes` are selected in all robustness best solutions for GA, PSO, and SA. Other features show method-specific behaviour. For instance, `sjit` is always selected by GA and SA but not by PSO, while `swin` is always selected by PSO but not by the other two methods. This supports two observations:

1. some variables appear to be broadly useful regardless of the search strategy; and
2. the optimisers are not simply converging to the same feature subset.

This makes the feature-selection stage meaningful rather than cosmetic. The metaheuristics are genuinely exploring different compact representations of the IDS problem.

### 6.4 Convergence Behaviour

![Convergence summary across seeds](../results/figures/convergence_summary_across_seeds_revised.png)

The convergence summary provides a process-level view of optimisation rather than a final-score-only comparison. GA and PSO both improve sharply in the early part of the budget, while SA exhibits a less stable progression. With only three robustness seeds, this figure should not be over-interpreted. However, it supports the broader robustness picture: GA and PSO are easier to defend as practical optimisers than SA under the current budget settings.

## 7. Critical Discussion

The most important result in this project is not simply that the optimised methods outperform the baseline on F1-score. The more important finding is that they improve the trade-off between detection and false alarms.

In IDS, recall is necessary but not sufficient. An IDS that detects most attacks but floods analysts with false alerts can still be operationally poor. The baseline demonstrates this clearly: it produces the highest recall, but its false positive rate is substantially worse than all three metaheuristics. From a security-operations perspective, this suggests that the baseline would generate more alert noise and therefore impose a heavier analyst burden.

The optimised methods accept a small recall reduction in exchange for lower FPR and smaller feature sets. That trade-off is defensible and, in this coursework context, desirable. The reduction in recall from the baseline to GA is small (`0.9852` to `0.9754`), but the reduction in FPR is large (`0.2670` to `0.1589`). Similarly, the feature set shrinks from `42` to `17`. This is the strongest single indication that metaheuristic optimisation is worthwhile for this IDS setting.

The comparison among GA, PSO, and SA is more nuanced than a single “best method” claim. GA is the strongest method in the primary comparison because it simultaneously achieves the best F1-score, the lowest FPR among the optimisers, and the smallest feature set. That is a compelling single-run result. However, the supporting robustness runs suggest that PSO may be the more stable method under tighter budgets. Its repeated-run FPR is the lowest on average, its optimisation cost is the lightest, and its feature count remains compact. SA, while still better than the baseline, is less persuasive in practice because it combines higher runtime variability with weaker FPR stability.

This leads to a careful conclusion. It would be too strong to claim that GA is universally best, because the robustness evidence does not fully support that claim. It would also be too strong to claim that PSO is definitively superior, because the primary `B=50, seed=0` comparison favours GA. The evidence supports a conditional judgment instead:

- **GA is the strongest method in the primary comparison.**
- **PSO appears more stable in the lighter repeated-seed robustness check.**
- **SA improves the baseline but is weaker in stability and computational cost.**

This distinction is important because it reflects the real limits of the evidence rather than smoothing them away.

## 8. Limitations and Future Work

Several limitations should be stated explicitly.

First, the strongest comparison in the repository is a single-seed run at `B = 50`. This is enough to compare methods, but it is not enough to make strong claims about stability.

Second, the robustness check uses only three seeds and a smaller budget (`B = 30`). It therefore tests repeated behaviour under a lighter setting, not under the exact same setting as the primary comparison.

Third, optimisation-time evidence is stronger for the robustness runs than for the main comparison. In the main `B = 50` results, optimisation wall-clock time is fully available for GA but not for PSO and SA. For that reason, runtime discussion in this paper relies more heavily on the robustness evidence.

Fourth, this repository-grounded draft does not include an external literature review or formal citations beyond what is implicit in the assessment sheet and repository evidence. That limits the depth of theoretical justification compared with a fully polished final submission.

Future work should therefore include:

1. repeated-seed evaluation at a budget closer to the main comparison;
2. fuller runtime logging for all main runs;
3. possible multi-objective extensions that model recall, FPR, feature count, and runtime jointly rather than through a single weighted objective;
4. a fuller literature-backed justification of the chosen metaheuristics.

## 9. Conclusion

This coursework demonstrates that metaheuristic optimisation can improve an IDS built on Random Forest by reducing false positives and feature count while retaining strong detection performance. Relative to the default baseline, GA, PSO, and SA all produce better overall trade-offs on UNSW-NB15.

The default baseline achieves the highest recall, but its false positive rate is too high to be attractive from an operational IDS perspective. In the main comparison (`B = 50`, `seed = 0`), GA provides the best overall result, combining the highest test F1-score, the lowest test FPR among the optimisers, and the smallest feature set. In the lighter robustness setting (`B = 30`, `seeds = 0,1,2`), PSO appears more stable and computationally efficient. SA improves on the baseline but is less convincing because of weaker stability and heavier optimisation cost.

The most defensible conclusion is therefore conditional rather than absolute. GA is the strongest reported method in the main comparison, while PSO is the more stable method in the repeated lighter-budget runs. That is a useful and evidence-based answer to the coursework problem, and it supports the broader claim that metaheuristic feature selection and hyperparameter optimisation can make IDS models more practical than a default untuned baseline.
