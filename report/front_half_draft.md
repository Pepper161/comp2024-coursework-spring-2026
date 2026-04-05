# Front-Half Paper Draft

## 1. Introduction

### 1.1 Background and Motivation

Network intrusion detection systems (IDS) are expected to identify malicious activity while maintaining an operationally acceptable false-alarm burden. This is difficult in contemporary network environments because traffic patterns are heterogeneous, attacks are diverse, and benchmark datasets contain many features with uneven predictive value. In practice, an IDS that achieves high recall at the cost of excessive false positives may still be operationally weak, because it overloads analysts with benign alerts and reduces trust in the system. For this reason, intrusion detection should be treated as a security trade-off problem rather than as an accuracy-only classification task.

The `UNSW-NB15` dataset remains a useful benchmark for this setting because it was designed to reflect modern network traffic and multiple attack families more faithfully than older datasets. It contains a mixture of categorical protocol/state attributes and derived traffic statistics, which makes it suitable for studying both model selection and feature-selection behaviour. Prior work on `UNSW-NB15` has shown that conventional classifiers such as Random Forest (RF) can already provide strong detection performance, but also that redundant or weakly informative features can increase noise, computational overhead, and overfitting risk [Kasongo and Sun, 2020; Dawood et al., 2024]. Recent studies further show that `UNSW-NB15` feature-selection pipelines often use RF-derived importance information, which reinforces the role of RF as a natural benchmark in this problem setting [Yin et al., 2023].

These characteristics motivate two linked optimisation tasks. The first is feature selection, where the goal is to remove unnecessary features while preserving detection quality. The second is hyperparameter optimisation, where the aim is to find better-performing model configurations than fixed default settings. Solving these tasks jointly is attractive because the usefulness of a feature subset depends on the classifier configuration used to exploit it. However, the joint search space of feature subsets and model hyperparameters is combinatorial and too large for exhaustive search. Metaheuristic optimisation therefore provides a practical way to explore this space under a fixed evaluation budget.

### 1.2 Research Objective

This study evaluates whether metaheuristic search can produce meaningful improvements over a non-metaheuristic RF baseline on `UNSW-NB15`. The comparison is not limited to predictive quality alone. Instead, the analysis explicitly considers `F1`, `Recall`, `FPR`, selected feature count, and runtime, because an IDS method should be judged by its overall security and operational trade-off rather than by a single metric. The project uses wrapper-based search, where candidate feature subsets and RF hyperparameters are evaluated directly through model performance on a validation split.

This study reports overall empirical results for all implemented methods, namely RF, GA, PSO, SA, Tabu Search, and VNS. However, the main comparative discussion focuses on RF, GA, PSO, and SA because these methods are supported more directly by recent IDS optimization literature, while Tabu Search and VNS are treated as additional comparators that help interpret the search landscape.

### 1.3 Contributions

This study makes four contributions. First, it provides a reproducible comparison of metaheuristic wrapper methods for `UNSW-NB15` under a common preprocessing pipeline, a shared RF search space, and explicit fairness rules. Second, it reports an overall empirical comparison across all implemented methods, rather than selectively presenting only a subset of them. Third, it separates the full empirical picture from the narrower main comparison, allowing the study to remain transparent about observed outcomes while keeping its interpretive focus aligned with the strongest IDS-specific literature support. Fourth, it analyses security trade-offs directly by discussing not only detection quality but also false positives, feature reduction, runtime, and recurring feature relevance patterns.

## 2. Related Work and Study Positioning

### 2.1 IDS on UNSW-NB15 and Conventional Baselines

`UNSW-NB15` has been used extensively to evaluate machine learning-based IDS models, especially in studies that compare conventional classifiers before or alongside feature-selection or optimisation techniques. In this literature, RF is a particularly defensible baseline. Kasongo and Sun compared conventional classifiers on `UNSW-NB15` and highlighted RF as one of the strongest overall conventional methods in that setting [Kasongo and Sun, 2020]. Dawood et al. similarly reported that RF performs strongly relative to other standard learners on `UNSW-NB15`, including in terms of false-alarm behaviour [Dawood et al., 2024]. Even when RF is not the final classifier under study, it remains common as a feature-importance mechanism inside `UNSW-NB15` feature-selection pipelines [Yin et al., 2023].

This pattern matters for the present work. The baseline in this paper is not intended to be weak or trivial. It is intended to be a credible non-metaheuristic benchmark that reflects a practical, widely recognisable IDS starting point. Using a default RF with all available grouped features therefore provides a strong reference level against which the value of optimisation can be measured.

### 2.2 Metaheuristic Optimisation for IDS

Recent IDS optimisation literature provides the strongest direct support for `GA` and `PSO`, with `SA` remaining supportable but somewhat less central. `GA` is well established in IDS feature-selection research and also appears in recent work on IDS hyperparameter tuning and hybrid optimisation pipelines [Halim et al., 2021; Bakır and Ceviz, 2024]. This makes it a natural evolutionary benchmark for the present study. `PSO` is similarly well supported in IDS-specific optimisation work, especially in wrapper-based feature selection and hyperparameter optimisation contexts [Chohra et al., 2022; Kilichev and Kim, 2023]. Together, `GA` and `PSO` can be described safely as widely used benchmark metaheuristics for IDS optimisation.

`SA` is somewhat different. It has weaker direct support in recent IDS literature than `GA` or `PSO`, but it still has relevant IDS-adjacent evidence. Huang et al. incorporated simulated annealing behaviour into an IDS-oriented hybrid feature-selection framework and showed that annealing-style acceptance mechanisms remain relevant to recent intrusion-detection optimisation [Huang et al., 2024]. More broadly, `SA` remains a recognised feature-selection metaheuristic in general optimisation literature, even if its IDS-specific presence is not as strong as that of `GA` and `PSO`. In this paper, `SA` is therefore retained as the stochastic local-search member of the primary comparison set rather than being presented as the dominant IDS optimiser.

### 2.3 Additional Local-Search Comparators

`Tabu Search` and `VNS` are also relevant to the optimisation problem considered here, but their literature position is weaker in direct IDS terms. `Tabu Search` has established historical use in feature selection and remains active in recent feature-selection research outside IDS [Huerta et al., 2002; Pacheco et al., 2023]. `VNS` is likewise a valid high-dimensional feature-selection strategy and has a strong general optimisation pedigree, but its recent direct support in IDS-specific literature appears weaker than that of `GA` or `PSO` [Consoli et al., 2016; Mladenović et al., 2017]. For that reason, neither method is used as a core literature-backed benchmark in the main comparative discussion.

This positioning does not imply that `Tabu Search` or `VNS` are invalid or unimportant. Instead, this study distinguishes between methods that are easiest to justify from recent IDS literature and methods that are still worth reporting because they can illuminate the behaviour of the broader search landscape. If either of these additional methods performs strongly in the empirical study, that outcome is reported directly and interpreted with the same metric framework used for the primary methods.

### 2.4 Positioning of This Study

The study is positioned as a comparison between a strong conventional baseline and three primary metaheuristics, while still reporting the outcomes of two additional local-search methods. Accordingly, the primary comparative focus is `RF`, `GA`, `PSO`, and `SA`. `Tabu Search` and `VNS` are included as additional comparators that broaden the empirical picture and help interpret the optimisation landscape, but they are not used as the central IDS-backed benchmark set. This framing keeps the empirical reporting transparent while avoiding stronger literature claims than the evidence can support.

## 3. Problem Formulation and Experimental Protocol

### 3.1 Task Definition

The task is binary intrusion-detection classification on `UNSW-NB15`. Given the provided training and test files, the goal is to learn a model that can distinguish normal from malicious traffic while maintaining a useful security trade-off. In this project, that trade-off is defined by a combination of strong detection quality, controlled false-positive behaviour, compact feature subsets, and reasonable runtime. The optimisation problem therefore goes beyond standard classification and instead seeks a balanced IDS configuration.

More formally, the search procedure jointly optimises a grouped original-feature subset and a Random Forest hyperparameter setting. Candidate solutions are evaluated through validation performance rather than test-set feedback, so the test set remains reserved for final one-shot evaluation after optimisation. This preserves test-set integrity and aligns the search process with a standard wrapper-optimisation protocol.

### 3.2 Data and Preprocessing

The project uses `dataset/UNSW_NB15_training-set.csv` and `dataset/UNSW_NB15_testing-set.csv`, with `label` as the binary target and `id` plus `attack_cat` removed from the feature space. The outer train/test split is therefore determined by the provided dataset files. Within the training portion, an additional validation split of `0.2` is used for optimisation. This inner split is stratified and seeded, and the same split is reused across the baseline and all compared metaheuristics for a given seed.

Preprocessing is leakage-safe. Numeric features are imputed using the median, while categorical features are imputed using the most frequent category. The categorical variables `proto`, `service`, and `state` are one-hot encoded with `handle_unknown="ignore"`. Importantly, the preprocessing pipeline is fit on the training partition only and then applied to the validation and test partitions via transformation. This design avoids leaking validation or test information into the fitted preprocessing state.

The feature-selection stage operates over grouped original features rather than over raw one-hot columns independently. This is important because it preserves the semantic meaning of the original variables and avoids degenerate solutions in which partial one-hot fragments of the same original feature are selected inconsistently.

### 3.3 Baseline Model

The baseline model is a default RF using all original grouped features and no metaheuristic search. The baseline hyperparameters are `n_estimators = 100`, `max_depth = None`, `min_samples_split = 2`, `min_samples_leaf = 1`, `max_features = "sqrt"`, and `class_weight = None`. This baseline serves two purposes. First, it provides a strong conventional reference point from the IDS literature. Second, it makes it possible to assess whether the additional computational cost of metaheuristic search is justified by improvements in detection quality, false-positive control, feature reduction, or runtime trade-off.

### 3.4 Search Representation

Each candidate solution combines two components: a grouped original-feature selection mask and six RF hyperparameter genes. The decoding logic is implemented in `src/representation.py`. At least `k_min = 8` original features must remain selected, which prevents degenerate near-empty subsets and ensures that the resulting classifiers remain meaningful. For the main local `B=120` experiments, the RF search space is restricted to practical ranges: `n_estimators` in `[100, 300]`, `max_depth` in `[6, 20]`, `min_samples_split` in `[2, 20]`, `min_samples_leaf` in `[1, 8]`, `max_features` in `{"sqrt", "log2"}`, and `class_weight` in `{None, "balanced"}`.

### 3.5 Fitness Function

The implemented fitness function rewards recall while penalising both excessive false positives and unnecessarily large feature subsets. In `src/evaluator.py`, fitness is defined as:

`fitness = recall - lambda_fpr * max(0, fpr - alpha_fpr) - lambda_feat * (k / d)`

where `alpha_fpr = 0.05`, `lambda_fpr = 20.0`, `lambda_feat = 0.2`, `k` is the number of selected original features, and `d` is the total number of original features. This design reflects the fact that IDS performance should not be judged by recall alone. A method that achieves high recall by producing too many false alarms is not necessarily operationally preferable. Likewise, a feature subset should not be rewarded merely for being small if that reduction is achieved through a serious loss in detection quality.

The results narrative therefore relies on directly interpretable metrics such as `F1`, `Recall`, `FPR`, selected feature count, and runtime rather than on an ambiguous composite headline score. This is especially important because one legacy logged `GA` main-run composite score is not fully aligned with the final evaluator definition.

### 3.6 Evaluation Criteria

The evaluation uses `Accuracy`, `Precision`, `Recall`, `F1`, `FPR`, selected feature count, and runtime. These metrics align both with common `UNSW-NB15` reporting practice and with the coursework requirement to discuss trade-offs between predictive performance, false positives, and feature count. Among these metrics, `F1`, `Recall`, and `FPR` are the most important for interpretation, because they jointly capture the balance between successful detection and operational alert noise. Selected feature count is included because the project is explicitly a feature-selection study, and runtime is included because an optimiser that is only marginally better but substantially more expensive may offer limited practical value.

### 3.7 Reproducibility and Fairness Policy

All compared methods use the same dataset split for a given seed, the same preprocessing pipeline, the same evaluator logic, the same feature-selection representation, and the same evaluation-budget accounting policy. Within each comparison family, the RF search-space policy is also kept fixed. The main comparison uses the strongest available local setting, `B = 120` and `seed = 0`, while lightweight robustness evidence is drawn separately from `B = 30` runs over seeds `0, 1, 2`. These robustness runs are used only to qualify stability and should not be interpreted as equivalent in strength to the main `B = 120` comparison.

## 4. Algorithms

### 4.1 Primary Algorithms

#### Genetic Algorithm

`GA` represents the evolutionary-search family in the primary comparison. It evolves a population of candidate solutions using selection, crossover, and mutation, which makes it suitable for mixed search spaces that contain both discrete feature decisions and structured hyperparameter choices. In the present work, `GA` is included because it is one of the strongest IDS-backed optimisation benchmarks and because its search behaviour offers a useful balance between broad exploration and exploitation of promising regions. It is expected to perform well when interactions between feature subsets and RF hyperparameters need to be discovered jointly.

#### Particle Swarm Optimization

`PSO` represents the swarm-intelligence family. It maintains a population of particles that move in the search space using both their own historical best position and the global best position discovered so far. In a mixed wrapper-optimisation setting, this makes `PSO` a strong contrast to `GA`: both are population-based, but they update solutions according to different search dynamics. `PSO` is included because recent IDS literature directly supports it as a widely used optimisation benchmark, especially for feature selection and hyperparameter tuning. In this study, it is expected to offer efficient broad search under a fixed evaluation budget.

#### Simulated Annealing

`SA` represents stochastic local search in the primary comparison. Unlike `GA` and `PSO`, it operates on a single current solution and accepts some worse moves probabilistically, with the acceptance tendency decreasing over time. This makes `SA` relevant when the search space is rugged and contains many local optima. It is included as the primary local-search comparator because it has more direct IDS-relevant support than other local-search candidates and because its probabilistic acceptance rule offers a conceptually different search behaviour from population-based methods. In the present work, `SA` serves as the primary benchmark for local improvement under controlled stochastic exploration.

### 4.2 Additional Algorithms

#### Tabu Search

`Tabu Search` is included as an additional memory-based local-search comparator. Its defining idea is to use a tabu list to prevent immediate revisiting of recently explored solutions, thereby encouraging diversification while still exploiting local structure. This is relevant to the current optimisation problem because grouped feature masks and RF hyperparameters create a mixed discrete search space in which local revisitation can be wasteful. `Tabu Search` is not treated as a primary method because its recent direct IDS literature support is weaker than that of `GA`, `PSO`, and `SA`. However, it remains methodologically meaningful and is retained to help interpret how memory-guided local search behaves relative to the main methods.

#### Variable Neighborhood Search

`VNS` is included as an additional neighborhood-changing local-search comparator. Its core principle is that a solution that is locally optimal under one neighborhood structure may not be locally optimal under another. By changing neighborhood structures systematically, `VNS` can escape local optima without relying on a population. This is attractive for high-dimensional feature-selection problems and is supported by general feature-selection literature. Nevertheless, `VNS` is treated as an additional method rather than a primary benchmark because recent direct IDS-specific support is weaker than for `GA` and `PSO`. `VNS` is therefore included as an additional empirical comparator that may still produce strong results, but without being overclaimed as a core IDS benchmark.

### 4.3 Algorithm Comparison Summary

Table X summarizes the practical differences among the compared optimizers. This is useful because the methods differ not only in literature support but also in how they traverse the same mixed search space under the same evaluation budget.

| Method | Search unit | Main exploration mechanism | Main exploitation mechanism | Memory / acceptance device | Stopping rule |
| --- | --- | --- | --- | --- | --- |
| `GA` | population of candidate solutions | crossover and mutation over mixed feature-hyperparameter encodings | selection pressure toward higher-fitness individuals | population diversity | evaluation budget `B` exhausted |
| `PSO` | swarm of particles | velocity-driven movement toward personal and global best regions | attraction to historically good positions | personal-best and global-best memory | evaluation budget `B` exhausted |
| `SA` | single incumbent solution | probabilistic acceptance of some worse moves early in search | gradual cooling toward local refinement | temperature-controlled acceptance rule | evaluation budget `B` exhausted |
| `Tabu Search` | single incumbent solution with neighborhood scan | forced neighborhood exploration with short-term move prohibition | best admissible local move selection | tabu list | evaluation budget `B` exhausted |
| `VNS` | single incumbent solution across multiple neighborhoods | systematic neighborhood changes to escape local optima | local improvement inside each neighborhood | neighborhood-switching schedule | evaluation budget `B` exhausted |
