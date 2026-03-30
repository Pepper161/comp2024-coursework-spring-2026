# Professor Checklist

1. **Required project objectives**
- Preprocess the IDS dataset and explore feature relevance.
- Apply at least **three metaheuristic algorithms** for feature selection and/or hyperparameter tuning.
- Evaluate performance using relevant metrics.
- Compare the optimized model against at least **one baseline non-metaheuristic method** with default feature set and default hyperparameters.
- Analyse trade-offs between detection accuracy, false positives, and number of features.
- Discuss the performance of the metaheuristics against the benchmark.

2. **What must appear in the paper**
- Problem formulation with objective and constraints.
- Dataset preprocessing.
- Clear description of the selected algorithms.
- Justification for why those algorithms were chosen.
- Experimental results comparing baseline and metaheuristics.
- Critical evaluation of strengths and weaknesses.
- Identification of the most effective method based on empirical findings.
- Discussion of security trade-offs.
- Well-labeled tables, charts, or figures.
- Conclusions and future work.
- The original conference template formatting must remain unchanged.

3. **What distinguishes Excellent vs Good vs Poor**
- **Excellent**
  - Strong rationale for chosen algorithms with support.
  - Thorough, logical discussion of results.
  - Clear interpretation of numerical outcomes.
  - Meaningful insights.
  - Well-executed charts/tables that support findings.
- **Good**
  - Reasonable discussion, but less depth.
  - Some interpretation, but somewhat generic.
  - Visuals present but limited or not fully supporting the discussion.
- **Poor**
  - Minimal discussion.
  - Weak or unsupported claims.
  - No meaningful interpretation of numerical results.
  - No useful visuals.
  - Incomplete or incorrect code / fewer than the required algorithms.

4. **Most mark-sensitive section**
- **Critical Analysis and Discussion of Findings for Performance Evaluation (25%)**
- This is the largest paper component and should dominate writing priorities.

5. **Common mistakes that would lose marks**
- Using fewer than 3 metaheuristics.
- Weak or missing baseline comparison.
- Reporting numbers without interpretation.
- Failing to discuss the trade-off between recall/detection and false positives.
- Ignoring the number of selected features.
- Weak or missing tables/figures.
- Making unsupported claims about the “best” method.
- Changing the provided conference template formatting.
- Submitting code that is hard to run or incomplete.

# Drafted Section

## Experimental Setup

This study evaluates feature selection and hyperparameter optimisation for intrusion detection using the UNSW-NB15 dataset in a binary classification setting. The target variable is `label`, and the fixed base classifier is Random Forest. The optimisation variables are the original grouped feature subset and the Random Forest hyperparameters. Three metaheuristic methods are considered: Genetic Algorithm (GA), Particle Swarm Optimisation (PSO), and Simulated Annealing (SA). The benchmark model is a default Random Forest using all available original features.

The reported evidence is divided into two levels. First, the main comparison uses a higher-budget setting with `evaluations_B = 50` and `seed = 0`. This provides the primary single-run comparison between the baseline and the three metaheuristic methods. Second, a lightweight robustness check uses `evaluations_B = 30` and `seeds = 0, 1, 2`. This second stage is not intended to replace the main comparison, but to test whether the observed behaviour remains broadly consistent under multiple random seeds and to provide optimisation-time evidence.

Performance is assessed using accuracy, precision, recall, F1-score, false positive rate (FPR), number of selected features, and runtime when supported by the available results. In the IDS context, this set of metrics is necessary because a method with very high recall may still be operationally weak if it produces too many false alarms, while an aggressively reduced feature set is only useful if detection performance remains acceptable.

## Results and Discussion

The baseline Random Forest achieves very high recall in the main comparison (`0.9852`), but its false positive rate is also high (`0.2670`), and it uses all `42` original features. This establishes a clear benchmark trade-off: the baseline is strong at detecting attacks, but weak at controlling false positives.

All three metaheuristic methods improve on the baseline in the main comparison. GA reaches test accuracy `0.9151`, test F1 `0.9267`, test FPR `0.1589`, and uses `17` features. PSO reaches test accuracy `0.9111`, test F1 `0.9238`, test FPR `0.1719`, and uses `18` features. SA reaches test accuracy `0.9061`, test F1 `0.9196`, test FPR `0.1784`, and uses `18` features. Directly observed from these numbers, all three optimisers reduce FPR substantially relative to the baseline while also reducing the feature set by more than half. Therefore, the optimised models provide a much more compact IDS representation without giving up most of the baseline recall.

Within the main comparison, GA shows the strongest overall balance. This is a direct observation from the reported metrics: GA has the highest test F1 among the metaheuristics, the lowest test FPR, and the smallest feature subset. PSO records the highest validation best score and slightly higher test recall than GA, but its false positive rate is worse and it keeps one additional feature. SA still improves over the baseline, but it is weaker than GA and PSO on the main test-set trade-off. Based on the main comparison alone, GA is the most convincing candidate under the reported `B = 50, seed = 0` setting.

However, that claim should be made cautiously because the main comparison is based on a single seed. The robustness check provides a more stable view across three lighter runs. In that setting, GA has mean validation best score `0.7998 ± 0.0350`, mean test F1 `0.9112 ± 0.0084`, mean test FPR `0.1678 ± 0.0124`, and mean selected features `23.00 ± 1.73`. PSO has mean validation best score `0.7461 ± 0.0657`, mean test F1 `0.9134 ± 0.0085`, mean test FPR `0.1608 ± 0.0054`, and mean selected features `18.67 ± 1.53`. SA has mean validation best score `0.4610 ± 0.6024`, mean test F1 `0.9132 ± 0.0165`, mean test FPR `0.1978 ± 0.0711`, and mean selected features `22.00 ± 4.36`.

The direct observation from the robustness table is that PSO appears more stable than GA and SA in this lower-budget repeated-seed setting. Its mean FPR is the lowest among the metaheuristics, its selected feature count is also the lowest on average, and its variability is smaller than SA on the reported metrics. GA still remains competitive, particularly in validation quality and overall balance, but the multi-seed evidence weakens any strong claim that GA is always superior. SA remains weaker from a stability perspective because its standard deviations are larger, especially on validation score and FPR.

The runtime evidence also improves the analysis. In the robustness check, the mean optimisation wall-clock time is `1166.64 ± 44.74` seconds for GA, `1130.81 ± 60.46` seconds for PSO, and `1263.21 ± 231.15` seconds for SA. Directly observed from these values, SA is the slowest and least stable in optimisation cost. PSO is the fastest on average, while GA is close but slightly heavier. This matters for IDS deployment because an optimisation method is more attractive when it not only improves predictive behaviour, but also reaches a good solution with lower search cost.

From a security perspective, the central trade-off is between detecting attacks and limiting false alarms. The baseline demonstrates one extreme of this trade-off: recall is highest, but the false positive rate is too high to be operationally comfortable. In practice, an IDS that generates too many false positives may overload analysts and reduce trust in the alerts. The metaheuristic methods accept a small reduction in recall in order to achieve much lower FPR and smaller feature subsets. That trade-off is defensible in this coursework because the reduction in false positives is substantial, while recall remains high across all three optimised models.

Each method therefore has a distinct profile. GA is strongest in the main single-run comparison and provides the best compact high-F1 solution under that setting. PSO appears more stable in the lightweight robustness check and offers a very competitive balance of FPR, feature count, and optimisation time. SA improves over the baseline, but its optimisation cost and variability make it less convincing as the preferred method in this study. Based on the available evidence, GA is the strongest main-run candidate, while PSO appears to be the more stable method under repeated lighter runs. The most appropriate conclusion is therefore conditional rather than absolute: GA is the best-performing method in the primary comparison, but PSO provides stronger evidence of robustness under lower-budget repeated runs.

## Conclusion and Future Work

This coursework shows that metaheuristic feature selection and hyperparameter optimisation can improve IDS performance relative to a default Random Forest baseline. All three metaheuristics reduce the false positive rate and the number of selected features while maintaining high recall. Under the primary `B = 50, seed = 0` comparison, GA provides the strongest overall trade-off. Under the lighter `B = 30, seeds = 0, 1, 2` robustness check, PSO appears more stable, while SA is less attractive because of both optimisation cost and instability.

The main limitation of the present evidence is that the strongest comparison uses a single seed, while the robustness check uses a smaller evaluation budget. Therefore, the paper should avoid claiming that any one method is universally best. A more defensible conclusion is that GA is the strongest method in the reported main comparison, whereas PSO is the most stable under the lighter repeated-seed setting.

Future work could strengthen this study by extending the repeated-seed analysis at a budget closer to the main experiment, by expanding the runtime analysis across all main runs, and by exploring more explicit multi-objective formulations that jointly control recall, false positives, feature count, and optimisation cost.

# Self-Audit Table

| Requirement | Satisfied? (Yes / Partly / No) | Evidence in the draft | What still needs improvement |
|---|---|---|---|
| At least 3 metaheuristics are clearly included | Yes | GA, PSO, and SA are all described and compared in Experimental Setup and Results | None in the text itself |
| Baseline comparison is explicit | Yes | Baseline Random Forest is described and compared numerically throughout Results | None |
| Trade-off between detection accuracy, false positives, and number of features is discussed | Yes | Multiple paragraphs contrast recall/FPR/features and explain IDS implications | Could add one dedicated figure reference in the final paper |
| Comparison against benchmark is clear | Yes | Baseline vs GA/PSO/SA numbers are directly interpreted | Final paper should present the corresponding table/figure in template form |
| Most effective method identified from empirical findings | Yes | Conclusion states GA is strongest in the main comparison, PSO more stable in robustness | Final paper should make this distinction visually clear in tables |
| Security trade-offs are discussed | Yes | Results section explains high recall vs alert burden from false positives | Could deepen with a short operational example sentence if space allows |
| Numerical results are clearly interpreted | Yes | Direct observations are separated from interpretation and tied to metric values | Could reference exact table numbers in the final formatted template |
| Strong tables / charts / figures are supported | Partly | Draft explicitly relies on main comparison and robustness tables | Final paper still needs the actual inserted charts/figures |
| Meaningful conclusions and future work are present | Yes | Conclusion and Future Work section is evidence-based and cautious | Could add one more future-work item if space allows |
| Writing is cautious and does not overclaim | Yes | Draft avoids significance claims and universal-best language | None |
| Computational cost is discussed when supported | Yes | Robustness optimisation runtime is used for GA/PSO/SA comparison | Main-run optimisation runtime is still incomplete for PSO/SA |
| Original conference template formatting unchanged | Partly | Draft text is written to fit standard sections | Still needs to be pasted into the provided template without changing format |
| Critical Analysis and Discussion depth is strong enough for the 25% category | Yes | Discussion contrasts methods, seeds, budgets, stability, runtime, and IDS implications | Final paper should add figure references to strengthen presentation further |
| Evidence is limited to the approved source set | Yes | All claims match the assessment sheet and generated results files | None |
