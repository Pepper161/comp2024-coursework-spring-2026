# Back-Half Paper Draft

## 5. Results

### 5.1 Overall Results for All Methods

Table A reports the overall results for all implemented methods under the strongest available main-run setting, `B = 120` and `seed = 0`. This full table is important because it prevents the analysis from selectively reporting only the methods that align most neatly with the literature review. The baseline `RF` result remains a strong conventional reference point, especially in recall, but it also exhibits a much higher false positive rate and uses all available grouped features. All optimised methods improve substantially over the baseline in `F1`, `FPR`, and feature count, indicating that metaheuristic search adds value beyond a default classifier configuration.

Within the overall comparison, the top-performing methods are `Tabu Search` and `VNS`. `Tabu Search` achieves the highest raw `F1` in the main comparison (`0.9259`), while `VNS` achieves nearly the same `F1` (`0.9250`) but with the lowest `FPR` among the top methods (`0.1658`) and the most compact feature subset (`10` selected original features). This result motivates a distinction between the method with the highest raw `F1` and the method with the best overall balance across security and efficiency criteria. Under that broader judgment, `VNS` emerges as the strongest overall method within the current evaluation setting.

This distinction matters because the project is not attempting to maximise a single headline metric in isolation. In IDS, a method that marginally improves `F1` but does so with a larger feature subset or a slightly worse false positive profile may not be the most operationally attractive option. The main-comparison evidence therefore supports the following careful interpretation: `Tabu Search` is the strongest method by raw `F1` in the main run, whereas `VNS` is the strongest method by balanced judgment.

### 5.2 Primary Comparison Focus

The primary comparative discussion focuses on `RF`, `GA`, `PSO`, and `SA`, because these methods form the core set established in the earlier sections. Within this primary group, all three metaheuristics outperform the `RF` baseline in `F1`, `FPR`, and feature count reduction. This confirms that the value of metaheuristic optimisation is not merely cosmetic: the search procedures materially improve the detection trade-off while producing much smaller feature subsets.

Among the primary methods, `SA` records the highest raw `F1` in the main comparison (`0.9208`), but `GA` provides the strongest balanced primary result because it combines near-best `F1` with the best `FPR` among the primary methods (`0.1672`) and the same compact `17`-feature subset as `PSO` and `SA`. `PSO` is the fastest primary optimiser in the main run, but this speed advantage does not compensate for its weaker `F1` relative to both `GA` and `SA`. Accordingly, the main-comparison interpretation for the primary set is that `GA` is the strongest primary method by balanced judgment, while `SA` is the strongest primary method by raw `F1` only.

This distinction matters because reporting only the highest raw `F1` would suppress the fact that the current project judges methods by security-aware trade-offs rather than by a single score. The primary comparison therefore reinforces the need to interpret `F1` jointly with `FPR`, selected feature count, and runtime.

### 5.3 Trade-off and Security Implications

The central trade-off in this study is between detection quality and operational alert burden. `RF` shows why this matters: it attains very high recall but at the cost of a much higher `FPR` and no feature reduction. In practice, an IDS that raises too many false alerts creates analyst fatigue and reduces trust in the system. Therefore, lower `FPR` is not a secondary aesthetic preference; it is part of the security usefulness of the method.

Against this background, the optimised methods show a more attractive balance. `GA`, `PSO`, and `SA` all reduce the feature count from `42` to roughly `17` while also producing much lower `FPR` values than the baseline. This means the optimised methods are not merely smaller or faster; they also produce less alert noise. `Tabu Search` and `VNS` show that local-search families can also be highly competitive in this mixed optimisation setting, especially when the search objective explicitly penalises excessive false positives and unnecessarily large feature subsets.

Feature count also matters in security terms. A smaller selected subset can simplify deployment, improve interpretability, and reduce computational overhead, but only if the reduction does not come at an unacceptable cost in detection quality. This is one reason `VNS` is notable in the overall results: it achieves a top-tier `F1` with only `10` selected features, making its trade-off especially strong. Runtime is also relevant, although it should remain secondary to security usefulness. A method that runs faster is not automatically preferable if it delivers a materially weaker detection trade-off.

Feature relevance is also visible in the cross-method selection patterns. The feature-selection frequency analysis from the robustness best solutions shows that several variables recur across multiple methods and seeds rather than appearing arbitrarily. In particular, `sbytes`, `dmean`, `ct_srv_dst`, and `synack` are repeatedly selected across the robustness best solutions, while features such as `ct_dst_src_ltm`, `spkts`, and `sjit` appear frequently in method-specific selections. This suggests that the optimisation layer is not merely shrinking the feature set at random; it is repeatedly preserving a smaller subset of traffic-volume, service-distribution, and timing-related variables that appear consistently informative under the current evaluation setting.

### 5.4 Additional Observations on Tabu Search and VNS

The additional methods provide useful evidence rather than mere side notes. `Tabu Search` demonstrates that a memory-based local-search strategy can be highly competitive in the main comparison, even outperforming all primary methods in raw `F1` under the `B = 120, seed = 0` setting. However, the lightweight robustness evidence qualifies this result. Under `B = 30` with seeds `0, 1, 2`, `Tabu Search` retains strong recall, but its average `FPR` and average feature count are clearly less attractive than those of `VNS`.

`VNS` is the more important additional result. It already performs strongly in the main comparison, where it combines near-best `F1` with the best `FPR` and the smallest selected subset among the strongest methods. The lightweight robustness evidence strengthens this interpretation rather than weakening it. Under `B = 30, seeds = 0, 1, 2`, `VNS` achieves the best overall robustness profile among the optimised methods by combined `F1`, `FPR`, and feature-count judgment. This does not automatically promote `VNS` into the primary IDS-backed set, but it does justify treating it as the strongest overall empirical method within the current study setting.

These results are therefore reported directly rather than treated as side notes. `Tabu Search` is a strong secondary method, especially in the main run. `VNS` is a strong overall contender whose empirical behaviour is more convincing than its IDS-specific literature footprint alone would suggest.

## 6. Discussion and Threats to Validity

### 6.1 Interpretation of Primary Results

The primary results are broadly consistent with the literature review. `GA` and `PSO` behave as expected from recent IDS optimisation studies: both are competitive, both substantially outperform the baseline on the most important trade-off metrics, and both operate effectively in the mixed feature-selection and hyperparameter space. `SA` also proves to be a meaningful member of the primary set. Although its direct IDS literature support is weaker than that of `GA` and `PSO`, its empirical behaviour in the main comparison justifies its inclusion as a stochastic local-search benchmark.

At the same time, the primary results show that literature support and empirical ranking are not identical. `GA` is the strongest primary method by balanced judgment in the main `B = 120, seed = 0` comparison, largely because it offers the best `FPR` among the primary methods while maintaining near-top `F1`. However, the lighter-budget robustness evidence suggests a more nuanced picture: under `B = 30, seeds = 0, 1, 2`, `PSO` becomes the strongest primary method by balanced judgment. This does not overturn the main result, because the project treats the stronger-budget `B = 120` runs as the main comparison and the lighter-budget runs as stability qualifiers. It does, however, show that the gap among the primary methods is not absolute.

### 6.2 Interpreting Additional Methods

The most interesting mismatch in the study lies in the additional methods. From a literature-support standpoint, `Tabu Search` and `VNS` are weaker than `GA` and `PSO`, especially in direct recent IDS support. From an empirical standpoint, however, they are highly competitive. The strongest example is `VNS`, which emerges as the best overall method by balanced judgment in the reported comparison setting. This outcome shows that restricting the analysis only to the primary set would have hidden one of the most meaningful results in the project.

`Tabu Search` is also instructive. Its strong single-run result indicates that memory-based local search can be effective in the present search space, but the robustness evidence suggests that this strength is less stable than that of `VNS`. Therefore, `Tabu Search` should be described as a strong secondary single-run performer rather than as the strongest overall secondary method. This interpretation is consistent with both the observed results and the weaker direct IDS evidence for `Tabu Search`.

Overall, the project supports a two-layer conclusion. The first layer concerns the core comparison, where `GA`, `PSO`, and `SA` all justify their inclusion and where `GA` currently provides the strongest balanced primary result in the main comparison. The second layer concerns the full empirical landscape, where `VNS` becomes the strongest overall method within the current evaluation setting once `F1`, `FPR`, feature count, and robustness context are considered together.

### 6.3 Threats to Validity

Several limitations should be acknowledged. First, the study is conducted on a single dataset, `UNSW-NB15`, so the conclusions should not be generalized automatically to other intrusion-detection datasets or operational network environments. Second, the main comparison uses a single seed at `B = 120`, which means the headline rankings should be treated with caution even though lightweight robustness evidence is available. Third, the robustness evidence itself uses a smaller budget (`B = 30`) and only three seeds, so it is useful for qualification but not strong enough to replace the main comparison.

There are also literature-related limitations. The support for `GA` and `PSO` in recent IDS literature is stronger than the support for `Tabu Search` and `VNS`, so the empirical strength of the latter should be interpreted more cautiously. Finally, the study does not include a broader family of popular optimisers such as `GWO`, so the results are best read as a focused comparison of implemented methods rather than as a definitive statement about the best possible IDS metaheuristic overall.

One additional technical limitation is metric consistency. A legacy logged `GA` main-run composite score appears not to be fully aligned with the final implemented evaluator definition. For that reason, the analysis relies on directly interpretable measures such as `F1`, `Recall`, `FPR`, selected feature count, and runtime rather than on that composite score.

## 7. Conclusion

### 7.1 Best-Performing Method Overall

Across all implemented methods, `VNS` emerges as the strongest overall method by balanced judgment within the current evaluation setting. Although `Tabu Search` achieves the highest raw `F1` in the main `B = 120, seed = 0` comparison, `VNS` combines near-best `F1` with the lowest `FPR`, the smallest selected feature set, and stronger lightweight robustness evidence. This makes `VNS` the most convincing overall result in the present study when the IDS trade-off is judged holistically rather than by a single metric.

### 7.2 Best-Performing Primary Method

Within the primary comparison set defined in this study, `GA` is the strongest method by balanced judgment in the main comparison, while `SA` is the strongest by raw `F1` alone. This distinction is important because the paper does not define success as the highest `F1` only. Instead, the primary comparison is interpreted through the combined lens of `F1`, `FPR`, feature count, and runtime.

### 7.3 Practical Takeaway

The experiments show that metaheuristic optimisation is worthwhile relative to a default RF baseline. All optimised methods substantially improve the baseline false-positive trade-off and reduce the number of selected features, demonstrating that joint feature selection and hyperparameter search can produce more operationally attractive IDS models than a fixed conventional baseline. At the same time, the study also shows that literature support and empirical strength do not always align perfectly: some methods that are easier to justify academically are not necessarily the strongest overall performers in practice.

### 7.4 Future Work

Future work should extend the evaluation to additional intrusion-detection datasets, increase the number of seeds in the stronger-budget setting, and include a broader family of optimisers such as `GWO` or other recent joint-optimisation methods. A more explicitly multi-objective version of the optimisation problem could also be useful, especially if the goal is to report Pareto-efficient trade-offs rather than single working best methods.
