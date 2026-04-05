# Chapter Goals

- **Introduction:** State the IDS optimisation problem, explain why feature selection and false-positive control matter, and position the study as a comparison between metaheuristic search and a non-metaheuristic Random Forest baseline.
- **Related Work and Study Positioning:** Briefly situate the chosen search methods as practical metaheuristic strategies for mixed feature-selection and hyperparameter optimisation without overclaiming novelty.
- **Problem Formulation and Experimental Protocol:** Define the binary IDS task, the optimisation objective, the leakage-safe split and preprocessing pipeline, and the fairness rules used across all compared methods.
- **Algorithms:** Explain the baseline, then justify and describe `GA`, `PSO`, `SA`, `Tabu Search`, and `VNS` in terms relevant to this mixed search space.
- **Results:** Present the main `B=120, seed=0` comparison and the lightweight `B=30, seeds=0,1,2` robustness evidence with clear metric-based comparisons.
- **Discussion and Threats to Validity:** Interpret the security trade-offs between F1, FPR, feature count, and runtime, identify the strongest overall and strongest primary methods, and note the limits of single-seed main comparisons and lightweight robustness.
- **Conclusion:** Summarise what the experiments show about IDS optimisation with metaheuristics and state the most defensible final conclusion without overstating generality.
