# Security Trade-off Rules

1. **Do not judge IDS quality by recall alone.**
   In this project, high recall is desirable, but a method that achieves it with a very high false positive rate can still be operationally weak because it produces too many false alarms.

2. **Treat FPR as a first-class security metric.**
   A lower FPR is valuable because an IDS that overwhelms analysts with benign alerts is difficult to trust and expensive to operate.

3. **Use F1 and FPR together when comparing methods.**
   F1 captures the precision-recall balance, while FPR captures alert noise; neither should be interpreted in isolation for final method judgement.

4. **Feature reduction only counts as a win if detection quality remains competitive.**
   A smaller selected feature set is useful for simplicity and efficiency, but not if it is achieved by a serious drop in F1 or a major increase in FPR.

5. **Runtime matters, but it is secondary to security usefulness.**
   Faster optimisation is beneficial, but a method should not be preferred solely because it runs faster if its IDS trade-off is materially worse.

6. **Use the main comparison for headline ranking, and the robustness runs as a qualifier.**
   The `B=120, seed=0` runs provide the strongest single-run comparison, while the `B=30, seeds=0,1,2` runs provide limited but useful stability evidence.

7. **Do not overclaim from robustness.**
   The robustness setting uses only three seeds and a smaller budget, so it should be used to qualify stability and variability, not to claim definitive superiority.

8. **Keep best-overall separate from best-primary.**
   The report should distinguish between:
   - the strongest method across `RF`, `GA`, `PSO`, `SA`, `Tabu Search`, and `VNS`
   - the strongest method within the agreed primary set `GA`, `PSO`, `SA`

9. **Interpret the current methods with their actual observed risk profiles.**
   - `RF`: very high recall, but high FPR and no feature reduction.
   - `GA`: strongest primary method by balanced judgement.
   - `PSO`: relatively fast in the primary set, but weaker on F1 than `GA` and `SA` in the main run.
   - `SA`: strong raw F1 in the primary set, but slower and slightly worse on FPR than `GA`.
   - `Tabu Search`: strong main-run F1, but any stability claim should be tied to the lightweight robustness evidence rather than assumed.
   - `VNS`: strongest balanced overall trade-off in the agreed result set.

10. **Do not rely on ambiguous composite scores when cleaner metrics are available.**
    In particular, the `GA` main-run `test_score_mean` appears inconsistent with the current fitness function in code, so conclusions should rely on directly interpretable metrics instead.
