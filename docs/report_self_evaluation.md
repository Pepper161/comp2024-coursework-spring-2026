# Report Self-Evaluation

This file records strict self-evaluation cycles for the final coursework draft against the assessment sheet.

## Iteration 1

### Estimated score by category

- Python Code (30): **26/30**
  - Functional correctness & completeness: **22/25**
  - Readability & documentation: **4/5**
- Documentation/Paper (50): **39/50**
  - Introduction and justification of chosen algorithms: **11/15**
  - Critical analysis and discussion of findings: **20/25**
  - Insights, conclusions, and future work: **8/10**
- Video readiness (20): **15/20**

**Estimated total: 80/100**

### Reasons for lost marks

1. The paper draft was initially too light on explicit problem formulation and fairness controls.
2. The distinction between the main result (`B=50`, seed 0) and robustness evidence (`B=30`, seeds 0,1,2) was not explicit enough.
3. Feature-selection content was underused; the paper discussed feature counts more than actual selected features.
4. Runtime discussion was not careful enough about where the stronger evidence comes from.
5. Algorithm justification was conceptually present, but still weaker than a literature-backed justification.

### Highest-impact fixes

1. Add the explicit optimisation objective and constraint interpretation.
2. Strengthen the methodology sections on representation, leakage control, fairness, and reproducibility.
3. Expand the results section to separate primary comparison evidence from robustness evidence.
4. Add a dedicated subsection on feature-selection content using the final heatmap.
5. Strengthen limitations and future work so the draft does not overclaim.

### Changes applied

- Added a formal problem formulation section with the exact fitness function.
- Added grouped-feature representation details and leakage-safe preprocessing explanation.
- Added explicit fairness controls and reproducibility details from the repository.
- Expanded the results section with separate main-comparison and robustness subsections.
- Added a feature-frequency discussion and a convergence subsection.
- Rewrote the conclusion to remain conditional and evidence-based.

## Iteration 2

### Estimated score by category

- Python Code (30): **26/30**
  - Functional correctness & completeness: **22/25**
  - Readability & documentation: **4/5**
- Documentation/Paper (50): **43/50**
  - Introduction and justification of chosen algorithms: **12/15**
  - Critical analysis and discussion of findings: **23/25**
  - Insights, conclusions, and future work: **8/10**
- Video readiness (20): **16/20**

**Estimated total: 85/100**

### Reasons for remaining lost marks

1. The repository does not contain the official conference template `.docx`, so the draft cannot yet be validated against the real submission layout.
2. The assessment sheet expects stronger literature-supported justification than the repository evidence can honestly provide.
3. The strongest reported comparison is still based on one seed.
4. The robustness evidence is only based on three seeds and a lower budget than the main comparison.
5. Optimisation wall-clock time is fully available for the robustness runs, but not equally complete for all three methods in the main `B=50` comparison.

### Highest-impact fixes considered

1. Add external literature references and related-work support.
2. Add more repeated-seed runs at the main budget.
3. Add fully comparable main-run optimisation runtime for PSO and SA.
4. Transfer the draft into the official template and polish layout.

### Why these fixes were not applied here

These improvements require either:

- new experiments,
- external literature not present in the repository,
- or the missing official `.docx` template file.

Applying them without new evidence would risk fabrication or overclaiming.

## Final stopping reason

The draft stops at an honest estimated **85/100** because further improvement would require new evidence that is not currently available in the cleaned repository. Within the available repository-grounded evidence, the current draft is strong enough to transfer into the official submission template and continue final formatting work.
