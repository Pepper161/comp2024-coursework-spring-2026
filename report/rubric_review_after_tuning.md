# Rubric Review After Tuning

Scope note:
- This review applies the extracted assessment rubric to `report/full_paper_draft.md`.
- The paper can only be reviewed directly against the documentation criteria. Code and video criteria are included only where the draft makes their evidence visible.
- The optimizer-specific light tuning evidence is included in this review because it affects whether the results discussion remains aligned with the strongest available evidence.

## Introduction and Justification of Chosen Algorithms (15%)

### Strengths
- The introduction establishes an IDS-specific problem context clearly and explains why false positives, feature count, and runtime matter alongside predictive quality.
- The draft justifies the baseline Random Forest well and explains why wrapper-based joint feature selection and hyperparameter tuning are appropriate.
- The literature positioning is careful about the distinction between the primary methods (`GA`, `PSO`, `SA`) and the secondary comparators (`Tabu Search`, `VNS`).
- The paper already avoids the common mistake of claiming that literature support and empirical ranking must match.

### Weaknesses
- The draft does not yet mention the limited optimizer-specific tuning follow-up, even though that is now part of the available evidence.
- Some justification language is stronger than necessary when it describes the main comparison as "broadly consistent with the literature review", because the strongest empirical overall method remains `VNS`, which is explicitly outside the most strongly supported primary set.
- The literature review is solid, but the paper does not explicitly acknowledge that a few older references are used as background support while more recent direct IDS evidence is concentrated on `GA`, `PSO`, and `SA`.

### Missing Evidence
- No explicit statement yet explains whether limited optimizer tuning changed the interpretation of weaker-than-expected methods.

### Revision Actions
- Add a short sensitivity-analysis subsection that reports the limited tuning screen and states clearly whether it changes the interpretation.
- Tighten any wording that implies empirical ranking is expected to mirror literature ranking.

## Critical Analysis and Discussion of Findings for Performance Evaluation (25%)

### Strengths
- The draft discusses `F1`, `FPR`, feature count, and runtime together instead of collapsing the discussion into one metric.
- The distinction between "best by raw F1" and "best by balanced judgement" is one of the strongest parts of the paper and matches the rubric’s emphasis on meaningful analysis.
- The main tables and figures are integrated into the prose rather than simply pasted without interpretation.
- The robustness section already uses lower-budget repeated runs as qualifying evidence rather than silently replacing the main comparison.

### Weaknesses
- The draft currently lacks any integration of the tuning evidence, so the results chapter is no longer complete relative to the evidence now available.
- The `Tabu Search` discussion currently relies on the untuned `B=30` robustness picture only; it does not note that a larger neighborhood setting materially improved the single-seed light-budget Tabu result.
- The draft does not explicitly state that the PSO and SA tuning screen failed to produce materially better variants, which is itself useful negative evidence.
- The paper refers to direct tables and figures correctly, but the repository’s generated robustness table has an encoding issue (`?` in place of `+/-`). Even though the draft table is readable, this is a consistency risk in the source artifacts.

### Missing Evidence
- No compact summary of the tuning screen exists in the paper yet.
- No explicit statement yet says whether a tuned setting should be promoted to a further robustness check.

### Unclear Claims
- "The primary results are broadly consistent with the literature review" is directionally fair, but it needs a little more caution because the strongest overall method is still outside the primary set.
- The current text says `Tabu Search` is a strong single-run method, but without the new tuning note it understates that part of Tabu’s weaker lightweight picture may be tied to fixed local settings.

### Mismatch Risks Between Text, Tables, and Figures
- The draft’s robustness table is consistent with the existing agreed robustness interpretation, but any future regenerated source table should fix the `+/-` encoding issue before submission.
- The tuning evidence does not overturn the main comparison, so the conclusions should remain centered on `VNS` overall and `GA` for the primary main-run comparison.

### Revision Actions
- Add a concise tuning-results paragraph and one small table that compares each method’s default screening result against its best screened tuned variant.
- State explicitly that PSO and SA did not materially improve under the limited light-tuning screen.
- State explicitly that Tabu improved under a larger neighborhood, but only at a light budget and single seed, so it does not silently replace the paper’s main findings.

## Insights, Conclusions, and Future Work (10%)

### Strengths
- The conclusion already separates best overall from best primary, which is a strong rubric-aligned choice.
- Threats to validity are not hidden; the draft already acknowledges single-dataset, limited-seed, and lighter-budget robustness limitations.
- Future work is plausible and does not overclaim.

### Weaknesses
- The conclusion does not yet mention the tuning follow-up, so it risks looking incomplete after the tuning study has been run.
- The threats-to-validity section should now mention fixed practical optimizer settings and the limited scope of the follow-up tuning evidence.

### Missing Evidence
- The conclusion does not yet say whether the tuning changed the paper’s headline claims.

### Revision Actions
- Add one short conclusion sentence stating that the limited tuning screen did not materially change the primary conclusions, although it identified a stronger light-budget Tabu neighborhood setting worth follow-up robustness checking.
- Add one sentence in threats to validity that the tuning evidence is single-seed and low-budget, so it is sensitivity evidence rather than a new main study.

## Cross-Cutting Evidence Review

### Strengths
- The draft is already recognizably a coursework IDS paper, not an optimisation benchmark paper.
- The metric framing is aligned with the assignment objective: detection quality, false positives, and feature count are all visible.
- The paper identifies practical takeaways instead of ending with raw metric restatement.

### Weaknesses
- The draft needs one final consistency pass after tuning integration to make sure prose still matches table values exactly.
- The current repository contains multiple sources of "default" settings for Tabu (`7/4` fallback in implementation versus `5/3` in the local run configs). The paper should avoid ambiguous wording here.

### Exact Revision Actions Needed
1. Insert a short "limited optimizer-specific sensitivity check" subsection in the results or discussion chapter.
2. Keep the main comparison unchanged unless the tuning evidence clearly overturns it; at present it does not.
3. Explicitly report that PSO and SA tuning produced no material balanced improvement.
4. Explicitly report that Tabu improved at `B=30, seed=0` under `tabu_tenure=5, neighborhood_size=5`, but treat that as a follow-up candidate rather than a replacement main result.
5. Add a threat-to-validity sentence about the limited scope of the tuning evidence.
6. Recheck table values, figure numbering, and citation coverage after the draft is updated.

## Overall Reviewer Judgement

Current rubric-aligned judgement before revision:
- The paper is already strong on context, justification, and metric-aware interpretation.
- The main weakness is not conceptual quality but evidence integration: the new tuning evidence has to be incorporated carefully and cautiously.
- If the tuning evidence is integrated without overclaiming, the paper should remain better aligned with the rubric because it will show stronger critical evaluation of method sensitivity rather than only reporting static fixed-setting outcomes.
