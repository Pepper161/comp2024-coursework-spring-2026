# Post-Tuning Integration Summary

## Did The Tuning Change The Paper's Conclusions?

No material conclusion changed.

- The limited light-tuning screen did not produce a meaningfully better `PSO` setting.
- The limited light-tuning screen did not produce a meaningfully better `SA` setting.
- `Tabu Search` did improve at the light-budget single-seed setting when `neighborhood_size` increased from `3` to `5`.
- The follow-up robustness check at `B = 30`, `seeds = 0,1,2` found only a marginal repeated-run improvement for tuned Tabu over original Tabu, not a strong enough shift to change the paper’s broader conclusions.
- The paper’s main conclusions therefore remain:
  - best overall by balanced judgment in the current evaluation setting: `VNS`
  - best overall by raw `F1` in the main `B = 120`, `seed = 0` comparison: `Tabu Search`
  - best primary method by balanced judgment in the main comparison: `GA`
  - best primary method by raw `F1` in the main comparison: `SA`

## Exact Light-Tuning Grids Used

Screening setting for all runs:
- `B = 30`
- `seed = 0`
- fixed local single-method RF search-space policy
- same dataset, preprocessing, grouped-feature representation, and evaluator logic as the corresponding local method runs

### PSO

Current default used for screening:
- `w = 0.7`
- `c1 = 1.5`
- `c2 = 1.5`

Candidate settings screened:
- default: `w = 0.7`, `c1 = 1.5`, `c2 = 1.5`
- lower inertia balanced: `w = 0.5`, `c1 = 1.5`, `c2 = 1.5`
- cognition-heavy: `w = 0.7`, `c1 = 1.8`, `c2 = 1.2`
- social-heavy: `w = 0.7`, `c1 = 1.2`, `c2 = 1.8`

### SA

Current default used for screening:
- `T0 = 1.0`
- `alpha = 0.995`

Candidate settings screened:
- default: `T0 = 1.0`, `alpha = 0.995`
- lower initial temperature: `T0 = 0.5`, `alpha = 0.995`
- higher initial temperature: `T0 = 2.0`, `alpha = 0.995`
- faster cooling: `T0 = 1.0`, `alpha = 0.99`
- slower cooling: `T0 = 1.0`, `alpha = 0.999`

### Tabu Search

Current default used for screening:
- `tabu_tenure = 5`
- `neighborhood_size = 3`

Candidate settings screened:
- default: `tabu_tenure = 5`, `neighborhood_size = 3`
- shorter tenure: `tabu_tenure = 3`, `neighborhood_size = 3`
- longer tenure: `tabu_tenure = 7`, `neighborhood_size = 3`
- smaller neighborhood: `tabu_tenure = 5`, `neighborhood_size = 2`
- larger neighborhood: `tabu_tenure = 5`, `neighborhood_size = 5`

## Light-Tuning Screening Results

| Method | Variant | F1 | FPR | Selected Features | Total Time (s) | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `PSO` | default | 0.9093 | 0.1603 | 20 | 1110.00 | current default |
| `PSO` | `w = 0.5, c1 = 1.5, c2 = 1.5` | 0.9093 | 0.1603 | 20 | 1069.03 | same solution as default |
| `PSO` | `w = 0.7, c1 = 1.8, c2 = 1.2` | 0.9093 | 0.1603 | 20 | 1067.07 | same solution as default |
| `PSO` | `w = 0.7, c1 = 1.2, c2 = 1.8` | 0.9093 | 0.1603 | 20 | 1066.51 | same solution as default |
| `SA` | default | 0.9221 | 0.1537 | 20 | 1514.41 | current default |
| `SA` | `T0 = 0.5, alpha = 0.995` | 0.8930 | 0.2069 | 20 | 1062.39 | materially worse |
| `SA` | `T0 = 2.0, alpha = 0.995` | 0.9220 | 0.1568 | 21 | 1104.32 | slightly worse balance |
| `SA` | `T0 = 1.0, alpha = 0.99` | 0.9221 | 0.1537 | 20 | 1438.58 | same solution as default |
| `SA` | `T0 = 1.0, alpha = 0.999` | 0.9221 | 0.1537 | 20 | 1496.30 | same solution as default |
| `Tabu Search` | default | 0.8600 | 0.3962 | 27 | 1699.89 | current default |
| `Tabu Search` | `tabu_tenure = 3, neighborhood_size = 3` | 0.8600 | 0.3962 | 27 | 1693.80 | same solution as default |
| `Tabu Search` | `tabu_tenure = 7, neighborhood_size = 3` | 0.8600 | 0.3962 | 27 | 1703.23 | same solution as default |
| `Tabu Search` | `tabu_tenure = 5, neighborhood_size = 2` | 0.8930 | 0.2774 | 27 | 2049.02 | clearly better than default, but slow and not more compact |
| `Tabu Search` | `tabu_tenure = 5, neighborhood_size = 5` | 0.8945 | 0.2773 | 24 | 1890.87 | clearest balanced improvement |

## Which Setting Deserves Follow-Up Robustness Checking?

Completed follow-up:
- `Tabu Search` with `tabu_tenure = 5`, `neighborhood_size = 5`

Follow-up outcome at `B = 30`, `seeds = 0,1,2`:
- tuned Tabu mean `F1`: `0.9109`
- original Tabu mean `F1`: `0.9106`
- tuned Tabu mean `FPR`: `0.2082`
- original Tabu mean `FPR`: `0.2098`
- tuned Tabu mean selected features: `23.33`
- original Tabu mean selected features: `24.00`
- tuned Tabu mean optimisation time: `1608.18 s`
- original Tabu mean optimisation time: `1438.84 s`

Interpretation:
- tuned Tabu is slightly better than original Tabu by balanced judgment
- the gain is small and does not change the broader robustness ranking
- `VNS` still remains clearly stronger overall at the lightweight robustness setting

## What Was Revised In The Paper

Files created:
- `report/assessment_rubric_extracted.md`
- `report/rubric_review_after_tuning.md`

Files revised:
- `report/full_paper_draft.md`
- `docs/generated/tables/robustness_comparison_b30.md`
- `report/checklist_inputs/robustness_results_table.md`
- `results/tabu/robustness_tuned_neigh5_b30_seeds012/notes.txt`

Paper revisions made:
- added a new subsection on the limited optimiser-specific sensitivity check
- added a compact tuning summary table to the draft
- clarified that the tuning evidence does not materially change the main conclusions
- refined the discussion of `Tabu Search` so the new sensitivity result is acknowledged without overstating it
- added a threats-to-validity sentence explaining the narrow scope of the tuning screen
- updated the robustness artifacts after the tuned Tabu repeated-run follow-up
- revised the draft so the tuned Tabu follow-up is reported as modest additional sensitivity evidence rather than as a pending check

## What Still Remains Weak

- The tuned `Tabu Search` repeated-run gain is small, so it is still not strong enough to justify a broader ranking change.
- The main comparison remains single-seed at `B = 120`, so the strongest claims must continue to be phrased carefully.
- The paper review covered the documentation rubric directly, but code and video grading still require separate human judgment.
- The assessment sheet contains a few template leftovers in the rubric text, so rubric interpretation should be read with that context in mind.

## Manual Checks Before Submission

- Open the final paper in the actual conference template and verify formatting, spacing, headings, and page limits visually.
- Confirm the final exported PDF uses black text only and preserves the original template style.
- Recheck that every figure renders cleanly and legibly in the exported PDF.
- Confirm citation and reference formatting matches the required paper style, not just the Markdown draft.
- If time permits, consider a stronger-budget or larger-seed follow-up for tuned Tabu rather than another single-seed tuning pass.
