# Method Policy

## Purpose

Freeze the final comparison policy before writing the paper body.

## Final Comparison Set

- Baseline: `Random Forest`
- Primary methods: `GA`, `PSO`, `SA`
- Secondary methods: `Tabu Search`, `VNS`

## Reporting Policy

- Overall comparison table must include all implemented methods:
  - `RF`
  - `GA`
  - `PSO`
  - `SA`
  - `Tabu Search`
  - `VNS`
- Primary comparative interpretation must focus on:
  - `RF`
  - `GA`
  - `PSO`
  - `SA`
- `Tabu Search` and `VNS` must remain visible in results and discussion.
- `Tabu Search` and `VNS` must not be framed as hidden, discarded, or invalid methods.

## Conclusion Policy

- Report `best overall` first.
- Report `best primary` second.
- Do not reverse this order.

## Safe Positioning Sentence

This study reports overall empirical results for all implemented methods, namely RF, GA, PSO, SA, Tabu Search, and VNS. However, the primary comparative discussion focuses on RF, GA, PSO, and SA because these methods are supported more directly by recent IDS optimization literature, while Tabu Search and VNS are treated as additional comparators that help interpret the search landscape.

## Interpretation Rules

- Do not identify the best method using `F1` alone.
- Judge methods using the balance of:
  - `F1`
  - `Recall`
  - `FPR`
  - selected feature count
  - runtime
- If a secondary method performs best empirically, report that result honestly and then explain its literature position separately.

## Non-Negotiables

- Keep the distinction between `primary` and `secondary` methods consistent across:
  - title
  - abstract
  - related work
  - methodology
  - results
  - discussion
  - conclusion
- Do not over-explain the distinction in every section.
- State it clearly once, then apply it consistently.
