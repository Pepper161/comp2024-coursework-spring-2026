# Figure and Table Selection Report

## Data audit
- Main B=120 runs found: Baseline, GA, PSO, SA, VNS, TS.
- No main B=120 method is missing among the chosen comparison set.
- Baseline rows are identical across the available B=120 main-run CSVs, so a single baseline reference is defensible for the main comparison.
- Robustness raw data exists for GA/PSO/SA, VNS, and TS across seeds 0,1,2 at B=30.
- Comparable convergence logs exist, but they were not selected for the main paper visual set due to page limits and lower value than feature-content visuals.
- Source files used for the selected assets:
  - `results\ga\b120_seed0\raw\all_runs.csv`
  - `results\pso\b120_seed0\raw\all_runs.csv`
  - `results\sa\b120_seed0\raw\all_runs.csv`
  - `results\vns\b120_seed0\raw\all_runs.csv`
  - `results\tabu\b120_seed0\raw\all_runs.csv`
  - `results\robustness\b30_seeds012\raw\seed_0_results.csv`
  - `results\robustness\b30_seed1\raw\all_runs.csv`
  - `results\robustness\b30_seed2\raw\all_runs.csv`
  - `results\vns\robustness_b30_seeds012\raw\all_runs.csv`
  - `results\tabu\robustness_b30_seeds012\raw\all_runs.csv`

## Chosen visuals
### Tables
- Main comparison table (B=120, seed=0) for Baseline, GA, PSO, SA, VNS, and TS; this is the strongest complete single-run comparison currently available.
- Robustness table (B=30, seeds=0,1,2) for Baseline, GA, PSO, SA, VNS, and TS; this adds stability evidence without requiring new heavy runs.

### Figures
- Main trade-off scatter (B=120): F1 vs FPR, marker size = selected features, to support the core IDS trade-off discussion.
- Robustness summary panel (B=30): F1, FPR, selected features, and optimization time across seeds 0,1,2.
- Feature selection frequency heatmap (B=30 robustness best solutions): shows which features are repeatedly selected across methods.

## Rejected visuals
- **Experimental setup table**: The setup is already concise in text/config and a full table would consume space without adding much analytical value.
- **Runtime-vs-F1 scatter**: Runtime is easier to compare in the tables and robustness panel; a separate scatter would be redundant for the page limit.
- **Selected-features-vs-F1 scatter**: The main trade-off scatter already encodes feature count and avoids splitting the same story across two figures.
- **Convergence curves**: Comparable logs exist, but they are lower value than the selected visuals for a short coursework paper and would crowd the Results section.
- **ILS in main comparison**: No B=120 ILS result is available, so including it would force an incomplete or unfair main-run table.

## Brief interpretation
### Main comparison table
- VNS and TS are the strongest single-run B=120 methods by test F1 in the currently available main comparison files.
- All optimized methods reduce FPR substantially relative to the baseline reference while also using fewer features.
- GA, SA, and VNS remain tightly clustered on F1, so FPR, feature count, and runtime are needed to separate them meaningfully.

### Robustness table
- The repeated-run robustness evidence is available for GA, PSO, SA, VNS, and TS at B=30, seeds 0,1,2.
- PSO and VNS have the lowest mean FPR values among the robustness methods currently available.
- TS shows the highest robustness recall but a weaker low-FPR profile than VNS or PSO in the lightweight repeated-run setting.

### Main trade-off scatter
- The baseline sits far from the tuned methods because its FPR is much higher while its feature count is also much larger.
- VNS and TS occupy the strongest high-F1/low-FPR region among the available B=120 main runs.
- GA and SA remain competitive, but VNS reaches this region with fewer selected features.

### Robustness summary panel
- The robustness panel shows actual seed-level spread rather than only mean +/- std values.
- VNS retains strong F1 and low FPR under B=30 repeated runs, which makes it more defensible than a single-seed-only claim.
- Optimization times are comparable enough to discuss cost, but they should still be treated as machine-specific wall-clock evidence.

### Feature selection frequency heatmap
- Several features appear repeatedly across multiple methods, which supports the claim that the feature-selection component is not arbitrary.
- The heatmap also shows method-specific preferences, so the optimizers are not converging to exactly the same subset.
- This figure adds value that the feature-count metrics alone cannot provide.
