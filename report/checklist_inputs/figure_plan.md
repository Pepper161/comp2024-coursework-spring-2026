# Figure Plan

## Figure 1 — Main comparison trade-off figure
- **Role:** Show the single-run main comparison at `B=120, seed=0` across `RF`, `GA`, `PSO`, `SA`, `Tabu Search`, and `VNS`.
- **Recommended design:** scatter plot with `FPR` on the x-axis and `F1` on the y-axis, with compact annotation of selected feature count and possibly total runtime.
- **Why it matters:** This is the fastest way to show the core coursework trade-off between detection quality, false positives, and compactness.
- **Source files:**
  - `results/ga/b120_seed0/raw/all_runs.csv`
  - `results/pso/b120_seed0/raw/all_runs.csv`
  - `results/sa/b120_seed0/raw/all_runs.csv`
  - `results/tabu/b120_seed0/raw/all_runs.csv`
  - `results/vns/b120_seed0/raw/all_runs.csv`
- **Notes:** Use the baseline row from the main-run CSVs as the benchmark point.

## Figure 2 — Robustness summary figure
- **Role:** Show repeated-run stability at `B=30, seeds=0,1,2` for `RF`, `GA`, `PSO`, `SA`, `Tabu Search`, and `VNS`.
- **Recommended design:** compact grouped summary panels or carefully selected box/point plots for `F1`, `FPR`, and selected-feature count; include runtime if space allows.
- **Why it matters:** The main comparison is single-seed, so this figure provides the necessary qualification on stability without pretending to be a full high-budget repeated trial.
- **Source files:**
  - `docs/generated/summary_robustness.csv`
  - `results/vns/robustness_b30_seeds012/summary.csv`
  - `results/tabu/robustness_b30_seeds012/summary.csv`
- **Notes:** Make it explicit in the caption that this is a lightweight robustness check and not the main headline comparison.

## Figure 3 — Feature selection frequency figure
- **Role:** Show which original grouped features are repeatedly retained across the robustness best solutions.
- **Recommended design:** heatmap or grouped bar chart covering `GA`, `PSO`, `SA`, `Tabu Search`, and `VNS`.
- **Why it matters:** The coursework is about feature selection as well as predictive performance, so the report should show not only how many features were selected, but also which ones recur.
- **Source files:**
  - robustness best-solution JSON files under:
    - `results/robustness/.../best_solutions/`
    - `results/vns/robustness_b30_seeds012/best_solutions/`
    - `results/tabu/robustness_b30_seeds012/best_solutions/`
- **Notes:** Prefer real feature names over indices and use only the top recurring features to keep the figure readable.

## Optional appendix figure — Convergence behaviour
- **Role:** Illustrate optimisation progress across evaluation steps if a short appendix or extra page space is available.
- **Recommended design:** mean or median best-so-far validation fitness over evaluations for robustness runs.
- **Why it matters:** It helps explain search behaviour, but it is less essential than the trade-off and robustness figures.
- **Source files:**
  - convergence logs under the robustness result folders for each method
- **Reason optional:** It is valuable but lower priority under a short conference-paper page limit.

## Figure choices not prioritised
- **Standalone runtime-only figure:** rejected as a primary figure because runtime can be conveyed in tables and the main trade-off figure already carries most of the paper's analytical weight.
- **Experimental setup diagram:** rejected because the methodology can be explained cleanly in text and configs.
- **Separate selected-features-vs-F1 scatter:** rejected as redundant if Figure 1 already encodes F1, FPR, and compactness together.
