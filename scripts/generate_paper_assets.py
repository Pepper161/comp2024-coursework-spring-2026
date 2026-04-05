
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_ROOT = ROOT / "docs" / "generated"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
REPORT_PATH = OUT_ROOT / "figure_selection_report.md"

METHOD_LABELS = {
    "baseline_rf_default": "Baseline",
    "ga": "GA",
    "pso": "PSO",
    "sa": "SA",
    "vns": "VNS",
    "tabu": "TS",
}
COLORS = {
    "baseline_rf_default": "#808080",
    "ga": "#4C78A8",
    "pso": "#F58518",
    "sa": "#54A24B",
    "vns": "#B279A2",
    "tabu": "#E45756",
}
MAIN_ORDER = ["baseline_rf_default", "ga", "pso", "sa", "vns", "tabu"]
ROBUSTNESS_ORDER = ["baseline_rf_default", "ga", "pso", "sa", "vns", "tabu"]
FEATURE_ORDER = ["ga", "pso", "sa", "vns", "tabu"]
MAIN_PATHS = {
    "ga": RESULTS / "ga" / "b120_seed0" / "raw" / "all_runs.csv",
    "pso": RESULTS / "pso" / "b120_seed0" / "raw" / "all_runs.csv",
    "sa": RESULTS / "sa" / "b120_seed0" / "raw" / "all_runs.csv",
    "vns": RESULTS / "vns" / "b120_seed0" / "raw" / "all_runs.csv",
    "tabu": RESULTS / "tabu" / "b120_seed0" / "raw" / "all_runs.csv",
}
ROBUSTNESS_PATHS = [
    RESULTS / "robustness" / "b30_seeds012" / "raw" / "seed_0_results.csv",
    RESULTS / "robustness" / "b30_seed1" / "raw" / "all_runs.csv",
    RESULTS / "robustness" / "b30_seed2" / "raw" / "all_runs.csv",
    RESULTS / "vns" / "robustness_b30_seeds012" / "raw" / "all_runs.csv",
    RESULTS / "tabu" / "robustness_b30_seeds012" / "raw" / "all_runs.csv",
]
FEATURE_JSONS = [
    RESULTS / "robustness" / "b30_seeds012" / "best_solutions" / "ga_seed_0.json",
    RESULTS / "robustness" / "b30_seed1" / "best_solutions" / "ga_seed_1.json",
    RESULTS / "robustness" / "b30_seed2" / "best_solutions" / "ga_seed_2.json",
    RESULTS / "robustness" / "b30_seeds012" / "best_solutions" / "pso_seed_0.json",
    RESULTS / "robustness" / "b30_seed1" / "best_solutions" / "pso_seed_1.json",
    RESULTS / "robustness" / "b30_seed2" / "best_solutions" / "pso_seed_2.json",
    RESULTS / "robustness" / "b30_seeds012" / "best_solutions" / "sa_seed_0.json",
    RESULTS / "robustness" / "b30_seed1" / "best_solutions" / "sa_seed_1.json",
    RESULTS / "robustness" / "b30_seed2" / "best_solutions" / "sa_seed_2.json",
    RESULTS / "vns" / "robustness_b30_seeds012" / "best_solutions" / "vns_seed_0.json",
    RESULTS / "vns" / "robustness_b30_seeds012" / "best_solutions" / "vns_seed_1.json",
    RESULTS / "vns" / "robustness_b30_seeds012" / "best_solutions" / "vns_seed_2.json",
    RESULTS / "tabu" / "robustness_b30_seeds012" / "best_solutions" / "tabu_seed_0.json",
    RESULTS / "tabu" / "robustness_b30_seeds012" / "best_solutions" / "tabu_seed_1.json",
    RESULTS / "tabu" / "robustness_b30_seeds012" / "best_solutions" / "tabu_seed_2.json",
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_main_df() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    baseline_rows = []
    for method, path in MAIN_PATHS.items():
        if not path.exists():
            missing.append(method)
            continue
        df = read_csv(path)
        df["source"] = str(path.relative_to(ROOT))
        base = df.loc[df["algorithm"] == "baseline_rf_default"].copy()
        if not base.empty:
            baseline_rows.append(base.iloc[0])
        target = df.loc[df["algorithm"] == method].copy()
        if not target.empty:
            rows.append(target.iloc[0])
    if not baseline_rows:
        raise RuntimeError("No baseline row found for B120 main comparison.")
    baseline_df = pd.DataFrame(baseline_rows)
    numeric_cols = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_fpr", "test_selected_features"]
    ref = baseline_df.iloc[0]
    for _, row in baseline_df.iloc[1:].iterrows():
        for col in numeric_cols:
            if not np.isclose(float(row[col]), float(ref[col])):
                raise RuntimeError(f"Baseline mismatch across B120 runs for column {col}.")
    main_df = pd.concat([pd.DataFrame([baseline_df.iloc[0]]), pd.DataFrame(rows)], ignore_index=True)
    main_df["label"] = main_df["algorithm"].map(METHOD_LABELS)
    main_df["order"] = main_df["algorithm"].map({k: i for i, k in enumerate(MAIN_ORDER)})
    main_df = main_df.sort_values("order").drop(columns=["order"])
    return main_df, missing


def load_robustness_df() -> pd.DataFrame:
    frames = []
    for path in ROBUSTNESS_PATHS:
        if not path.exists():
            raise FileNotFoundError(path)
        df = read_csv(path)
        df["source"] = str(path.relative_to(ROOT))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["algorithm", "seed", "budget_b"], keep="first")
    df = df[df["algorithm"].isin(ROBUSTNESS_ORDER)].copy()
    df["label"] = df["algorithm"].map(METHOD_LABELS)
    return df


def summarise_robustness(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "test_accuracy", "test_precision", "test_recall", "test_f1", "test_fpr",
        "test_selected_features", "optimization_wall_time_sec", "total_run_wall_time_sec",
        "val_best_score",
    ]
    summary = df.groupby("algorithm")[metrics].agg(["mean", "std"]) 
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["n_runs"] = df.groupby("algorithm").size().values
    summary["label"] = summary["algorithm"].map(METHOD_LABELS)
    summary["order"] = summary["algorithm"].map({k: i for i, k in enumerate(ROBUSTNESS_ORDER)})
    return summary.sort_values("order").drop(columns=["order"])


def load_feature_frequency() -> pd.DataFrame:
    rows = []
    for path in FEATURE_JSONS:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        algo = payload["algorithm"]
        if algo not in FEATURE_ORDER:
            continue
        selected = payload.get("selected_mask", {})
        for feat, chosen in selected.items():
            rows.append({"algorithm": algo, "feature": feat, "selected": int(bool(chosen))})
    if not rows:
        raise RuntimeError("No feature-selection JSONs found for robustness heatmap.")
    df = pd.DataFrame(rows)
    freq = df.groupby(["algorithm", "feature"], as_index=False)["selected"].mean()
    pivot = freq.pivot(index="algorithm", columns="feature", values="selected").fillna(0.0)
    overall = pivot.mean(axis=0).sort_values(ascending=False)
    top_features = list(overall.head(10).index)
    pivot = pivot[top_features]
    pivot = pivot.reindex(FEATURE_ORDER)
    pivot.index = [METHOD_LABELS[idx] for idx in pivot.index]
    return pivot


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = [
        '| ' + ' | '.join(cols) + ' |',
        '| ' + ' | '.join(['---'] * len(cols)) + ' |',
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append('| ' + ' | '.join(vals) + ' |')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_main_table(main_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "label", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_fpr",
        "test_selected_features", "optimization_wall_time_sec", "total_run_wall_time_sec"
    ]
    table = main_df[cols].copy()
    table.columns = ["Method", "Accuracy", "Precision", "Recall", "F1", "FPR", "Features", "Optimization Time (s)", "Total Time (s)"]
    return table


def build_robustness_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append({
            "Method": row["label"],
            "Runs": int(row["n_runs"]),
            "Test F1 (mean+/-std)": f"{row['test_f1_mean']:.4f} ? {row['test_f1_std']:.4f}",
            "Test FPR (mean+/-std)": f"{row['test_fpr_mean']:.4f} ? {row['test_fpr_std']:.4f}",
            "Features (mean+/-std)": f"{row['test_selected_features_mean']:.2f} ? {row['test_selected_features_std']:.2f}",
            "Optimization Time (s)": f"{row['optimization_wall_time_sec_mean']:.2f} ? {row['optimization_wall_time_sec_std']:.2f}",
        })
    return pd.DataFrame(rows)


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig: plt.Figure, stem: str) -> list[Path]:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_main_tradeoff(main_df: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    offsets = {
        "baseline_rf_default": (-26, 14),
        "ga": (-18, -28),
        "pso": (8, 10),
        "sa": (10, -16),
        "vns": (-108, 22),
        "tabu": (34, -8),
    }
    for _, row in main_df.iterrows():
        algo = row["algorithm"]
        size = 60 + 7 * float(row["test_selected_features"])
        ax.scatter(
            float(row["test_fpr"]),
            float(row["test_f1"]),
            s=size,
            color=COLORS[algo],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.92 if algo != "baseline_rf_default" else 0.75,
            zorder=3,
        )
        label = f"{METHOD_LABELS[algo]}\nF1={row['test_f1']:.3f}, FPR={row['test_fpr']:.3f}, k={int(row['test_selected_features'])}"
        dx, dy = offsets.get(algo, (8, 8))
        ax.annotate(label, (float(row["test_fpr"]), float(row["test_f1"])), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS[algo], alpha=0.9),
                    arrowprops=dict(arrowstyle='-', lw=0.8, color=COLORS[algo]))
    ax.set_xlabel("Test FPR")
    ax.set_ylabel("Test F1")
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_title("Main comparison trade-off", pad=16)
    ax.text(0.5, 1.01, "B=120 | seed=0 | marker size = selected feature count", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, color="#444")
    return save_fig(fig, "main_tradeoff_b120")


def plot_robustness_summary(robust_df: pd.DataFrame, summary_df: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    panels = [
        ("test_f1", "Test F1", False),
        ("test_fpr", "Test FPR", False),
        ("test_selected_features", "Selected features", False),
        ("optimization_wall_time_sec", "Optimization time (s)", True),
    ]
    x = np.arange(len(ROBUSTNESS_ORDER))
    for ax, (metric, title, baseline_note) in zip(axes.flat, panels):
        for idx, algo in enumerate(ROBUSTNESS_ORDER):
            sub = robust_df[robust_df["algorithm"] == algo]
            vals = sub[metric].to_numpy(dtype=float)
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) else np.array([])
            ax.scatter(np.full(len(vals), idx) + jitter, vals, color=COLORS[algo], alpha=0.45, s=28, zorder=2)
            row = summary_df[summary_df["algorithm"] == algo].iloc[0]
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            ax.errorbar(idx, mean, yerr=std, fmt='o', color=COLORS[algo], ecolor=COLORS[algo],
                        elinewidth=1.3, capsize=4, markersize=6, zorder=3)
            ax.annotate(f"{mean:.3f}" if metric != 'optimization_wall_time_sec' else f"{mean:.0f}",
                        (idx, mean), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
        ax.set_xticks(x, [METHOD_LABELS[a] for a in ROBUSTNESS_ORDER])
        ax.set_title(title, pad=10)
        ax.grid(alpha=0.18, linestyle='--', axis='y')
        if baseline_note and True:
            ax.text(0.98, 0.03, 'Baseline runtime is 0 by design\nand omitted from interpretation.',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, color='#555')
    fig.suptitle("Robustness summary", y=0.98, fontsize=13)
    fig.text(0.5, 0.945, "B=30 | seeds=0,1,2 | points = runs, circles = mean +/- std", ha='center', fontsize=8.5, color='#444')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return save_fig(fig, "robustness_summary_b30")


def plot_feature_frequency(pivot: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.0, 3.9))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap='Blues', aspect='auto', vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=30, ha='right')
    ax.set_yticks(np.arange(pivot.shape[0]), pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8,
                    color='white' if val >= 0.6 else '#1f1f1f')
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Selection frequency')
    ax.set_title('Feature selection frequency', pad=14)
    ax.text(0.5, 1.01, 'Best robustness solutions | B=30 | seeds=0,1,2 | top 10 features by overall frequency',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8.3, color='#444')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save_fig(fig, 'feature_frequency_robustness_b30')


def write_report(audit: dict[str, object], chosen: dict[str, list[str]], rejected: dict[str, str], brief_notes: dict[str, list[str]]) -> None:
    lines = []
    lines.append('# Figure and Table Selection Report\n')
    lines.append('## Data audit')
    lines.append(f"- Main B=120 runs found: {', '.join(audit['main_found'])}.")
    if audit['main_missing']:
        lines.append(f"- Main B=120 runs missing: {', '.join(audit['main_missing'])}.")
    else:
        lines.append('- No main B=120 method is missing among the chosen comparison set.')
    lines.append('- Baseline rows are identical across the available B=120 main-run CSVs, so a single baseline reference is defensible for the main comparison.')
    lines.append('- Robustness raw data exists for GA/PSO/SA, VNS, and TS across seeds 0,1,2 at B=30.')
    lines.append('- Comparable convergence logs exist, but they were not selected for the main paper visual set due to page limits and lower value than feature-content visuals.')
    lines.append('- Source files used for the selected assets:')
    for rel in audit['source_files']:
        lines.append(f'  - `{rel}`')
    lines.append('')
    lines.append('## Chosen visuals')
    for section, items in chosen.items():
        lines.append(f'### {section}')
        for item in items:
            lines.append(f'- {item}')
        lines.append('')
    lines.append('## Rejected visuals')
    for name, why in rejected.items():
        lines.append(f'- **{name}**: {why}')
    lines.append('')
    lines.append('## Brief interpretation')
    for name, bullets in brief_notes.items():
        lines.append(f'### {name}')
        for bullet in bullets:
            lines.append(f'- {bullet}')
        lines.append('')
    REPORT_PATH.write_text('\n'.join(lines).strip() + '\n', encoding='utf-8')


def main() -> None:
    ensure_dirs()
    setup_style()
    main_df, missing = load_main_df()
    robust_df = load_robustness_df()
    robust_summary = summarise_robustness(robust_df)
    feature_pivot = load_feature_frequency()

    main_table = build_main_table(main_df[main_df['algorithm'].isin(MAIN_ORDER)].copy())
    robust_table = build_robustness_table(robust_summary)
    main_table.to_csv(TABLE_DIR / 'main_comparison_b120.csv', index=False)
    robust_table.to_csv(TABLE_DIR / 'robustness_comparison_b30.csv', index=False)
    save_markdown_table(main_table, TABLE_DIR / 'main_comparison_b120.md')
    save_markdown_table(robust_table, TABLE_DIR / 'robustness_comparison_b30.md')

    created = []
    created += [str(p.relative_to(ROOT)) for p in plot_main_tradeoff(main_df[main_df['algorithm'].isin(MAIN_ORDER)].copy())]
    created += [str(p.relative_to(ROOT)) for p in plot_robustness_summary(robust_df, robust_summary)]
    created += [str(p.relative_to(ROOT)) for p in plot_feature_frequency(feature_pivot)]

    audit = {
        'main_found': [METHOD_LABELS[k] for k in MAIN_ORDER if k == 'baseline_rf_default' or MAIN_PATHS.get(k, Path()).exists()],
        'main_missing': [METHOD_LABELS[k] for k in missing if k in METHOD_LABELS],
        'source_files': [
            str(p.relative_to(ROOT)) for p in MAIN_PATHS.values() if p.exists()
        ] + [str(p.relative_to(ROOT)) for p in ROBUSTNESS_PATHS if p.exists()],
    }
    chosen = {
        'Tables': [
            'Main comparison table (B=120, seed=0) for Baseline, GA, PSO, SA, VNS, and TS; this is the strongest complete single-run comparison currently available.',
            'Robustness table (B=30, seeds=0,1,2) for Baseline, GA, PSO, SA, VNS, and TS; this adds stability evidence without requiring new heavy runs.',
        ],
        'Figures': [
            'Main trade-off scatter (B=120): F1 vs FPR, marker size = selected features, to support the core IDS trade-off discussion.',
            'Robustness summary panel (B=30): F1, FPR, selected features, and optimization time across seeds 0,1,2.',
            'Feature selection frequency heatmap (B=30 robustness best solutions): shows which features are repeatedly selected across methods.',
        ],
    }
    rejected = {
        'Experimental setup table': 'The setup is already concise in text/config and a full table would consume space without adding much analytical value.',
        'Runtime-vs-F1 scatter': 'Runtime is easier to compare in the tables and robustness panel; a separate scatter would be redundant for the page limit.',
        'Selected-features-vs-F1 scatter': 'The main trade-off scatter already encodes feature count and avoids splitting the same story across two figures.',
        'Convergence curves': 'Comparable logs exist, but they are lower value than the selected visuals for a short coursework paper and would crowd the Results section.',
        'ILS in main comparison': 'No B=120 ILS result is available, so including it would force an incomplete or unfair main-run table.',
    }
    brief_notes = {
        'Main comparison table': [
            'VNS and TS are the strongest single-run B=120 methods by test F1 in the currently available main comparison files.',
            'All optimized methods reduce FPR substantially relative to the baseline reference while also using fewer features.',
            'GA, SA, and VNS remain tightly clustered on F1, so FPR, feature count, and runtime are needed to separate them meaningfully.',
        ],
        'Robustness table': [
            'The repeated-run robustness evidence is available for GA, PSO, SA, VNS, and TS at B=30, seeds 0,1,2.',
            'PSO and VNS have the lowest mean FPR values among the robustness methods currently available.',
            'TS shows the highest robustness recall but a weaker low-FPR profile than VNS or PSO in the lightweight repeated-run setting.',
        ],
        'Main trade-off scatter': [
            'The baseline sits far from the tuned methods because its FPR is much higher while its feature count is also much larger.',
            'VNS and TS occupy the strongest high-F1/low-FPR region among the available B=120 main runs.',
            'GA and SA remain competitive, but VNS reaches this region with fewer selected features.',
        ],
        'Robustness summary panel': [
            'The robustness panel shows actual seed-level spread rather than only mean +/- std values.',
            'VNS retains strong F1 and low FPR under B=30 repeated runs, which makes it more defensible than a single-seed-only claim.',
            'Optimization times are comparable enough to discuss cost, but they should still be treated as machine-specific wall-clock evidence.',
        ],
        'Feature selection frequency heatmap': [
            'Several features appear repeatedly across multiple methods, which supports the claim that the feature-selection component is not arbitrary.',
            'The heatmap also shows method-specific preferences, so the optimizers are not converging to exactly the same subset.',
            'This figure adds value that the feature-count metrics alone cannot provide.',
        ],
    }
    write_report(audit, chosen, rejected, brief_notes)

    print('Generated tables:')
    for p in sorted(TABLE_DIR.glob('*')):
        print(' -', p.relative_to(ROOT))
    print('Generated figures:')
    for p in sorted(FIG_DIR.glob('*')):
        print(' -', p.relative_to(ROOT))
    print('Decision memo:')
    print(' -', REPORT_PATH.relative_to(ROOT))


if __name__ == '__main__':
    main()
