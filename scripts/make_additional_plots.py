"""Generate additional report figures from saved experiment outputs.

This script keeps the original figures intact and writes revised, paper-ready
variants with clearer annotation and experiment context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MAIN_RAW_PATHS = [
    RESULTS_DIR / "ga" / "b50_seed0" / "raw" / "all_runs.csv",
    RESULTS_DIR / "pso" / "b50_seed0" / "raw" / "all_runs.csv",
    RESULTS_DIR / "sa" / "b50_seed0" / "raw" / "all_runs.csv",
]

ROBUSTNESS_RAW_PATHS = [
    RESULTS_DIR / "robustness" / "b30_seeds012" / "raw" / "seed_0_results.csv",
    RESULTS_DIR / "robustness" / "b30_seed1" / "raw" / "all_runs.csv",
    RESULTS_DIR / "robustness" / "b30_seed2" / "raw" / "all_runs.csv",
]

ROBUSTNESS_CONVERGENCE_GLOBS = [
    RESULTS_DIR / "robustness" / "b30_seeds012" / "convergence" / "*.csv",
    RESULTS_DIR / "robustness" / "b30_seed1" / "convergence" / "*.csv",
    RESULTS_DIR / "robustness" / "b30_seed2" / "convergence" / "*.csv",
]

BEST_SOLUTION_PATHS = [
    RESULTS_DIR / "robustness" / "b30_seeds012" / "best_solutions" / "ga_seed_0.json",
    RESULTS_DIR / "robustness" / "b30_seeds012" / "best_solutions" / "pso_seed_0.json",
    RESULTS_DIR / "robustness" / "b30_seeds012" / "best_solutions" / "sa_seed_0.json",
    RESULTS_DIR / "robustness" / "b30_seed1" / "best_solutions" / "ga_seed_1.json",
    RESULTS_DIR / "robustness" / "b30_seed1" / "best_solutions" / "pso_seed_1.json",
    RESULTS_DIR / "robustness" / "b30_seed1" / "best_solutions" / "sa_seed_1.json",
    RESULTS_DIR / "robustness" / "b30_seed2" / "best_solutions" / "ga_seed_2.json",
    RESULTS_DIR / "robustness" / "b30_seed2" / "best_solutions" / "pso_seed_2.json",
    RESULTS_DIR / "robustness" / "b30_seed2" / "best_solutions" / "sa_seed_2.json",
]

METHOD_LABELS = {
    "baseline_rf_default": "Baseline",
    "ga": "GA",
    "pso": "PSO",
    "sa": "SA",
}

METHOD_COLORS = {
    "baseline_rf_default": "#7A7A7A",
    "ga": "#4C78A8",
    "pso": "#F58518",
    "sa": "#54A24B",
}

METHOD_MARKERS = {
    "baseline_rf_default": "X",
    "ga": "o",
    "pso": "^",
    "sa": "s",
}

TRADEOFF_LABEL_OFFSETS = {
    "baseline_rf_default": (-96, -18),
    "ga": (14, -18),
    "pso": (-70, 12),
    "sa": (16, 10),
}

BOX_MEAN_OFFSETS = [-10, 10, -10, 10]


def _apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _load_raw_runs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required raw results file not found: {path}")
        df = pd.read_csv(path)
        df["source_path"] = str(path.relative_to(ROOT))
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["order"] = np.arange(len(combined))
    combined = combined.sort_values("order")
    combined = combined.drop_duplicates(subset=["algorithm", "seed", "budget_b"], keep="first")
    return combined.drop(columns=["order"])


def _load_best_solution_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _load_convergence_frames() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pattern in ROBUSTNESS_CONVERGENCE_GLOBS:
        for path in sorted(pattern.parent.glob(pattern.name)):
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No robustness convergence CSV files were found.")
    return pd.concat(frames, ignore_index=True)


def _size_from_features(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 28.0 + arr * 7.0


def _apply_title(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    *,
    top: float = 0.84,
    title_y: float = 0.965,
    subtitle_y: float = 0.905,
    title_size: float = 13.0,
    subtitle_size: float = 8.4,
) -> None:
    fig.subplots_adjust(top=top)
    fig.suptitle(title, y=title_y, fontsize=title_size)
    fig.text(0.5, subtitle_y, subtitle, ha="center", va="bottom", fontsize=subtitle_size, color="#4A4A4A")


def _save_figure(fig: plt.Figure, filename: str) -> Path:
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tradeoff_scatter_revised(robustness_df: pd.DataFrame) -> tuple[bool, str]:
    plot_df = robustness_df.copy()
    plot_df["label"] = plot_df["algorithm"].map(METHOD_LABELS)

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    mean_rows: list[dict[str, Any]] = []

    for algorithm in ["baseline_rf_default", "ga", "pso", "sa"]:
        sub = plot_df.loc[plot_df["algorithm"] == algorithm].copy()
        if sub.empty:
            continue
        color = METHOD_COLORS[algorithm]
        marker = METHOD_MARKERS[algorithm]

        ax.scatter(
            sub["test_fpr"],
            sub["test_f1"],
            s=_size_from_features(sub["test_selected_features"]),
            alpha=0.18 if algorithm != "baseline_rf_default" else 0.28,
            color=color,
            marker=marker,
            edgecolor=color,
            linewidth=0.5,
            zorder=2,
        )

        mean_row = {
            "algorithm": algorithm,
            "test_fpr": sub["test_fpr"].mean(),
            "test_f1": sub["test_f1"].mean(),
            "test_selected_features": sub["test_selected_features"].mean(),
        }
        mean_rows.append(mean_row)

        ax.scatter(
            [mean_row["test_fpr"]],
            [mean_row["test_f1"]],
            s=float(_size_from_features(np.array([mean_row["test_selected_features"]]))[0] * 1.65),
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )

        label_text = (
            f"{METHOD_LABELS[algorithm]}\n"
            f"F1={mean_row['test_f1']:.3f}, "
            f"FPR={mean_row['test_fpr']:.3f}, "
            f"k={mean_row['test_selected_features']:.1f}"
        )
        dx, dy = TRADEOFF_LABEL_OFFSETS[algorithm]
        ax.annotate(
            label_text,
            xy=(mean_row["test_fpr"], mean_row["test_f1"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.7,
            color="#222222" if algorithm != "baseline_rf_default" else "#444444",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.92,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 0.8,
                "shrinkA": 2,
                "shrinkB": 4,
            },
            zorder=5,
        )

    tuned = plot_df.loc[plot_df["algorithm"] != "baseline_rf_default"]
    x_min = min(plot_df["test_fpr"].min() - 0.008, tuned["test_fpr"].min() - 0.006)
    x_max = plot_df["test_fpr"].max() + 0.01
    y_min = plot_df["test_f1"].min() - 0.004
    y_max = plot_df["test_f1"].max() + 0.004
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Test FPR")
    ax.set_ylabel("Test F1")
    _apply_title(
        fig,
        "F1-FPR-feature trade-off",
        "Robustness runs only | B=30 | seeds=0,1,2 | faded markers = individual runs",
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[algo],
            color="w",
            label=METHOD_LABELS[algo],
            markerfacecolor=METHOD_COLORS[algo],
            markeredgecolor="black",
            markersize=9,
            linewidth=0,
        )
        for algo in ["baseline_rf_default", "ga", "pso", "sa"]
    ]
    legend_methods = ax.legend(
        handles=method_handles,
        title="Method means",
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
    )
    ax.add_artist(legend_methods)

    size_reference = [18, 23, 42]
    size_handles = [
        ax.scatter([], [], s=float(_size_from_features(np.array([size]))[0] * 1.1), color="#CFCFCF", edgecolor="black")
        for size in size_reference
    ]
    ax.legend(
        size_handles,
        [f"k={size}" for size in size_reference],
        title="Marker size",
        frameon=True,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.02),
    )

    ax.text(
        0.02,
        0.04,
        "Baseline is a reference model, not an optimized method.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )

    out_path = _save_figure(fig, "tradeoff_scatter_revised.png")
    return True, f"Created {out_path.relative_to(ROOT)} from robustness per-seed raw outputs."


def _distribution_plot_revised(
    robustness_df: pd.DataFrame,
    metric: str,
    y_label: str,
    title: str,
    out_name: str,
    include_baseline: bool = True,
) -> tuple[bool, str]:
    if include_baseline:
        order = ["baseline_rf_default", "ga", "pso", "sa"]
    else:
        order = ["ga", "pso", "sa"]

    labels = [METHOD_LABELS[a] for a in order]
    values = [robustness_df.loc[robustness_df["algorithm"] == a, metric].to_numpy(dtype=float) for a in order]

    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    bp = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "#333333", "linewidth": 1.6},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
        boxprops={"linewidth": 1.1, "edgecolor": "#666666"},
    )

    for patch, algorithm in zip(bp["boxes"], order):
        patch.set_facecolor(METHOD_COLORS[algorithm])
        patch.set_alpha(0.32)

    for i, (algorithm, arr) in enumerate(zip(order, values), start=1):
        if arr.size == 0:
            continue
        xs = np.full(arr.shape[0], i, dtype=float) + np.linspace(-0.07, 0.07, arr.shape[0])
        ax.scatter(
            xs,
            arr,
            color=METHOD_COLORS[algorithm],
            edgecolor="black",
            linewidth=0.7,
            s=52,
            zorder=3,
        )

        mean_val = float(np.mean(arr))
        ax.scatter(
            [i],
            [mean_val],
            marker="D",
            s=62,
            color=METHOD_COLORS[algorithm],
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

    y_all = np.concatenate([arr for arr in values if arr.size > 0])
    y_min = float(y_all.min())
    y_max = float(y_all.max())
    pad = max((y_max - y_min) * 0.16, 0.004 if metric != "optimization_wall_time_sec" else 35.0)
    ax.set_ylim(y_min - pad * 0.35, y_max + pad)

    for i, arr in enumerate(values, start=1):
        if arr.size == 0:
            continue
        mean_val = float(np.mean(arr))
        offset = pad * (0.12 + 0.02 * (i % 2))
        ax.text(
            i,
            mean_val + offset,
            f"mean={mean_val:.3f}" if metric != "optimization_wall_time_sec" else f"mean={mean_val:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.95},
        )

    ax.set_ylabel(y_label)
    _apply_title(
        fig,
        title,
        "Robustness runs | B=30 | seeds=0,1,2 | boxes show distribution, diamonds show mean",
    )
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.8)

    if include_baseline:
        note = "Baseline is shown as a repeated reference run; it is not a tuned method."
    else:
        note = "Baseline omitted here: optimization time applies only to tuned metaheuristics."
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=8.3, color="#555555")
    ax.text(0.99, 0.02, "n=3 runs per method", transform=ax.transAxes, ha="right", fontsize=8.3, color="#555555")

    out_path = _save_figure(fig, out_name)
    return True, f"Created {out_path.relative_to(ROOT)} from robustness per-seed raw outputs."


def plot_feature_selection_frequency_revised(best_solution_records: list[dict[str, Any]]) -> tuple[bool, str]:
    if not best_solution_records:
        return False, "Could not create revised feature selection frequency figure because no best-solution JSON files were found."

    rows: list[dict[str, Any]] = []
    for record in best_solution_records:
        algorithm = record["algorithm"]
        if algorithm == "baseline_rf_default":
            continue
        for feature_name, is_selected in record["selected_mask"].items():
            rows.append(
                {
                    "algorithm": algorithm,
                    "feature_name": feature_name,
                    "selected": int(bool(is_selected)),
                }
            )

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        return False, "Could not create revised feature selection frequency figure because no optimizer feature masks were available."

    freq_df = (
        feature_df.groupby(["algorithm", "feature_name"])["selected"]
        .mean()
        .reset_index(name="selection_frequency")
    )

    overall = (
        freq_df.groupby("feature_name")["selection_frequency"]
        .agg(["sum", "max"])
        .sort_values(["sum", "max"], ascending=[False, False])
        .head(10)
        .index.tolist()
    )

    top_df = freq_df[freq_df["feature_name"].isin(overall)].copy()
    pivot = top_df.pivot(index="algorithm", columns="feature_name", values="selection_frequency").fillna(0.0)
    pivot = pivot.reindex(index=["ga", "pso", "sa"])
    pivot = pivot[overall]

    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([METHOD_LABELS[idx] for idx in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=32, ha="right")
    _apply_title(
        fig,
        "Feature selection frequency",
        "Best robustness solutions | B=30 | seeds=0,1,2 | top 10 features by overall frequency",
        top=0.82,
        title_y=0.952,
        subtitle_y=0.868,
        title_size=10.8,
        subtitle_size=8.0,
    )

    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = float(pivot.iloc[y, x])
            ax.text(
                x,
                y,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color="white" if val >= 0.62 else "#1E1E1E",
            )

    ax.set_xticks(np.arange(-0.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Selection frequency")
    out_path = _save_figure(fig, "feature_selection_frequency_revised.png")
    return True, f"Created {out_path.relative_to(ROOT)} from robustness best-solution JSON files."


def plot_convergence_summary_revised(convergence_df: pd.DataFrame) -> tuple[bool, str]:
    if convergence_df.empty:
        return False, "Could not create revised convergence summary because no convergence logs were found."

    fig, ax = plt.subplots(figsize=(9.1, 5.8))
    final_labels: list[str] = []

    for algorithm in ["ga", "pso", "sa"]:
        sub = convergence_df.loc[convergence_df["algorithm"] == algorithm].copy()
        if sub.empty:
            continue

        curve = sub.groupby(["seed", "evaluation"])["best_score"].first().reset_index()
        stats = curve.groupby("evaluation")["best_score"].agg(
            median="median",
            q1=lambda x: np.quantile(x, 0.25),
            q3=lambda x: np.quantile(x, 0.75),
        ).reset_index()

        color = METHOD_COLORS[algorithm]
        ax.fill_between(
            stats["evaluation"].to_numpy(dtype=float),
            stats["q1"].to_numpy(dtype=float),
            stats["q3"].to_numpy(dtype=float),
            color=color,
            alpha=0.08,
            zorder=1,
        )
        ax.plot(
            stats["evaluation"],
            stats["median"],
            label=METHOD_LABELS[algorithm],
            color=color,
            linewidth=2.4,
            zorder=3,
        )
        final_labels.append(f"{METHOD_LABELS[algorithm]}: {stats['median'].iloc[-1]:.3f}")

    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Best validation fitness")
    _apply_title(
        fig,
        "Convergence during optimization",
        "Validation fitness | robustness runs | B=30 | seeds=0,1,2 | line = median, band = IQR",
    )
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", bbox_to_anchor=(0.985, 0.22), frameon=True)
    ax.text(
        0.98,
        0.06,
        "Final median best fitness\n" + "\n".join(final_labels),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.95},
    )

    out_path = _save_figure(fig, "convergence_summary_across_seeds_revised.png")
    return True, f"Created {out_path.relative_to(ROOT)} from robustness convergence CSV logs."


def write_revision_notes() -> None:
    lines = [
        "# Revision Notes For Additional Plots",
        "",
        "## tradeoff_scatter_revised.png",
        "- Changed mean markers to be visually dominant and per-seed points to be lighter.",
        "- Replaced the large combined legend with a compact method legend and a separate marker-size legend.",
        "- Added compact mean labels with leader lines to avoid overlap.",
        "- Kept the baseline visible, but labeled it more quietly so it does not dominate the tuned-method comparison.",
        "",
        "## convergence_summary_across_seeds_revised.png",
        "- Replaced the heavier mean +/- std presentation with a median line and lighter IQR band.",
        "- Added an explicit context subtitle stating that this is validation fitness during optimization for B=30 and seeds 0,1,2.",
        "- Moved final values into a small summary box instead of placing overlapping labels on the curves.",
        "",
        "## distribution_test_f1_revised.png",
        "## distribution_test_fpr_revised.png",
        "## distribution_selected_features_revised.png",
        "- Kept the boxplot + individual points structure, but added mean markers and compact mean labels only.",
        "- Did not label every raw point because that would add clutter with only three runs per method.",
        "- Added explicit robustness-run context and a note clarifying the role of the baseline.",
        "",
        "## distribution_runtime_revised.png",
        "- Kept the same metric but excluded the baseline from the boxplot because optimization time applies only to tuned methods.",
        "- Added an embedded note explaining that omission so the comparison remains technically honest.",
        "",
        "## feature_selection_frequency_revised.png",
        "- Reduced the display to the top 10 features by overall selection frequency to improve readability.",
        "- Kept the heatmap concept and numeric cell annotations, but tightened spacing and reduced colorbar prominence.",
        "- Retained real feature names and sorted them by overall frequency so shared versus method-specific choices are easier to see.",
        "",
        "## Label omissions",
        "- Raw-point numeric labels were intentionally omitted in the distribution plots to avoid collisions and unnecessary clutter.",
        "- The convergence summary uses a final-value box instead of multiple endpoint labels for the same reason.",
    ]
    (FIGURES_DIR / "revision_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _apply_style()

    created: list[str] = []
    skipped: list[str] = []

    # Keep the main raw load in place so the script still validates the main-run
    # outputs are present, even though the revised figures are built from the more
    # comparable robustness results.
    _ = _load_raw_runs(MAIN_RAW_PATHS)
    robustness_raw = _load_raw_runs(ROBUSTNESS_RAW_PATHS)
    best_solution_records = _load_best_solution_records(BEST_SOLUTION_PATHS)
    convergence_df = _load_convergence_frames()

    for success, message in [
        plot_tradeoff_scatter_revised(robustness_raw),
        _distribution_plot_revised(
            robustness_raw,
            "test_f1",
            "Test F1",
            "Test F1 across methods",
            "distribution_test_f1_revised.png",
            include_baseline=True,
        ),
        _distribution_plot_revised(
            robustness_raw,
            "test_fpr",
            "Test FPR",
            "Test FPR across methods",
            "distribution_test_fpr_revised.png",
            include_baseline=True,
        ),
        _distribution_plot_revised(
            robustness_raw,
            "test_selected_features",
            "Selected original features",
            "Selected feature count across methods",
            "distribution_selected_features_revised.png",
            include_baseline=True,
        ),
        _distribution_plot_revised(
            robustness_raw,
            "optimization_wall_time_sec",
            "Optimization wall time (s)",
            "Optimization time across methods",
            "distribution_runtime_revised.png",
            include_baseline=False,
        ),
        plot_feature_selection_frequency_revised(best_solution_records),
        plot_convergence_summary_revised(convergence_df),
    ]:
        (created if success else skipped).append(message)

    write_revision_notes()

    print("Created revised figures:")
    for item in created:
        print(f"- {item}")
    if skipped:
        print("Skipped or limited:")
        for item in skipped:
            print(f"- {item}")
    print(f"- Created { (FIGURES_DIR / 'revision_notes.md').relative_to(ROOT) } with figure revision notes.")


if __name__ == "__main__":
    main()
