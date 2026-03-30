"""Generate paper-ready summary CSV/Markdown tables from raw experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


MAIN_COLUMNS = [
    "algorithm",
    "seed",
    "budget_b",
    "val_best_score",
    "val_recall",
    "val_fpr",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_fpr",
    "test_selected_features",
    "optimization_wall_time_sec",
    "test_runtime_sec",
    "total_run_wall_time_sec",
    "best_params_json",
]

ROBUSTNESS_METRICS = [
    "val_best_score",
    "val_recall",
    "val_fpr",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_fpr",
    "test_selected_features",
    "optimization_wall_time_sec",
    "test_runtime_sec",
    "total_run_wall_time_sec",
]

ALGORITHM_ORDER = {
    "baseline_rf_default": 0,
    "ga": 1,
    "pso": 2,
    "sa": 3,
}


def _read_runs(paths: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw results CSV not found: {path}")
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("No input CSVs were provided.")
    combined = pd.concat(frames, ignore_index=True)
    for col in MAIN_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined["_order"] = range(len(combined))
    combined = combined.sort_values("_order")
    combined = combined.drop_duplicates(subset=["algorithm", "seed", "budget_b"], keep="first")
    combined["algorithm_order"] = combined["algorithm"].map(ALGORITHM_ORDER).fillna(999)
    return combined.sort_values(["algorithm_order", "seed"]).drop(columns=["_order"])


def _format_value(value: float | int | str | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _format_mean_std(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None or pd.isna(mean):
        return "-"
    if std is None or pd.isna(std):
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def _best_params_from_best_val(group: pd.DataFrame) -> str:
    numeric_scores = pd.to_numeric(group["val_best_score"], errors="coerce")
    if numeric_scores.notna().any():
        idx = numeric_scores.idxmax()
    else:
        idx = group.index[0]
    raw = group.loc[idx, "best_params_json"]
    try:
        parsed = json.loads(raw)
        return json.dumps(parsed, sort_keys=True)
    except Exception:
        return str(raw)


def _write_main_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    main_df = df[MAIN_COLUMNS + ["source_csv"]].copy()
    main_df.to_csv(out_dir / "summary_main.csv", index=False)

    lines = [
        "| Method | Seed | Eval B | Val Best Score | Val Recall | Val FPR | Test Accuracy | Test Precision | Test Recall | Test F1 | Test FPR | Features | Optimization Time (s) | Final Test Time (s) | Total Run Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in main_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["algorithm"]),
                    _format_value(row["seed"], digits=0),
                    _format_value(row["budget_b"], digits=0),
                    _format_value(row["val_best_score"]),
                    _format_value(row["val_recall"]),
                    _format_value(row["val_fpr"]),
                    _format_value(row["test_accuracy"]),
                    _format_value(row["test_precision"]),
                    _format_value(row["test_recall"]),
                    _format_value(row["test_f1"]),
                    _format_value(row["test_fpr"]),
                    _format_value(row["test_selected_features"], digits=0),
                    _format_value(row["optimization_wall_time_sec"]),
                    _format_value(row["test_runtime_sec"]),
                    _format_value(row["total_run_wall_time_sec"]),
                ]
            )
            + " |"
        )
    (out_dir / "paper_table_main.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_robustness_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        (out_dir / "paper_table_robustness.md").write_text(
            "Robustness results are not available yet.\n",
            encoding="utf-8",
        )
        pd.DataFrame().to_csv(out_dir / "summary_robustness.csv", index=False)
        return

    grouped = df.groupby("algorithm", dropna=False)
    summary = grouped[ROBUSTNESS_METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "algorithm" if col[0] == "algorithm" else f"{col[0]}_{col[1]}"
        for col in summary.columns
    ]
    counts = grouped.size().reset_index(name="n_runs")
    best_params = grouped.apply(_best_params_from_best_val).reset_index(name="best_params_from_best_val")
    robustness_df = counts.merge(summary, on="algorithm").merge(best_params, on="algorithm")
    robustness_df["algorithm_order"] = robustness_df["algorithm"].map(ALGORITHM_ORDER).fillna(999)
    robustness_df = robustness_df.sort_values("algorithm_order").drop(columns=["algorithm_order"])
    robustness_df.to_csv(out_dir / "summary_robustness.csv", index=False)

    lines = [
        "| Method | Runs | Val Best Score | Test Recall | Test FPR | Test F1 | Features | Optimization Time (s) | Total Run Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in robustness_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["algorithm"]),
                    _format_value(row["n_runs"], digits=0),
                    _format_mean_std(row["val_best_score_mean"], row["val_best_score_std"]),
                    _format_mean_std(row["test_recall_mean"], row["test_recall_std"]),
                    _format_mean_std(row["test_fpr_mean"], row["test_fpr_std"]),
                    _format_mean_std(row["test_f1_mean"], row["test_f1_std"]),
                    _format_mean_std(row["test_selected_features_mean"], row["test_selected_features_std"], digits=2),
                    _format_mean_std(row["optimization_wall_time_sec_mean"], row["optimization_wall_time_sec_std"], digits=2),
                    _format_mean_std(row["total_run_wall_time_sec_mean"], row["total_run_wall_time_sec_std"], digits=2),
                ]
            )
            + " |"
        )
    (out_dir / "paper_table_robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper-ready summary tables from raw result CSVs.")
    parser.add_argument(
        "--main",
        nargs="+",
        required=True,
        help="One or more raw all_runs.csv paths for the main comparison table.",
    )
    parser.add_argument(
        "--robustness",
        nargs="*",
        default=[],
        help="Optional raw all_runs.csv paths for robustness aggregation.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/generated",
        help="Output directory for generated CSV/Markdown files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_df = _read_runs(args.main)
    _write_main_outputs(main_df, out_dir)

    if args.robustness:
        robustness_df = _read_runs(args.robustness)
    else:
        robustness_df = pd.DataFrame()
    _write_robustness_outputs(robustness_df, out_dir)

    print(f"Generated report tables in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
