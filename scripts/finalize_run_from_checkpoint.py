"""Finalize normal run artifacts from a saved optimizer checkpoint.

This recovers `all_runs.csv`, per-seed CSV, `summary.csv`, and `run_status.json`
after a run that completed optimization but failed before writing final CSVs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    results_dir = Path(args.results_dir).resolve()
    raw_dir = results_dir / "raw"
    incremental_path = raw_dir / "all_runs_incremental.csv"
    checkpoint_path = Path(args.checkpoint).resolve()

    if not incremental_path.exists():
        raise FileNotFoundError(f"Missing incremental results CSV: {incremental_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint JSON: {checkpoint_path}")

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("stage") != "final_test_completed":
        raise RuntimeError(
            f"Checkpoint is not final-test complete: stage={checkpoint.get('stage')!r}"
        )

    all_runs = pd.read_csv(incremental_path)
    algorithm = str(checkpoint["algorithm"])
    seed = int(args.seed)
    params_json = json.dumps(checkpoint["params"], sort_keys=True)
    test_metrics = dict(checkpoint["test_metrics"])
    val_metrics = dict(checkpoint["validation_metrics"])

    d_features = len(checkpoint["selected_mask"])
    alpha = float(config["fitness"]["alpha_fpr"])
    lambda_fpr = float(config["fitness"]["lambda_fpr"])
    lambda_feat = float(config["fitness"]["lambda_feat"])
    recall = float(test_metrics["recall"])
    fpr = float(test_metrics["fpr"])
    k = int(test_metrics["selected_features"])
    test_score = recall - lambda_fpr * max(0.0, fpr - alpha) - lambda_feat * (k / d_features)

    result_row = {
        "algorithm": algorithm,
        "seed": seed,
        "budget_b": int(config["budget"]["evaluations_B"]),
        "evaluations_used": int(config["budget"]["evaluations_B"]),
        "val_best_score": float(checkpoint["validation_best_score"]),
        "val_recall": float(val_metrics["recall"]),
        "val_fpr": float(val_metrics["fpr"]),
        "test_score": float(test_score),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_fpr": float(test_metrics["fpr"]),
        "test_selected_features": int(test_metrics["selected_features"]),
        "optimization_wall_time_sec": float(checkpoint["optimization_wall_time_sec"]),
        "test_runtime_sec": float(test_metrics["runtime_sec"]),
        "test_fit_time_sec": float(test_metrics["fit_time_sec"]),
        "test_predict_time_sec": float(test_metrics["predict_time_sec"]),
        "total_run_wall_time_sec": float(checkpoint["optimization_wall_time_sec"])
        + float(test_metrics["runtime_sec"]),
        "best_params_json": params_json,
    }

    mask = (all_runs["algorithm"] == algorithm) & (all_runs["seed"] == seed)
    if mask.any():
        for key, value in result_row.items():
            all_runs.loc[mask, key] = value
    else:
        all_runs = pd.concat([all_runs, pd.DataFrame([result_row])], ignore_index=True)

    all_runs = all_runs.sort_values(["seed", "algorithm"], kind="stable").reset_index(drop=True)
    all_runs.to_csv(raw_dir / "all_runs.csv", index=False)

    seed_df = all_runs.loc[all_runs["seed"] == seed].copy()
    seed_df.to_csv(raw_dir / f"seed_{seed}_results.csv", index=False)

    numeric_cols = [
        "val_best_score",
        "val_recall",
        "val_fpr",
        "test_score",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_fpr",
        "test_selected_features",
        "optimization_wall_time_sec",
        "test_runtime_sec",
        "test_fit_time_sec",
        "test_predict_time_sec",
        "total_run_wall_time_sec",
    ]
    summary_df = all_runs.groupby("algorithm")[numeric_cols].agg(["mean", "std"]).reset_index()
    summary_df.columns = [
        "algorithm" if col[0] == "algorithm" else f"{col[0]}_{col[1]}"
        for col in summary_df.columns
    ]
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    (results_dir / "run_status.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "algorithm": algorithm,
                "stage": "final_test_completed",
                "evaluations_used": int(config["budget"]["evaluations_B"]),
                "validation_best_score": float(checkpoint["validation_best_score"]),
                "optimization_wall_time_sec": float(checkpoint["optimization_wall_time_sec"]),
                "test_runtime_sec": float(test_metrics["runtime_sec"]),
                "checkpoint": str(checkpoint_path),
                "all_runs_csv": str(raw_dir / "all_runs.csv"),
                "summary_csv": str(results_dir / "summary.csv"),
                "summary_rows": int(len(summary_df)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("Run artifacts finalized from checkpoint.")
    print(f"all_runs.csv: {raw_dir / 'all_runs.csv'}")
    print(f"summary.csv: {results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
