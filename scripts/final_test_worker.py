"""Run final full-train/test evaluation in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--solution-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data import load_unsw_nb15
    from src.evaluator import _build_rf, compute_fitness
    from src.metrics import compute_binary_metrics
    from src.preprocess import fit_preprocessor
    from src.representation import CandidateSolution
    from src.runner import load_experiment_config

    config = load_experiment_config(Path(args.config).resolve())
    solution_payload = json.loads(Path(args.solution_json).read_text(encoding="utf-8"))

    data = load_unsw_nb15(project_root=project_root, config=config)
    feature_cols = data.feature_cols
    categorical_cols = list(config["preprocess"].get("categorical_cols", []))
    numeric_impute = str(config["preprocess"].get("numeric_impute", "median"))
    onehot_handle_unknown = str(config["preprocess"].get("onehot_handle_unknown", "ignore"))

    full_y = data.train_df[data.target_col].astype(int).to_numpy(copy=False)
    test_y = data.test_df[data.target_col].astype(int).to_numpy(copy=False)
    solution = CandidateSolution(
        mask=np.asarray(solution_payload["mask"], dtype=bool),
        params=dict(solution_payload["params"]),
    )

    selected_original_features = [
        feature for feature, keep in zip(feature_cols, solution.mask.tolist()) if keep
    ]
    if not selected_original_features:
        raise RuntimeError("Selected feature set is empty for final evaluation.")

    # Use the same preprocessing semantics, but only materialize columns for selected features.
    selected_pre = fit_preprocessor(
        train_features_df=data.train_df.loc[:, selected_original_features],
        original_features=selected_original_features,
        categorical_cols=categorical_cols,
        numeric_impute=numeric_impute,
        onehot_handle_unknown=onehot_handle_unknown,
    )
    x_train_selected = selected_pre.transform(data.train_df.loc[:, selected_original_features])
    x_test_selected = selected_pre.transform(data.test_df.loc[:, selected_original_features])

    model = _build_rf(solution.params, seed=args.seed, n_jobs=1)
    start = time.perf_counter()
    fit_start = time.perf_counter()
    model.fit(x_train_selected, full_y)
    fit_time = time.perf_counter() - fit_start
    pred_start = time.perf_counter()
    y_pred = model.predict(x_test_selected)
    pred_time = time.perf_counter() - pred_start
    runtime = time.perf_counter() - start

    metrics = compute_binary_metrics(
        y_true=test_y,
        y_pred=y_pred,
        selected_features=solution.k,
        total_features=len(feature_cols),
        runtime_sec=runtime,
        fit_time_sec=fit_time,
        predict_time_sec=pred_time,
    )
    payload = {
        "score": compute_fitness(
            recall=metrics["recall"],
            fpr=metrics["fpr"],
            k=solution.k,
            d=len(feature_cols),
            alpha=float(config["fitness"]["alpha_fpr"]),
            lambda_fpr=float(config["fitness"]["lambda_fpr"]),
            lambda_feat=float(config["fitness"]["lambda_feat"]),
        ),
        "metrics": metrics,
        "solution": {
            "mask": solution.mask.tolist(),
            "params": dict(solution.params),
        },
    }
    Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
