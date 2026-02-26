"""Experiment runner orchestrating leakage-safe protocol and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .baseline import run_baseline_test
from .data import DatasetBundle, load_unsw_nb15
from .evaluator import ObjectiveEvaluator, evaluate_solution_on_test
from .optimizers.ga import run_ga
from .optimizers.pso import run_pso
from .optimizers.sa import run_sa
from .preprocess import fit_preprocessor, split_xy
from .representation import SearchSpace, build_search_space


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    required_keys = [
        "dataset",
        "preprocess",
        "selection",
        "fitness",
        "budget",
        "model",
        "optimizers",
        "output",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    return config


def _ensure_dirs(base_results_dir: Path) -> dict[str, Path]:
    raw_dir = base_results_dir / "raw"
    convergence_dir = base_results_dir / "convergence"
    plots_dir = base_results_dir / "plots"
    for d in [base_results_dir, raw_dir, convergence_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"results": base_results_dir, "raw": raw_dir, "convergence": convergence_dir, "plots": plots_dir}


def _save_run_repro_artifacts(
    out_dirs: dict[str, Path],
    config: dict[str, Any],
    seeds: list[int],
    config_raw_text: str | None = None,
) -> None:
    # Persist exact run inputs so results can be reproduced without guessing runtime flags.
    run_config_path = out_dirs["results"] / "run_config.yaml"
    if config_raw_text is not None:
        run_config_path.write_text(config_raw_text, encoding="utf-8")
    else:
        with run_config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)

    seed_list_path = out_dirs["results"] / "seed_list.txt"
    seed_list_path.write_text("\n".join(str(s) for s in seeds) + "\n", encoding="utf-8")


def _append_row_csv(path: Path, row: dict[str, Any]) -> None:
    # Flush each completed result row to disk to survive Colab/runtime interruptions.
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", index=False, header=not path.exists())


def _optimizer_rng(seed: int, algorithm: str) -> np.random.Generator:
    offsets = {"ga": 11, "pso": 23, "sa": 37}
    return np.random.default_rng(seed * 10_000 + offsets[algorithm])


def _save_convergence(history: list[dict[str, Any]], out_path: Path, algorithm: str, seed: int) -> None:
    if not history:
        return
    hist_df = pd.DataFrame(history)
    hist_df["algorithm"] = algorithm
    hist_df["seed"] = seed
    hist_df.to_csv(out_path, index=False)


def _aggregate_summary(all_runs: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    numeric_cols = [
        "val_best_score",
        "test_score",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_fpr",
        "test_selected_features",
        "test_runtime_sec",
        "test_fit_time_sec",
        "test_predict_time_sec",
    ]
    grouped = all_runs.groupby("algorithm")[numeric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "algorithm" if col[0] == "algorithm" else f"{col[0]}_{col[1]}"
        for col in grouped.columns
    ]
    grouped.to_csv(out_path, index=False)
    return grouped


def _plot_box(
    all_runs: pd.DataFrame,
    metric_col: str,
    y_label: str,
    out_path: Path,
) -> None:
    ordered_algorithms = sorted(all_runs["algorithm"].unique().tolist())
    values = [
        all_runs.loc[all_runs["algorithm"] == alg, metric_col].to_numpy()
        for alg in ordered_algorithms
    ]
    plt.figure(figsize=(10, 5))
    plt.boxplot(values, labels=ordered_algorithms, showmeans=True)
    plt.ylabel(y_label)
    plt.title(f"{y_label} by algorithm")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_convergence(convergence_dir: Path, out_path: Path) -> None:
    files = sorted(convergence_dir.glob("*.csv"))
    if not files:
        return
    frames = [pd.read_csv(p) for p in files]
    conv_df = pd.concat(frames, ignore_index=True)

    plt.figure(figsize=(10, 5))
    for alg, sub in conv_df.groupby("algorithm"):
        mean_curve = sub.groupby("evaluation")["best_score"].mean().sort_index()
        plt.plot(mean_curve.index, mean_curve.values, label=alg)
    plt.xlabel("Objective evaluations")
    plt.ylabel("Best validation fitness")
    plt.title("Mean convergence by algorithm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _create_plots(all_runs: pd.DataFrame, convergence_dir: Path, plots_dir: Path) -> None:
    _plot_box(all_runs, "test_recall", "Test Recall", plots_dir / "box_test_recall.png")
    _plot_box(all_runs, "test_fpr", "Test FPR", plots_dir / "box_test_fpr.png")
    _plot_box(
        all_runs,
        "test_selected_features",
        "Selected Original Features",
        plots_dir / "box_selected_features.png",
    )
    _plot_convergence(convergence_dir, plots_dir / "convergence_mean_best_score.png")


def _run_single_optimizer(
    algorithm: str,
    evaluator: ObjectiveEvaluator,
    budget_b: int,
    n_feature_genes: int,
    seed: int,
    optimizer_cfg: dict[str, Any],
) -> dict[str, Any]:
    # All optimizers consume the same objective-evaluation budget for fair comparison.
    rng = _optimizer_rng(seed=seed, algorithm=algorithm)
    if algorithm == "ga":
        return run_ga(
            evaluator=evaluator,
            budget_b=budget_b,
            n_feature_genes=n_feature_genes,
            rng=rng,
            ga_cfg=optimizer_cfg,
        )
    if algorithm == "pso":
        return run_pso(
            evaluator=evaluator,
            budget_b=budget_b,
            n_feature_genes=n_feature_genes,
            rng=rng,
            pso_cfg=optimizer_cfg,
        )
    if algorithm == "sa":
        return run_sa(
            evaluator=evaluator,
            budget_b=budget_b,
            n_feature_genes=n_feature_genes,
            rng=rng,
            sa_cfg=optimizer_cfg,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_coursework_experiment(
    project_root: Path,
    config: dict[str, Any],
    budget_override: int | None = None,
    max_seeds: int | None = None,
    skip_plots: bool = False,
    config_raw_text: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data: DatasetBundle = load_unsw_nb15(project_root=project_root, config=config)
    feature_cols = data.feature_cols
    d_features = len(feature_cols)
    if d_features <= 0:
        raise RuntimeError("No usable feature columns resolved.")

    budget_b = int(budget_override or config["budget"]["evaluations_B"])
    if budget_b <= 0:
        raise ValueError("Evaluation budget B must be > 0.")

    seeds: list[int] = [int(s) for s in config["budget"]["seeds"]]
    if max_seeds is not None:
        seeds = seeds[:max_seeds]

    out_dirs = _ensure_dirs(project_root / config["output"]["results_dir"])
    _save_run_repro_artifacts(
        out_dirs=out_dirs,
        config=config,
        seeds=seeds,
        config_raw_text=config_raw_text,
    )
    incremental_path = out_dirs["raw"] / "all_runs_incremental.csv"
    if incremental_path.exists():
        incremental_path.unlink()

    search_space: SearchSpace = build_search_space(config["model"])
    k_min = int(config["selection"]["k_min"])
    val_size = float(config.get("protocol", {}).get("val_size", 0.2))

    all_rows: list[dict[str, Any]] = []
    categorical_cols = list(config["preprocess"].get("categorical_cols", []))
    numeric_impute = str(config["preprocess"].get("numeric_impute", "median"))
    onehot_handle_unknown = str(config["preprocess"].get("onehot_handle_unknown", "ignore"))

    # Full train/test arrays are built once per seed to keep seed-specific RF randomness.
    full_x_df, full_y = split_xy(data.train_df, feature_cols, data.target_col)
    test_x_df, test_y = split_xy(data.test_df, feature_cols, data.target_col)

    for seed in seeds:
        # Split only training CSV into inner train/validation for model selection (no test leakage).
        train_inner_idx, val_inner_idx = train_test_split(
            np.arange(len(data.train_df)),
            test_size=val_size,
            random_state=seed,
            stratify=full_y,
        )
        train_inner_df = data.train_df.iloc[train_inner_idx].reset_index(drop=True)
        val_inner_df = data.train_df.iloc[val_inner_idx].reset_index(drop=True)
        train_inner_x_df, y_train_inner = split_xy(train_inner_df, feature_cols, data.target_col)
        val_inner_x_df, y_val_inner = split_xy(val_inner_df, feature_cols, data.target_col)

        # Leakage-safe: fit preprocessing ONLY on train_inner for optimization.
        pre_inner = fit_preprocessor(
            train_features_df=train_inner_x_df,
            original_features=feature_cols,
            categorical_cols=categorical_cols,
            numeric_impute=numeric_impute,
            onehot_handle_unknown=onehot_handle_unknown,
        )
        x_train_inner = pre_inner.transform(train_inner_x_df)
        x_val_inner = pre_inner.transform(val_inner_x_df)

        # Final one-shot evaluation uses preprocessing fit on full training data.
        pre_full = fit_preprocessor(
            train_features_df=full_x_df,
            original_features=feature_cols,
            categorical_cols=categorical_cols,
            numeric_impute=numeric_impute,
            onehot_handle_unknown=onehot_handle_unknown,
        )
        x_train_full = pre_full.transform(full_x_df)
        x_test = pre_full.transform(test_x_df)

        baseline = run_baseline_test(
            seed=seed,
            x_train_full=x_train_full,
            y_train_full=full_y,
            x_test=x_test,
            y_test=test_y,
            original_features=feature_cols,
            group_to_indices=pre_full.group_to_indices,
            fitness_cfg=config["fitness"],
        )
        baseline_row = {
            "algorithm": "baseline_rf_default",
            "seed": seed,
            "budget_b": 0,
            "evaluations_used": 0,
            "val_best_score": np.nan,
            "val_recall": np.nan,
            "val_fpr": np.nan,
            "test_score": baseline.score,
            "test_accuracy": baseline.metrics["accuracy"],
            "test_precision": baseline.metrics["precision"],
            "test_recall": baseline.metrics["recall"],
            "test_f1": baseline.metrics["f1"],
            "test_fpr": baseline.metrics["fpr"],
            "test_selected_features": baseline.metrics["selected_features"],
            "test_runtime_sec": baseline.metrics["runtime_sec"],
            "test_fit_time_sec": baseline.metrics["fit_time_sec"],
            "test_predict_time_sec": baseline.metrics["predict_time_sec"],
            "best_params_json": json.dumps(baseline.solution.params, sort_keys=True),
        }
        all_rows.append(baseline_row)
        _append_row_csv(incremental_path, baseline_row)

        for algorithm in ["ga", "pso", "sa"]:
            # New evaluator instance per (algorithm, seed) keeps cache isolation explicit.
            evaluator = ObjectiveEvaluator(
                algorithm_name=algorithm,
                seed=seed,
                x_train_inner=x_train_inner,
                y_train_inner=y_train_inner,
                x_val_inner=x_val_inner,
                y_val_inner=y_val_inner,
                original_features=feature_cols,
                group_to_indices=pre_inner.group_to_indices,
                search_space=search_space,
                k_min=k_min,
                fitness_cfg=config["fitness"],
                n_jobs=1,
            )

            opt_result = _run_single_optimizer(
                algorithm=algorithm,
                evaluator=evaluator,
                budget_b=budget_b,
                n_feature_genes=d_features,
                seed=seed,
                optimizer_cfg=config["optimizers"][algorithm],
            )
            if opt_result["best_solution"] is None:
                raise RuntimeError(f"{algorithm} produced no best solution.")

            _save_convergence(
                history=opt_result["history"],
                out_path=out_dirs["convergence"] / f"{algorithm}_seed_{seed}.csv",
                algorithm=algorithm,
                seed=seed,
            )

            final_test = evaluate_solution_on_test(
                solution=opt_result["best_solution"],
                x_train_full=x_train_full,
                y_train_full=full_y,
                x_test=x_test,
                y_test=test_y,
                original_features=feature_cols,
                group_to_indices=pre_full.group_to_indices,
                fitness_cfg=config["fitness"],
                seed=seed,
                n_jobs=1,
            )

            opt_row = {
                    "algorithm": algorithm,
                    "seed": seed,
                    "budget_b": budget_b,
                    "evaluations_used": opt_result["evaluations"],
                    "val_best_score": opt_result["best_score"],
                    "val_recall": opt_result["best_metrics"]["recall"],
                    "val_fpr": opt_result["best_metrics"]["fpr"],
                    "test_score": final_test.score,
                    "test_accuracy": final_test.metrics["accuracy"],
                    "test_precision": final_test.metrics["precision"],
                    "test_recall": final_test.metrics["recall"],
                    "test_f1": final_test.metrics["f1"],
                    "test_fpr": final_test.metrics["fpr"],
                    "test_selected_features": final_test.metrics["selected_features"],
                    "test_runtime_sec": final_test.metrics["runtime_sec"],
                    "test_fit_time_sec": final_test.metrics["fit_time_sec"],
                    "test_predict_time_sec": final_test.metrics["predict_time_sec"],
                    "best_params_json": json.dumps(
                        opt_result["best_solution"].params,
                        sort_keys=True,
                    ),
                }
            all_rows.append(opt_row)
            _append_row_csv(incremental_path, opt_row)

        seed_df = pd.DataFrame([r for r in all_rows if r["seed"] == seed])
        seed_df.to_csv(out_dirs["raw"] / f"seed_{seed}_results.csv", index=False)

    all_runs_df = pd.DataFrame(all_rows)
    all_runs_df.to_csv(out_dirs["raw"] / "all_runs.csv", index=False)
    summary_df = _aggregate_summary(all_runs_df, out_dirs["results"] / "summary.csv")

    if not skip_plots and bool(config["output"].get("save_plots", True)):
        _create_plots(
            all_runs=all_runs_df,
            convergence_dir=out_dirs["convergence"],
            plots_dir=out_dirs["plots"],
        )

    return all_runs_df, summary_df
