"""Baseline Random Forest evaluation (all features, default hyperparameters)."""

from __future__ import annotations

import numpy as np

from .evaluator import EvaluationRecord, evaluate_solution_on_test
from .representation import CandidateSolution


def build_baseline_solution(n_original_features: int) -> CandidateSolution:
    return CandidateSolution(
        mask=np.ones(n_original_features, dtype=bool),
        params={
            # Baseline must stay non-metaheuristic with default RF settings for fair benchmarking.
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": None,
        },
    )


def run_baseline_test(
    seed: int,
    x_train_full,
    y_train_full,
    x_test,
    y_test,
    original_features,
    group_to_indices,
    fitness_cfg,
) -> EvaluationRecord:
    solution = build_baseline_solution(len(original_features))
    return evaluate_solution_on_test(
        solution=solution,
        x_train_full=x_train_full,
        y_train_full=y_train_full,
        x_test=x_test,
        y_test=y_test,
        original_features=original_features,
        group_to_indices=group_to_indices,
        fitness_cfg=fitness_cfg,
        seed=seed,
        n_jobs=1,
    )
