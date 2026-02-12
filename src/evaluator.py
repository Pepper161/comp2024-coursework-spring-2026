"""Objective evaluator with per-(algorithm,seed) cache isolation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .metrics import compute_binary_metrics
from .representation import (
    CandidateSolution,
    SearchSpace,
    decode_solution,
    mask_to_encoded_indices,
    solution_key,
)


@dataclass
class EvaluationRecord:
    score: float
    metrics: dict[str, Any]
    solution: CandidateSolution
    cache_hit: bool


def _build_rf(
    params: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=seed,
        n_jobs=n_jobs,
    )


def compute_fitness(
    recall: float,
    fpr: float,
    k: int,
    d: int,
    alpha: float,
    lambda_fpr: float,
    lambda_feat: float,
) -> float:
    return float(
        recall - lambda_fpr * max(0.0, fpr - alpha) - lambda_feat * (k / max(d, 1))
    )


class ObjectiveEvaluator:
    """Validation-only evaluator used by metaheuristics."""

    def __init__(
        self,
        algorithm_name: str,
        seed: int,
        x_train_inner: np.ndarray,
        y_train_inner: np.ndarray,
        x_val_inner: np.ndarray,
        y_val_inner: np.ndarray,
        original_features: list[str],
        group_to_indices: dict[str, list[int]],
        search_space: SearchSpace,
        k_min: int,
        fitness_cfg: dict[str, Any],
        n_jobs: int = 1,
    ) -> None:
        self.algorithm_name = algorithm_name
        self.seed = seed
        self.x_train_inner = x_train_inner
        self.y_train_inner = y_train_inner
        self.x_val_inner = x_val_inner
        self.y_val_inner = y_val_inner
        self.original_features = original_features
        self.group_to_indices = group_to_indices
        self.search_space = search_space
        self.k_min = k_min
        self.n_jobs = n_jobs
        self.alpha = float(fitness_cfg["alpha_fpr"])
        self.lambda_fpr = float(fitness_cfg["lambda_fpr"])
        self.lambda_feat = float(fitness_cfg["lambda_feat"])
        self.total_features = len(original_features)
        # Cache is isolated per (algorithm, seed) by object lifetime.
        self._cache: dict[str, tuple[float, dict[str, Any], CandidateSolution]] = {}

    def evaluate(self, vector: np.ndarray) -> EvaluationRecord:
        solution = decode_solution(
            vector=vector,
            n_features=self.total_features,
            k_min=self.k_min,
            search_space=self.search_space,
        )
        key = solution_key(solution)
        if key in self._cache:
            cached_score, cached_metrics, cached_solution = self._cache[key]
            return EvaluationRecord(
                score=cached_score,
                metrics=dict(cached_metrics),
                solution=CandidateSolution(
                    mask=cached_solution.mask.copy(),
                    params=dict(cached_solution.params),
                ),
                cache_hit=True,
            )

        selected_indices = mask_to_encoded_indices(
            mask=solution.mask,
            original_features=self.original_features,
            group_to_indices=self.group_to_indices,
        )
        if selected_indices.size == 0:
            raise RuntimeError("Decoded empty feature set. k_min enforcement failed.")

        model = _build_rf(solution.params, seed=self.seed, n_jobs=self.n_jobs)
        start = time.perf_counter()
        fit_start = time.perf_counter()
        model.fit(self.x_train_inner[:, selected_indices], self.y_train_inner)
        fit_time = time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        y_pred = model.predict(self.x_val_inner[:, selected_indices])
        pred_time = time.perf_counter() - pred_start
        runtime = time.perf_counter() - start

        metrics = compute_binary_metrics(
            y_true=self.y_val_inner,
            y_pred=y_pred,
            selected_features=solution.k,
            total_features=self.total_features,
            runtime_sec=runtime,
            fit_time_sec=fit_time,
            predict_time_sec=pred_time,
        )
        score = compute_fitness(
            recall=metrics["recall"],
            fpr=metrics["fpr"],
            k=solution.k,
            d=self.total_features,
            alpha=self.alpha,
            lambda_fpr=self.lambda_fpr,
            lambda_feat=self.lambda_feat,
        )

        self._cache[key] = (
            score,
            dict(metrics),
            CandidateSolution(mask=solution.mask.copy(), params=dict(solution.params)),
        )
        return EvaluationRecord(score=score, metrics=metrics, solution=solution, cache_hit=False)


def evaluate_solution_on_test(
    solution: CandidateSolution,
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    original_features: list[str],
    group_to_indices: dict[str, list[int]],
    fitness_cfg: dict[str, Any],
    seed: int,
    n_jobs: int = 1,
) -> EvaluationRecord:
    """Final one-shot test evaluation performed after optimization."""
    selected_indices = mask_to_encoded_indices(
        mask=solution.mask,
        original_features=original_features,
        group_to_indices=group_to_indices,
    )
    if selected_indices.size == 0:
        raise RuntimeError("Attempted test evaluation with empty selected feature set.")

    model = _build_rf(solution.params, seed=seed, n_jobs=n_jobs)
    start = time.perf_counter()
    fit_start = time.perf_counter()
    model.fit(x_train_full[:, selected_indices], y_train_full)
    fit_time = time.perf_counter() - fit_start
    pred_start = time.perf_counter()
    y_pred = model.predict(x_test[:, selected_indices])
    pred_time = time.perf_counter() - pred_start
    runtime = time.perf_counter() - start

    total_features = len(original_features)
    metrics = compute_binary_metrics(
        y_true=y_test,
        y_pred=y_pred,
        selected_features=solution.k,
        total_features=total_features,
        runtime_sec=runtime,
        fit_time_sec=fit_time,
        predict_time_sec=pred_time,
    )
    score = compute_fitness(
        recall=metrics["recall"],
        fpr=metrics["fpr"],
        k=solution.k,
        d=total_features,
        alpha=float(fitness_cfg["alpha_fpr"]),
        lambda_fpr=float(fitness_cfg["lambda_fpr"]),
        lambda_feat=float(fitness_cfg["lambda_feat"]),
    )
    return EvaluationRecord(score=score, metrics=metrics, solution=solution, cache_hit=False)

