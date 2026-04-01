"""Iterated Local Search optimizer with fixed objective-evaluation budget."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..representation import clamp_vector, random_vector


def _propose_neighbor(
    base: np.ndarray,
    n_feature_genes: int,
    rng: np.random.Generator,
    feature_flip_range: tuple[int, int],
    param_step_scale: float,
) -> np.ndarray:
    candidate = base.copy()
    lo, hi = feature_flip_range
    n_flips = int(rng.integers(low=lo, high=hi + 1))
    flip_idx = rng.choice(n_feature_genes, size=min(n_flips, n_feature_genes), replace=False)
    candidate[flip_idx] = 1.0 - candidate[flip_idx]
    for idx in range(n_feature_genes, len(candidate)):
        if rng.random() < 0.5:
            candidate[idx] += rng.normal(loc=0.0, scale=param_step_scale)
    return clamp_vector(candidate)


def run_ils(
    evaluator,
    budget_b: int,
    n_feature_genes: int,
    rng: np.random.Generator,
    ils_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run lightweight ILS with bounded hill-climbing and perturbation."""
    local_trials = int(ils_cfg.get("local_trials", 3))
    perturb_feature_flips = int(ils_cfg.get("perturb_feature_flips", 3))
    param_step_scale = float(ils_cfg.get("param_step_scale", 0.12))

    current = random_vector(rng, n_feature_genes)
    rec_current = evaluator.evaluate(current)
    evaluations = 1

    current_score = rec_current.score
    best_score = current_score
    best_solution = rec_current.solution
    best_metrics = dict(rec_current.metrics)
    best_vector = current.copy()
    history: list[dict[str, Any]] = [
        {
            "evaluation": evaluations,
            "score": current_score,
            "best_score": best_score,
            "recall": rec_current.metrics["recall"],
            "fpr": rec_current.metrics["fpr"],
            "selected_features": rec_current.metrics["selected_features"],
            "best_recall": best_metrics["recall"],
            "best_fpr": best_metrics["fpr"],
            "best_selected_features": best_metrics["selected_features"],
            "cache_hit": int(rec_current.cache_hit),
        }
    ]

    while evaluations < budget_b:
        improved = False
        local_best_score = current_score
        local_best_vector = current.copy()

        for _ in range(local_trials):
            if evaluations >= budget_b:
                break
            proposal = _propose_neighbor(
                base=current,
                n_feature_genes=n_feature_genes,
                rng=rng,
                feature_flip_range=(1, 2),
                param_step_scale=param_step_scale,
            )
            rec = evaluator.evaluate(proposal)
            evaluations += 1

            if rec.score > best_score:
                best_score = rec.score
                best_solution = rec.solution
                best_metrics = dict(rec.metrics)
                best_vector = proposal.copy()

            history.append(
                {
                    "evaluation": evaluations,
                    "score": rec.score,
                    "best_score": best_score,
                    "recall": rec.metrics["recall"],
                    "fpr": rec.metrics["fpr"],
                    "selected_features": rec.metrics["selected_features"],
                    "best_recall": best_metrics["recall"],
                    "best_fpr": best_metrics["fpr"],
                    "best_selected_features": best_metrics["selected_features"],
                    "cache_hit": int(rec.cache_hit),
                }
            )

            if rec.score > local_best_score:
                improved = True
                local_best_score = rec.score
                local_best_vector = proposal.copy()

        if improved:
            current = local_best_vector
            current_score = local_best_score
            continue

        if evaluations >= budget_b:
            break

        current = current.copy()
        flip_count = min(perturb_feature_flips, n_feature_genes)
        flip_idx = rng.choice(n_feature_genes, size=flip_count, replace=False)
        current[flip_idx] = 1.0 - current[flip_idx]
        for idx in range(n_feature_genes, len(current)):
            if rng.random() < 0.75:
                current[idx] += rng.normal(loc=0.0, scale=param_step_scale * 1.5)
        current = clamp_vector(current)
        rec_perturb = evaluator.evaluate(current)
        evaluations += 1
        current_score = rec_perturb.score

        if rec_perturb.score > best_score:
            best_score = rec_perturb.score
            best_solution = rec_perturb.solution
            best_metrics = dict(rec_perturb.metrics)
            best_vector = current.copy()

        history.append(
            {
                "evaluation": evaluations,
                "score": rec_perturb.score,
                "best_score": best_score,
                "recall": rec_perturb.metrics["recall"],
                "fpr": rec_perturb.metrics["fpr"],
                "selected_features": rec_perturb.metrics["selected_features"],
                "best_recall": best_metrics["recall"],
                "best_fpr": best_metrics["fpr"],
                "best_selected_features": best_metrics["selected_features"],
                "cache_hit": int(rec_perturb.cache_hit),
            }
        )

    return {
        "best_score": float(best_score),
        "best_solution": best_solution,
        "best_metrics": best_metrics,
        "best_vector": best_vector,
        "history": history,
        "evaluations": evaluations,
    }
