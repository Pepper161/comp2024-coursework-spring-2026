"""Variable Neighborhood Search optimizer with fixed objective-evaluation budget."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..representation import clamp_vector, random_vector


def _shake(
    current: np.ndarray,
    n_feature_genes: int,
    rng: np.random.Generator,
    neighborhood_size: int,
    param_step_scale: float,
) -> np.ndarray:
    candidate = current.copy()
    n_flips = min(max(1, neighborhood_size), n_feature_genes)
    flip_idx = rng.choice(n_feature_genes, size=n_flips, replace=False)
    candidate[flip_idx] = 1.0 - candidate[flip_idx]

    for idx in range(n_feature_genes, len(candidate)):
        if rng.random() < min(0.4 + 0.1 * neighborhood_size, 0.9):
            candidate[idx] += rng.normal(loc=0.0, scale=param_step_scale * neighborhood_size)
    return clamp_vector(candidate)


def run_vns(
    evaluator,
    budget_b: int,
    n_feature_genes: int,
    rng: np.random.Generator,
    vns_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run VNS with increasing neighborhood size until improvement is found."""
    neighborhoods = [int(v) for v in vns_cfg.get("neighborhood_sizes", [1, 2, 3])]
    local_trials = int(vns_cfg.get("local_trials_per_k", 1))
    param_step_scale = float(vns_cfg.get("param_step_scale", 0.08))

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

    k_idx = 0
    while evaluations < budget_b:
        k = neighborhoods[k_idx]
        improved = False

        for _ in range(local_trials):
            if evaluations >= budget_b:
                break
            proposal = _shake(
                current=current,
                n_feature_genes=n_feature_genes,
                rng=rng,
                neighborhood_size=k,
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

            if rec.score > current_score:
                current = proposal
                current_score = rec.score
                improved = True
                break

        if improved:
            k_idx = 0
        else:
            k_idx = (k_idx + 1) % len(neighborhoods)

    return {
        "best_score": float(best_score),
        "best_solution": best_solution,
        "best_metrics": best_metrics,
        "best_vector": best_vector,
        "history": history,
        "evaluations": evaluations,
    }
