"""Tabu Search optimizer with fixed objective-evaluation budget."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..representation import clamp_vector, random_vector, solution_key


def _propose_neighbor(
    current: np.ndarray,
    n_feature_genes: int,
    rng: np.random.Generator,
    feature_flip_range: tuple[int, int],
    param_step_scale: float,
) -> np.ndarray:
    neighbor = current.copy()
    lo, hi = feature_flip_range
    n_flips = int(rng.integers(low=lo, high=hi + 1))
    flip_idx = rng.choice(n_feature_genes, size=min(n_flips, n_feature_genes), replace=False)
    neighbor[flip_idx] = 1.0 - neighbor[flip_idx]

    for idx in range(n_feature_genes, len(neighbor)):
        if rng.random() < 0.5:
            neighbor[idx] += rng.normal(loc=0.0, scale=param_step_scale)
    return clamp_vector(neighbor)


def run_tabu(
    evaluator,
    budget_b: int,
    n_feature_genes: int,
    rng: np.random.Generator,
    tabu_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run single-solution Tabu Search with aspiration on global-best improvement."""
    tabu_tenure = int(tabu_cfg.get("tabu_tenure", 7))
    neighborhood_size = int(tabu_cfg.get("neighborhood_size", 4))
    flips = tabu_cfg.get("feature_flips", [1, 3])
    feature_flip_range = (int(flips[0]), int(flips[1]))
    param_step_scale = float(tabu_cfg.get("param_step_scale", 0.12))

    current = random_vector(rng, n_feature_genes)
    rec_current = evaluator.evaluate(current)
    evaluations = 1

    current_score = rec_current.score
    best_score = current_score
    best_solution = rec_current.solution
    best_metrics = dict(rec_current.metrics)
    best_vector = current.copy()
    tabu_signatures: deque[str] = deque(maxlen=max(1, tabu_tenure))
    tabu_signatures.append(solution_key(rec_current.solution))

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
        best_admissible = None
        best_admissible_vector = None
        best_any = None
        best_any_vector = None

        for _ in range(neighborhood_size):
            if evaluations >= budget_b:
                break
            proposal = _propose_neighbor(
                current=current,
                n_feature_genes=n_feature_genes,
                rng=rng,
                feature_flip_range=feature_flip_range,
                param_step_scale=param_step_scale,
            )
            rec = evaluator.evaluate(proposal)
            evaluations += 1
            signature = solution_key(rec.solution)
            is_tabu = signature in tabu_signatures
            aspiration = rec.score > best_score

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

            if best_any is None or rec.score > best_any.score:
                best_any = rec
                best_any_vector = proposal.copy()
            if (not is_tabu) or aspiration:
                if best_admissible is None or rec.score > best_admissible.score:
                    best_admissible = rec
                    best_admissible_vector = proposal.copy()

        chosen = best_admissible if best_admissible is not None else best_any
        chosen_vector = best_admissible_vector if best_admissible is not None else best_any_vector
        if chosen is None or chosen_vector is None:
            break

        current = chosen_vector
        current_score = chosen.score
        tabu_signatures.append(solution_key(chosen.solution))

    return {
        "best_score": float(best_score),
        "best_solution": best_solution,
        "best_metrics": best_metrics,
        "best_vector": best_vector,
        "history": history,
        "evaluations": evaluations,
    }
