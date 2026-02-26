"""Hybrid Binary/Continuous PSO with fixed objective-evaluation budget."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..representation import clamp_vector, random_vector


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def run_pso(
    evaluator,
    budget_b: int,
    n_feature_genes: int,
    rng: np.random.Generator,
    pso_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run PSO; cache hits still count as evaluations."""
    swarm_size = int(pso_cfg.get("swarm_size", 30))
    w = float(pso_cfg.get("w", 0.7))
    c1 = float(pso_cfg.get("c1", 1.5))
    c2 = float(pso_cfg.get("c2", 1.5))

    dim = n_feature_genes + 6
    positions = np.vstack([random_vector(rng, n_feature_genes) for _ in range(swarm_size)])
    velocities = rng.normal(loc=0.0, scale=0.2, size=(swarm_size, dim))
    pbest_positions = positions.copy()
    pbest_scores = np.full(swarm_size, -np.inf, dtype=np.float64)

    best_score = -np.inf
    best_solution = None
    best_metrics = None
    best_vector = None
    history: list[dict[str, Any]] = []
    evaluations = 0

    for i in range(swarm_size):
        if evaluations >= budget_b:
            break
        rec = evaluator.evaluate(positions[i])
        pbest_scores[i] = rec.score
        evaluations += 1
        if rec.score > best_score:
            best_score = rec.score
            best_solution = rec.solution
            best_metrics = dict(rec.metrics)
            best_vector = positions[i].copy()
        history.append(
            {
                "evaluation": evaluations,
                "score": rec.score,
                "best_score": best_score,
                "cache_hit": int(rec.cache_hit),
            }
        )

    if best_vector is None:
        raise RuntimeError("PSO failed to initialize best vector.")

    while evaluations < budget_b:
        for i in range(swarm_size):
            if evaluations >= budget_b:
                break

            r1 = rng.random(dim)
            r2 = rng.random(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (best_vector - positions[i])
            )

            proposal = positions[i] + velocities[i]
            proposal = clamp_vector(proposal)

            # Binary update for feature genes and continuous update for hyperparameter genes.
            mask_probs = _sigmoid(velocities[i, :n_feature_genes])
            proposal[:n_feature_genes] = (
                rng.random(n_feature_genes) < mask_probs
            ).astype(np.float64)

            positions[i] = proposal
            rec = evaluator.evaluate(positions[i])
            evaluations += 1

            if rec.score > pbest_scores[i]:
                pbest_scores[i] = rec.score
                pbest_positions[i] = positions[i].copy()
            if rec.score > best_score:
                best_score = rec.score
                best_solution = rec.solution
                best_metrics = dict(rec.metrics)
                best_vector = positions[i].copy()

            history.append(
                {
                    "evaluation": evaluations,
                    "score": rec.score,
                    "best_score": best_score,
                    "cache_hit": int(rec.cache_hit),
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
