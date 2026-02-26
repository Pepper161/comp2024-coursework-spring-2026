"""Simulated Annealing optimizer with fixed objective-evaluation budget."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..representation import clamp_vector, random_vector


def _propose_neighbor(
    current: np.ndarray,
    n_feature_genes: int,
    rng: np.random.Generator,
    bit_flips_range: tuple[int, int],
    param_step_prob: float,
    temperature_ratio: float,
) -> np.ndarray:
    neighbor = current.copy()

    lo, hi = bit_flips_range
    n_flips = int(rng.integers(low=lo, high=hi + 1))
    flip_idx = rng.choice(n_feature_genes, size=min(n_flips, n_feature_genes), replace=False)
    neighbor[flip_idx] = 1.0 - neighbor[flip_idx]

    step_scale = 0.12 * max(temperature_ratio, 0.1)
    for idx in range(n_feature_genes, len(neighbor)):
        if rng.random() < param_step_prob:
            neighbor[idx] += rng.normal(loc=0.0, scale=step_scale)
    return clamp_vector(neighbor)


def run_sa(
    evaluator,
    budget_b: int,
    n_feature_genes: int,
    rng: np.random.Generator,
    sa_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run SA; each evaluator call consumes one budget unit."""
    t0 = float(sa_cfg.get("T0", 1.0))
    cooling = float(sa_cfg.get("alpha", 0.995))
    neighbor_cfg = sa_cfg.get("neighbor", {})
    flips = neighbor_cfg.get("bit_flips", [1, 3])
    bit_flips_range = (int(flips[0]), int(flips[1]))
    param_step_prob = float(neighbor_cfg.get("param_step_prob", 0.5))

    current = random_vector(rng, n_feature_genes)
    rec_current = evaluator.evaluate(current)
    evaluations = 1
    current_score = rec_current.score
    current_solution = rec_current.solution

    best_score = current_score
    best_solution = current_solution
    best_metrics = dict(rec_current.metrics)
    best_vector = current.copy()
    history: list[dict[str, Any]] = [
        {
            "evaluation": evaluations,
            "score": current_score,
            "best_score": best_score,
            "cache_hit": int(rec_current.cache_hit),
        }
    ]

    step = 0
    while evaluations < budget_b:
        step += 1
        temperature = t0 * (cooling**step)
        proposal = _propose_neighbor(
            current=current,
            n_feature_genes=n_feature_genes,
            rng=rng,
            bit_flips_range=bit_flips_range,
            param_step_prob=param_step_prob,
            temperature_ratio=temperature / max(t0, 1e-9),
        )
        rec_proposal = evaluator.evaluate(proposal)
        evaluations += 1

        delta = rec_proposal.score - current_score
        if delta >= 0:
            accept = True
        else:
            # Metropolis acceptance allows occasional worse moves to escape local minima.
            acceptance_prob = math.exp(delta / max(temperature, 1e-9))
            accept = rng.random() < acceptance_prob

        if accept:
            current = proposal
            current_score = rec_proposal.score
            current_solution = rec_proposal.solution

        if rec_proposal.score > best_score:
            best_score = rec_proposal.score
            best_solution = rec_proposal.solution
            best_metrics = dict(rec_proposal.metrics)
            best_vector = proposal.copy()

        history.append(
            {
                "evaluation": evaluations,
                "score": rec_proposal.score,
                "best_score": best_score,
                "cache_hit": int(rec_proposal.cache_hit),
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
