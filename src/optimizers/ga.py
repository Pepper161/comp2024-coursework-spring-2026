"""Genetic Algorithm optimizer with fixed objective-evaluation budget."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..representation import clamp_vector, random_vector


def _tournament_index(
    scores: np.ndarray, k: int, rng: np.random.Generator
) -> int:
    candidates = rng.integers(low=0, high=len(scores), size=max(2, k))
    best = candidates[0]
    for idx in candidates[1:]:
        if scores[idx] > scores[best]:
            best = idx
    return int(best)


def _crossover_and_mutate(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    n_feature_genes: int,
    crossover_rate: float,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    child = parent_a.copy()
    if rng.random() < crossover_rate:
        cut = int(rng.integers(1, max(2, n_feature_genes)))
        child[:n_feature_genes] = np.concatenate(
            [parent_a[:cut], parent_b[cut:n_feature_genes]]
        )
        for idx in range(n_feature_genes, len(child)):
            child[idx] = parent_a[idx] if rng.random() < 0.5 else parent_b[idx]

    for idx in range(n_feature_genes):
        if rng.random() < mutation_rate:
            child[idx] = 1.0 - child[idx]
    for idx in range(n_feature_genes, len(child)):
        if rng.random() < mutation_rate:
            child[idx] += rng.normal(loc=0.0, scale=0.12)
    return clamp_vector(child)


def run_ga(
    evaluator,
    budget_b: int,
    n_feature_genes: int,
    rng: np.random.Generator,
    ga_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run GA; every call to evaluator consumes one budget unit (cache hit included)."""
    pop_size = int(ga_cfg.get("pop_size", 30))
    tournament_k = int(ga_cfg.get("tournament_k", 3))
    crossover_rate = float(ga_cfg.get("crossover_rate", 0.9))
    mutation_rate = float(ga_cfg.get("mutation_rate", 0.05))

    population = np.vstack([random_vector(rng, n_feature_genes) for _ in range(pop_size)])
    scores = np.full(pop_size, -np.inf, dtype=np.float64)
    best_score = -np.inf
    best_solution = None
    best_metrics = None
    best_vector = None
    history: list[dict[str, Any]] = []

    evaluations = 0
    for i in range(pop_size):
        if evaluations >= budget_b:
            break
        rec = evaluator.evaluate(population[i])
        scores[i] = rec.score
        evaluations += 1
        if rec.score > best_score:
            best_score = rec.score
            best_solution = rec.solution
            best_metrics = dict(rec.metrics)
            best_vector = population[i].copy()
        history.append(
            {
                "evaluation": evaluations,
                "score": rec.score,
                "best_score": best_score,
                "cache_hit": int(rec.cache_hit),
            }
        )

    while evaluations < budget_b:
        idx_a = _tournament_index(scores, tournament_k, rng)
        idx_b = _tournament_index(scores, tournament_k, rng)
        child = _crossover_and_mutate(
            parent_a=population[idx_a],
            parent_b=population[idx_b],
            n_feature_genes=n_feature_genes,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            rng=rng,
        )
        rec = evaluator.evaluate(child)
        evaluations += 1

        worst_idx = int(np.argmin(scores))
        population[worst_idx] = child
        scores[worst_idx] = rec.score

        if rec.score > best_score:
            best_score = rec.score
            best_solution = rec.solution
            best_metrics = dict(rec.metrics)
            best_vector = child.copy()

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

