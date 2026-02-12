"""Solution representation and decoding for feature+hyperparameter search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PARAM_GENE_COUNT = 6


@dataclass(frozen=True)
class SearchSpace:
    n_estimators_min: int
    n_estimators_max: int
    max_depth_min: int
    max_depth_max: int
    allow_none_depth: bool
    min_samples_split_min: int
    min_samples_split_max: int
    min_samples_leaf_min: int
    min_samples_leaf_max: int
    max_features_options: list[str]
    class_weight_options: list[str | None]


@dataclass
class CandidateSolution:
    mask: np.ndarray
    params: dict[str, Any]

    @property
    def k(self) -> int:
        return int(self.mask.sum())


def vector_dim(n_features: int) -> int:
    return n_features + PARAM_GENE_COUNT


def random_vector(rng: np.random.Generator, n_features: int) -> np.ndarray:
    return rng.random(vector_dim(n_features), dtype=np.float64)


def clamp_vector(vec: np.ndarray) -> np.ndarray:
    return np.clip(vec, 0.0, 1.0)


def build_search_space(model_cfg: dict[str, Any]) -> SearchSpace:
    search_cfg = model_cfg["search_space"]
    depth_cfg = search_cfg["max_depth"]
    allow_none_depth = any(v is None for v in depth_cfg)

    class_weight_options: list[str | None] = []
    for item in search_cfg["class_weight"]:
        class_weight_options.append(item)

    return SearchSpace(
        n_estimators_min=int(search_cfg["n_estimators"][0]),
        n_estimators_max=int(search_cfg["n_estimators"][1]),
        max_depth_min=int(depth_cfg[0]),
        max_depth_max=int(depth_cfg[1]),
        allow_none_depth=allow_none_depth,
        min_samples_split_min=int(search_cfg["min_samples_split"][0]),
        min_samples_split_max=int(search_cfg["min_samples_split"][1]),
        min_samples_leaf_min=int(search_cfg["min_samples_leaf"][0]),
        min_samples_leaf_max=int(search_cfg["min_samples_leaf"][1]),
        max_features_options=[str(v) for v in search_cfg["max_features"]],
        class_weight_options=class_weight_options,
    )


def _decode_int(gene: float, lower: int, upper: int) -> int:
    return int(round(lower + gene * (upper - lower)))


def _decode_choice(gene: float, choices: list[Any]) -> Any:
    idx = min(int(gene * len(choices)), len(choices) - 1)
    return choices[idx]


def _enforce_k_min(mask_scores: np.ndarray, mask: np.ndarray, k_min: int) -> np.ndarray:
    enforced = mask.copy()
    need = max(1, min(k_min, mask_scores.shape[0]))
    if int(enforced.sum()) >= need:
        return enforced
    top_idx = np.argsort(mask_scores)[-need:]
    enforced[:] = False
    enforced[top_idx] = True
    return enforced


def decode_solution(
    vector: np.ndarray,
    n_features: int,
    k_min: int,
    search_space: SearchSpace,
) -> CandidateSolution:
    """Decode [0,1] genes into feature mask + RF parameter dictionary."""
    vec = clamp_vector(vector)
    mask_scores = vec[:n_features]
    mask = mask_scores >= 0.5
    mask = _enforce_k_min(mask_scores, mask, k_min)

    g = vec[n_features : n_features + PARAM_GENE_COUNT]

    depth_choices: list[int | None] = list(
        range(search_space.max_depth_min, search_space.max_depth_max + 1)
    )
    if search_space.allow_none_depth:
        depth_choices = [None] + depth_choices

    params: dict[str, Any] = {
        "n_estimators": _decode_int(
            g[0], search_space.n_estimators_min, search_space.n_estimators_max
        ),
        "max_depth": _decode_choice(g[1], depth_choices),
        "min_samples_split": _decode_int(
            g[2], search_space.min_samples_split_min, search_space.min_samples_split_max
        ),
        "min_samples_leaf": _decode_int(
            g[3], search_space.min_samples_leaf_min, search_space.min_samples_leaf_max
        ),
        "max_features": _decode_choice(g[4], search_space.max_features_options),
        "class_weight": _decode_choice(g[5], search_space.class_weight_options),
    }

    return CandidateSolution(mask=mask.astype(bool), params=params)


def solution_key(solution: CandidateSolution) -> str:
    mask_bits = "".join("1" if bit else "0" for bit in solution.mask.tolist())
    ordered_params = [
        str(solution.params["n_estimators"]),
        str(solution.params["max_depth"]),
        str(solution.params["min_samples_split"]),
        str(solution.params["min_samples_leaf"]),
        str(solution.params["max_features"]),
        str(solution.params["class_weight"]),
    ]
    return f"{mask_bits}|{'|'.join(ordered_params)}"


def mask_to_encoded_indices(
    mask: np.ndarray,
    original_features: list[str],
    group_to_indices: dict[str, list[int]],
) -> np.ndarray:
    indices: list[int] = []
    for is_selected, feature in zip(mask.tolist(), original_features):
        if is_selected:
            indices.extend(group_to_indices[feature])
    return np.asarray(indices, dtype=np.int32)

