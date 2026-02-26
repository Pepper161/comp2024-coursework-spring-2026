"""Leakage-safe preprocessing fitted only on train folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _build_one_hot_encoder(handle_unknown: str) -> OneHotEncoder:
    # Compatible across sklearn versions where sparse_output may not exist.
    try:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)


@dataclass
class FittedPreprocessor:
    """Fitted preprocessor plus mapping from original to encoded indices."""

    transformer: ColumnTransformer
    original_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    group_to_indices: dict[str, list[int]]
    encoded_feature_names: list[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        transformed = self.transformer.transform(df[self.original_features])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=np.float32)


def fit_preprocessor(
    train_features_df: pd.DataFrame,
    original_features: list[str],
    categorical_cols: list[str],
    numeric_impute: str = "median",
    onehot_handle_unknown: str = "ignore",
) -> FittedPreprocessor:
    """Fit preprocessing on train-only data and return encoded mapping."""
    # Categorical membership is config-driven to keep feature grouping stable across runs.
    categorical = [c for c in categorical_cols if c in original_features]
    numeric = [c for c in original_features if c not in categorical]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=numeric_impute)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _build_one_hot_encoder(onehot_handle_unknown)),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    # Caller controls leakage boundary; this fit must only see training fold data.
    transformer.fit(train_features_df[original_features])

    group_to_indices: dict[str, list[int]] = {}
    encoded_names: list[str] = []
    cursor = 0

    for col in numeric:
        group_to_indices[col] = [cursor]
        encoded_names.append(col)
        cursor += 1

    if categorical:
        ohe = transformer.named_transformers_["cat"].named_steps["onehot"]
        for col, categories in zip(categorical, ohe.categories_):
            width = len(categories)
            group_to_indices[col] = list(range(cursor, cursor + width))
            for cat in categories:
                encoded_names.append(f"{col}={cat}")
            cursor += width

    transformed_dim = transformer.transform(
        train_features_df.iloc[[0]][original_features]
    ).shape[1]
    if cursor != transformed_dim:
        raise RuntimeError(
            f"Encoded index mapping mismatch. mapped={cursor}, actual={transformed_dim}"
        )

    return FittedPreprocessor(
        transformer=transformer,
        original_features=original_features,
        numeric_features=numeric,
        categorical_features=categorical,
        group_to_indices=group_to_indices,
        encoded_feature_names=encoded_names,
    )


def split_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract X dataframe and y array (int) from a dataframe."""
    x_df = df[feature_cols].copy()
    y = df[target_col].astype(int).to_numpy()
    return x_df, y
