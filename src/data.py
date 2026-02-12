"""Dataset loading utilities for UNSW-NB15 coursework experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetBundle:
    """Container for train/test data and resolved feature metadata."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target_col: str
    feature_cols: list[str]


def _resolve_target_col(configured_target: str | None, test_df: pd.DataFrame) -> str:
    if configured_target and configured_target in test_df.columns:
        return configured_target
    if "label" in test_df.columns:
        return "label"
    raise ValueError(
        "Unable to resolve target column. Set dataset.target in config "
        "to a valid column name present in testing CSV."
    )


def load_unsw_nb15(project_root: Path, config: dict[str, Any]) -> DatasetBundle:
    """Load train/test CSVs and resolve the exact feature columns."""
    dataset_cfg = config["dataset"]
    train_path = (project_root / dataset_cfg["train_csv"]).resolve()
    test_path = (project_root / dataset_cfg["test_csv"]).resolve()

    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing CSV not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = _resolve_target_col(dataset_cfg.get("target"), test_df)
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training CSV.")

    drop_cols = set(dataset_cfg.get("drop_cols", []))
    drop_cols.add(target_col)

    feature_cols = [col for col in train_df.columns if col not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after applying drop_cols/target.")

    missing_in_test = [col for col in feature_cols if col not in test_df.columns]
    if missing_in_test:
        raise ValueError(
            "Testing CSV is missing required feature columns: "
            + ", ".join(missing_in_test[:10])
        )

    return DatasetBundle(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        feature_cols=feature_cols,
    )

