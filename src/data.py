"""
data.py

This module centralizes dataset ingestion and splitting logic.

I implemented schema validation and logging to ensure data integrity and reproducibility.
This allows the pipeline to fail early when unexpected data arrives and provides traceability
for dataset characteristics and model training steps.
"""

from __future__ import annotations

import logging
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Reproducibility (Global Seed)
# A global seed helps reproduce results across runs, machines, and environments.
RANDOM_STATE: int = 42


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """
    Set a global seed for reproducibility across common random number generators.

    Note:
    - This makes train/test splits and many ML steps reproducible.
    - Some multi-threaded algorithms can still show tiny variations, but this greatly reduces noise.
    """
    random.seed(seed)
    np.random.seed(seed)


# Logging (Traceability)
# Logging gives visibility into what the pipeline is doing (shapes, missingness, class balance).
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Dataset Schema Expectations
TARGET_COL: str = "Recommended IND"

REQUIRED_COLS = {
    "Clothing ID",
    "Age",
    "Title",
    "Review Text",
    "Positive Feedback Count",
    "Division Name",
    "Department Name",
    "Class Name",
    TARGET_COL,
}


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate the minimum expected schema for this project.

    1) Required columns exist
    2) Target values are binary (0/1)
    3) Dataset is non-empty

    - Since data often changes, failing fast with clear errors saves time and prevents silent pipeline failures.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty or could not be loaded.")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Target sanity check
    unique_y = set(df[TARGET_COL].dropna().unique().tolist())
    if not unique_y.issubset({0, 1}):
        raise ValueError(f"Unexpected target values in '{TARGET_COL}': {sorted(unique_y)}")


def log_basic_data_profile(df: pd.DataFrame) -> None:
    """
    Audit dataset health by logging missingness and target label statistics.
    """
    logger.info("Loaded dataset with shape=%s", df.shape)

    # Missingness overview (top columns with most nulls)
    null_counts = df.isna().sum().sort_values(ascending=False)
    top_nulls = null_counts[null_counts > 0].head(10)
    if len(top_nulls) > 0:
        logger.info("Top columns with missing values:\n%s", top_nulls.to_string())
    else:
        logger.info("No missing values detected in dataset.")

    # Target distribution
    target_mean = df[TARGET_COL].mean()
    target_counts = df[TARGET_COL].value_counts(dropna=False).to_dict()
    logger.info("Target '%s' mean (recommend rate)=%.4f", TARGET_COL, target_mean)
    logger.info("Target '%s' counts=%s", TARGET_COL, target_counts)


def load_data(path: str, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Load raw dataset from CSV + enforce basic data quality guarantees.
    """
    set_global_seed(seed)

    df = pd.read_csv(path)
    validate_schema(df)
    log_basic_data_profile(df)
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y).
    """
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    return X, y


def train_test_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = RANDOM_STATE,
):
    """
    Create a stratified train/test split to preserve class balance.
    """
    set_global_seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    logger.info("Split done: X_train=%s | X_test=%s", X_train.shape, X_test.shape)
    logger.info("Target mean: train=%.4f | test=%.4f", y_train.mean(), y_test.mean())

    return X_train, X_test, y_train, y_test