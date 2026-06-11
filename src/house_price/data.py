"""Data loading and schema checks."""

from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV and fail with a useful message when it is unavailable."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def validate_columns(data: pd.DataFrame, required_columns: set[str]) -> None:
    """Validate the minimal schema required by the pipeline."""
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_training_data(
    path: Path,
    target_column: str,
    id_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load training features and target."""
    data = load_csv(path)
    validate_columns(data, {id_column, target_column})
    X = data.drop(columns=target_column)
    y = data[target_column].copy()
    if y.isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values")
    if (y <= 0).any():
        raise ValueError(f"Target column {target_column!r} must contain positive values")
    return X, y


def load_test_data(path: Path, id_column: str) -> pd.DataFrame:
    """Load competition test data."""
    data = load_csv(path)
    validate_columns(data, {id_column})
    return data

