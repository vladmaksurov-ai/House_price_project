"""Metrics matching the competition objective."""

import numpy as np


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared logarithmic error with non-negative predictions."""
    predictions = np.clip(np.asarray(y_pred, dtype=float), a_min=0, a_max=None)
    actual = np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(np.square(np.log1p(predictions) - np.log1p(actual)))))

