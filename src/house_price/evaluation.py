"""Evaluate regression pipelines with reproducible cross-validation."""

from time import perf_counter

import numpy as np
from sklearn.model_selection import cross_val_score


def evaluate_model(
    pipeline,
    X,
    y,
    folds,
) -> dict[str, object]:
    """Evaluate a pipeline and return serializable cross-validation metrics."""
    started_at = perf_counter()

    scores = -cross_val_score(
        pipeline,
        X,
        y,
        cv=folds,
        scoring="neg_root_mean_squared_log_error",
        n_jobs=-1,
    )

    elapsed_seconds = perf_counter() - started_at
    return {
        "cv_rmsle_scores": scores.tolist(),
        "cv_rmsle_mean": float(np.mean(scores)),
        "cv_rmsle_std": float(np.std(scores)),
        "elapsed_seconds": float(elapsed_seconds),
    }
