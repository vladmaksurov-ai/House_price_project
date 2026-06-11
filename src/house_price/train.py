"""Evaluate and train the final model."""

import json
import platform
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold, cross_val_score

from house_price.config import SETTINGS, Settings
from house_price.data import load_training_data
from house_price.pipeline import build_pipeline


def train(settings: Settings = SETTINGS) -> dict[str, object]:
    """Cross-validate, train on all rows, and persist reproducible artifacts."""
    X, y = load_training_data(settings.train_path, settings.target_column, settings.id_column)
    model = build_pipeline(settings.id_column, settings.random_state, settings.model_params)
    folds = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state)

    scores = -cross_val_score(
        model,
        X,
        y,
        cv=folds,
        scoring="neg_root_mean_squared_log_error",
        n_jobs=-1,
    )
    model.fit(X, y)

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.model_path)

    report: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(X),
        "raw_feature_count": X.shape[1],
        "cv_folds": settings.cv_folds,
        "cv_rmsle_scores": scores.tolist(),
        "cv_rmsle_mean": float(np.mean(scores)),
        "cv_rmsle_std": float(np.std(scores)),
        "model": "HistGradientBoostingRegressor",
        "model_params": settings.model_params,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    settings.report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = train()
    print(f"CV RMSLE: {report['cv_rmsle_mean']:.5f} +/- {report['cv_rmsle_std']:.5f}")
    print(f"Model saved to: {SETTINGS.model_path}")
    print(f"Report saved to: {SETTINGS.report_path}")


if __name__ == "__main__":
    main()

