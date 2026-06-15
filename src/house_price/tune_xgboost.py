"""Search for promising XGBoost hyperparameters with cross-validation."""

import json

from sklearn.model_selection import KFold, RandomizedSearchCV

from house_price.config import SETTINGS, Settings
from house_price.data import load_training_data
from house_price.models import build_model
from house_price.pipeline import build_pipeline


XGBOOST_PARAMETER_DISTRIBUTIONS = {
    "regressor__model__n_estimators": [300, 500, 800, 1200],
    "regressor__model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "regressor__model__max_depth": [2, 3, 4, 5, 6],
    "regressor__model__min_child_weight": [1, 3, 5, 10],
    "regressor__model__subsample": [0.7, 0.8, 0.9, 1.0],
    "regressor__model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "regressor__model__reg_alpha": [0, 0.01, 0.1, 1],
    "regressor__model__reg_lambda": [1, 3, 5, 10],
}


def tune_xgboost(settings: Settings = SETTINGS) -> dict[str, object]:
    """Run randomized parameter search and save its best result."""
    X, y = load_training_data(
        settings.train_path,
        settings.target_column,
        settings.id_column,
    )

    model = build_model(
        "xgboost",
        settings.random_state,
        settings.model_params["xgboost"],
    )
    pipeline = build_pipeline(settings.id_column, model)

    folds = KFold(
        n_splits=settings.cv_folds,
        shuffle=True,
        random_state=settings.random_state,
    )

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=XGBOOST_PARAMETER_DISTRIBUTIONS,
        n_iter=30,
        scoring="neg_root_mean_squared_log_error",
        cv=folds,
        random_state=settings.random_state,
        n_jobs=-1,
        verbose=1,
    )

    random_search.fit(X, y)

    tuning_report = {
        "best_cv_rmsle": -float(random_search.best_score_),
        "best_params": random_search.best_params_,
    }

    report_path = settings.artifacts_dir / "xgboost_tuning.json"
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(tuning_report, indent=2),
        encoding="utf-8",
    )

    return tuning_report


def main() -> None:
    """Run XGBoost tuning and print the best cross-validation result."""
    tuning_report = tune_xgboost()

    print(f"Best CV RMSLE: {tuning_report['best_cv_rmsle']:.5f}")
    print("Best parameters:")
    print(json.dumps(tuning_report["best_params"], indent=2))


if __name__ == "__main__":
    main()
