"""Compare configured regression models using identical cross-validation splits."""

import json

from sklearn.model_selection import KFold

from house_price.config import SETTINGS, Settings
from house_price.data import load_training_data
from house_price.evaluation import evaluate_model
from house_price.models import build_model
from house_price.pipeline import build_pipeline

MODEL_NAMES = (
    "hist_gradient_boosting",
    "random_forest",
    "xgboost",
)


def compare_models(settings: Settings = SETTINGS) -> dict[str, dict[str, object]]:
    """Evaluate every configured model identically and persist the results."""
    results: dict[str, dict[str, object]] = {}
    X, y = load_training_data(
        settings.train_path,
        settings.target_column,
        settings.id_column,
    )
    folds = KFold(
        n_splits=settings.cv_folds,
        shuffle=True,
        random_state=settings.random_state,
    )

    for model_name in MODEL_NAMES:
        model = build_model(model_name, settings.random_state, settings.model_params[model_name])
        pipeline = build_pipeline(settings.id_column, model)
        results[model_name] = evaluate_model(pipeline, X, y, folds)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.comparison_report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results


def main() -> None:
    """Run the configured model comparison and print a concise summary."""
    results = compare_models()
    for model_name, metrics in results.items():
        print(
            f"{model_name}: "
            f"{metrics['cv_rmsle_mean']:.5f} +/- {metrics['cv_rmsle_std']:.5f} "
            f"({metrics['elapsed_seconds']:.1f}s)"
        )
    print(f"Comparison saved to: {SETTINGS.comparison_report_path}")


if __name__ == "__main__":
    main()
