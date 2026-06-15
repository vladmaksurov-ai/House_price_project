"""Compare configured regression models using identical cross-validation splits."""

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
    """Evaluate every configured model on the same data and CV splits."""
    results = {}
    X, y = load_training_data(settings.train_path,
                              settings.target_column,
                              settings.id_column)
    folds = KFold(n_splits=settings.cv_folds,
                  shuffle=True,
                  random_state=settings.random_state)

    for model_name in MODEL_NAMES:
        model = build_model(model_name, settings.random_state, settings.model_params[model_name])
        pipeline = build_pipeline(settings.id_column, model)
        results[model_name] = evaluate_model(pipeline, X, y, folds)
    return results
