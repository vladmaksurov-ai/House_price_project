import json

from house_price.config import Settings
from house_price.tune_xgboost import XGBOOST_PARAMETER_DISTRIBUTIONS, tune_xgboost


def test_tune_xgboost_runs_search_and_saves_best_result(tmp_path, monkeypatch) -> None:
    X = object()
    y = object()
    model = object()
    pipeline = object()
    search_arguments = {}
    fitted_data = []

    class FakeRandomizedSearchCV:
        def __init__(self, **kwargs):
            search_arguments.update(kwargs)
            self.best_score_ = -0.123
            self.best_params_ = {"regressor__model__max_depth": 3}

        def fit(self, passed_X, passed_y):
            fitted_data.append((passed_X, passed_y))
            return self

    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        submissions_dir=tmp_path / "submissions",
        random_state=42,
        cv_folds=3,
        model_params={
            "xgboost": {
                "n_estimators": 2,
                "n_jobs": 1,
            }
        },
    )

    monkeypatch.setattr("house_price.tune_xgboost.load_training_data", lambda *args: (X, y))
    monkeypatch.setattr("house_price.tune_xgboost.build_model", lambda *args: model)
    monkeypatch.setattr("house_price.tune_xgboost.build_pipeline", lambda *args: pipeline)
    monkeypatch.setattr("house_price.tune_xgboost.RandomizedSearchCV", FakeRandomizedSearchCV)

    tuning_report = tune_xgboost(settings)

    assert search_arguments["estimator"] is pipeline
    assert search_arguments["param_distributions"] == XGBOOST_PARAMETER_DISTRIBUTIONS
    assert search_arguments["n_iter"] == 30
    assert search_arguments["scoring"] == "neg_root_mean_squared_log_error"
    assert search_arguments["cv"].n_splits == settings.cv_folds
    assert search_arguments["random_state"] == settings.random_state
    assert fitted_data == [(X, y)]
    assert tuning_report == {
        "best_cv_rmsle": 0.123,
        "best_params": {"regressor__model__max_depth": 3},
    }

    saved_report = json.loads(
        (settings.artifacts_dir / "xgboost_tuning.json").read_text(encoding="utf-8")
    )
    assert saved_report == tuning_report
