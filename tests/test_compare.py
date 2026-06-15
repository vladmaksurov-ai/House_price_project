import json

from house_price.compare import MODEL_NAMES, compare_models
from house_price.config import Settings


def test_compare_models_evaluates_all_models_with_identical_inputs(tmp_path, monkeypatch) -> None:
    X = object()
    y = object()
    created_models = []
    evaluated_pipelines = []
    received_folds = []

    model_params = {
        "hist_gradient_boosting": {"max_iter": 2},
        "random_forest": {"n_estimators": 2},
        "xgboost": {"n_estimators": 2},
    }
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        submissions_dir=tmp_path / "submissions",
        random_state=42,
        cv_folds=3,
        model_params=model_params,
    )

    def fake_build_model(name, random_state, params):
        created_models.append((name, random_state, params))
        return f"model:{name}"

    def fake_build_pipeline(id_column, model):
        return f"pipeline:{id_column}:{model}"

    def fake_evaluate_model(pipeline, passed_X, passed_y, folds):
        assert passed_X is X
        assert passed_y is y
        evaluated_pipelines.append(pipeline)
        received_folds.append(folds)
        return {"cv_rmsle_mean": 0.1}

    monkeypatch.setattr("house_price.compare.load_training_data", lambda *args: (X, y))
    monkeypatch.setattr("house_price.compare.build_model", fake_build_model)
    monkeypatch.setattr("house_price.compare.build_pipeline", fake_build_pipeline)
    monkeypatch.setattr("house_price.compare.evaluate_model", fake_evaluate_model)

    results = compare_models(settings)

    assert tuple(results) == MODEL_NAMES
    assert [name for name, _, _ in created_models] == list(MODEL_NAMES)
    assert all(random_state == settings.random_state for _, random_state, _ in created_models)
    assert [params for _, _, params in created_models] == [
        settings.model_params[name] for name in MODEL_NAMES
    ]
    assert len(evaluated_pipelines) == len(MODEL_NAMES)
    assert all(folds is received_folds[0] for folds in received_folds)

    assert settings.comparison_report_path.exists()
    saved_results = json.loads(settings.comparison_report_path.read_text(encoding="utf-8"))
    assert saved_results == results
