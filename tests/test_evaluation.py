import numpy as np

from house_price.evaluation import evaluate_model


def test_evaluate_model_returns_cv_metrics(monkeypatch) -> None:
    expected_scores = np.array([-0.12, -0.14, -0.13])
    pipeline = object()
    X = object()
    y = object()
    folds = object()

    def fake_cross_val_score(estimator, passed_X, passed_y, **kwargs):
        assert estimator is pipeline
        assert passed_X is X
        assert passed_y is y
        assert kwargs["cv"] is folds
        assert kwargs["scoring"] == "neg_root_mean_squared_log_error"
        assert kwargs["n_jobs"] == -1
        return expected_scores

    monkeypatch.setattr(
        "house_price.evaluation.cross_val_score",
        fake_cross_val_score,
    )

    result = evaluate_model(
        pipeline=pipeline,
        X=X,
        y=y,
        folds=folds,
    )

    assert result["cv_rmsle_scores"] == [0.12, 0.14, 0.13]
    assert result["cv_rmsle_mean"] == 0.13
    assert result["cv_rmsle_std"] == np.std([0.12, 0.14, 0.13])
    assert result["elapsed_seconds"] >= 0


