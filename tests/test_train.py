import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor

from house_price.config import Settings
from house_price.train import train


def make_training_sample() -> tuple[pd.DataFrame, pd.Series]:
    row = {
        "YrSold": 2010,
        "YearBuilt": 2000,
        "YearRemodAdd": 2005,
        "GarageYrBlt": 2001,
        "TotalBsmtSF": 500,
        "1stFlrSF": 800,
        "2ndFlrSF": 200,
        "OpenPorchSF": 10,
        "EnclosedPorch": 20,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "WoodDeckSF": 30,
        "FullBath": 2,
        "HalfBath": 1,
        "BsmtFullBath": 1,
        "BsmtHalfBath": 0,
        "GarageType": "Attchd",
        "PoolArea": 0,
        "Fireplaces": 1,
        "MSSubClass": 20,
        "MoSold": 6,
        "Neighborhood": "NAmes",
    }
    X = pd.DataFrame([{**row, "Id": index} for index in range(1, 21)])
    y = pd.Series([150_000 + index * 1_000 for index in range(20)])
    return X, y


def test_train_evaluates_fits_and_saves_complete_pipeline(tmp_path, monkeypatch) -> None:
    X, y = make_training_sample()
    evaluated_estimators = []

    def fake_cross_val_score(estimator, *args, **kwargs):
        evaluated_estimators.append(estimator)
        return np.array([-0.12, -0.14])

    monkeypatch.setattr("house_price.train.load_training_data", lambda *args: (X, y))
    monkeypatch.setattr("house_price.train.cross_val_score", fake_cross_val_score)

    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        submissions_dir=tmp_path / "submissions",
        cv_folds=2,
        model_params={
            "hist_gradient_boosting": {
                "max_iter": 2,
                "min_samples_leaf": 2,
            },
            "random_forest": {
                "n_estimators": 2,
                "n_jobs": 1,
            },
            "xgboost": {
                "n_estimators": 2,
                "max_depth": 2,
                "n_jobs": 1,
            },
        },
    )

    report = train(model_name="hist_gradient_boosting", settings=settings)
    saved_pipeline = joblib.load(settings.model_path)
    predictions = saved_pipeline.predict(X.head(3))

    assert isinstance(evaluated_estimators[0], TransformedTargetRegressor)
    assert isinstance(saved_pipeline, TransformedTargetRegressor)
    assert report["model"] == "hist_gradient_boosting"
    assert report["model_params"] == settings.model_params["hist_gradient_boosting"]
    assert report["cv_rmsle_scores"] == [0.12, 0.14]
    assert len(predictions) == 3
    assert (predictions > 0).all()
