import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from house_price.pipeline import build_pipeline


def test_pipeline_fits_and_predicts_on_small_sample() -> None:
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
    pipeline = build_pipeline(
        id_column="Id",
        model=HistGradientBoostingRegressor(
            random_state=42,
            max_iter=2,
            min_samples_leaf=2,
        ),
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict(X.head(3))

    assert len(predictions) == 3
    assert (predictions > 0).all()
