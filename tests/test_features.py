import pandas as pd

from house_price.features import HouseFeatureEngineer


def test_feature_engineer_creates_expected_values() -> None:
    data = pd.DataFrame(
        {
            "YrSold": [2010],
            "YearBuilt": [2000],
            "YearRemodAdd": [2005],
            "GarageYrBlt": [2001],
            "TotalBsmtSF": [500],
            "1stFlrSF": [800],
            "2ndFlrSF": [200],
            "OpenPorchSF": [10],
            "EnclosedPorch": [20],
            "3SsnPorch": [30],
            "ScreenPorch": [40],
            "WoodDeckSF": [50],
            "FullBath": [2],
            "HalfBath": [1],
            "BsmtFullBath": [1],
            "BsmtHalfBath": [1],
            "GarageType": ["Attchd"],
            "PoolArea": [0],
            "Fireplaces": [1],
            "MSSubClass": [20],
            "MoSold": [6],
        }
    )

    result = HouseFeatureEngineer().fit_transform(data)

    assert result.loc[0, "HouseAge"] == 10
    assert result.loc[0, "RemodAge"] == 5
    assert result.loc[0, "TotalSF"] == 1500
    assert result.loc[0, "TotalPorchSF"] == 150
    assert result.loc[0, "TotalBath"] == 4
    assert result.loc[0, "HasGarage"] == 1
    assert result.loc[0, "HasPool"] == 0


def test_feature_engineer_does_not_mutate_input() -> None:
    data = pd.DataFrame(
        {
            "YrSold": [2010],
            "YearBuilt": [2000],
            "YearRemodAdd": [2000],
            "GarageYrBlt": [2000],
            "TotalBsmtSF": [0],
            "1stFlrSF": [1],
            "2ndFlrSF": [0],
            "OpenPorchSF": [0],
            "EnclosedPorch": [0],
            "3SsnPorch": [0],
            "ScreenPorch": [0],
            "WoodDeckSF": [0],
            "FullBath": [1],
            "HalfBath": [0],
            "BsmtFullBath": [0],
            "BsmtHalfBath": [0],
            "GarageType": [None],
            "PoolArea": [0],
            "Fireplaces": [0],
            "MSSubClass": [20],
            "MoSold": [1],
        }
    )

    HouseFeatureEngineer().fit_transform(data)

    assert "HouseAge" not in data.columns

