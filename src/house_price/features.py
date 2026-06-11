"""Domain feature engineering implemented as a reusable sklearn transformer."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create stable domain features without learning from the target."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HouseFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()

        data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
        data["RemodAge"] = data["YrSold"] - data["YearRemodAdd"]
        data["GarageAge"] = data["YrSold"] - data["GarageYrBlt"]
        data["WasRemodeled"] = (data["YearBuilt"] != data["YearRemodAdd"]).astype("int8")

        data["TotalSF"] = data[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].sum(axis=1)
        data["TotalPorchSF"] = data[
            ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
        ].sum(axis=1)
        data["TotalBath"] = (
            data["FullBath"]
            + 0.5 * data["HalfBath"]
            + data["BsmtFullBath"]
            + 0.5 * data["BsmtHalfBath"]
        )

        data["HasGarage"] = data["GarageType"].notna().astype("int8")
        data["HasPool"] = (data["PoolArea"] > 0).astype("int8")
        data["HasFireplace"] = (data["Fireplaces"] > 0).astype("int8")
        data["HasBasement"] = (data["TotalBsmtSF"] > 0).astype("int8")

        # Numeric codes in these columns represent labels, not magnitudes.
        for column in ("MSSubClass", "MoSold"):
            data[column] = data[column].astype("string")

        return data


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop known non-feature columns while preserving the DataFrame schema."""

    def __init__(self, columns: tuple[str, ...]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DropColumns":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=list(self.columns), errors="ignore")
