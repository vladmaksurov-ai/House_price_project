"""Build the complete preprocessing and regression pipeline."""

from collections.abc import Mapping

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_price.features import DropColumns, HouseFeatureEngineer


def build_pipeline(
    id_column: str,
    random_state: int,
    model_params: Mapping[str, int | float],
) -> TransformedTargetRegressor:
    """Return one object containing features, preprocessing, and model."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )
    regressor = Pipeline(
        steps=[
            ("features", HouseFeatureEngineer()),
            ("drop_id", DropColumns((id_column,))),
            ("preprocess", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(random_state=random_state, **model_params),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=True,
    )
