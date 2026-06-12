"""Create configured regression models for training and comparison."""

from collections.abc import Mapping

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor


def build_model(name: str, random_state:str, model_params: Mapping[str, int | float]):
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=random_state, **model_params)

    if name == "random_forest":
        return RandomForestRegressor(random_state=random_state, **model_params)

    if name == "xgboost":
        return XGBRegressor(random_state=random_state, **model_params)

    raise ValueError(f"Unknown model: {name}")
