import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from house_price.models import build_model


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("hist_gradient_boosting", HistGradientBoostingRegressor),
        ("random_forest", RandomForestRegressor),
        ("xgboost", XGBRegressor),
    ],
)
def test_build_model_returns_expected_type(name: str, expected_type: type) -> None:
    model = build_model(name, random_state=42, model_params={})

    assert isinstance(model, expected_type)


def test_build_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown", random_state=42, model_params={})


def test_build_model_applies_parameters() -> None:
    model = build_model(
        "random_forest",
        random_state=42,
        model_params={"n_estimators": 10},
    )

    assert model.random_state == 42
    assert model.n_estimators == 10
