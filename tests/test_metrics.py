import numpy as np

from house_price.metrics import rmsle


def test_rmsle_is_zero_for_perfect_predictions() -> None:
    values = np.array([100_000, 200_000])
    assert rmsle(values, values) == 0.0


def test_rmsle_clips_negative_predictions() -> None:
    score = rmsle(np.array([1.0]), np.array([-5.0]))
    assert np.isfinite(score)

