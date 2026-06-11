"""Create a Kaggle-ready submission from a persisted model."""

import joblib
import numpy as np
import pandas as pd

from house_price.config import SETTINGS, Settings
from house_price.data import load_test_data


def create_submission(settings: Settings = SETTINGS) -> pd.DataFrame:
    """Load the trained model and write predictions in Kaggle format."""
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {settings.model_path}. Run `python -m house_price.train` first."
        )

    test_data = load_test_data(settings.test_path, settings.id_column)
    model = joblib.load(settings.model_path)
    predictions = np.clip(model.predict(test_data), a_min=0, a_max=None)
    submission = pd.DataFrame(
        {
            settings.id_column: test_data[settings.id_column],
            settings.target_column: predictions,
        }
    )

    settings.submissions_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(settings.submission_path, index=False)
    return submission


def main() -> None:
    submission = create_submission()
    print(f"Created {len(submission)} predictions: {SETTINGS.submission_path}")


if __name__ == "__main__":
    main()

