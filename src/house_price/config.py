"""Project configuration kept in one discoverable place."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Immutable settings used by training and prediction commands."""

    data_dir: Path = PROJECT_DIR / "data"
    artifacts_dir: Path = PROJECT_DIR / "artifacts"
    submissions_dir: Path = PROJECT_DIR / "submissions"
    target_column: str = "SalePrice"
    id_column: str = "Id"
    random_state: int = 42
    cv_folds: int = 5
    default_model_name: str = "xgboost"
    model_params: dict[str, dict[str, int | float]] = field(
        default_factory=lambda: {
            "hist_gradient_boosting": {
                "learning_rate": 0.05,
                "max_iter": 500,
                "max_leaf_nodes": 31,
                "l2_regularization": 1.0,
                "min_samples_leaf": 20,
            },
            "random_forest": {
                "n_estimators": 300,
                "n_jobs": 1,
            },
            "xgboost": {
                "subsample": 1.0,
                "reg_lambda": 1,
                "reg_alpha": 1,
                "n_estimators": 800,
                "min_child_weight": 3,
                "max_depth": 2,
                "learning_rate": 0.1,
                "colsample_bytree": 0.7,
                "n_jobs": 1,
            },
        }
    )

    @property
    def train_path(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def test_path(self) -> Path:
        return self.data_dir / "test.csv"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "model.joblib"

    @property
    def report_path(self) -> Path:
        return self.artifacts_dir / "training_report.json"

    @property
    def submission_path(self) -> Path:
        return self.submissions_dir / "submission.csv"

    @property
    def comparison_report_path(self) -> Path:
        return self.artifacts_dir / "model_comparison.json"


SETTINGS = Settings()
