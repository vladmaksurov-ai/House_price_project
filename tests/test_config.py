from house_price.config import PROJECT_DIR, SETTINGS


def test_default_paths_are_relative_to_project_root() -> None:
    assert SETTINGS.data_dir == PROJECT_DIR / "data"
    assert SETTINGS.train_path == PROJECT_DIR / "data" / "train.csv"
    assert SETTINGS.test_path == PROJECT_DIR / "data" / "test.csv"
    assert SETTINGS.artifacts_dir == PROJECT_DIR / "artifacts"
    assert SETTINGS.submissions_dir == PROJECT_DIR / "submissions"
    assert SETTINGS.default_model_name == "xgboost"
    assert SETTINGS.comparison_report_path == PROJECT_DIR / "artifacts" / "model_comparison.json"
