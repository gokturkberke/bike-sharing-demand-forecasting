"""Contracts for config loading and raw data ingestion."""

from pathlib import Path

import pandas as pd
import pytest

from bike_sharing.config import load_config
from bike_sharing.data import load_raw_test, load_raw_train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

EXPECTED_TRAIN_COLUMNS = {
    "datetime",
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "casual",
    "registered",
    "count",
}
EXPECTED_TEST_COLUMNS = EXPECTED_TRAIN_COLUMNS - {"casual", "registered", "count"}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG_PATH)


def test_config_loads_required_keys(cfg):
    assert cfg["seed"] == 42
    assert cfg["target"] == "count"
    assert cfg["datetime_col"] == "datetime"
    assert {"casual", "registered"}.issubset(cfg["drop_columns"])


def test_config_resolves_paths_to_absolute(cfg):
    raw_train_path = Path(cfg["paths"]["raw_train"])
    assert raw_train_path.is_absolute()
    assert raw_train_path.name == "train.csv"
    assert raw_train_path.parent.name == "raw"


def test_load_raw_train_shape_and_columns(cfg):
    df = load_raw_train(cfg)
    assert df.shape == (10886, 12)
    assert set(df.columns) == EXPECTED_TRAIN_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])


def test_load_raw_test_shape_and_columns(cfg):
    df = load_raw_test(cfg)
    assert df.shape == (6493, 9)
    assert set(df.columns) == EXPECTED_TEST_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])


def test_casual_plus_registered_equals_count(cfg):
    df = load_raw_train(cfg)
    assert (df["casual"] + df["registered"] == df["count"]).all()


def test_missing_raw_file_raises_clear_error(tmp_path, cfg):
    broken_cfg = {
        "datetime_col": cfg["datetime_col"],
        "paths": {"raw_train": str(tmp_path / "does_not_exist.csv")},
    }
    with pytest.raises(FileNotFoundError, match="kaggle.com"):
        load_raw_train(broken_cfg)
