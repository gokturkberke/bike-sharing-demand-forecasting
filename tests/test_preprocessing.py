"""Contracts for preprocessing: leakage removal and target transforms."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bike_sharing.config import load_config
from bike_sharing.data import load_raw_train
from bike_sharing.preprocessing import (
    drop_leakage_columns,
    from_log1p,
    to_log1p_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG_PATH)


def test_drop_leakage_columns_removes_casual_and_registered(cfg):
    df = pd.DataFrame(
        {
            "datetime": ["2011-01-01 00:00:00"],
            "casual": [3],
            "registered": [13],
            "count": [16],
            "temp": [9.84],
        }
    )
    out = drop_leakage_columns(df, cfg)
    assert "casual" not in out.columns
    assert "registered" not in out.columns
    assert {"datetime", "count", "temp"}.issubset(out.columns)


def test_drop_leakage_columns_safe_when_columns_absent(cfg):
    df = pd.DataFrame({"datetime": ["2011-01-20"], "temp": [10.66]})
    out = drop_leakage_columns(df, cfg)
    assert set(out.columns) == {"datetime", "temp"}


def test_drop_leakage_columns_returns_copy(cfg):
    df = pd.DataFrame({"casual": [1], "temp": [9.0]})
    out = drop_leakage_columns(df, cfg)
    assert "casual" in df.columns
    assert "casual" not in out.columns


def test_real_train_has_no_leakage_columns_after_drop(cfg):
    df = load_raw_train(cfg)
    out = drop_leakage_columns(df, cfg)
    assert "casual" not in out.columns
    assert "registered" not in out.columns
    assert "count" in out.columns


def test_log1p_roundtrip_is_lossless():
    y = np.array([0, 1, 5, 100, 977], dtype=float)
    np.testing.assert_allclose(from_log1p(to_log1p_target(y)), y, atol=1e-9)


def test_from_log1p_clips_negative_predictions_to_zero():
    y_log = np.array([-5.0, -0.01, 0.0, 0.5, 2.0])
    inverted = from_log1p(y_log)
    assert (inverted >= 0).all()
    assert inverted[0] == 0.0
    assert inverted[1] == 0.0
    assert inverted[3] > 0
