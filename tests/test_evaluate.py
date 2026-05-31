"""Contracts for the evaluation metrics."""

import numpy as np
import pytest

from bike_sharing.evaluate import mae, r2, report, rmse, rmsle


def test_mae_zero_on_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([2.0, 0.0, 0.0])
    # sqrt(mean([4, 0, 0])) = sqrt(4/3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(4 / 3))


def test_r2_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == pytest.approx(1.0)


def test_r2_mean_predictor_yields_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.full_like(y, y.mean())
    assert r2(y, pred) == pytest.approx(0.0)


def test_rmsle_zero_on_perfect_predictions():
    y = np.array([0.0, 10.0, 100.0, 977.0])
    assert rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_clips_negative_predictions():
    # A model that predicts -5 (impossible demand) should not crash and
    # should be treated as 0 inside log1p.
    y_true = np.array([0.0])
    y_pred_neg = np.array([-5.0])
    y_pred_zero = np.array([0.0])
    assert rmsle(y_true, y_pred_neg) == pytest.approx(rmsle(y_true, y_pred_zero))


def test_report_returns_full_metric_set():
    y = np.array([0.0, 1.0, 4.0, 9.0])
    pred = np.array([1.0, 1.0, 1.0, 9.0])
    out = report(y, pred)
    assert set(out) == {"rmsle", "rmse", "mae", "r2"}
    for value in out.values():
        assert isinstance(value, float)
