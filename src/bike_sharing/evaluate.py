"""Evaluation metrics for Bike Sharing Demand models.

All four metrics (RMSLE, RMSE, MAE, R2) are computed on the original
``count`` scale and reported together; no single one is decisive. RMSLE
is informative because the target is right-skewed and it penalizes
under-prediction relative to over-prediction; it applies ``log1p`` to
both the predicted and the true value, so predictions are clipped at
zero first (negative hourly demand is undefined and ``log1p`` of a
negative number is NaN).
"""

from typing import Any

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(diff * diff)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), a_min=0.0, a_max=None)
    diff = np.log1p(y_pred) - np.log1p(y_true)
    return float(np.sqrt(np.mean(diff * diff)))


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the project's full metric set in one call.

    Returns a flat dict with keys ``rmsle``, ``rmse``, ``mae``, ``r2``,
    all on the original count scale and meant to be read together. RMSLE
    clips predictions at zero internally.
    """
    return {
        "rmsle": rmsle(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
