"""Feature engineering for the Bike Sharing Demand dataset.

Single responsibility: turn raw rows into a numeric feature frame.
Leakage exclusion lives in ``preprocessing.drop_leakage_columns``; target
transformation lives in ``preprocessing.to_log1p_target``. This module
adds time-derived features and cyclic encodings only.

The raw ``datetime`` column is preserved on the returned frame because
it is needed to label the test-set prediction artifact. Models drop it
from ``X`` at fit time, not here.
"""

from typing import Any

import numpy as np
import pandas as pd

# `day` is intentionally excluded: the Kaggle split puts days 1-19 in
# train and days 20-31 in test, so day-of-month has zero overlap between
# the two sets. Including it would let trees split on out-of-distribution
# values at test time and force linear models to extrapolate. See
# AGENTS.md s2 for the original data-contract note.
TIME_FEATURE_COLUMNS = (
    "hour",
    "dayofweek",
    "month",
    "year",
    "is_weekend",
)
CYCLIC_FEATURE_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
)
ADDED_FEATURE_COLUMNS = TIME_FEATURE_COLUMNS + CYCLIC_FEATURE_COLUMNS


def build_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return a copy of ``df`` with time-derived and cyclic features added.

    Adds the columns in ``ADDED_FEATURE_COLUMNS``. Does not drop
    ``datetime``, ``casual``, ``registered``, or ``count`` — leakage
    exclusion is the job of ``preprocessing.drop_leakage_columns`` and
    the target column lives until fit time.
    """
    datetime_col = cfg["datetime_col"]
    out = df.copy()
    ts = out[datetime_col]

    out["hour"] = ts.dt.hour.astype("int16")
    out["dayofweek"] = ts.dt.dayofweek.astype("int16")
    out["month"] = ts.dt.month.astype("int16")
    out["year"] = ts.dt.year.astype("int16")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24).astype("float32")
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12).astype("float32")
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12).astype("float32")

    return out
