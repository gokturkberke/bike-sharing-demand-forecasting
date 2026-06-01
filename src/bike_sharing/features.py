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
    # Second-harmonic hour terms. experiment:
    # 2026-06-01_leakage-safe-feature-sweep.md - promoted with the
    # workingday-gated terms below; together they let the linear baseline
    # represent the bimodal daily demand shape.
    "hour_sin2",
    "hour_cos2",
)
# Workingday-gated cyclic terms. experiment:
# 2026-06-01_leakage-safe-feature-sweep.md - the interaction_harmonic group
# cut Ridge holdout RMSLE 0.91 -> 0.72 while every tree held or improved.
INTERACTION_FEATURE_COLUMNS = (
    "hour_sin_workday",
    "hour_cos_workday",
)
ADDED_FEATURE_COLUMNS = (
    TIME_FEATURE_COLUMNS + CYCLIC_FEATURE_COLUMNS + INTERACTION_FEATURE_COLUMNS
)


def build_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return a copy of ``df`` with time-derived and cyclic features added.

    Adds the columns in ``ADDED_FEATURE_COLUMNS``. Does not drop
    ``datetime``, ``casual``, ``registered``, or ``count`` — leakage
    exclusion is the job of ``preprocessing.drop_leakage_columns`` and
    the target column lives until fit time.

    Requires an already-parsed datetime column (``cfg["datetime_col"]``) and
    ``workingday`` (used by the workingday-gated cyclic terms); both are
    present in the raw Kaggle train and test frames. Missing or unparsed
    inputs raise rather than fail with an opaque ``KeyError``/``.dt`` error.
    """
    datetime_col = cfg["datetime_col"]
    required = {datetime_col, "workingday"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_features requires columns {sorted(required)}; "
            f"missing: {sorted(missing)}."
        )
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        raise TypeError(
            f"build_features expects {datetime_col!r} already parsed to "
            "datetime (load it via data.load_raw_train / load_raw_test)."
        )
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

    # Second harmonic of the hour cycle: a single sin/cos pair has one peak
    # per day; the second harmonic adds the structure needed for two.
    out["hour_sin2"] = np.sin(2 * 2 * np.pi * out["hour"] / 24).astype("float32")
    out["hour_cos2"] = np.cos(2 * 2 * np.pi * out["hour"] / 24).astype("float32")
    # Cyclic hour gated by workingday: a linear-safe encoding of the
    # hour x workingday interaction (the two daily shapes).
    out["hour_sin_workday"] = (out["hour_sin"] * out["workingday"]).astype("float32")
    out["hour_cos_workday"] = (out["hour_cos"] * out["workingday"]).astype("float32")

    return out


# --- Experimental candidate features (feature sweep) ---------------------
# NOT part of the production feature set. These are the candidates from
# docs/experiments/2026-06-01_leakage-safe-feature-sweep.md that did NOT
# clear the promotion rule: the workingday-gated cyclic + second-harmonic
# group was promoted into build_features above; the columns below were
# dropped (peaks were redundant with that encoding, the environmental
# products regressed the trees, the year flag was marginal). They are kept
# so scripts/run_feature_experiment.py stays reproducible. All are
# leakage-safe and computable on train and test alike, split by how the
# experimental Ridge routes them (scaled numeric vs passthrough).
CANDIDATE_NUMERIC_COLUMNS = ("feels_like_gap", "temp_humidity_interaction")
CANDIDATE_PASSTHROUGH_COLUMNS = (
    "is_morning_peak",
    "is_evening_peak",
    "is_rush_hour",
    "is_2012",
    "bad_weather",
)
CANDIDATE_FEATURE_COLUMNS = CANDIDATE_NUMERIC_COLUMNS + CANDIDATE_PASSTHROUGH_COLUMNS


def build_candidate_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return ``build_features(df)`` plus the (un-promoted) candidate columns.

    Experiment-only (see the module note above): production code calls
    ``build_features``, not this. Composes ``build_features`` first because
    the candidates depend on its ``hour``/``year`` outputs.
    """
    out = build_features(df, cfg)
    hour = out["hour"]

    out["is_morning_peak"] = hour.isin([7, 8, 9]).astype("int8")
    out["is_evening_peak"] = hour.isin([16, 17, 18, 19]).astype("int8")
    out["is_rush_hour"] = (out["is_morning_peak"] | out["is_evening_peak"]).astype("int8")

    out["is_2012"] = (out["year"] == 2012).astype("int8")
    out["feels_like_gap"] = (out["atemp"] - out["temp"]).astype("float32")
    out["temp_humidity_interaction"] = (out["temp"] * out["humidity"]).astype("float32")
    out["bad_weather"] = (out["weather"] >= 3).astype("int8")

    return out
