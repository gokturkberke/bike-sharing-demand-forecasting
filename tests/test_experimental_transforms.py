"""Contracts for the experiment-only environmental recalibration transforms.

The key property is leakage safety: each transform learns its parameters from
the fit data only, so a held-out row is transformed with training-derived
values, never its own.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from bike_sharing.experimental_transforms import AtempRecalibrator, WindspeedZeroImputer


def _frame(windspeed, temp=None, humidity=None, atemp=None):
    n = len(windspeed)
    return pd.DataFrame(
        {
            "temp": temp if temp is not None else np.linspace(5, 35, n),
            "atemp": atemp if atemp is not None else np.linspace(5, 40, n),
            "humidity": humidity if humidity is not None else np.linspace(20, 90, n),
            "windspeed": np.asarray(windspeed, dtype=float),
            "other": np.arange(n),  # an untouched column, to check preservation
        }
    )


def test_windspeed_imputer_learns_train_median_and_applies_to_holdout():
    train = _frame([0.0, 10.0, 20.0, 30.0])  # non-zero median = 20
    imp = WindspeedZeroImputer().fit(train)
    assert imp.fill_value_ == pytest.approx(20.0)
    # A held-out zero is filled with the TRAIN median, not anything of its own.
    holdout = _frame([0.0, 5.0])
    out = imp.transform(holdout)
    assert out["windspeed"].iloc[0] == pytest.approx(20.0)
    assert out["windspeed"].iloc[1] == pytest.approx(5.0)  # non-zero untouched


def test_windspeed_imputer_fit_depends_only_on_fit_data():
    a = WindspeedZeroImputer().fit(_frame([0.0, 10.0, 20.0, 30.0]))    # median 20
    b = WindspeedZeroImputer().fit(_frame([0.0, 2.0, 4.0, 6.0, 8.0]))  # median 4
    assert a.fill_value_ != b.fill_value_


def test_atemp_recalibrator_uses_fit_coefficients():
    rng = np.random.default_rng(0)
    n = 50
    temp = rng.uniform(0, 40, n)
    humidity = rng.uniform(20, 90, n)
    atemp = 2.0 + 0.9 * temp + 0.05 * humidity  # exactly linear
    train = _frame(np.ones(n), temp=temp, humidity=humidity, atemp=atemp)
    out = AtempRecalibrator().fit(train).transform(train)
    assert np.allclose(out["atemp"].to_numpy(), atemp, atol=1e-6)


def test_transforms_preserve_columns_and_order():
    df = _frame([0.0, 10.0, 20.0])
    for transformer in (WindspeedZeroImputer, AtempRecalibrator):
        out = transformer().fit(df).transform(df)
        assert list(out.columns) == list(df.columns)
        assert out.shape == df.shape
        assert (out["other"].to_numpy() == df["other"].to_numpy()).all()


def test_clone_then_fit_on_train_fold_is_leakage_safe():
    # Mimic per-fold use: clone, fit on the train fold only. The learned fill
    # must equal the train fold's median, uninfluenced by held-out rows.
    full = _frame([0.0, 10.0, 20.0, 30.0, 1000.0])  # last row is a held-out outlier
    train_fold = full.iloc[:4]
    imp = clone(WindspeedZeroImputer()).fit(train_fold)
    assert imp.fill_value_ == pytest.approx(20.0)  # median of [10, 20, 30], not 1000
