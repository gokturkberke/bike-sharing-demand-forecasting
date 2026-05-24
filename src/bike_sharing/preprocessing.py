"""Preprocessing primitives for the Bike Sharing Demand dataset.

Owns the two contracts that downstream modules rely on:

1. Leakage removal: ``casual`` and ``registered`` sum to ``count`` and are
   absent from ``test.csv``, so they must never enter the feature matrix
   for the direct-count modeling path.
2. Target transformation: training is done on ``log1p(count)`` and
   predictions are inverted via ``expm1`` and clipped at zero so a
   submission can never carry negative demand.
"""

from typing import Any

import numpy as np
import pandas as pd


def drop_leakage_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return a copy of ``df`` with ``cfg['drop_columns']`` removed.

    Safe to call on ``test.csv``, which does not contain the leakage
    columns: only columns that are actually present are dropped.
    """
    to_drop = [c for c in cfg["drop_columns"] if c in df.columns]
    return df.drop(columns=to_drop)


def to_log1p_target(y: pd.Series | np.ndarray) -> np.ndarray:
    """Apply ``log1p`` to a non-negative target column."""
    return np.log1p(np.asarray(y))


def from_log1p(y_log: np.ndarray) -> np.ndarray:
    """Invert :func:`to_log1p_target` and clip negative predictions to 0.

    Models that fit in log space can produce slightly negative outputs;
    when inverted with ``expm1`` these become negative demand, which is
    nonsensical for hourly bike counts. Per AGENTS.md the prediction
    pipeline clips at zero before the submission stage.
    """
    return np.clip(np.expm1(np.asarray(y_log)), a_min=0.0, a_max=None)
