"""Test-set prediction and prediction-artifact writing.

Single responsibility: turn a fitted model plus the processed test frame
into a ``datetime,count`` prediction artifact, in the dataset's sample
format. Predictions are already non-negative because every real model
inverts its log1p target with ``from_log1p`` (expm1 + clip-at-0); this
module preserves the original test row order and writes the schema.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUBMISSION_COLUMNS = ["datetime", "count"]


def make_prediction_frame(
    model, test_df: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Return a ``datetime,count`` frame for the processed test rows.

    ``test_df`` is the feature-engineered test frame (from
    ``prepare_data.py``): it carries the datetime column plus the model
    features and no target. Row order is preserved so the artifact lines
    up with the original test file.
    """
    datetime_col = cfg["datetime_col"]
    target = cfg["target"]
    features = test_df.drop(columns=[datetime_col])
    preds = np.asarray(model.predict(features), dtype=float)
    # Defensive: the model contract already clips at zero, but never let a
    # negative demand reach the artifact.
    preds = np.clip(preds, a_min=0.0, a_max=None)
    return pd.DataFrame(
        {datetime_col: test_df[datetime_col].to_numpy(), target: preds}
    )


def write_prediction_artifact(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the prediction frame as CSV with the ``datetime,count`` schema."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"prediction frame must have columns {SUBMISSION_COLUMNS}, "
            f"got {list(frame.columns)}."
        )
    frame.to_csv(out_path, index=False)
    return out_path
