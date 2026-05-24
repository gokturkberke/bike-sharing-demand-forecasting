"""Raw data loading for the Kaggle Bike Sharing Demand dataset."""

from pathlib import Path
from typing import Any

import pandas as pd

KAGGLE_DATA_URL = "https://www.kaggle.com/c/bike-sharing-demand/data"


def load_raw_train(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the raw training CSV with datetime parsed."""
    return _read_csv(Path(cfg["paths"]["raw_train"]), cfg["datetime_col"])


def load_raw_test(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the raw test CSV with datetime parsed."""
    return _read_csv(Path(cfg["paths"]["raw_test"]), cfg["datetime_col"])


def _read_csv(path: Path, datetime_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. "
            f"Download the dataset from {KAGGLE_DATA_URL} and place the "
            f"CSV under data/raw/. See the README's Data setup section."
        )
    return pd.read_csv(path, parse_dates=[datetime_col])
