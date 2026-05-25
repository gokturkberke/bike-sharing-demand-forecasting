"""Build the processed feature matrices for train and test.

Thin orchestrator: loads the raw CSVs, runs ``build_features``, applies
``drop_leakage_columns`` to the train frame (test has no leakage columns
to begin with), and writes parquet files under ``data/processed/``.

No feature engineering or modeling logic lives here; that all sits in
``src/bike_sharing/{features,preprocessing}.py``.

Run from the project root:

    .venv/bin/python scripts/prepare_data.py
"""

from pathlib import Path

from bike_sharing.config import load_config
from bike_sharing.data import load_raw_test, load_raw_train
from bike_sharing.features import build_features
from bike_sharing.preprocessing import drop_leakage_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_raw_train(cfg)
    test_raw = load_raw_test(cfg)

    train_features = build_features(train_raw, cfg)
    test_features = build_features(test_raw, cfg)

    train_safe = drop_leakage_columns(train_features, cfg)

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    train_safe.to_parquet(train_path, index=False)
    test_features.to_parquet(test_path, index=False)

    print(f"wrote {train_path.relative_to(PROJECT_ROOT)} shape={train_safe.shape}")
    print(f"wrote {test_path.relative_to(PROJECT_ROOT)} shape={test_features.shape}")


if __name__ == "__main__":
    main()
