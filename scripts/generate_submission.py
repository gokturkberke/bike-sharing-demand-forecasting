"""Generate a test-set prediction artifact (datetime,count) for a model.

Thin orchestrator. Loads a persisted estimator (``models/<name>.joblib``,
produced by ``train_model.py``) and the processed test parquet (produced
by ``prepare_data.py``), then writes a ``datetime,count`` CSV under
``reports/submissions/`` in the dataset's sample format. This is a
project deliverable, not a leaderboard submission.

Run from the project root, after prepare_data.py and train_model.py:

    .venv/bin/python scripts/generate_submission.py --model xgboost
"""

import argparse
from datetime import datetime as _dt
from pathlib import Path

import joblib
import pandas as pd

from bike_sharing.config import load_config
from bike_sharing.models import MODEL_FACTORIES
from bike_sharing.predict import make_prediction_frame, write_prediction_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def main(model_name: str, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    cfg = load_config(config_path)

    model_path = Path(cfg["paths"]["models_dir"]) / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            f"Run `python scripts/train_model.py --model {model_name}` first."
        )
    test_path = Path(cfg["paths"]["processed_dir"]) / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Processed test data not found at {test_path}. "
            f"Run `python scripts/prepare_data.py` first."
        )

    model = joblib.load(model_path)
    test_df = pd.read_parquet(test_path)
    frame = make_prediction_frame(model, test_df, cfg)

    stamp = _dt.now().strftime("%Y%m%d-%H%M%S")
    out_path = (
        Path(cfg["paths"]["reports_dir"])
        / "submissions"
        / f"{model_name}_{stamp}.csv"
    )
    write_prediction_artifact(frame, out_path)

    print(f"model={model_name}")
    print(f"  rows: {len(frame)}  columns: {list(frame.columns)}")
    print(f"  count range: [{frame['count'].min():.2f}, {frame['count'].max():.2f}]")
    print(f"  wrote: {out_path.relative_to(PROJECT_ROOT)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_FACTORIES),
        help="Which trained model to generate the artifact from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.model)
