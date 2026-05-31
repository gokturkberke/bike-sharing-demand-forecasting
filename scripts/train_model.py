"""Train a single model end-to-end and persist its metrics + artifact.

Thin orchestrator. Loads the processed train parquet (produced by
``prepare_data.py``), runs two leakage-safe validations - a chronological
``TimeSeriesSplit`` (``train.fit_and_cv``) and the day-of-month holdout
(``train.evaluate_holdout``) - fits the model on the full train set,
writes ``models/<name>.joblib``, and updates ``reports/metrics.json``
with this run's entry.

Run from the project root, after ``prepare_data.py``:

    .venv/bin/python scripts/train_model.py --model ridge
    .venv/bin/python scripts/train_model.py --model mean_baseline
    .venv/bin/python scripts/train_model.py --model hourly_mean_baseline
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from bike_sharing.config import load_config
from bike_sharing.models import MODEL_FACTORIES, get_model
from bike_sharing.train import evaluate_holdout, fit_and_cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def main(model_name: str, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    cfg = load_config(config_path)
    target = cfg["target"]
    datetime_col = cfg["datetime_col"]

    train_path = Path(cfg["paths"]["processed_dir"]) / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Processed training data not found at {train_path}. "
            f"Run `python scripts/prepare_data.py` first."
        )

    df = pd.read_parquet(train_path)
    y = df[target].to_numpy(dtype=float)
    datetime = df[datetime_col]
    X = df.drop(columns=[target, datetime_col])

    model = get_model(model_name, cfg)
    cv_summary = fit_and_cv(model, X, y, datetime, cfg)
    holdout_summary = evaluate_holdout(get_model(model_name, cfg), X, y, datetime, cfg)

    # Fit on full train for the persisted artifact.
    fitted = get_model(model_name, cfg).fit(X, y)

    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{model_name}.joblib"
    joblib.dump(fitted, model_path)

    metrics_path = reports_dir / "metrics.json"
    metrics = _load_metrics(metrics_path)
    metrics[model_name] = {"cv": cv_summary, "day_of_month_holdout": holdout_summary}
    _save_metrics(metrics_path, metrics)

    mean = cv_summary["mean"]
    holdout = holdout_summary["metrics"]
    print(f"model={model_name}")
    print(f"  cv mean:             rmsle={mean['rmsle']:.4f} rmse={mean['rmse']:.2f} "
          f"mae={mean['mae']:.2f} r2={mean['r2']:.3f}")
    print(f"  day-of-month holdout: rmsle={holdout['rmsle']:.4f} rmse={holdout['rmse']:.2f} "
          f"mae={holdout['mae']:.2f} r2={holdout['r2']:.3f}")
    print(f"  saved estimator: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"  updated metrics: {metrics_path.relative_to(PROJECT_ROOT)}")


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_metrics(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_FACTORIES),
        help="Which model to train.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.model)
