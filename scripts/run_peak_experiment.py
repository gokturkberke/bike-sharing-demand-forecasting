"""Peak-demand underprediction: Duan's smearing on the day-of-month holdout.

Experiment orchestrator for
docs/experiments/2026-06-05_peak-underprediction.md. Fits the deployed model
(xgboost) on the day-of-month train rows (days 1-15), computes Duan's smearing
factor theta from the TRAINING log-residuals, applies it to the held-out days
16-19, and compares uncorrected vs smeared predictions - overall (RMSLE, RMSE,
MAE, R2) and stratified by demand quintile (RMSE, mean bias). Writes a JSON
next to the plan file. Touches no production config, model artifact, or
prediction path.

Run from project root, after prepare_data.py:
    .venv/bin/python scripts/run_peak_experiment.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bike_sharing.config import load_config, load_models_config
from bike_sharing.evaluate import report
from bike_sharing.models import get_model
from bike_sharing.postprocess import apply_smearing, compute_smearing_factor
from bike_sharing.train import day_of_month_holdout_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "docs" / "experiments" / "2026-06-05_peak-underprediction.json"
MODEL = "xgboost"
N_QUANTILES = 5


def _per_quintile(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    df = pd.DataFrame({"actual": y_true, "pred": y_pred})
    df["resid"] = df["pred"] - df["actual"]
    df["bin"] = pd.qcut(df["actual"], N_QUANTILES, duplicates="drop")
    rows = []
    for label, g in df.groupby("bin", observed=True):
        rows.append({
            "quantile": str(label),
            "n": int(len(g)),
            "rmse": round(float(np.sqrt(np.mean(g["resid"] ** 2))), 2),
            "mean_bias": round(float(g["resid"].mean()), 2),
        })
    return rows


def main(force: bool = False) -> None:
    if OUT_PATH.exists() and not force:
        raise FileExistsError(
            f"{OUT_PATH.relative_to(PROJECT_ROOT)} already exists - pass --force "
            "to overwrite the recorded experiment."
        )

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    params = load_models_config(PROJECT_ROOT / "config" / "models.yaml").get(MODEL, {})
    target = cfg["target"]
    datetime_col = cfg["datetime_col"]

    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "train.parquet")
    y = df[target].to_numpy(float)
    dt = df[datetime_col]
    X = df.drop(columns=[target, datetime_col])

    tr_idx, ho_idx = day_of_month_holdout_split(dt)
    model = get_model(MODEL, cfg, params).fit(X.iloc[tr_idx], y[tr_idx])

    # Duan's theta from TRAINING log-residuals (days 1-15) only.
    log_pred_train = model.regressor_.predict(X.iloc[tr_idx])
    e = np.log1p(y[tr_idx]) - log_pred_train
    theta = compute_smearing_factor(e)

    # Holdout (days 16-19): uncorrected (the production inverse) vs smeared.
    y_true = y[ho_idx]
    log_pred_ho = model.regressor_.predict(X.iloc[ho_idx])
    uncorrected = np.asarray(model.predict(X.iloc[ho_idx]), dtype=float)
    smeared = apply_smearing(log_pred_ho, theta)

    results = {
        "model": MODEL,
        "theta": round(float(theta), 4),
        "n_holdout": int(len(ho_idx)),
        "uncorrected": {
            "overall": {k: round(v, 4) for k, v in report(y_true, uncorrected).items()},
            "by_quintile": _per_quintile(y_true, uncorrected),
        },
        "smeared": {
            "overall": {k: round(v, 4) for k, v in report(y_true, smeared).items()},
            "by_quintile": _per_quintile(y_true, smeared),
        },
    }

    u = results["uncorrected"]["overall"]
    s = results["smeared"]["overall"]
    print(f"model={MODEL}  theta={theta:.4f}  n_holdout={len(ho_idx)}")
    print(f"  uncorrected: rmsle={u['rmsle']:.4f} rmse={u['rmse']:.2f} "
          f"mae={u['mae']:.2f} r2={u['r2']:.3f}")
    print(f"  smeared:     rmsle={s['rmsle']:.4f} rmse={s['rmse']:.2f} "
          f"mae={s['mae']:.2f} r2={s['r2']:.3f}")
    print(f"  top-quintile mean bias: "
          f"uncorrected={results['uncorrected']['by_quintile'][-1]['mean_bias']:+.2f} "
          f"smeared={results['smeared']['by_quintile'][-1]['mean_bias']:+.2f}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Duan smearing peak-underprediction experiment."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite the existing experiment JSON."
    )
    args = parser.parse_args()
    main(force=args.force)
