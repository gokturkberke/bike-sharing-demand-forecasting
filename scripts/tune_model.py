"""Hyperparameter tuning for the boosting models (xgboost, gradient_boosting).

Experiment orchestrator for
docs/experiments/2026-06-05_xgb-gbm-tuning.md. RandomizedSearchCV over a
chronological TimeSeriesSplit, scored on count-scale RMSLE. The search runs
ONLY on the day-of-month train subset (days 1-15); the day-of-month holdout
(days 16-19) is never seen by the search and is used once at the end as the
final confirmation check (fit days 1-15, score days 16-19). Writes a sweep
JSON next to the plan file and prints a baseline-vs-tuned table. Does NOT
modify config/models.yaml or reports/metrics.json - promotion is a separate,
gated decision recorded in the plan file.

Run from project root, after prepare_data.py:
    .venv/bin/python scripts/tune_model.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from bike_sharing.config import load_config, load_models_config
from bike_sharing.evaluate import rmsle_scorer
from bike_sharing.models import get_model
from bike_sharing.train import day_of_month_holdout_split, evaluate_holdout, fit_and_cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "docs" / "experiments" / "2026-06-05_xgb-gbm-tuning.json"

# Keys are regressor__<param> because get_model returns a
# TransformedTargetRegressor wrapping the estimator.
PARAM_DISTRIBUTIONS = {
    "xgboost": {
        "regressor__n_estimators": [200, 400, 600, 800],
        "regressor__max_depth": [3, 4, 5, 6, 8],
        "regressor__learning_rate": [0.02, 0.05, 0.1],
        "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
        "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "regressor__min_child_weight": [1, 3, 5],
        "regressor__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    },
    "gradient_boosting": {
        "regressor__n_estimators": [200, 300, 500],
        "regressor__learning_rate": [0.02, 0.05, 0.1],
        "regressor__max_depth": [2, 3, 4],
        "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
        "regressor__min_samples_leaf": [1, 3, 5],
        "regressor__max_features": ["sqrt", 0.8, None],
    },
}
N_ITER = {"xgboost": 40, "gradient_boosting": 30}
MODELS = ("xgboost", "gradient_boosting")


def _strip_prefix(params: dict) -> dict:
    return {k.replace("regressor__", "", 1): v for k, v in params.items()}


def _eval(name: str, params: dict, cfg: dict, X, y, dt) -> dict:
    """Baseline-comparable evaluation: day-of-month holdout + CV view, on the
    full data, exactly as scripts/train_model.py records them."""
    ho = evaluate_holdout(get_model(name, cfg, params), X, y, dt, cfg)["metrics"]
    cv = fit_and_cv(get_model(name, cfg, params), X, y, dt, cfg)["mean"]
    return {
        "holdout": {k: round(v, 4) for k, v in ho.items()},
        "cv_rmsle": round(cv["rmsle"], 4),
    }


def main(force: bool = False) -> None:
    if OUT_PATH.exists() and not force:
        raise FileExistsError(
            f"{OUT_PATH.relative_to(PROJECT_ROOT)} already exists - pass --force "
            "to overwrite the recorded tuning sweep."
        )

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_params = load_models_config(PROJECT_ROOT / "config" / "models.yaml")
    target = cfg["target"]
    datetime_col = cfg["datetime_col"]
    seed = int(cfg["seed"])
    n_splits = int(cfg["cv"]["n_splits"])
    gap = int(cfg["cv"].get("gap", 0))

    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "train.parquet")
    y = df[target].to_numpy(float)
    dt = df[datetime_col]
    X = df.drop(columns=[target, datetime_col])

    # Search only on the day-of-month TRAIN subset (days 1-15), sorted by time,
    # so the holdout (days 16-19) is never seen during tuning.
    train_idx, _ = day_of_month_holdout_split(dt)
    order = np.argsort(dt.iloc[train_idx].values)
    search_pos = np.asarray(train_idx)[order]
    X_search = X.iloc[search_pos].reset_index(drop=True)
    y_search = y[search_pos]

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scorer = rmsle_scorer()

    results = {"search_rows_days_1_15": int(len(search_pos)), "models": {}}
    for name in MODELS:
        search = RandomizedSearchCV(
            estimator=get_model(name, cfg, {}),
            param_distributions=PARAM_DISTRIBUTIONS[name],
            n_iter=N_ITER[name],
            scoring=scorer,
            cv=splitter,
            random_state=seed,
            refit=False,
            n_jobs=-1,
        )
        search.fit(X_search, y_search)
        best_params = _strip_prefix(search.best_params_)

        baseline_eval = _eval(name, model_params.get(name, {}), cfg, X, y, dt)
        tuned_eval = _eval(name, best_params, cfg, X, y, dt)
        improvement = round(
            baseline_eval["holdout"]["rmsle"] - tuned_eval["holdout"]["rmsle"], 4
        )
        results["models"][name] = {
            "baseline_params": model_params.get(name, {}),
            "best_params": best_params,
            "baseline": baseline_eval,
            "tuned": tuned_eval,
            "holdout_rmsle_improvement": improvement,
            "search_best_cv_rmsle": round(-float(search.best_score_), 4),
        }
        print(
            f"{name:18s} holdout RMSLE baseline={baseline_eval['holdout']['rmsle']:.4f} "
            f"tuned={tuned_eval['holdout']['rmsle']:.4f}  improvement={improvement:+.4f}"
        )

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune xgboost/gradient_boosting.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing tuning sweep JSON.",
    )
    args = parser.parse_args()
    main(force=args.force)
