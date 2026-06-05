"""Environmental recalibration + Humidex experiment (experiment-only).

Orchestrator for docs/experiments/2026-06-05_env-recalibration-and-humidex.md.
For ridge/random_forest/gradient_boosting/xgboost on both leakage-safe
validation views, evaluates four arms:
  - baseline: production features + production model
  - comfort:  production features + the Humidex comfort_index candidate
  - recalib:  production features, model wrapped with the fold-safe
              WindspeedZeroImputer + AtempRecalibrator transforms
  - all:      comfort feature + recalib transforms
Writes a JSON next to the plan file. Touches no production config, features,
model artifact, or reports/metrics.json - it only measures.

Run from project root, after prepare_data.py:
    .venv/bin/python scripts/run_env_experiment.py
"""

import argparse
import json
from pathlib import Path

from sklearn.pipeline import Pipeline

from bike_sharing.config import load_config, load_models_config
from bike_sharing.data import load_raw_train
from bike_sharing.experimental_transforms import AtempRecalibrator, WindspeedZeroImputer
from bike_sharing.features import build_candidate_features, build_features
from bike_sharing.models import build_experimental_ridge, get_model
from bike_sharing.preprocessing import drop_leakage_columns
from bike_sharing.train import evaluate_holdout, fit_and_cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "2026-06-05_env-recalibration-and-humidex.json"
)
MODELS = ("ridge", "random_forest", "gradient_boosting", "xgboost")
COMFORT = "comfort_index"
ARMS = ("baseline", "comfort", "recalib", "all")


def _recalib(model):
    """Prepend the fold-safe transforms so per-fold cloning fits them on the
    train fold only."""
    return Pipeline(
        [
            ("impute", WindspeedZeroImputer()),
            ("recalib", AtempRecalibrator()),
            ("model", model),
        ]
    )


def _build(name, cfg, params, arm):
    """Estimator for a (model, arm) pair. Ridge needs comfort_index routed into
    its ColumnTransformer via build_experimental_ridge; trees see it as a
    column in X."""
    use_comfort = arm in ("comfort", "all")
    if name == "ridge":
        extra = (COMFORT,) if use_comfort else ()
        model = build_experimental_ridge(cfg, params, extra_numeric=extra)
    else:
        model = get_model(name, cfg, params)
    if arm in ("recalib", "all"):
        model = _recalib(model)
    return model


def _summarize(cv, ho):
    return {
        "cv_rmsle": round(cv["mean"]["rmsle"], 4),
        "holdout": {k: round(v, 4) for k, v in ho["metrics"].items()},
    }


def main(force: bool = False) -> None:
    if OUT_PATH.exists() and not force:
        raise FileExistsError(
            f"{OUT_PATH.relative_to(PROJECT_ROOT)} already exists - pass --force "
            "to overwrite the recorded experiment."
        )

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_params = load_models_config(PROJECT_ROOT / "config" / "models.yaml")
    target = cfg["target"]
    datetime_col = cfg["datetime_col"]

    raw = load_raw_train(cfg)
    base = drop_leakage_columns(build_features(raw, cfg), cfg)
    cand = drop_leakage_columns(build_candidate_features(raw, cfg), cfg)
    base_cols = [c for c in base.columns if c not in (target, datetime_col)]

    y = base[target].to_numpy(float)
    dt = base[datetime_col]
    X_base = base[base_cols]
    X_comfort = cand[base_cols + [COMFORT]]

    results = {"arms": list(ARMS), "models": {}}
    for name in MODELS:
        params = model_params.get(name, {})
        results["models"][name] = {}
        for arm in ARMS:
            X = X_comfort if arm in ("comfort", "all") else X_base
            model = _build(name, cfg, params, arm)
            summary = _summarize(
                fit_and_cv(model, X, y, dt, cfg),
                evaluate_holdout(model, X, y, dt, cfg),
            )
            base_ho = (
                results["models"][name].get("baseline", {}).get("holdout", {}).get("rmsle")
            )
            if base_ho is not None:
                summary["holdout_rmsle_improvement"] = round(
                    base_ho - summary["holdout"]["rmsle"], 4
                )
            results["models"][name][arm] = summary
            imp = summary.get("holdout_rmsle_improvement")
            imp_s = f"  improvement={imp:+.4f}" if imp is not None else ""
            print(
                f"{name:18s} {arm:9s} holdout RMSLE={summary['holdout']['rmsle']:.4f}{imp_s}"
            )

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Environmental recalibration + Humidex experiment."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite the existing experiment JSON."
    )
    args = parser.parse_args()
    main(force=args.force)
