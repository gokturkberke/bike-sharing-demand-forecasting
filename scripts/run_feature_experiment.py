"""Leakage-safe feature sweep: baseline vs candidate features.

Experiment orchestrator for
docs/experiments/2026-06-01_leakage-safe-feature-sweep.md. Calls into
src/bike_sharing only and writes a metrics JSON next to the plan file. It
does NOT modify production features, config, or reports/metrics.json - it
only measures whether the candidate features would help, to inform the
promotion decision recorded in the plan file.

Run from project root:
    .venv/bin/python scripts/run_feature_experiment.py
"""

import json
from pathlib import Path

from bike_sharing.config import load_config, load_models_config
from bike_sharing.data import load_raw_train
from bike_sharing.features import (
    CANDIDATE_NUMERIC_COLUMNS,
    build_candidate_features,
    build_features,
)
from bike_sharing.models import build_experimental_ridge, get_model
from bike_sharing.preprocessing import drop_leakage_columns
from bike_sharing.train import evaluate_holdout, fit_and_cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "2026-06-01_leakage-safe-feature-sweep.json"
)
MODELS = ("ridge", "random_forest", "gradient_boosting", "xgboost")

# Candidate feature groups, evaluated independently so the minimal subset
# that drives any gain is identifiable (not just the all-in set). The
# interaction_harmonic group (hour_sin2/hour_cos2/hour_sin_workday/
# hour_cos_workday) was promoted into build_features and is now part of the
# baseline; the groups below are the candidates it did not promote.
FEATURE_GROUPS = {
    "peaks": ["is_morning_peak", "is_evening_peak", "is_rush_hour"],
    "environmental": ["feels_like_gap", "temp_humidity_interaction", "bad_weather"],
    "year_trend": ["is_2012"],
    "all": None,  # filled with the full candidate set below
}


def _summarize(cv: dict, ho: dict) -> dict:
    return {
        "cv_rmsle": round(cv["mean"]["rmsle"], 4),
        "holdout": {k: round(v, 4) for k, v in ho["metrics"].items()},
    }


def _model_for(name, cfg, params, extra_cols):
    """Production estimator, except Ridge, which needs the candidate columns
    routed into its ColumnTransformer to see them at all."""
    if name != "ridge":
        return get_model(name, cfg, params)
    extra_numeric = tuple(c for c in extra_cols if c in CANDIDATE_NUMERIC_COLUMNS)
    extra_pass = tuple(c for c in extra_cols if c not in CANDIDATE_NUMERIC_COLUMNS)
    return build_experimental_ridge(cfg, params, extra_numeric, extra_pass)


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_params = load_models_config(PROJECT_ROOT / "config" / "models.yaml")
    datetime_col = cfg["datetime_col"]
    target = cfg["target"]

    cand = drop_leakage_columns(build_candidate_features(load_raw_train(cfg), cfg), cfg)
    base_cols = [
        c
        for c in drop_leakage_columns(build_features(load_raw_train(cfg), cfg), cfg).columns
        if c not in (target, datetime_col)
    ]
    y = cand[target].to_numpy(float)
    dt = cand[datetime_col]
    candidate_cols = [c for c in cand.columns if c not in base_cols + [target, datetime_col]]
    FEATURE_GROUPS["all"] = candidate_cols

    results = {"baseline": {}, "groups": {}}
    for name in MODELS:
        params = model_params.get(name, {})
        model = _model_for(name, cfg, params, extra_cols=[])
        X_base = cand[base_cols]
        results["baseline"][name] = _summarize(
            fit_and_cv(model, X_base, y, dt, cfg),
            evaluate_holdout(model, X_base, y, dt, cfg),
        )
        print(f"baseline {name:18s} holdout RMSLE={results['baseline'][name]['holdout']['rmsle']:.4f}")

    for group, cols in FEATURE_GROUPS.items():
        results["groups"][group] = {}
        X_group = cand[base_cols + cols]
        for name in MODELS:
            params = model_params.get(name, {})
            model = _model_for(name, cfg, params, extra_cols=cols)
            summary = _summarize(
                fit_and_cv(model, X_group, y, dt, cfg),
                evaluate_holdout(model, X_group, y, dt, cfg),
            )
            base_rmsle = results["baseline"][name]["holdout"]["rmsle"]
            improvement = round(base_rmsle - summary["holdout"]["rmsle"], 4)
            summary["holdout_rmsle_improvement"] = improvement
            results["groups"][group][name] = summary
            print(
                f"{group:22s} {name:18s} holdout RMSLE={summary['holdout']['rmsle']:.4f}"
                f"  improvement={improvement:+.4f}"
            )

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
