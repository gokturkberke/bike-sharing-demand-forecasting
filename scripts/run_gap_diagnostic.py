"""Validation gap diagnostic: TimeSeriesSplit CV at gap=0 vs gap=48.

Experiment orchestrator for
docs/experiments/2026-06-04_validation-gap-diagnostics.md. Calls into
src/bike_sharing only and writes a metrics JSON next to the plan file. It
does NOT modify production config, features, or reports/metrics.json - it
only measures whether a chronological gap between TimeSeriesSplit train and
validation folds moves the CV estimate, which is a leakage robustness check
on the chronological CV view. The gap=0 column should reproduce
reports/metrics.json CV exactly (a harness self-check).

Run from project root:
    .venv/bin/python scripts/run_gap_diagnostic.py
"""

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from bike_sharing.config import load_config, load_models_config
from bike_sharing.models import get_model
from bike_sharing.train import fit_and_cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "2026-06-04_validation-gap-diagnostics.json"
)
MODELS = ("ridge", "random_forest", "gradient_boosting", "xgboost")
GAPS = (0, 48)


def _cfg_with_gap(cfg: dict, gap: int) -> dict:
    out = copy.deepcopy(cfg)
    out["cv"] = {**out["cv"], "gap": gap}
    return out


def main(force: bool = False) -> None:
    if OUT_PATH.exists() and not force:
        raise FileExistsError(
            f"{OUT_PATH.relative_to(PROJECT_ROOT)} already exists - pass --force "
            "to overwrite the recorded diagnostic."
        )

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_params = load_models_config(PROJECT_ROOT / "config" / "models.yaml")
    target = cfg["target"]
    datetime_col = cfg["datetime_col"]

    train_path = Path(cfg["paths"]["processed_dir"]) / "train.parquet"
    df = pd.read_parquet(train_path)
    y = df[target].to_numpy(float)
    dt = df[datetime_col]
    X = df.drop(columns=[target, datetime_col])

    lo, hi = str(GAPS[0]), str(GAPS[1])
    results = {"gaps": list(GAPS), "models": {}}
    for name in MODELS:
        params = model_params.get(name, {})
        by_gap = {}
        for gap in GAPS:
            cfg_gap = _cfg_with_gap(cfg, gap)
            summary = fit_and_cv(get_model(name, cfg_gap, params), X, y, dt, cfg_gap)
            by_gap[str(gap)] = {k: round(v, 4) for k, v in summary["mean"].items()}
        delta = round(by_gap[hi]["rmsle"] - by_gap[lo]["rmsle"], 4)
        results["models"][name] = {
            "by_gap": by_gap,
            "cv_rmsle_delta_gap48_minus_gap0": delta,
        }
        print(
            f"{name:18s} CV RMSLE gap{lo}={by_gap[lo]['rmsle']:.4f} "
            f"gap{hi}={by_gap[hi]['rmsle']:.4f}  delta={delta:+.4f}"
        )

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimeSeriesSplit gap diagnostic.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing diagnostic JSON.",
    )
    args = parser.parse_args()
    main(force=args.force)
