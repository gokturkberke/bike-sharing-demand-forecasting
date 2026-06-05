"""Dual-target experiment (optional stretch): casual + registered, summed.

Orchestrator for docs/experiments/2026-06-05_dual-target.md. Compares the
dual-target approach (separate casual/registered models summed on the count
scale) against the direct-count baseline from reports/metrics.json, for
xgboost and gradient_boosting, on both leakage-safe validation views. Reads
RAW train (the processed parquet drops casual/registered). Writes a JSON next
to the plan file. Touches no production config, model artifact, or
reports/metrics.json - promotion is a separate, gated decision.

Run from project root:
    .venv/bin/python scripts/train_dual_target.py
"""

import argparse
import json
from pathlib import Path

from bike_sharing.config import load_config, load_models_config
from bike_sharing.data import load_raw_train
from bike_sharing.dual_target import evaluate_dual_target
from bike_sharing.features import build_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "docs" / "experiments" / "2026-06-05_dual-target.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
MODELS = ("xgboost", "gradient_boosting")


def main(force: bool = False) -> None:
    if OUT_PATH.exists() and not force:
        raise FileExistsError(
            f"{OUT_PATH.relative_to(PROJECT_ROOT)} already exists - pass --force "
            "to overwrite the recorded experiment."
        )

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_params = load_models_config(PROJECT_ROOT / "config" / "models.yaml")
    direct = json.loads(METRICS_PATH.read_text())

    df = build_features(load_raw_train(cfg), cfg)

    results = {"models": {}}
    for name in MODELS:
        dual = evaluate_dual_target(name, cfg, model_params.get(name, {}), df)
        d_ho = direct[name]["day_of_month_holdout"]["metrics"]
        d_cv = direct[name]["cv"]["mean"]["rmsle"]
        dual_ho = dual["holdout"]["metrics"]
        improvement = round(d_ho["rmsle"] - dual_ho["rmsle"], 4)
        results["models"][name] = {
            "direct_count": {
                "holdout": {k: round(v, 4) for k, v in d_ho.items()},
                "cv_rmsle": round(d_cv, 4),
            },
            "dual_target": {
                "holdout": {k: round(v, 4) for k, v in dual_ho.items()},
                "cv_rmsle": round(dual["cv"]["mean"]["rmsle"], 4),
            },
            "holdout_rmsle_improvement": improvement,
        }
        print(
            f"{name:18s} direct holdout RMSLE={d_ho['rmsle']:.4f}  "
            f"dual={dual_ho['rmsle']:.4f}  improvement={improvement:+.4f}"
        )

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-target casual+registered experiment."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite the existing experiment JSON."
    )
    args = parser.parse_args()
    main(force=args.force)
