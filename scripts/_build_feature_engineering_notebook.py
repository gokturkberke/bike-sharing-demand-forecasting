"""One-shot helper to (re)build notebooks/02_feature_engineering.ipynb.

Run from project root:
    .venv/bin/python scripts/_build_feature_engineering_notebook.py

Build tool, not a runtime pipeline step.
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_ROOT / "notebooks" / "02_feature_engineering.ipynb"


SETUP_CODE = """\
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bike_sharing.config import load_config
from bike_sharing.data import load_raw_train
from bike_sharing.features import (
    ADDED_FEATURE_COLUMNS,
    CYCLIC_FEATURE_COLUMNS,
    build_features,
)

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "config" / "config.yaml").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not locate config/config.yaml above cwd.")
    PROJECT_ROOT = PROJECT_ROOT.parent

CFG = load_config(PROJECT_ROOT / "config" / "config.yaml")
FIG_DIR = Path(CFG["paths"]["reports_dir"]) / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="notebook", style="whitegrid")
print("added feature columns:", list(ADDED_FEATURE_COLUMNS))
"""

BUILD_CODE = """\
raw = load_raw_train(CFG)
df = build_features(raw, CFG)
print("raw shape:", raw.shape, "  featured shape:", df.shape)
print("new columns added:")
for col in ADDED_FEATURE_COLUMNS:
    print(f"  {col:12s}  dtype={df[col].dtype}  range=[{df[col].min()}, {df[col].max()}]")
df.head()
"""

CYCLIC_CODE = """\
# Cyclic encoding sanity: each hour maps to a unique point on the unit
# circle, and hour 23 is adjacent to hour 0 in (sin, cos) space.
hour_summary = (
    df[["hour", "hour_sin", "hour_cos"]].drop_duplicates().sort_values("hour")
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(hour_summary["hour"], hour_summary["hour_sin"], "o-", label="sin")
axes[0].plot(hour_summary["hour"], hour_summary["hour_cos"], "o-", label="cos")
axes[0].set_title("Cyclic hour encoding components")
axes[0].set_xlabel("hour of day")
axes[0].set_ylabel("encoding value")
axes[0].set_xticks(range(0, 24, 2))
axes[0].legend()

axes[1].plot(hour_summary["hour_sin"], hour_summary["hour_cos"], "o-")
for _, row in hour_summary.iterrows():
    axes[1].annotate(int(row["hour"]),
                     (row["hour_sin"], row["hour_cos"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=8)
axes[1].set_title("Hours on the unit circle")
axes[1].set_xlabel("hour_sin")
axes[1].set_ylabel("hour_cos")
axes[1].set_aspect("equal")
fig.tight_layout()
fig.savefig(FIG_DIR / "06_cyclic_hour_encoding.png", dpi=120, bbox_inches="tight")
plt.show()
"""

HOUR_INTERACTION_CODE = """\
# Mean count at every hour, split by workingday. The Phase 2 EDA showed
# this interaction was the strongest temporal signal. First-harmonic cyclic
# terms preserve hour wrap-around but cannot express the two distinct
# workingday curves; the workingday-gated cyclic terms added later
# (feature experiment) encode exactly this for the linear model.
mean_by_hour_workday = (
    df.groupby(["hour", "workingday"], as_index=False)["count"].mean()
)
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.lineplot(
    data=mean_by_hour_workday, x="hour", y="count",
    hue="workingday", marker="o", ax=ax,
)
ax.set_title("Mean count by hour and working-day status (post feature engineering)")
ax.set_xlabel("hour of day")
ax.set_ylabel("mean count")
ax.set_xticks(range(0, 24))
ax.legend(title="workingday")
fig.tight_layout()
fig.savefig(FIG_DIR / "07_hour_workingday_interaction.png", dpi=120, bbox_inches="tight")
plt.show()
"""

CORR_CODE = """\
# Correlation of new features with the target on the original scale.
# Cyclic encodings should show non-trivial correlations; year captures
# yearly growth.
feature_cols = list(ADDED_FEATURE_COLUMNS) + ["count"]
corr = df[feature_cols].corr()["count"].drop("count").sort_values()

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(x=corr.values, y=corr.index, hue=corr.index, legend=False, palette="vlag", ax=ax)
ax.set_title("Pearson correlation of new features with count")
ax.set_xlabel("correlation")
ax.set_ylabel("")
ax.axvline(0, color="black", linewidth=0.8)
fig.tight_layout()
fig.savefig(FIG_DIR / "08_new_feature_target_correlation.png", dpi=120, bbox_inches="tight")
plt.show()
"""


CELLS = [
    new_markdown_cell(
        "# Phase 3 — Feature Engineering\n"
        "\n"
        "Builds the time-derived and cyclic features defined in "
        "`src/bike_sharing/features.py` and verifies them against the "
        "training data. Three checks: cyclic encoding sanity, the "
        "hour × workingday interaction strength from Phase 2 EDA, and a "
        "first-pass correlation of each new feature with `count`. The "
        "production parquet is produced by `scripts/prepare_data.py`, "
        "not by this notebook."
    ),
    new_markdown_cell("## Setup"),
    new_code_cell(SETUP_CODE),
    new_markdown_cell(
        "## 1. Apply `build_features` to the training frame\n"
        "\n"
        "Confirms the engineered columns are added with sensible dtypes and "
        "ranges: the calendar/time features, the cyclic encodings (including "
        "the second-harmonic `hour_sin2`/`hour_cos2`), and the "
        "workingday-gated cyclic terms `hour_sin_workday`/`hour_cos_workday`. "
        "`datetime` is preserved so the same pipeline can later be applied to "
        "the test set and label its predictions."
    ),
    new_code_cell(BUILD_CODE),
    new_markdown_cell(
        "## 2. Cyclic encoding sanity\n"
        "\n"
        "Each of the 24 hours maps to a unique point on the unit circle. "
        "The right-hand plot visualizes that mapping: hour 0 sits at "
        "(0, 1), hour 6 at (1, 0), and hour 23 is geometrically adjacent "
        "to hour 0 — the property the encoding exists to provide."
    ),
    new_code_cell(CYCLIC_CODE),
    new_markdown_cell(
        "## 3. Hour × workingday interaction\n"
        "\n"
        "Re-plots the strongest temporal signal from Phase 2 on the "
        "engineered frame to confirm nothing was distorted. Working days "
        "still show a sharp double rush-hour pattern; non-working days a "
        "smoother afternoon peak."
    ),
    new_code_cell(HOUR_INTERACTION_CODE),
    new_markdown_cell(
        "## 4. New-feature correlations with target\n"
        "\n"
        "First-pass diagnostic — Pearson correlation of each engineered "
        "feature with `count`. Sign and magnitude are sanity checks, not "
        "modeling decisions: feature value is judged by Phase 4 CV, not "
        "by raw correlations."
    ),
    new_code_cell(CORR_CODE),
    new_markdown_cell(
        "## Findings (feed into Phase 4 modeling)\n"
        "\n"
        "- The cyclic hour encoding represents cyclicity: hour 23 is "
        "adjacent to hour 0 on the unit circle (verified by "
        "`test_cyclic_encoding_continuity`). First-harmonic sin/cos "
        "alone can express at most one peak per 24h — it does **not** "
        "give a linear model the bimodal commuter pattern for free.\n"
        "- The hour × workingday interaction shows two distinct hourly "
        "shapes: a sharp morning + evening double-peak on working days "
        "and a smoother single afternoon peak on non-working days. Tree "
        "models can pick this up from raw `hour` + `workingday`. To give "
        "the linear model a fair shot at it, the feature set now includes "
        "a second hour harmonic (`hour_sin2`, `hour_cos2`) and "
        "workingday-gated cyclic terms (`hour_sin_workday`, "
        "`hour_cos_workday`).\n"
        "- Those additions came from the gated experiment in "
        "`docs/experiments/2026-06-01_leakage-safe-feature-sweep.md`, which "
        "promoted this interaction/second-harmonic group (it cut Ridge "
        "holdout RMSLE 0.91 -> 0.72 with no tree regression) and dropped "
        "the candidates that did not clear the bar (peaks, environmental "
        "products, year flag).\n"
        "- Among the features, `hour`, `year`, and the cyclic columns "
        "show the strongest single-feature correlations with `count`. "
        "`is_weekend` is weakly correlated because `workingday` already "
        "captures most weekend behavior.\n"
        "- `day` (day-of-month) is intentionally excluded from the "
        "feature set: the dataset's split puts days 1-19 in train and "
        "20-31 in test, so the feature would be out-of-distribution at "
        "test time. The schema-contract test "
        "`test_train_and_test_predictor_schemas_match` prevents this "
        "regression from reappearing."
    ),
]


def main() -> None:
    nb = new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (bike_sharing)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NB_PATH.open("w") as f:
        nbformat.write(nb, f)
    print(f"wrote {NB_PATH.relative_to(PROJECT_ROOT)} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
