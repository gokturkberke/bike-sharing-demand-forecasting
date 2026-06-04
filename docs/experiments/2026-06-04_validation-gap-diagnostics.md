# Validation gap diagnostics (TimeSeriesSplit gap)

- **Date:** 2026-06-04
- **Topic:** Does inserting a chronological gap between TimeSeriesSplit train and validation folds change the CV estimate?
- **Motivation:** Roadmap Phase 9 (`docs/audit/2026-06-04_improvement-roadmap.md`) and both research PDFs recommend a `gap` in `TimeSeriesSplit` (the PDFs suggest `gap=48`, i.e. a ~48-hour chronological buffer) to keep train and validation blocks from touching at the fold boundary. The current `fit_and_cv` (`src/bike_sharing/train.py:52`) uses `TimeSeriesSplit(n_splits=5)` with no gap. This experiment quantifies whether a gap moves the CV-mean RMSLE, which doubles as a leakage robustness check on the chronological CV view.
  - **Baseline reference:** commit `2ec3a00`, `reports/metrics.json` CV-mean RMSLE (gap=0): gradient_boosting 0.4528, xgboost 0.4633, random_forest 0.5147, ridge 0.8064. (Day-of-month holdout is unaffected by this change and stays the primary view; only the CV view is touched.)
- **Hypothesis (measurable):** with `gap=48` the CV-mean RMSLE of the tree/boosting models (xgboost, gradient_boosting, random_forest) changes by **< 0.01** versus gap=0, because the models carry **no autoregressive/lag features** - so train/validation adjacency cannot leak the target, and the only adjacency effect is mild shared weather autocorrelation. A change of **>= 0.02** on the trees would instead flag hidden adjacency leakage worth investigating.
- **Decision rule (agreed):** this is a *confirm-robustness* experiment, not a promotion. If gap=48 confirms a negligible change (< ~0.01 on the trees), keep `cv.gap: 0` as the production default - it uses all rows and reproduces the existing `metrics.json` exactly - and report the gap=48 result as a robustness check. If the gap materially worsens CV (>= 0.02 on the trees), investigate the cause before changing any default. Either way the outcome is recorded here as DONE.
- **Preconditions:** processed data present (`data/processed/train.parquet`); `reports/metrics.json` current at commit `2ec3a00`; `pytest` green. The day-of-month holdout view and all production features/models stay unchanged.

## 1) Add a leakage-safe `gap` option to fit_and_cv

- **Goal:** make the chronological gap configurable without changing default behavior (gap=0 must reproduce today's CV numbers).
- **Files:**
  - `config/config.yaml`: add `gap: 0` under `cv:` with an inline `# experiment:` back-reference to this file.
  - `src/bike_sharing/train.py`: `fit_and_cv` reads `int(cfg["cv"].get("gap", 0))` and passes it to `TimeSeriesSplit(n_splits=n_splits, gap=gap)`. No other behavior changes; `config.py` validation is untouched (only `n_splits` is required, so `gap` is optional and back-compatible).
  - `tests/test_train.py`: add (a) a monkeypatch test that `fit_and_cv` passes the configured `gap` to `TimeSeriesSplit` (e.g. `cfg["cv"]["gap"]=48` -> `TimeSeriesSplit` called with `gap=48`), and (b) a back-compat test that a config with no `gap` key calls it with `gap=0`.
- **Steps:**
  - Note that `gap` is measured in **samples**; with hourly, datetime-sorted rows it is approximately `gap` hours within a contiguous run (month boundaries make it approximate).
- **Test / verification:** new tests green; full `pytest` green; gap=0 path unchanged so existing CV contracts still pass.
- **Expected outcome:** `fit_and_cv` honors `cv.gap`; default 0 is behavior-identical.
- **DONE (commit `90f0445`):** Added `cv.gap` to `config/config.yaml` (default 0, with an `# experiment:` back-reference); `fit_and_cv` now reads `int(cfg['cv'].get('gap', 0))` and passes it to `TimeSeriesSplit(n_splits=..., gap=gap)`; added two contracts to `tests/test_train.py` (a monkeypatch spy proving `cv.gap=48` reaches the splitter, and a back-compat test that a missing key yields `gap=0`). `config.py` validation untouched. `pytest` green (82 passed); the gap=0 path is behavior-identical.

## 2) Run the gap=0 vs gap=48 ablation and record results

- **Goal:** quantify CV-mean RMSLE at gap=0 and gap=48 for the four real models (ridge, random_forest, gradient_boosting, xgboost).
- **Files:** `scripts/run_gap_diagnostic.py` (thin orchestrator calling `fit_and_cv`, mirroring `scripts/run_feature_experiment.py`), writing `docs/experiments/2026-06-04_validation-gap-diagnostics.json` and printing a comparison table.
- **Steps:**
  - Load `data/processed/train.parquet`; for each model run `fit_and_cv` with a config whose `cv.gap` is 0, then 48; record CV-mean RMSLE/RMSE/MAE/R2 and the gap delta.
  - The gap=0 column must reproduce `reports/metrics.json` CV (a harness self-check, exactly as the feature sweep's baseline reproduced production).
- **Test / verification:** sweep JSON produced; comparison table pasted below.
- **Expected outcome:** decide per the decision rule above.
- **DONE (commit `90f0445`):** Ran `scripts/run_gap_diagnostic.py`; the gap=0 column reproduced `reports/metrics.json` CV exactly for all four models (harness self-check passed).
  - Metric / result (CV-mean RMSLE):

    | model | gap=0 | gap=48 | delta (gap48 - gap0) |
    |---|---|---|---|
    | xgboost | 0.4633 | 0.4669 | +0.0036 |
    | random_forest | 0.5147 | 0.5228 | +0.0081 |
    | gradient_boosting | 0.4528 | 0.4808 | +0.0280 |
    | ridge | 0.8064 | 0.9542 | +0.1478 |

  - Investigation (the decision rule requires attributing any tree movement >= 0.02; gradient_boosting crossed it). Per-fold deltas localize the effect. Ridge's +0.1478 is almost entirely fold 1 (0.6642 -> 1.3643, +0.70): the smallest train window (1768 rows), a winter->summer level ramp, and the linear model's known extrapolation fragility. The boosting models spike on fold 3 (GBM +0.122, XGB +0.132); the 48 rows the gap drops there are `2011-12-18..2012-01-01`, which bridge the 2011->2012 year-over-year level shift (fold-3 val mean count ~185 vs late-2011 train). XGBoost's net stays ~0 only because its fold-1 error falls (-0.078) to offset the fold-3 rise; random_forest is mild on every fold.
  - Root cause: **not target leakage** - the models carry no lag features and cannot see validation targets. The gap removes training rows temporally adjacent to each validation block, which were aiding extrapolation across large seasonal/annual level shifts; the movement therefore concentrates at the small first fold (ridge) and the year-boundary fold (boosting), not uniformly.
  - Result artifact: `docs/experiments/2026-06-04_validation-gap-diagnostics.json`
  - Decision: hypothesis partially confirmed - the deployed XGBoost (+0.0036) and random_forest (+0.0081) move < 0.01, so the chronological CV is not inflated by adjacency leakage for the models that matter. The larger Ridge/GBM movements are explained boundary artifacts, fed into item 3.

## 3) Report the robustness check and set the default

- **Goal:** surface the result in the report and lock the production default.
- **Files:** `scripts/_build_baseline_and_linear_notebook.py` (a "Validation robustness: chronological gap" cell that reads the sweep JSON and plots `reports/figures/24_cv_gap_sensitivity.png`, guarded so the notebook still executes with a one-line note if the JSON is absent); rebuild + execute `notebooks/03_baseline_and_linear.ipynb`; one sentence in `reports/RESULTS.md` under "How models were evaluated".
- **Steps:**
  - Keep `cv.gap: 0` as the default unless item 2 argues otherwise; the figure/sentence frame gap=48 as a robustness confirmation, noting the no-lag-feature reasoning.
- **Test / verification:** notebook executes clean; figure 24 written; `pytest` green; `metrics.json` unchanged (no retrain needed since the default stays gap=0).
- **Expected outcome:** the report carries an honest validation-robustness result; the production CV default is documented and unchanged.
- **DONE (commit `90f0445`):** Added a "Validation robustness: chronological gap" cell to `scripts/_build_baseline_and_linear_notebook.py` that reads the diagnostic JSON and plots `reports/figures/24_cv_gap_sensitivity.png` (guarded - prints a note and skips if the JSON is absent); rebuilt and executed `notebooks/03_baseline_and_linear.ipynb`; added one sentence to `reports/RESULTS.md` under "How models were evaluated".
  - Decision: keep `cv.gap: 0` as the production default. It is the standard TimeSeriesSplit, uses all rows, and reproduces the recorded `metrics.json`; the deployed model's CV is robust to the gap; the larger movements are explained boundary artifacts, not leakage; and the day-of-month holdout - the primary view - is unaffected and does not straddle a year boundary. `reports/metrics.json` unchanged (no retrain needed). The gap=48 run is retained as a documented robustness diagnostic.
