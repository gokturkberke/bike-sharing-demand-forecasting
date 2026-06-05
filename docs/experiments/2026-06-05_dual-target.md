# Dual-target: separate casual + registered models (optional stretch)

- **Date:** 2026-06-05
- **Topic:** Does predicting `casual` and `registered` with separate models and summing them beat the direct-`count` model?
- **Motivation:** The proposal and CLAUDE.md scope a stretch strategy: model `casual` and `registered` separately and sum, because the two user types follow different regimes (casual = leisure, weather-sensitive; registered = commute, weather-resistant), so a single `count` model may blur two behaviors. This experiment quantifies it against the deployed direct-count models, leakage-safely.
  - **Baseline reference:** commit `87f844e`, direct-count day-of-month holdout / CV RMSLE: xgboost 0.3064 / 0.4633, gradient_boosting 0.3120 / 0.4528. Holdout RMSE/MAE/R2 for xgboost: 47.54 / 28.00 / 0.933.
- **Hypothesis (measurable):** the dual-target sum lowers the deployed XGBoost holdout RMSLE by **>= 0.01** with **no regression** in holdout RMSE/MAE/R2. Outcome is genuinely uncertain - separating the regimes can help, but two models also means two error sources that may not cancel.
- **Scope:** dual-target for the boosting models (xgboost, gradient_boosting). The XGB+GBM blend is a separate idea and is out of scope here; Ridge dual-target is not pursued (Ridge is not competitive).
- **Leakage discipline (non-negotiable):** the feature matrix `X` excludes `count`, `casual`, and `registered` (and `datetime`); neither sub-target is ever a feature for the other or for count (CLAUDE.md stretch rule). Each sub-model trains on `X` and its own `log1p` target only; predictions are inverted with the project's non-negative `from_log1p` and then summed on the original count scale. Because the processed parquet drops `casual`/`registered`, the experiment reads **raw** train data (`load_raw_train` + `build_features`), exactly like the other experiment orchestrators. Validation uses the same leakage-safe TimeSeriesSplit CV and day-of-month holdout as the direct-count models. The CV is **fold-safe**: for every TimeSeriesSplit fold, separate casual and registered models are cloned and fit on that fold's train rows only, then used to predict that fold's validation rows; their count-scale predictions are summed and scored against the true count. No validation (or holdout) row ever influences a fitted model.
- **Decision rule (pre-registered):** adopt dual-target as the headline approach only if it improves the deployed XGBoost holdout RMSLE by **>= 0.01** with no RMSE/MAE/R2 regression (CV consistency considered as a secondary check). If it clears the bar, the promotion (and the summed `datetime,count` artifact wiring) is **referred to the user**, not auto-applied. Otherwise DROPPED with the comparison recorded. Either way the finding is noted in `reports/RESULTS.md`, since it answers the proposal's optional-stretch question.
- **Preconditions:** raw data present (`data/raw/train.csv`); `reports/metrics.json` current at `87f844e`; `pytest` green; production direct-count path, config, models, and `metrics.json` unchanged until (and unless) an explicitly approved promotion.

## 1) Add a leakage-safe dual-target module

- **Goal:** the dual-target split, summed prediction, and evaluation as single-responsibility, testable `src` logic (not script logic, per the module-boundary rule).
- **Files:**
  - `src/bike_sharing/dual_target.py` (new): `dual_target_split(df, cfg)` returns `(X, casual, registered, count, datetime)` with `X` excluding `count`/`casual`/`registered`/`datetime`; `fit_and_predict_dual_target(name, cfg, params, X_train, casual_train, registered_train, X_eval)` (named for what it does - it both fits and predicts) fits a casual and a registered model via `get_model` (each on its `log1p` target), inverts each with the model's non-negative inverse, and returns their summed count-scale prediction; `evaluate_dual_target(name, cfg, params, df)` returns `{cv, holdout}` using the same `TimeSeriesSplit(n_splits, gap)` (rows sorted by datetime) and `day_of_month_holdout_split` + `evaluate.report` the direct-count path uses, with the per-fold fit-on-train-only behavior described above. If a pure count-scale summing helper is later useful, the name `summed_prediction` is reserved for it.
  - `tests/test_dual_target.py` (new): `dual_target_split` excludes `count`/`casual`/`registered` from `X`; `fit_and_predict_dual_target` returns a non-negative prediction equal to casual_pred + registered_pred; the evaluation returns the four-metric set on both views.
- **Test / verification:** new tests green; full `pytest` green.
- **Expected outcome:** a tested, leakage-safe dual-target evaluator that reuses the project's split + metric primitives.
- **DONE / DROPPED:**

## 2) Run the dual-target comparison

- **Goal:** compare dual-target vs direct-count for xgboost and gradient_boosting on both views.
- **Files:** `scripts/train_dual_target.py` (thin orchestrator), writing `docs/experiments/2026-06-05_dual-target.json` and printing a direct-count-vs-dual table.
- **Steps:**
  - Load raw train, `build_features`; for each model run `evaluate_dual_target`; pull the direct-count baseline from `reports/metrics.json`; record dual-target CV-mean RMSLE and holdout RMSLE/RMSE/MAE/R2 plus the holdout-RMSLE delta vs direct-count.
  - The script touches no production config, model artifact, or `metrics.json`.
- **Test / verification:** JSON produced; results table pasted into item 3.
- **Expected outcome:** per the decision rule.
- **DONE / DROPPED:**

## 3) Decision and report

- **Goal:** record the outcome; keep production unchanged unless the bar is cleared and the user approves.
- **Files (reporting, always):** `reports/RESULTS.md` ("Sequential-data objective" / a stretch note) - record the dual-target result and decision, answering the proposal's optional-stretch question. Rebuild + execute a notebook only if a figure/number there changes.
- **Files (only if it clears and the user approves):** summed-artifact wiring in `scripts/generate_submission.py` / `predict.py` - designed in full only on approval (avoid speculative production changes).
- **Test / verification:** `pytest` green; if wired, the artifact stays non-negative `datetime,count` with 6493 rows; `git diff` on production paths is empty otherwise.
- **Expected outcome:** if dual-target clears the bar it is referred to the user for promotion; otherwise DROPPED with the comparison and a RESULTS.md note. Production unchanged either way absent explicit approval.
- **DONE / DROPPED:**
