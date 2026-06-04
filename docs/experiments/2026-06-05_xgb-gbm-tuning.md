# XGBoost / Gradient Boosting hyperparameter tuning

- **Date:** 2026-06-05
- **Topic:** Can a RandomizedSearchCV over TimeSeriesSplit improve the two boosting models beyond their hand-chosen defaults, without overfitting the day-of-month holdout?
- **Motivation:** Roadmap Phase 10 (`docs/audit/2026-06-04_improvement-roadmap.md`) and `reports/RESULTS.md` "Limitations" both note the hyperparameters are sensible defaults, not tuned. This experiment tunes `xgboost` and `gradient_boosting` (the holdout and CV leaders) and decides, against a strict guardrail, whether to ship any tuned set.
  - **Baseline reference:** commit `2ec3a00` (Phase 9 changes pending commit but do not alter `reports/metrics.json`), production hyperparameters in `config/models.yaml`. Current day-of-month holdout / CV-mean RMSLE: xgboost 0.3064 / 0.4633, gradient_boosting 0.3120 / 0.4528. Day-of-month holdout RMSE/MAE/R2 for the guardrail: xgboost 47.54 / 28.00 / 0.933, gradient_boosting 51.7 / 31.0 / 0.920.
- **Hypothesis (measurable):** a RandomizedSearchCV (scored on count-scale RMSLE) lowers `xgboost` day-of-month holdout RMSLE by **>= 0.01** with **no regression** in holdout RMSE/MAE/R2. Expectation is that gains are marginal - the trio is already near-optimal and the three trees are a near-tie - so the likely outcome is DROPPED.
- **Scoring (non-negotiable):** the search scorer is **RMSLE on the original `count` scale** - `make_scorer(evaluate.rmsle, greater_is_better=False)`, applied to the model's predictions (already inverted to count scale by `TransformedTargetRegressor`). The search must NOT optimize a log-space MSE or the default R2; that is a different objective than the reported metric.
- **Holdout isolation (non-negotiable):** the search runs **only on the day-of-month train subset (days 1-15)**, with TimeSeriesSplit CV over those rows. The day-of-month holdout rows (days 16-19) are **never** seen by the search - they are used once, at the end, as the final confirmation check (fit best params on days 1-15, score days 16-19), so the holdout cannot be overfit by the search. This is the rigorous reading of "holdout excluded from the search".
- **Promotion rule (agreed):** promote a model's tuned params to `config/models.yaml` only if its day-of-month holdout RMSLE improves by **>= 0.01** with **no regression** in holdout RMSE/MAE/R2. Promotion is per-model and independent. If neither clears the bar, the item is DROPPED with the table recorded and production is unchanged. Any promoted value carries an inline `# experiment: 2026-06-05_xgb-gbm-tuning.md` comment.
- **Preconditions:** processed data present (`data/processed/train.parquet`); `reports/metrics.json` current; `pytest` green; production `config/models.yaml`, features, and models unchanged until (and unless) promotion.

## 1) Add a reusable count-scale RMSLE scorer to evaluate.py

- **Goal:** keep the scoring logic in `src/` (single source of truth), so the tuning script stays a thin orchestrator.
- **Files:**
  - `src/bike_sharing/evaluate.py`: add `rmsle_scorer()` returning `make_scorer(rmsle, greater_is_better=False)`.
  - `tests/test_evaluate.py`: add a contract that the scorer is sign-correct on the count scale (a closer prediction yields a higher, i.e. less negative, score) and is built on `rmsle`.
- **Steps:**
  - One bullet: the scorer wraps the existing `rmsle` (count scale, clips negatives); no new metric logic.
- **Test / verification:** new test green; full `pytest` green.
- **Expected outcome:** a reusable, tested scorer.
- **DONE (commit `64e5d84`):** Added `rmsle_scorer()` to `src/bike_sharing/evaluate.py` (`make_scorer(rmsle, greater_is_better=False)`) and a sign-correctness contract to `tests/test_evaluate.py` (a closer fit scores higher; the score equals the negated count-scale RMSLE, not a log-space loss). `pytest` green (83 passed).

## 2) Add scripts/tune_model.py and run the search

- **Goal:** RandomizedSearchCV for `xgboost` and `gradient_boosting`, then a clean holdout check, recorded as a sweep artifact.
- **Files:** `scripts/tune_model.py` (thin orchestrator calling `src/`), writing `docs/experiments/2026-06-05_xgb-gbm-tuning.json` and printing a baseline-vs-tuned table.
- **Steps:**
  - Load `data/processed/train.parquet`; restrict the search to the day-of-month train subset (days 1-15), sorted by datetime.
  - For each model, build the estimator from `get_model(name, cfg, {})` and run `RandomizedSearchCV` with `regressor__`-prefixed param distributions (the estimator is a `TransformedTargetRegressor`), `cv=TimeSeriesSplit(n_splits=cfg.cv.n_splits, gap=cfg.cv.gap)`, `scoring=rmsle_scorer()`, `random_state=cfg.seed`, `refit=False`.
  - Param distributions (described, not pasted): XGBoost over n_estimators/max_depth/learning_rate/subsample/colsample_bytree/min_child_weight/reg_lambda (n_iter ~40); GradientBoosting over n_estimators/learning_rate/max_depth/subsample/min_samples_leaf/max_features (n_iter ~30).
  - Take `best_params_`, strip the `regressor__` prefix, rebuild via `get_model(name, cfg, best_params)`, and run the final check: `evaluate_holdout` (fit days 1-15, score 16-19) plus `fit_and_cv` on the full data for a CV view comparable to `metrics.json`.
  - Record baseline vs tuned holdout RMSLE/RMSE/MAE/R2, the CV-mean RMSLE, and the chosen params; print the per-model improvement.
- **Test / verification:** sweep JSON produced; results table pasted below; the script touches no production config or `metrics.json`.
- **Expected outcome:** decide per the promotion rule.
- **DONE (commit `64e5d84`):** Added `scripts/tune_model.py` and ran it. The search ran on the day-of-month train subset only (8600 rows, days 1-15) with TimeSeriesSplit; the baseline column reproduced `reports/metrics.json` holdout exactly (xgb 0.3064, gbm 0.3120) - harness valid.
  - Metric / result (day-of-month holdout, baseline vs tuned; CV-mean RMSLE alongside):

    | model | holdout RMSLE | holdout RMSE | holdout MAE | holdout R2 | CV RMSLE |
    |---|---|---|---|---|---|
    | xgboost baseline | 0.3064 | 47.54 | 28.00 | 0.933 | 0.4633 |
    | xgboost tuned | 0.3056 | 48.54 | 28.62 | 0.930 | 0.4094 |
    | gradient_boosting baseline | 0.3120 | 51.66 | 31.05 | 0.920 | 0.4528 |
    | gradient_boosting tuned | 0.3143 | 50.48 | 29.78 | 0.924 | 0.4143 |

  - Key observation: tuning clearly improved the **chronological CV** RMSLE for both models (xgb 0.4633 -> 0.4094, gbm 0.4528 -> 0.4143), but the gain did **not** transfer to the day-of-month holdout: XGBoost is flat on holdout RMSLE (+0.0008) and slightly worse on RMSE/MAE/R2, and gradient_boosting's holdout RMSLE regresses (-0.0023, though its RMSE/MAE/R2 improve). Because the holdout was excluded from the search, this is honest evidence that CV-optimal params overfit the chronological folds relative to the day-of-month generalization the holdout (and the real Kaggle test) represent.
  - Best params (recorded in the JSON): xgb {n_estimators 600, max_depth 3, learning_rate 0.1, subsample 0.8, colsample_bytree 0.9, min_child_weight 5, reg_lambda 2.0}; gbm {n_estimators 200, max_depth 4, learning_rate 0.1, subsample 0.8, min_samples_leaf 5, max_features sqrt}.
  - Result artifact: `docs/experiments/2026-06-05_xgb-gbm-tuning.json`
  - Decision: neither model clears the promotion guardrail (>= 0.01 holdout RMSLE gain with no RMSE/MAE/R2 regression). Fed into item 3.

## 3) (Conditional) Promote tuned params to production

- **Goal:** ship only the model(s) that clear the promotion rule.
- **Files (only if item 2 passes for a model):** `config/models.yaml` (replace that model's block, with an inline `# experiment:` back-reference); retrain that model via `scripts/train_model.py --model NAME` -> `reports/metrics.json` + `models/<name>.joblib`; rebuild + execute `notebooks/04_tree_models.ipynb` and `notebooks/05_results_and_interpretation.ipynb`; update `reports/RESULTS.md` model-comparison numbers and the "Limitations" tuning note.
- **Test / verification:** `pytest` green; `metrics.json` diff shows the recorded gain and no regression beyond the guardrail.
- **Expected outcome:** production improves per the rule; otherwise this item is DROPPED and production is unchanged.
- **DROPPED (2026-06-05):** No promotion. XGBoost's holdout RMSLE gain (+0.0008) is below the 0.01 bar and it regresses on RMSE/MAE/R2; gradient_boosting's holdout RMSLE regresses outright. The hand-chosen defaults in `config/models.yaml` are already well-calibrated for the day-of-month holdout (and the real Kaggle test it mirrors), and the tuned params only helped the chronological CV, which does not generalize here. Production is unchanged - `config/models.yaml`, `reports/metrics.json`, the saved estimators, and the notebook metric tables all stay as they were. The only report change is a one-line accuracy update to the "Limitations" note in `reports/RESULTS.md` and notebook 05, so the report no longer claims the models are untuned; it now records that tuning was run and the defaults held.
