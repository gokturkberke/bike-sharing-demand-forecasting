# Technical Audit and Improvement Roadmap - Bike Sharing Demand Forecasting

Authored 2026-06-04. Approved improvement roadmap that extends the Phase 7 (final) pipeline. Numbering continues at Phase 8 so it adds to, rather than overwrites, the delivered Phase 7.

## Context

This is a planning/audit deliverable, not an implementation record. The repository is a CMP4336 school project (Kaggle Bike Sharing Demand, hourly `count` forecasting) that CLAUDE.md records as Phase 7 (final), complete: leakage-safe pipeline, baselines + Ridge + RF + GBM + XGBoost, two leakage-safe validation views, four co-equal metrics, 22 figures, a written `reports/RESULTS.md`, and a passing test suite. Two Turkish research PDFs were added under `docs/`; this roadmap is a cautious, phase-by-phase plan to improve the project without breaking the existing clean pipeline, prioritizing the report / explainability over leaderboard score.

Both PDFs were read in full and the codebase was mapped before writing this. Confirmed scope decisions: Balanced ambition (rigor + explainability + gated additive experiments; dual-target/ensembling as clearly separated optional stretch), windspeed/atemp recalibration is experiment-only (never auto-promoted), and deep-learning / extra boosting libs (N-BEATS, TFT, LightGBM, CatBoost) are out of scope.

Numbering note: the new work is numbered Phase 8-14 so it extends rather than overwrites the delivered Phase 7.

---

## 1. Current Project Audit

### Pipeline summary
- Config (`config.py`, `config/config.yaml`, `config/models.yaml`): seed, target, datetime col, `drop_columns: [casual, registered]`, `cv.n_splits: 5`, paths; model hyperparameters live in `models.yaml`. Config is validated for required keys.
- Data (`data.py`): loads local `data/raw/train.csv` (10886x12) and `test.csv` (6493x9), parses `datetime`, fails loudly if files are missing. No auto-download.
- Preprocessing (`preprocessing.py`): `drop_leakage_columns` removes `casual`/`registered`; `to_log1p_target` / `from_log1p` (expm1 + clip-at-0) are the target-transform contract.
- Features (`features.py`): `build_features` adds 13 columns - calendar (`hour, dayofweek, month, year, is_weekend`), cyclic (`hour_sin/cos`, `month_sin/cos`, second harmonic `hour_sin2/cos2`), and the promoted workingday-gated interaction (`hour_sin_workday`, `hour_cos_workday`). Day-of-month is deliberately excluded. `build_candidate_features` + `CANDIDATE_*` tuples hold experiment-only features (`is_morning_peak/evening_peak/rush_hour`, `feels_like_gap`, `temp_humidity_interaction`, `bad_weather`, `is_2012`).
- Models (`models.py`): factory for `mean_baseline`, `hourly_mean_baseline`, `ridge`, `random_forest`, `gradient_boosting`, `xgboost` (lazy import). Every real model trains on `log1p(count)` via `TransformedTargetRegressor` and inverts with `from_log1p`. Ridge is a `ColumnTransformer` (scale numerics, drop raw ordinals, one-hot `season`/`weather`); trees use the full feature set.
- Train/validation (`train.py`): `fit_and_cv` (`TimeSeriesSplit(n_splits=5)`, rows sorted by datetime) and `evaluate_holdout` (`day_of_month_holdout_split`: train days 1-15, validate 16-19 - mirrors Kaggle's own 1-19 / 20+ structure).
- Evaluate (`evaluate.py`): `report` returns RMSLE, RMSE, MAE, R2 together; RMSLE clips negatives; R2 guards zero-variance targets.
- Predict (`predict.py`): `make_prediction_frame` / `write_prediction_artifact` produce the `datetime,count` artifact (6493 rows, original order, non-negative).
- Scripts: `prepare_data.py`, `train_model.py --model`, `generate_submission.py --model`, `run_feature_experiment.py` (the gated feature-sweep harness).
- Notebooks 01-05 + `reports/RESULTS.md` + `reports/metrics.json` + 22 figures. Experiment log `docs/experiments/2026-06-01_leakage-safe-feature-sweep.md` (DONE, fully documented).

### Current results (day-of-month holdout; CV in parentheses)

| Model | RMSLE | RMSE | MAE | R2 |
|---|---|---|---|---|
| xgboost | 0.306 (CV 0.463) | 47.5 | 28.0 | 0.933 |
| gradient_boosting | 0.312 (CV 0.453) | 51.7 | 31.0 | 0.920 |
| random_forest | 0.328 (CV 0.515) | 51.8 | 29.9 | 0.920 |
| ridge | 0.718 (CV 0.806) | 140.8 | 89.1 | 0.408 |
| hourly_mean_baseline | 0.755 | 125.9 | 86.1 | 0.527 |
| mean_baseline | 1.531 | 183.1 | 142.6 | -0.001 |

XGBoost leads holdout; GBM narrowly leads CV; the three trees are a near-tie. Documented weakness: error scales with demand and bias turns negative (about -35) in the top quintile - the model underpredicts peak hours (figure 21), concentrated in working-day commute peaks (figure 22).

### What is already strong (preserve it)
- Strict, three-layer leakage protection (config + preprocessing + tests).
- Two genuinely leakage-safe validation views, both reported.
- Principled `log1p` target with non-negative inversion.
- A disciplined gated-experiment harness with documented promotion guardrails - unusually mature, and the right place to land most PDF ideas.
- Honest, four-metric, no-single-winner reporting with a written interpretation against the proposal's three questions.

Verdict: the existing structure should be preserved. The roadmap is additive and reversible; no module needs rewriting.

---

## 2. Ideas from the PDFs

Both PDFs converge on the same recommendations. Critically, several are already implemented or already tested-and-rejected in this repo.

| Recommendation (PDF) | Already implemented? | Expected benefit | Risk | Difficulty | Now / Later / Skip |
|---|---|---|---|---|---|
| Cyclic sin/cos hour & month encoding | Yes (production), incl. 2nd harmonic | - | - | - | Done |
| Workingday x hour interaction | Yes (promoted to production), headline win | - | - | - | Done |
| Rush-hour indicators (`is_rush_hour`, peak dummies) | Built as candidates + tested in 2026-06-01 sweep; not promoted (redundant w/ cyclic+interaction) | Low (already captured) | Low | Low | Skip / note as tested |
| Day-type triple (weekday/weekend/holiday) | Yes - `workingday`+`holiday`+`is_weekend` present | - | Low | Low | Skip (redundant) |
| log1p target + clip-at-0 inversion | Yes (production) | - | - | - | Done |
| Avoid random K-Fold; time-aware split | Yes - TimeSeriesSplit + day-of-month holdout | - | - | - | Done |
| Avoid UCI public leak (test truths online) | Yes - local Kaggle data only, no downloader | Critical (already safe) | - | - | Done / affirm in report |
| Permutation importance / SHAP | No (only impurity-based) | High for the report (unbiased importance) | Low | Low (perm. is in sklearn, no new dep) | Now (Phase 8) |
| TimeSeriesSplit `gap` (e.g. 48h) | No (`gap=0`) | Medium (robustness check) | Low | Low | Now (Phase 9) |
| Peak-error / demand-quantile stratification | Partly (figures 21/22 exist) | Medium (deepen diagnostics) | Low | Low | Now (Phase 9) |
| Duan's smearing / retransformation-bias correction | No | Medium (targets peak underprediction bias) | Medium (RMSLE vs RMSE tradeoff) | Medium | Later, gated (Phase 11) |
| Sample weighting for high-demand | No | Medium (reduces peak bias) | Medium (can hurt low-demand) | Medium | Later, gated (Phase 11) |
| Hyperparameter tuning (XGB/GBM) | No (hand-chosen defaults) | Low-Medium (trio already near-optimal) | Medium (overfit small holdout) | Medium | Later, gated (Phase 10) |
| Comfort index / Humidex (exp formula) | Proxy only (`feels_like_gap`, `temp_humidity_interaction` tested + dropped) | Low (env. tier is secondary; proxies regressed trees) | Low | Low | Later, one gated sweep item (Phase 12) |
| Periodic SplineTransformer for cyclic time | No (2nd-harmonic Fourier instead) | Low (Ridge not competitive; trees don't need it) | Low | Medium | Later/Skip (Phase 12, optional) |
| Windspeed-zero imputation (sensor floor -> missing) | No | Low-Medium (cleaner splits) | Medium-High (mutates validated pipeline; imputation leakage if cross-split) | Medium | Later, experiment-only (Phase 12) |
| Atemp recalibration (regress on temp+humidity) | No | Low (temp/atemp r~0.98; little new signal) | Medium-High (same) | Medium | Later, experiment-only (Phase 12) |
| Dual-target casual + registered | No | Medium (different behavior regimes) | Medium (2x models, leakage discipline) | Medium-High | Optional stretch (Phase 13) |
| Ensemble / blend XGB + GBM | No | Low-Medium (near-tied models) | Low-Medium | Low-Medium | Optional stretch (Phase 13) |
| Shift parameter `log(y + c)`, c=100 | No (uses log1p, c=1) | Low (variance smoothing) | Medium (breaks RMSLE geometry/story) | Low | Skip / note |
| `is_event` indicators (festival/strike) | No | None (no such data in this dataset) | - | - | Skip (not applicable) |
| GLM Poisson/Gamma log-link (no bias correction) | No | Low (alt. to smearing) | Low | Medium | Skip / research note |
| Asymmetric loss for peaks | No | Medium | Medium-High (custom objective, harder to validate) | High | Later/optional (subset of Phase 11) |
| N-BEATS / TFT / LightGBM / CatBoost | No | Low-Medium | Out of scope (heavy deps, scope creep) | High | Skip - research note only |

Critical read: the PDFs are sound, but for this clean, leak-free dataset many ideas are either already done, already tested-and-rejected with documented reasons, or redundant. The genuinely new, report-valuable items are: permutation importance, `gap` diagnostics, deeper peak-error analysis, and a gated smearing/weighting experiment for the one documented weakness. Everything else is optional.

---

## 3. Protect the Current Project

Files/modules to keep stable (do not rename keys, columns, metric names, or artifact schema):
- `config.py` and the config keys; `data.py`; `preprocessing.py` target-transform contract (`to_log1p_target`/`from_log1p`).
- Public APIs of `models.py`, `train.py`, `evaluate.py`, `predict.py`.
- The production `build_features` column set and `ADDED_FEATURE_COLUMNS`. New work extends `build_candidate_features` / `FEATURE_GROUPS`, not the production set, until a sweep clears the guardrail.
- The `datetime,count` artifact schema (6493 rows).

Tests to run after every phase (must stay green):

```
.venv/bin/python -m pytest
```

Pay attention to: `test_features.py` (schema parity, day-of-month exclusion), `test_preprocessing.py` (leakage), `test_predict.py` (artifact schema + 6493 rows), `test_train.py` (split correctness), `test_evaluate.py`, `test_models.py`. Each behavioral change adds/updates at least one contract test (per CLAUDE.md section 4).

Artifacts/figures to regenerate (only what a phase actually touches):
- Features changed -> `scripts/prepare_data.py` -> `data/processed/*.parquet`.
- Model/metric changed -> `scripts/train_model.py --model NAME` -> `reports/metrics.json` + `models/*.joblib`.
- Then rebuild affected notebooks (02-05) and `reports/figures/*`, and update `reports/RESULTS.md`.
- Final artifact via `scripts/generate_submission.py --model xgboost`.

How to avoid changing too much at once:
- One phase = one or a few small commits. Additive, behind config flags, reversible.
- Per CLAUDE.md section 7, every modeling experiment gets a `docs/experiments/{date}_{name}.md` plan file before any code, with baseline reference, hypothesis, decision criterion, and a DONE/DROPPED marker.
- Never mutate production features/transform without a passing sweep that clears the guardrail (XGB improves >= 0.01 holdout RMSLE, OR a clear non-tree win with no tree regression > 0.005).
- The direct-count XGBoost pipeline stays the canonical deliverable; dual-target/ensembling live in clearly separated experiment scripts.

---

## 4. Recommended Roadmap

Each phase is small and commit-friendly. Per CLAUDE.md section 7, every change judged against a metric - modeling experiments and validation ablations (Phases 9-13) - gets a `docs/experiments/{date}_{name}.md` plan file before any code.

### Phase 8 - Documentation + audit cleanup + explainability
- Goal: capture this audit; upgrade the feature-importance story from impurity-based to model-agnostic permutation importance (PDF recommendation, zero new deps via `sklearn.inspection.permutation_importance`).
- SHAP is NOT a default and adds no dependency. Permutation importance is the deliverable and is sufficient. SHAP stays strictly optional - used only if it is already installed or specifically requested; it is never added to `requirements.txt` for this.
- Files: `reports/RESULTS.md`, `notebooks/05_results_and_interpretation.ipynb` (or a small new analysis section), optionally a single-responsibility `src/bike_sharing/explain.py` helper; `reports/figures/` (new permutation-importance figure); this audit committed under `docs/`.
- Tasks: compute permutation importance on the day-of-month holdout for XGBoost (scored with count-scale RMSLE); compare against the impurity ranking; document that impurity is biased toward high-cardinality/continuous features.
- Expected outcome: an unbiased importance figure + narrative; report explicitly notes the bias caveat now addressed.
- Tests: full suite green; if `explain.py` is added, one small test that it returns a ranking over the known feature names.
- Risks: very low (read-only analysis).
- Commit: `docs(report): add permutation importance and audit roadmap; note impurity-importance bias`

### Phase 9 - Validation diagnostics improvements
- Goal: add a `gap` to TimeSeriesSplit as a robustness diagnostic and deepen the peak/quantile error analysis.
- Plan file: `docs/experiments/{date}_validation-gap-diagnostics.md` first - the `gap=0` vs `gap=48` comparison is a metric-judged validation ablation, so it gets an experiment log (baseline reference, hypothesis, decision criterion, DONE/DROPPED) per CLAUDE.md section 7.
- Files: `config/config.yaml` (`cv.gap`, default 0 to preserve current behavior), `src/bike_sharing/train.py` (`fit_and_cv` reads `cfg["cv"].get("gap", 0)` and passes to `TimeSeriesSplit(gap=...)`), `tests/test_train.py`, notebook 03/05 + figures.
- Tasks: run CV at `gap=0` and `gap=48`, report sensitivity; extend the demand-quantile/working-day-hour error breakdown (figures 21-22) with explicit bias-per-quintile numbers.
- Expected outcome: a short "validation robustness" subsection; expectation that `gap=48` changes little because the models use no autoregressive/lag features (adjacency leakage is limited to mild environmental autocorrelation) - itself a useful, honest finding.
- Tests: `test_train.py` asserts `gap` is honored and folds shrink accordingly; full suite green.
- Risks: low. Keep `gap=0` as the reported default unless the sensitivity check argues otherwise; document either way.
- Commit: `feat(train): optional TimeSeriesSplit gap diagnostic; deepen peak-error analysis`

### Phase 10 - XGBoost / Gradient Boosting tuning (gated)
- Goal: replace hand-chosen hyperparameters with a tuned set only if it clears the guardrail.
- Plan file: `docs/experiments/{date}_xgb-gbm-tuning.md` first.
- Files: new `scripts/tune_model.py` (orchestrator using sklearn `RandomizedSearchCV` over `TimeSeriesSplit` - no new deps), then `config/models.yaml` (only if promoted, with an `# experiment:` provenance comment), `reports/metrics.json`, notebook 04.
- Scoring (explicit): the search scorer must be RMSLE computed on the original `count` scale - a custom `make_scorer(..., greater_is_better=False)` that takes the model's predictions (already inverted to count scale by `TransformedTargetRegressor`) and applies `evaluate.rmsle`. Do not score in log-space or let the search optimize a log-space MSE; that is a different objective and would not match the reported metric.
- Tasks: search only over `TimeSeriesSplit` CV folds. The day-of-month holdout is excluded from the search entirely and is used solely as the final confirmation check on the single chosen configuration, so the holdout cannot be overfit by the search. Report all four metrics.
- Expected outcome: likely small gains (the trio is already near-optimal). Promote only if XGB holdout RMSLE improves >= 0.01 with no metric regression; else DROPPED with the table recorded.
- Tests: full suite; tuning script is a thin orchestrator (logic stays in `src`).
- Risks: overfitting the small (2286-row) holdout; mitigated by tuning on CV and confirming on holdout. Fully reversible (config-only change).
- Commit: `experiment(tuning): RandomizedSearchCV for xgboost/gbm; promote iff guardrail clears`

### Phase 11 - Peak-demand underprediction experiments (gated) - see section 5
- Goal: reduce the documented high-quintile underprediction via (a) Duan's smearing and/or (b) sample weighting, evaluated on all four metrics.
- Plan file: `docs/experiments/{date}_peak-underprediction.md` first.
- Files: new `src/bike_sharing/postprocess.py`, `src/bike_sharing/train.py` (optional `sample_weight` routing), a `scripts/run_peak_experiment.py` orchestrator, `tests/test_postprocess.py`.
- Exact smearing math (important - the correction is NOT a multiply of the existing `expm1` output):
  - Residuals are taken in the log1p training space on the train rows only: `e_i = log1p(y_true_train) - log_pred_train`, where `log_pred` is the inner regressor's log-space prediction (`TransformedTargetRegressor.regressor_.predict`), not the inverted count-scale output.
  - Factor: `theta = mean(exp(e_i))`.
  - Corrected prediction operates on `exp(log_pred)` before subtracting 1: `y_corrected = max(theta * exp(log_pred) - 1, 0)`. This is `theta * (expm1(log_pred) + 1) - 1`, which is NOT equal to `theta * expm1(log_pred)` - so `apply_smearing` must take `log_pred`, reconstruct `exp(log_pred)`, scale, then subtract 1 and clip. `compute_smearing_factor` takes log-space residuals; `apply_smearing(log_pred, theta)` returns the clipped count-scale prediction.
- Tasks: compute `theta` on training log-residuals only; apply at prediction; separately try sample weights proportional to demand. Report RMSLE/RMSE/MAE/bias on the holdout, stratified by quintile.
- Expected outcome: smearing should cut the negative peak bias and likely improve RMSE/MAE, possibly at a small RMSLE cost (RMSLE-optimal predictions are near the conditional median; smearing targets the mean). Decision weighs the four metrics jointly; document the tradeoff explicitly.
- Tests: `test_postprocess.py` asserts the exact formula `apply_smearing(log_pred, theta) == max(theta*exp(log_pred) - 1, 0)` on a hand-computed fixture, and that the output is non-negative. It must not assume `theta >= 1` (theta depends on the residual definition and is not guaranteed >= 1) and must not accept the incorrect `theta*expm1(log_pred)` form.
- Risks: smearing can over-correct and worsen RMSLE; sample weights can hurt low-demand hours. Gated, off by default, reversible.
- Commit: `experiment(peak): Duan smearing + demand weighting for peak underprediction`

### Phase 12 - Safe feature additions, experiment-only (gated)
- Goal: test Humidex/comfort-index and the windspeed/atemp recalibration as experiment-only ideas (never auto-promoted). These are two different classes of change and must be implemented differently:
  - (12a) `comfort_index` (Humidex) is a pure, pointwise, closed-form function of `temp`/`humidity` - no fitting, no cross-row dependence. This one is appropriate for `build_candidate_features` + a new `FEATURE_GROUPS` entry, exactly like the existing candidate columns.
  - (12b) Windspeed-zero imputation and atemp recalibration are FITTED transforms (the imputer learns fill values; the calibration learns `atemp ~ temp + humidity` coefficients). They must NOT go in `build_candidate_features` - that function does pure feature generation and is currently called once on the whole frame in the sweep harness, so fitting there would learn from validation/holdout rows and leak.
- Plan file: `docs/experiments/{date}_env-recalibration-and-humidex.md` first.
- Files:
  - 12a: `src/bike_sharing/features.py` (`build_candidate_features` gains `comfort_index`), `scripts/run_feature_experiment.py` (new group), `tests/test_features.py`.
  - 12b: a new experiment-only module `src/bike_sharing/experimental_transforms.py` holding sklearn-style transformers (`fit`/`transform`), e.g. `WindspeedZeroImputer` and `AtempRecalibrator`. These are prepended to the model pipeline so that per-fold cloning in `fit_and_cv` / `evaluate_holdout` fits them on the training fold only and applies them to train/val - fold-safe by construction. (Equivalent fallback: an explicit `fit(train) -> transform(train/val)` loop inside the experiment script. Either way the fit never sees validation rows.) New `tests/test_experimental_transforms.py`.
- Tasks: 12a add `comfort_index` and sweep it through the existing harness; 12b implement the two fitted transformers, wire them at the head of the pipeline behind a flag, and evaluate on both validation views. The imputer uses train-fit fill values / temporal-neighbor logic that never references future-of-validation rows.
- Expected outcome: most likely no promotion (env. tier is secondary; proxies already regressed trees; temp/atemp r~0.98 leaves little new signal) - recorded as DROPPED with the table, itself a clean methodological result. Production pipeline untouched regardless.
- Tests: `test_features.py` for `comfort_index` (no NaN, schema parity). `test_experimental_transforms.py` proves the transforms are fit-on-train-only: fitting on a train subset then transforming yields outputs that do not change when held-out rows are added/removed (no peeking).
- Risks: imputation/calibration leakage if fit globally - eliminated by the transformer-in-pipeline design and enforced by the leakage test.
- Commit: `experiment(features): humidex candidate + fold-safe windspeed/atemp transforms (experiment-only)`

### Phase 13 - Optional stretch: dual-target casual + registered
- Goal: a clearly separated experimental branch predicting `casual` and `registered` independently and summing on the original scale; compare against direct-count XGBoost.
- Plan file: `docs/experiments/{date}_dual-target.md` first.
- Files: new `scripts/train_dual_target.py` (reuses `models.get_model`, two models on `log1p(casual)` / `log1p(registered)`), reuses `evaluate.report`, `train.evaluate_holdout`. No `src` API changes that affect the direct-count path.
- Tasks: train two leakage-safe models (neither target as a feature for the other or for count), validate with the same day-of-month holdout and CV, sum predictions, compare four metrics.
- Expected outcome: documented comparison; adopt as the headline model only if it clearly beats direct-count on the holdout. Direct-count stays canonical otherwise.
- Tests: a contract test that neither dual-target model sees `count`/the other target as a feature; artifact still `datetime,count` with 6493 rows.
- Risks: doubles model surface and leakage discipline; kept fully separate so it cannot destabilize the main pipeline. (Ensemble/blend of XGB+GBM can be a small sibling item here if time allows.)
- Commit: `experiment(dual-target): separate casual/registered models, summed; compare to direct count`

### Phase 14 - Final report and reproducibility cleanup
- Goal: fold every adopted change into `reports/RESULTS.md` + notebook 05; confirm full reproducibility from raw to artifact.
- Files: `reports/RESULTS.md`, notebooks 02-05, `README.md` model-comparison section, `reports/metrics.json`, `reports/figures/`.
- Tasks: end-to-end rerun (`prepare_data` -> `train_model` for each model -> notebooks -> `generate_submission`); ensure every promoted config value carries an `# experiment:` provenance comment; verify all experiment plans are closed (DONE/DROPPED).
- Expected outcome: a coherent final report reflecting the audit; clean clone reproduces metrics and the 6493-row artifact.
- Tests: full suite; artifact schema check.
- Commit: `docs(report): finalize results, interpretation, and reproducibility after Phase 8-13`

---

## 5. High-Demand Underprediction Plan (focused)

The documented failure (figure 21): top-quintile RMSE about 79, mean bias about -35 - the log1p model is median-ish on the original scale and biased low for the mean of skewed peaks, concentrated in working-day commute peaks (figure 22).

Ranked options (impact / difficulty / leakage risk):

| Rank | Intervention | Expected impact | Difficulty | Leakage risk | Notes |
|---|---|---|---|---|---|
| 1 | Duan's smearing (theta on train log-residuals, applied at predict) | Medium-High on bias/RMSE/MAE | Low-Medium | None (train residuals only) | Most theory-grounded fix for log-retransform bias; may cost a little RMSLE - decide on the four-metric balance. |
| 2 | Ensemble/blend XGB + GBM (average in log space) | Low-Medium | Low | Low | Trio is near-tied; averaging reduces variance at peaks. Cheap and reversible. |
| 3 | Sample weighting proportional to demand (or weight top quantiles) | Medium | Medium | Low | Refits the model to care more about peaks; watch low-demand degradation. |
| 4 | Residual-correction model (2nd model on holdout residuals) | Medium | Medium-High | Medium | Easy to overfit; needs its own clean split - lower priority. |
| 5 | Demand-regime / richer rush-hour features | Low | Low | Low | Largely already captured by the promoted interaction terms; expect little. |
| 6 | Asymmetric loss (penalize underprediction more) | Medium | High | Medium | Custom objective, harder to validate; only if 1-3 are insufficient. |
| 7 | Post-hoc per-quantile calibration | Low-Medium | Medium | Medium-High | Calibrating on the eval split risks leakage; must use nested/train-only calibration. |

Recommended sequence: smearing first (Phase 11) -> if peak bias persists, add XGB+GBM blend -> only then sample weighting. Validate every step on the untouched day-of-month holdout, stratified by demand quintile, reporting all four metrics so an RMSLE/RMSE tradeoff is visible. Do not auto-adopt any of these; each is a gated experiment with a plan file.

---

## 6. Validation and Leakage Review (strict)

- Chronological CV (`TimeSeriesSplit(5)`) and day-of-month holdout (1-15 / 16-19): both are reasonable and leakage-safe. The holdout faithfully mirrors Kaggle's own 1-19 / 20+ axis within labeled data. Keep both as co-reported views.
- Random K-Fold: must be avoided (both PDFs agree) - it lets the model borrow from temporally adjacent rows and inflates scores. The project already avoids it. Do not introduce it.
- `gap` in TimeSeriesSplit: a reasonable robustness diagnostic (Phase 9). But note: because the models carry no lag/autoregressive features, train/validation adjacency cannot leak a target - the only adjacency effect is mild shared weather autocorrelation. So `gap=48` is expected to confirm robustness, not change conclusions. The day-of-month holdout's day-15 to day-16 adjacency is similarly low-risk and is faithful to Kaggle's day-19 to day-20 adjacency.
- Lag / rolling features: dangerous for this exact Kaggle structure. Test = days 20-31 with no observed counts, so `count(t-1)` cannot be built at inference without true future targets. Only two safe paradigms (PDF 2): (a) recursive generation - predict hour-by-hour and feed predictions forward, with matching recursive validation; (b) structural-gap lags - only lags >= the full test horizon (e.g. >= 12 days) so the lag value is always in the labeled past. Both are complex and easy to get wrong; the project correctly defers them. If ever attempted, document data-generation, inference behavior, validation protocol, and the non-lag baseline comparison (CLAUDE.md requirement).
- Dual-target casual/registered: acceptable as a clearly separated experiment iff neither target is used as a feature for the other or for count, both are validated with the same leakage-safe splits, and predictions are summed on the original scale. AGENTS.md explicitly allows this stretch.
- UCI full dataset: using the UCI variant (which contains the test-period true counts) is exactly the "public leak" both PDFs warn about - it yields artificial ~0.0 RMSLE and destroys generalization validity. Must be avoided. The project already uses only local Kaggle CSVs with no downloader - compliant; affirm this explicitly in the report.
- Fitted preprocessing transforms (imputation, recalibration, scaling) must be fit per-fold on training rows only. Any transform that learns parameters (windspeed-zero fill values, `atemp ~ temp + humidity` coefficients, target-residual smearing theta) cannot be precomputed once on the full frame or placed in the pure `build_features`/`build_candidate_features` path - that would learn from validation/holdout rows. Such transforms belong in the model pipeline (so per-fold cloning fits them on train only) or in an explicit in-fold `fit(train) -> transform(train/val)` loop. Pointwise closed-form features (cyclic encodings, `comfort_index`, interaction products) carry no such risk and stay in the feature builders.
- More honest Kaggle-like validation: the day-of-month holdout already does this. Optionally add a forward-chaining-by-month view (train months 1..k, validate month k+1) as a third honest lens. Always generate the final artifact by training on all labeled data, then predicting test days 20-31.

---

## 7. Implementation Order (safest)

1. Phase 8 - audit + permutation importance (pure additive, high report value, near-zero risk).
2. Phase 9 - validation diagnostics (`gap`, deeper peak analysis; config-flag default preserves behavior).
3. Phase 11 - peak-underprediction smearing experiment (addresses the one real weakness; gated, reversible). Prioritized above tuning because it has a clearer report narrative and lower overfitting risk.
4. Phase 10 - XGB/GBM tuning (gated; promote only if guardrail clears).
5. Phase 12 - experiment-only env recalibration + Humidex (likely DROPPED, but a clean methodological result).
6. Phase 13 - optional dual-target / ensemble stretch (clearly separated; only if time and curiosity allow).
7. Phase 14 - final report + reproducibility pass.

Rationale: do the zero-risk, high-report-value work first; tackle the one documented weakness early; keep all model-touching work gated and reversible; defer the broadest/riskiest items to the end where they cannot destabilize the deliverable.

---

## 8. Final Recommendation

Top 3 to do next:
1. Phase 8 - permutation importance + commit this audit. Upgrades the feature-importance story (currently impurity-biased) with no new dependency and no leakage risk; directly strengthens the report.
2. Phase 9 - validation diagnostics (`gap` sensitivity + deeper peak/quantile error analysis). Cheap, honest, and exactly the rigor the PDFs emphasize.
3. Phase 11 - gated Duan's smearing experiment for peak underprediction. Targets the single documented weakness with a theory-grounded, reversible change, and produces a "diagnosed and addressed it" narrative - evaluated on all four metrics.

Postpone: hyperparameter tuning (Phase 10 - the trio is already near-optimal; small holdout overfitting risk), env recalibration + Humidex (Phase 12 - experiment-only, low expected gain), dual-target and ensembling (Phase 13 - optional stretch).

Avoid: the UCI full-data leak; random K-Fold; target-lag features as ordinary inputs; `log(y+100)` as a production default (breaks the RMSLE story); deep-learning / new heavy dependencies (N-BEATS, TFT, LightGBM, CatBoost); auto-promoting smearing/weighting without checking the RMSLE tradeoff; deleting or rewriting any working module.

Is it good enough already? Yes. The project is already a clean, leakage-safe, multi-model, multi-view, well-documented and explainable pipeline with a written results report - it is already sufficient for a strong academic/portfolio submission. Every roadmap item is an enhancement that deepens rigor and explainability, not a fix for a deficiency. The highest-leverage additions for the report are permutation importance, the validation-robustness diagnostics, and the peak-underprediction experiment; the rest is genuinely optional.

---

## Verification (for the implementation phases, when approved)

- After each phase: `.venv/bin/python -m pytest` stays green; `.venv/bin/python -c "import bike_sharing; print(bike_sharing.__version__)"` works.
- Feature/transform changes: rerun `scripts/prepare_data.py`, then `scripts/train_model.py --model NAME` for affected models; confirm `reports/metrics.json` updates and `reports/figures/*` regenerate.
- Experiments: a `docs/experiments/{date}_{name}.md` plan exists before code, baseline recorded, closed with DONE/DROPPED + result table.
- Final artifact: `scripts/generate_submission.py --model xgboost` produces a `datetime,count` CSV with exactly 6493 rows.
- No leakage regression: `casual`/`registered` never enter any count feature matrix (enforced by `test_preprocessing.py` + `test_features.py`).
