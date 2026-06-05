# Environmental recalibration and Humidex (experiment-only)

- **Date:** 2026-06-05
- **Topic:** Do a Humidex comfort index and fold-safe windspeed/atemp recalibration help any model, or are they (as expected) in the already-weak environmental tier?
- **Motivation:** Both research PDFs recommend a comfort index (Humidex) and "fixing" the windspeed-zero sensor floor and atemp sensor drift. Roadmap Phase 12 (`docs/audit/2026-06-04_improvement-roadmap.md`) scopes these as **experiment-only**, never auto-promoted, because the environmental tier is secondary and the prior sweep already rejected environmental products. This experiment quantifies them with leakage-safe machinery.
  - **Baseline reference:** commit `bf8b719`, production day-of-month holdout / CV RMSLE: ridge 0.7184 / 0.8064, random_forest 0.3284 / 0.5147, gradient_boosting 0.3120 / 0.4528, xgboost 0.3064 / 0.4633.
  - **Prior evidence:** the 2026-06-01 sweep found the `environmental` group (`feels_like_gap`, `temp_humidity_interaction`, `bad_weather`) regressed every tree (xgb -0.006, gbm -0.009) and barely moved Ridge (+0.001); it was dropped. `temp` and `atemp` are ~0.98 correlated (EDA figure 05), so a recalibrated atemp largely duplicates temp.
- **Hypothesis (measurable):** neither the Humidex comfort index nor the windspeed/atemp recalibration clears the promotion guardrail on either validation view - the deployed XGBoost does not improve by >= 0.01 holdout RMSLE and Ridge does not improve by >= 0.03 without a tree regression. Expected outcome: DROPPED, production untouched.
- **Two classes of change (implemented differently - this is the crux):**
  - **(12a) `comfort_index` (Humidex)** is a pure, pointwise, closed-form function of `temp`/`humidity` - no fitting, no cross-row dependence - so it belongs in `build_candidate_features` as a candidate **feature column**, exactly like the existing experiment-only candidates.
    - Formula: `comfort_index = temp + (5/9) * (6.11 * exp(17.27*temp/(237.7+temp)) * humidity/100 - 10)`.
  - **(12b) Windspeed-zero imputation and atemp recalibration are FITTED transforms** (the imputer learns a fill value; the recalibrator learns `atemp ~ temp + humidity` coefficients). They must **NOT** go in `build_candidate_features` - that builder runs once on the whole frame, so fitting there would learn from validation/holdout rows and leak. They live in a new experiment-only module of sklearn-style transformers, prepended to the model pipeline so per-fold cloning fits them on the **training fold only**.
- **Decision rule (agreed):** reuse the 2026-06-01 promotion guardrail (a group qualifies if XGBoost improves >= 0.01 holdout RMSLE, OR Ridge improves >= 0.03 with no tree regression > 0.005), evaluated on both views. Per the experiment-only decision, **nothing auto-promotes**: if an arm clears the bar, the promotion is referred to the user (as in Phase 11), not applied automatically. Production `build_features`, `config`, models, and `reports/metrics.json` are untouched by this experiment regardless.
- **Preconditions:** processed/raw data present; `reports/metrics.json` current at `bf8b719`; `pytest` green; production `build_features` and `prepare_data` output unchanged.

## 1) Add the Humidex comfort index as a pointwise candidate feature

- **Goal:** make `comfort_index` computable on train and test without touching the production feature set.
- **Files:**
  - `src/bike_sharing/features.py`: add `comfort_index` inside `build_candidate_features` and to `CANDIDATE_NUMERIC_COLUMNS` (it is a scaled numeric for Ridge). `build_features` (production) is unchanged.
  - `tests/test_features.py`: the existing candidate contracts (no-NaN, train/test parity, numeric-union-passthrough coverage) cover it automatically; add a focused check that `comfort_index` is finite and varies, and is not in the production set.
- **Test / verification:** new/updated `test_features.py` green; full `pytest` green; production schema-parity test still passes.
- **Expected outcome:** `comfort_index` builds cleanly on train and test; production untouched.
- **DONE (commit pending):** Added `comfort_index` to `build_candidate_features` and `CANDIDATE_NUMERIC_COLUMNS` (production `build_features` / `ADDED_FEATURE_COLUMNS` untouched); added a `test_features.py` check that it is finite, varies, and is candidate-only. `pytest` green (93 passed).

## 2) Add fold-safe windspeed/atemp transformers

- **Goal:** leakage-safe, DataFrame-preserving fitted transforms usable inside a model pipeline.
- **Files:**
  - `src/bike_sharing/experimental_transforms.py` (new): `WindspeedZeroImputer` (treats `windspeed == 0` as the sensor floor / missing; `fit` learns the median of non-zero training windspeed; `transform` replaces zeros with it) and `AtempRecalibrator` (`fit` learns a linear `atemp ~ temp + humidity` on training rows; `transform` replaces `atemp` with the fitted prediction). Both are sklearn `BaseEstimator`/`TransformerMixin`, return a column-preserving DataFrame copy. (A simple train-median / linear fit is used deliberately; k-NN / temporal-neighbor imputation is a heavier alternative with its own inference-time complexity, not pursued here.)
  - `tests/test_experimental_transforms.py` (new): prove fit-on-train-only - the imputer's fill and the recalibrator's coefficients are learned from the fit data only; transforming a held-out row uses the train-learned values (e.g. a held-out `windspeed == 0` is filled with the train median regardless of the held-out rows' own distribution); both preserve columns and shape; fitting on different data yields different learned parameters.
- **Test / verification:** new tests green; full `pytest` green.
- **Expected outcome:** transforms fit on train only, no peeking; safe to wrap around any model.
- **DONE (commit pending):** Added `src/bike_sharing/experimental_transforms.py` (`WindspeedZeroImputer` learns the train median of non-zero windspeed; `AtempRecalibrator` learns `atemp ~ temp + humidity` on train) and `tests/test_experimental_transforms.py` proving fit-on-train-only (a held-out `windspeed==0` is filled with the train median; a cloned transformer fit on a train fold ignores a held-out outlier; coefficients depend only on fit data; columns/order preserved).

## 3) Run the environmental experiment

- **Goal:** quantify each arm against baseline for ridge, random_forest, gradient_boosting, xgboost on both validation views.
- **Files:** `scripts/run_env_experiment.py` (thin orchestrator), writing `docs/experiments/2026-06-05_env-recalibration-and-humidex.json` and printing a baseline-vs-arm table.
- **Steps:**
  - Arms: `baseline` (production features + production model), `comfort` (production + `comfort_index`; trees see the column, Ridge routes it via `build_experimental_ridge(extra_numeric=("comfort_index",))`), `recalib` (production features, model wrapped in `Pipeline([WindspeedZeroImputer, AtempRecalibrator, model])`), and `all` (comfort feature + recalib transforms).
  - For each arm/model record CV-mean RMSLE and holdout RMSLE/RMSE/MAE/R2 plus the holdout-RMSLE delta. The baseline arm must reproduce `reports/metrics.json` (harness self-check).
- **Test / verification:** JSON produced; results table pasted into item 4; the script touches no production config, features, or `metrics.json`.
- **Expected outcome:** per the decision rule.
- **DONE (commit pending):** Ran `scripts/run_env_experiment.py`; the baseline arm reproduced `reports/metrics.json` holdout exactly for all four models (harness valid).
  - Metric / result (holdout RMSLE improvement vs baseline, positive = better; cvΔ = change in CV-mean RMSLE, negative = better):

    | arm | ridge | random_forest | gradient_boosting | xgboost |
    |---|---|---|---|---|
    | baseline (RMSLE) | 0.7184 | 0.3284 | 0.3120 | 0.3064 |
    | comfort | +0.0021 (cvΔ -0.0038) | +0.0009 (cvΔ +0.0011) | +0.0000 (cvΔ +0.0031) | -0.0020 (cvΔ +0.0003) |
    | recalib | +0.0096 (cvΔ +0.0002) | +0.0052 (cvΔ +0.0024) | -0.0001 (cvΔ -0.0003) | +0.0026 (cvΔ -0.0004) |
    | all | +0.0120 (cvΔ -0.0012) | +0.0048 (cvΔ +0.0043) | +0.0000 (cvΔ +0.0017) | +0.0045 (cvΔ -0.0101) |

  - Result artifact: `docs/experiments/2026-06-05_env-recalibration-and-humidex.json`
  - Decision: per the guardrail (item 4).

## 4) Decision

- **Goal:** record the outcome and (almost certainly) confirm production is unchanged.
- **Files (reporting):** record the table and decision here; if (unexpectedly) an arm clears the bar, refer promotion to the user rather than auto-applying.
- **Test / verification:** `pytest` green; `git diff` on `config/`, `src/bike_sharing/features.py` production set, and `reports/metrics.json` is empty.
- **Expected outcome:** expected DROPPED with the table recorded; production untouched. The experiment-only candidate/transform code stays for reproducibility.
- **DONE (commit pending):** Decision = **no promotion** (production untouched). No arm clears the pre-registered guardrail (best XGBoost is `all` at +0.0045 holdout, below the 0.01 bar; best Ridge is +0.0120, below 0.03). Findings: (1) `comfort_index` alone is neutral-to-slightly-harmful (xgb -0.0020), confirming the environmental tier is secondary - consistent with the 2026-06-01 rejection of environmental products. (2) The fold-safe windspeed/atemp recalibration gives small, mostly consistent sub-threshold gains, best for the deployed XGBoost (`all`: holdout RMSLE 0.3064 -> 0.3019, RMSE 47.54 -> 45.80, MAE 28.00 -> 26.98, R2 0.933 -> 0.937; CV 0.4633 -> 0.4532) - a different result from the 2026-06-01 environmental products (`feels_like_gap`, `temp_humidity_interaction`, `bad_weather`), which were themselves leakage-safe but redundant/noisy derived columns appended to the feature matrix and regressed every tree. The recalibration differs in kind: instead of adding redundant products it fold-safely transforms the existing sensor fields (windspeed, atemp) in place, which is why it nudges the trees up slightly rather than down. But the gains sit below the bar, the cross-model CV picture is mixed (rf/gbm CV slightly worse under `all`), and they are within plausible single-split noise. Per the rule and the experiment-only / explicit-approval-required constraint, production (`config`, `build_features`, models.yaml, `metrics.json`, prediction path) is unchanged. The candidate + transform code stays for reproducibility; the XGBoost recalibration gain is surfaced to the user, who may approve promotion if desired.
