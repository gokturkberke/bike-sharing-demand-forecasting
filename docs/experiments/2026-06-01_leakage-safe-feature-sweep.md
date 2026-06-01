# Leakage-safe feature enrichment sweep

- **Date:** 2026-06-01
- **Topic:** Do leakage-safe engineered features improve or explain the models?
- **Motivation:** The instructor asked for more comprehensive feature engineering that
  explains or improves the model (see `reports/RESULTS.md` "Temporal patterns" and
  "Environmental impact"). The current production feature set is `hour, dayofweek, month,
  year, is_weekend, hour_sin, hour_cos, month_sin, month_cos`. The trees already capture the
  `hour x workingday` interaction from the raw columns, so the open question is whether
  explicit engineered features (a) meaningfully help the strong models and/or (b) let the
  interpretable Ridge baseline represent the bimodal daily pattern it currently cannot.
  - **Baseline reference:** commit `ba2b4ba`, `reports/metrics.json` day-of-month holdout
    RMSLE / CV RMSLE: xgboost 0.3088 / 0.4469, random_forest 0.3296 / 0.5135,
    gradient_boosting 0.3336 / 0.4713, ridge 0.9055 / 0.9868. (Track A enrichment added no
    feature/model change, so this is the live production baseline.)
- **Hypothesis (measurable):** second-harmonic Fourier (`hour_sin2/hour_cos2`),
  workingday-gated cyclic terms (`hour_sin*workingday`, `hour_cos*workingday`), and `is_2012`
  lower **Ridge** day-of-month holdout RMSLE by **>= 0.03**, while the best tree (XGBoost)
  holdout RMSLE does **not** regress by more than **0.005**.
- **Promotion rule (agreed):** a group *qualifies* if **XGBoost improves meaningfully
  (>= 0.01 holdout RMSLE)** OR **Ridge improves by >= 0.03 with no tree regression > 0.005**.
  When more than one group qualifies, promote the **minimal** subset, preferring the one with
  **no tree regression at all** (every model holds or improves) that most directly encodes the
  documented bimodal-shape limitation. If only Ridge improves, frame it as a linear-baseline /
  explainability improvement, not a decisive final-model improvement. Do not run promotion
  (item 3) unless the sweep clearly passes this rule.
- **Preconditions:** processed/raw data present under `data/raw/`; all 6 models trained with
  `reports/metrics.json` current; `pytest` green; production `build_features` and
  `prepare_data` output must stay unchanged until (and unless) promotion.

## 1) Build candidate features and an experiment-only Ridge routing

- **Goal:** make the candidate features computable for both train and test without touching
  the production feature set, and let Ridge actually consume them in the experiment.
- **Files:**
  - `src/bike_sharing/features.py`: add `build_candidate_features(df, cfg)` plus
    `CANDIDATE_NUMERIC_COLUMNS` / `CANDIDATE_PASSTHROUGH_COLUMNS` / `CANDIDATE_FEATURE_COLUMNS`.
    `build_features` is unchanged.
  - `src/bike_sharing/models.py`: add experiment-only `build_experimental_ridge(cfg, params,
    extra_numeric, extra_passthrough)`. Production `get_model('ridge')` / `_build_ridge`
    unchanged.
- **Steps:**
  - Candidate columns (all leakage-safe, derivable from columns present in train AND test):
    `is_morning_peak`, `is_evening_peak`, `is_rush_hour` (from `hour`); `hour_sin2`,
    `hour_cos2` (second harmonic); `hour_sin_workday`, `hour_cos_workday` (workingday-gated
    cyclic); `is_2012` (from `year`); `feels_like_gap = atemp - temp`;
    `temp_humidity_interaction = temp * humidity`; `bad_weather = weather >= 3`.
  - Route candidate numerics through StandardScaler and binaries/bounded through passthrough
    in the experimental Ridge transformer.
- **Test / verification:** `tests/test_features.py` extended with candidate-column contracts
  (binaries in {0,1}, second-harmonic cyclics in [-1,1], no NaN, train/test parity for the
  candidate set).
- **Expected outcome:** candidate frame builds cleanly on train and test; production schema
  test still passes (production `build_features` untouched).
- **DONE (commit `9f65041`):** Added `build_candidate_features` + `CANDIDATE_*` tuples to
  `features.py` and the experiment-only `build_experimental_ridge` to `models.py`; added
  candidate-feature contracts to `tests/test_features.py`. Production `build_features` was
  unchanged at this stage.

## 2) Run the sweep against both validation views

- **Goal:** quantify baseline-vs-candidate metrics for ridge, random_forest,
  gradient_boosting, xgboost on both `fit_and_cv` (TimeSeriesSplit) and `evaluate_holdout`
  (day-of-month).
- **Files:** `scripts/run_feature_experiment.py` (thin orchestrator calling `src/`),
  writes `docs/experiments/2026-06-01_leakage-safe-feature-sweep.json`.
- **Steps:**
  - Trees: baseline on the production feature matrix, candidate on the production matrix +
    candidate columns (trees consume the full frame).
  - Ridge: baseline via production `get_model('ridge')`; candidate via
    `build_experimental_ridge` with the candidate numerics/passthroughs routed in.
  - Record CV-mean RMSLE and holdout RMSLE/RMSE/MAE/R^2 for each, plus the holdout-RMSLE delta.
- **Test / verification:** sweep JSON produced; results table pasted below.
- **Expected outcome:** decide per the promotion rule.
- **DONE (commit `9f65041`):** Ran `scripts/run_feature_experiment.py` as a per-group
  ablation across both validation views. The baseline reproduced production `metrics.json`
  exactly (ridge 0.9055, xgb 0.3088, rf 0.3296, gbm 0.3336), confirming the harness.
  - Metric / result (day-of-month holdout RMSLE, with improvement vs baseline):

    | group | ridge | random_forest | gradient_boosting | xgboost |
    |---|---|---|---|---|
    | baseline | 0.9055 | 0.3296 | 0.3336 | 0.3088 |
    | interaction_harmonic | 0.7184 (+0.187) | 0.3284 (+0.001) | 0.3120 (+0.022) | 0.3064 (+0.002) |
    | peaks | 0.7705 (+0.135) | 0.3309 (-0.001) | 0.3332 (+0.000) | 0.3068 (+0.002) |
    | environmental | 0.9048 (+0.001) | 0.3345 (-0.005) | 0.3429 (-0.009) | 0.3146 (-0.006) |
    | year_trend | 0.8739 (+0.032) | 0.3298 (-0.000) | 0.3337 (-0.000) | 0.3093 (-0.001) |
    | all | 0.6402 (+0.265) | 0.3303 (-0.001) | 0.3154 (+0.018) | 0.3091 (-0.000) |

  - Result artifact: `docs/experiments/2026-06-01_leakage-safe-feature-sweep.json`
  - Decision: under the literal guardrail (Ridge >= 0.03, no tree regression > 0.005),
    **four** groups qualify - interaction_harmonic, peaks, year_trend, and all; only
    `environmental` fails (Ridge +0.001 < 0.03, and xgb -0.006 beyond the 0.005 guardrail).
    Applying the tiebreak: **interaction_harmonic** is the only qualifying group with *no*
    tree regression at all (rf +0.001, gbm +0.022, xgb +0.002), it is parsimonious (4 columns),
    and it directly encodes the bimodal-shape limitation. The runners-up are rejected on
    parsimony/robustness, not on the guardrail: `all` passes on net but bundles the
    environmental columns that regress every tree in isolation; `peaks` are redundant with the
    cyclic encoding and nick rf (-0.001); `year_trend` clears Ridge only marginally (+0.032,
    just over the bar) and dips every tree slightly. Promote interaction_harmonic only; drop
    the rest. Fed into item 3.

## 3) (Conditional) Promote the winning subset to production

- **Goal:** ship only the subset that clears the promotion rule.
- **Files (only if item 2 passes):** `src/bike_sharing/features.py` (move winners into
  `build_features` / `ADDED_FEATURE_COLUMNS`, second-harmonic cyclics into
  `CYCLIC_FEATURE_COLUMNS`); `src/bike_sharing/models.py` (winners into the `LINEAR_*` lists so
  Ridge consumes them); `tests/test_features.py` (promote the candidate contracts);
  `config/models.yaml` (only if any param retuned, with a back-reference comment to this file);
  retrain all 6 models -> `reports/metrics.json`; rebuild + execute notebooks 02-05; update
  `reports/RESULTS.md`.
- **Test / verification:** `pytest` green; `metrics.json` diff shows the recorded gain and no
  tree regression beyond the guardrail.
- **Expected outcome:** production metrics improve per the rule; otherwise this item is DROPPED.
- **DONE (commits `9f65041`, `b722888`):** Promoted the four interaction_harmonic columns into
  `build_features` (`CYCLIC_FEATURE_COLUMNS` gains `hour_sin2`/`hour_cos2`, new
  `INTERACTION_FEATURE_COLUMNS` holds `hour_sin_workday`/`hour_cos_workday`), added them to
  Ridge's `LINEAR_PASSTHROUGH_COLUMNS`, updated tests, shrank the candidate set to the
  un-promoted columns, re-ran `prepare_data.py` + retrained all six models, and rebuilt and
  executed notebooks 02-05 with updated narratives/figures. `reports/RESULTS.md` updated.
  - Metric / result (day-of-month holdout RMSLE): ridge 0.905 -> **0.718** (now beats the
    hourly-mean baseline 0.755), gradient_boosting 0.334 -> 0.312, xgboost 0.309 -> **0.306**
    (still best, no regression), random_forest 0.330 -> 0.328. Guardrail held (no tree
    regression > 0.005).
  - Result artifact: `reports/metrics.json`
  - Decision: **shipped** to production as a linear-baseline / explainability improvement. The
    deployed XGBoost is essentially unchanged and the model ranking is unchanged; the headline
    gain is that engineered features let the linear baseline finally edge past the hourly-mean
    benchmark, confirming the bimodal daily shape (not linearity) was its binding constraint.
