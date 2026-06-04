# Peak-demand underprediction: Duan's smearing

- **Date:** 2026-06-05
- **Topic:** Does Duan's smearing estimator reduce the documented top-quintile underprediction of the deployed XGBoost, and at what cost to RMSLE?
- **Motivation:** `reports/RESULTS.md` and notebook 05 (figure 21) document that the best model's held-out error scales with demand and the mean bias turns negative (~ -35) in the top demand quintile - it under-predicts the busiest hours. This is the expected consequence of training on `log1p(count)`: by Jensen's inequality, `expm1` of an (approximately) unbiased log prediction underestimates the conditional mean of the right-skewed target. Both research PDFs recommend Duan's non-parametric smearing as the correction. Roadmap Phase 11 (`docs/audit/2026-06-04_improvement-roadmap.md`, section 5) ranks it the #1 intervention.
  - **Baseline reference:** commit `64e5d84`, production XGBoost day-of-month holdout: RMSLE 0.3064, RMSE 47.54, MAE 28.00, R2 0.9325; documented top-quintile mean bias ~ -35 (figure 21). The experiment recomputes the exact per-quintile bias as its baseline.
- **Hypothesis (measurable):** smearing with `theta = mean(exp(e_i))` over the days-1-15 log-residuals raises predictions multiplicatively, so on the day-of-month holdout it (a) shrinks the magnitude of the top-quintile mean bias toward 0 and (b) lowers holdout RMSE and MAE, at the cost of (c) a modest RMSLE increase - because RMSLE-optimal point predictions sit near the conditional median, while smearing targets the conditional mean.
- **Scope:** Duan's smearing only, on the deployed model (XGBoost). Per the roadmap section-5 sequence (smearing first; XGB+GBM blend and sample weighting only if smearing is insufficient), the blend (Phase 13) and sample weighting are deliberately **out of scope** here.
- **Exact smearing math (the correction is NOT a multiply of the existing `expm1` output):**
  - Log-space residuals on the training rows only: `e_i = log1p(y_train) - log_pred_train`, where `log_pred_train = model.regressor_.predict(X_train)` (the inner regressor's log-space output), not `model.predict`.
  - Factor: `theta = mean(exp(e_i))` (not guaranteed >= 1).
  - Corrected count prediction: `y_corrected = max(theta * exp(log_pred) - 1, 0)`, i.e. `theta` multiplies `exp(log_pred)` *before* the -1 that undoes `log1p`. This is **not** `theta * expm1(log_pred)`.
- **Decision rule (pre-registered, four metrics read together):** evaluate uncorrected vs smeared on the day-of-month holdout (fit days 1-15, theta from days 1-15 residuals, score days 16-19), reporting overall RMSLE/RMSE/MAE/R2 plus per-demand-quintile RMSE and mean bias. Promote smearing into the production prediction path (config-gated, default off until promoted) **only if** it improves holdout RMSE **and** MAE **and** reduces the top-quintile |bias|, while RMSLE worsens by **<= 0.005**. Otherwise keep smearing as a tested, documented, off-by-default capability and record the tradeoff. Either way the analysis is reported, since it directly addresses a documented weakness.
- **Preconditions:** processed data present (`data/processed/train.parquet`); `reports/metrics.json` current at `64e5d84`; `pytest` green; the production prediction path (`predict.py`) unchanged until (and unless) promotion.

## 1) Add the smearing transform to a new postprocess module

- **Goal:** a single-responsibility, tested correction that operates on log-space predictions.
- **Files:**
  - `src/bike_sharing/postprocess.py`: `compute_smearing_factor(log_residuals) -> float` (`mean(exp(e))`) and `apply_smearing(log_pred, theta) -> np.ndarray` (`max(theta*exp(log_pred) - 1, 0)`).
  - `tests/test_postprocess.py`: assert `compute_smearing_factor` equals `mean(exp(residuals))` on a fixture; assert `apply_smearing` matches the **exact** formula on a hand-computed fixture, clips negatives to 0, and is **not** equal to the incorrect `theta*expm1(log_pred)` form. Must **not** assume `theta >= 1`.
- **Test / verification:** new tests green; full `pytest` green.
- **Expected outcome:** a reusable, exact, non-negative smearing correction.
- **DONE (commit pending):** Added `src/bike_sharing/postprocess.py` (`compute_smearing_factor`, `apply_smearing`) and `tests/test_postprocess.py` (exact-formula fixture; clips negatives to 0; `theta < 1` allowed; distinct from the wrong `theta*expm1` form; `theta=1` no-op). `pytest` green (87 passed).

## 2) Run the smearing experiment on the day-of-month holdout

- **Goal:** quantify the RMSLE <-> RMSE/MAE/bias tradeoff and the per-quintile bias change for XGBoost.
- **Files:** `scripts/run_peak_experiment.py` (thin orchestrator), writing `docs/experiments/2026-06-05_peak-underprediction.json` and printing an uncorrected-vs-smeared table.
- **Steps:**
  - Fit XGBoost on the day-of-month train rows (days 1-15); get `log_pred_train` via `model.regressor_.predict`; compute `theta` from the training log-residuals.
  - Get `log_pred` on the holdout (days 16-19); uncorrected = `model.predict` (= `from_log1p`); smeared = `apply_smearing(log_pred, theta)`.
  - Report overall RMSLE/RMSE/MAE/R2 (uncorrected vs smeared) and, stratified by holdout demand quintile, RMSE and mean bias for both; record `theta`.
- **Test / verification:** sweep JSON produced; table pasted below; the script touches no production config, model artifact, or prediction path.
- **Expected outcome:** decide per the decision rule.
- **DONE (commit pending):** Added `scripts/run_peak_experiment.py` and ran it. theta = 1.0215 (a ~2% upward correction). The uncorrected column reproduced the production XGBoost holdout exactly (RMSLE 0.3064, RMSE 47.54, MAE 28.00, R2 0.9325) - harness valid.
  - Metric / result (day-of-month holdout, uncorrected vs smeared):

    | metric | uncorrected | smeared |
    |---|---|---|
    | RMSLE | 0.3064 | 0.3053 |
    | RMSE | 47.54 | 47.08 |
    | MAE | 28.00 | 27.66 |
    | R2 | 0.9325 | 0.9340 |
    | top-quintile mean bias | -32.93 | -22.97 |

  - Result: contrary to the hypothesized RMSLE cost, smearing was a small **Pareto improvement** - every overall metric improved slightly and the top-quintile underprediction shrank ~30% (bias -32.93 -> -22.97). The model under-predicts in log space by ~2% on average (theta 1.0215), and correcting that helped count-scale and log-scale metrics alike on this holdout. The all-metrics gains are small (RMSLE -0.0011, RMSE -0.46) and within plausible single-split noise; the bias reduction is the robust, directionally-certain effect.
  - Result artifact: `docs/experiments/2026-06-05_peak-underprediction.json`
  - Decision: clears the pre-registered guardrail (RMSE and MAE improve, top-quintile |bias| reduced, RMSLE does not worsen). Promotion scope (item 3) referred to the user given the small magnitude and the production surface area.

## 3) Decision: report the tradeoff and (conditionally) wire smearing into production

- **Goal:** record the finding against the documented weakness and lock the production behavior.
- **Files (reporting, always):** `reports/RESULTS.md` "Where the model errs" / "Limitations" - record whether smearing was adopted and the measured tradeoff; rebuild + execute notebook 05 only if a figure/number there changes.
- **Files (only if the guardrail clears):** wire smearing into the production prediction path behind a config flag (default off) - `theta` computed at train time in `scripts/train_model.py` and applied in `predict.make_prediction_frame`; this plumbing is designed in full only if item 2 clears the bar (avoid speculative production changes).
- **Test / verification:** `pytest` green; if wired, a `predict` contract that smearing is off by default and that the artifact stays non-negative `datetime,count` with 6493 rows.
- **Expected outcome:** if smearing clears the bar, it ships config-gated; otherwise it stays a documented, tested, off-by-default capability and the tradeoff is recorded. Production prediction default is explicit either way.
- **DONE (commit pending):** Decision = **document-only** (user-chosen). Smearing clears the guardrail, but the all-metric gain is small and within plausible single-split noise, so it is **not** wired into the production prediction path: `config/config.yaml`, `predict.py`, `reports/metrics.json`, the saved estimators, and the notebooks/figures all stay tied to the plain model. Smearing ships as a tested, documented correction (`src/bike_sharing/postprocess.py`) plus a finding added to `reports/RESULTS.md` ("Where the model errs"): the peak underprediction is a correctable `log1p` retransformation bias (theta ~1.02, ~30% top-quintile bias reduction), not a modeling failure. Production wiring was deliberately declined in favor of the leanest honest outcome.
