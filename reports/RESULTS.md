# Results and Interpretation

CMP4336 — Bike Sharing Demand Forecasting. This is the project's written results report. It summarizes the model comparison, the best model's error analysis, and what the results say about the proposal's three questions (environmental impact, temporal patterns, sequential-data objective). The numbers come from `reports/metrics.json`; the figures from `reports/figures/`; the full analysis from `notebooks/05_results_and_interpretation.ipynb`.

## How models were evaluated

Every model is scored on the original `count` scale with four metrics read **together** — RMSLE, RMSE, MAE, R² — and on **two** leakage-safe validation views:

- **CV**: chronological `TimeSeriesSplit` (5 folds), simulating forecasting future months from past months.
- **Day-of-month holdout**: train on days 1-15 of each month, validate on days 16-19. This mirrors the dataset's own train/test structure (labeled data covers days 1-19; the unlabeled test set is days 20+), so it is the more realistic generalization estimate.

`casual` and `registered` are never used as features (they sum to `count` and are absent from the test set). The target is modeled as `log1p(count)` and inverted with `expm1` clipped at zero, so no prediction is ever negative.

A train/test distribution check (figures 17-18) confirms the day-of-month holdout is a fair proxy for the real test set: the numeric predictors (temp, atemp, humidity, windspeed) have near-identical means and spreads across the labeled days (1-19) and the later test days (20+), and the categorical/temporal shares (season, weather, holiday, workingday, hour, month) line up almost exactly. There is no meaningful covariate shift — the test set is simply later days from the same regime.

## Model comparison

Day-of-month holdout (primary view) with the CV RMSLE alongside:

| Model | RMSLE | RMSE | MAE | R² | CV RMSLE |
|---|---|---|---|---|---|
| XGBoost | 0.306 | 47.5 | 28.0 | 0.93 | 0.463 |
| Gradient Boosting | 0.312 | 51.7 | 31.0 | 0.92 | 0.453 |
| Random Forest | 0.328 | 51.8 | 29.9 | 0.92 | 0.515 |
| Ridge (cyclic + log1p) | 0.718 | 140.8 | 89.1 | 0.41 | 0.806 |
| Hourly-mean baseline | 0.755 | 125.9 | 86.1 | 0.53 | 0.739 |
| Mean baseline | 1.531 | 183.1 | 142.6 | -0.00 | 1.402 |

The three tree/boosting models roughly **halve** the error of the best baseline and reach R² ≈ 0.92-0.93 on the harder holdout. XGBoost is narrowly the strongest on both views, but Random Forest and Gradient Boosting are close — treat the three as a near-tie rather than reading a decisive winner from one metric. The Ridge row reflects the feature experiment in `docs/experiments/2026-06-01_leakage-safe-feature-sweep.md`: adding second-harmonic and workingday-gated cyclic features lowered its holdout RMSLE from 0.905 to 0.718, enough to edge past the hourly-mean baseline. This is a linear-baseline / explainability gain — it does not change the model ranking, and the deployed XGBoost is essentially unchanged (0.309 -> 0.306).

## What the results say about the proposal's questions

### Temporal patterns (the primary signal)

Demand is driven first by time-of-day, and the daily shape differs sharply between working and non-working days: a bimodal morning + evening commuter peak on working days versus a single smoother afternoon peak otherwise. This `hour × workingday` interaction is the crux of the problem:

- A linear model (Ridge) with only first-harmonic cyclic hour features **cannot** represent two peaks in one day, so it originally beat the global mean but lost to the trivial hour-of-day average (RMSLE 0.905 vs 0.755). The feature experiment (`docs/experiments/2026-06-01_leakage-safe-feature-sweep.md`) added a second hour harmonic and workingday-gated cyclic terms — a linear-safe encoding of the `hour × workingday` interaction — and Ridge fell to 0.718, just past the hourly-mean baseline. It is still far behind the trees, but it demonstrates that the bimodal shape, not linearity itself, was the linear model's binding constraint.
- The tree/boosting models capture the interaction non-linearly and cut the error in half without any explicit interaction feature.
- Feature importance is dominated by `hour` (and its cyclic encodings), `workingday`, and `year`. The residuals-by-hour figure for the best model (figure 15) hug zero across the whole day — the opposite of Ridge's hour-structured residuals in notebook 03.

### Environmental impact (a real but secondary signal)

Temperature, humidity, weather category, and season form the next tier of importance. They modulate demand around the dominant daily rhythm rather than setting it. The per-condition error analysis (figure 16) shows how the best model's absolute error varies across weather categories and seasons. This matches the EDA: demand rises with temperature, falls with humidity, and declines as weather worsens (categories 1→3). The bivariate EDA (figures 19-20) sharpens the picture: temperature lifts demand mainly when humidity is low — the warm, dry band carries the highest mean count — and weather's drag is not uniform, biting hardest during commute hours and varying by season.

### Where the model errs (demand level and commute hours)

Stratifying the best model's held-out residuals by demand level (figure 21) shows error scales with demand: RMSE climbs from about 10 count units in the lowest-demand quintile to about 79 in the highest, and the mean bias turns negative at the top (about -35), i.e. the model slightly under-predicts the busiest hours — expected given the `log1p` target and the high variance of peak demand. The hour × workingday error heatmap (figure 22) localizes those errors to the working-day commute peaks, while the flat overnight hours are predicted almost perfectly. This is the error-side confirmation of the temporal story above.

### Sequential-data objective

The proposal raised a sequential-analysis goal (target lags such as `count(t-1)`). We deliberately did **not** add target-lag features. The test set covers later days of each month with no observed counts, so a target lag cannot be constructed at inference time without true future values, and using it in validation would leak. The temporal signal is instead captured through calendar and cyclic features. A leakage-safe sequential experiment would require recursive multi-step prediction at inference plus a matching recursive validation protocol; it is scoped as optional future work and would be documented under `docs/experiments/` before any code, per the repository's experiment-log rules.

## Limitations

- **Hyperparameters are sensible defaults, not tuned.** A tuning sweep would be a separate documented experiment; current values live in `config/models.yaml`.
- **Feature importances are impurity-based** — a quick diagnostic, biased toward continuous/high-cardinality features and able to split correlated inputs (e.g. `temp`/`atemp`) arbitrarily. Permutation importance on the holdout is the stronger next step.
- **`day`-of-month is intentionally excluded** as a feature because train (days 1-19) and test (days 20+) do not overlap on it; a schema-contract test enforces that train and test predictors stay identical.
- **The three trees are statistically close**; the "best model" label should not be over-read.

## Reproduce

```bash
python scripts/prepare_data.py
python scripts/train_model.py --model mean_baseline
python scripts/train_model.py --model hourly_mean_baseline
python scripts/train_model.py --model ridge
python scripts/train_model.py --model random_forest
python scripts/train_model.py --model gradient_boosting
python scripts/train_model.py --model xgboost
python scripts/generate_submission.py --model xgboost   # datetime,count artifact
```

Then rebuild and run the notebooks via their `scripts/_build_*_notebook.py` builders. Metrics land in `reports/metrics.json`; figures in `reports/figures/`.
