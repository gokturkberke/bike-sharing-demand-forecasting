# Results and Interpretation

CMP4336 — Bike Sharing Demand Forecasting. This is the project's written results report. It summarizes the model comparison, the best model's error analysis, and what the results say about the proposal's three questions (environmental impact, temporal patterns, sequential-data objective). The numbers come from `reports/metrics.json`; the figures from `reports/figures/`; the full analysis from `notebooks/05_results_and_interpretation.ipynb`.

## How models were evaluated

Every model is scored on the original `count` scale with four metrics read **together** — RMSLE, RMSE, MAE, R² — and on **two** leakage-safe validation views:

- **CV**: chronological `TimeSeriesSplit` (5 folds), simulating forecasting future months from past months.
- **Day-of-month holdout**: train on days 1-15 of each month, validate on days 16-19. This mirrors the dataset's own train/test structure (labeled data covers days 1-19; the unlabeled test set is days 20+), so it is the more realistic generalization estimate.

`casual` and `registered` are never used as features (they sum to `count` and are absent from the test set). The target is modeled as `log1p(count)` and inverted with `expm1` clipped at zero, so no prediction is ever negative.

## Model comparison

Day-of-month holdout (primary view) with the CV RMSLE alongside:

| Model | RMSLE | RMSE | MAE | R² | CV RMSLE |
|---|---|---|---|---|---|
| XGBoost | 0.309 | 46.5 | 27.6 | 0.94 | 0.447 |
| Random Forest | 0.330 | 51.9 | 30.4 | 0.92 | 0.514 |
| Gradient Boosting | 0.334 | 59.0 | 36.5 | 0.90 | 0.471 |
| Hourly-mean baseline | 0.755 | 125.9 | 86.1 | 0.53 | 0.739 |
| Ridge (cyclic + log1p) | 0.905 | 162.2 | 106.2 | 0.21 | 0.987 |
| Mean baseline | 1.531 | 183.1 | 142.6 | -0.00 | 1.402 |

The three tree/boosting models roughly **halve** the error of the best baseline and reach R² ≈ 0.90-0.94 on the harder holdout. XGBoost is narrowly the strongest on both views, but Random Forest and Gradient Boosting are close — treat the three as a near-tie rather than reading a decisive winner from one metric.

## What the results say about the proposal's questions

### Temporal patterns (the primary signal)

Demand is driven first by time-of-day, and the daily shape differs sharply between working and non-working days: a bimodal morning + evening commuter peak on working days versus a single smoother afternoon peak otherwise. This `hour × workingday` interaction is the crux of the problem:

- A linear model (Ridge) with first-harmonic cyclic hour features **cannot** represent two peaks in one day. It beat the global mean but lost to the trivial hour-of-day average (RMSLE 0.905 vs 0.755).
- The tree/boosting models capture the interaction non-linearly and cut the error in half.
- Feature importance is dominated by `hour` (and its cyclic encodings), `workingday`, and `year`. The residuals-by-hour figure for the best model (figure 15) hug zero across the whole day — the opposite of Ridge's hour-structured residuals in notebook 03.

### Environmental impact (a real but secondary signal)

Temperature, humidity, weather category, and season form the next tier of importance. They modulate demand around the dominant daily rhythm rather than setting it. The per-condition error analysis (figure 16) shows how the best model's absolute error varies across weather categories and seasons. This matches the EDA: demand rises with temperature, falls with humidity, and declines as weather worsens (categories 1→3).

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
