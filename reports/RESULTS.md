# Results and Interpretation

CMP4336 — Bike Sharing Demand Forecasting. This is the project's written results report. It summarizes the model comparison, the best model's error analysis, and what the results say about the proposal's three questions (environmental impact, temporal patterns, sequential-data objective). The numbers come from `reports/metrics.json`; the figures from `reports/figures/`; the full analysis from `notebooks/05_results_and_interpretation.ipynb`.

## How models were evaluated

Every model is scored on the original `count` scale with four metrics read **together** — RMSLE, RMSE, MAE, R² — and on **two** leakage-safe validation views:

- **CV**: chronological `TimeSeriesSplit` (5 folds), simulating forecasting future months from past months. A robustness check (`docs/experiments/2026-06-04_validation-gap-diagnostics.md`, figure 24) confirms that inserting a ~48-hour chronological gap between folds barely moves the deployed XGBoost's CV RMSLE (+0.004) — expected, since the models use no lag features, so the CV estimate is not inflated by train/validation adjacency. (Ridge and gradient boosting are more gap-sensitive, but that traces to the smallest fold and the 2011→2012 year-boundary fold, not to leakage.)
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

The three tree/boosting models roughly **halve** the error of the best baseline and reach R² ≈ 0.92-0.93 on the harder holdout. XGBoost leads the day-of-month holdout (the primary view), while Gradient Boosting is marginally ahead on the chronological CV (0.453 vs 0.463); with Random Forest just behind, the three are best read as a near-tie rather than a decisive winner from one metric. The Ridge row reflects the feature experiment in `docs/experiments/2026-06-01_leakage-safe-feature-sweep.md`: adding second-harmonic and workingday-gated cyclic features lowered its holdout RMSLE from 0.905 to 0.718, enough to edge past the hourly-mean baseline. This is a linear-baseline / explainability gain — it does not change the best (deployed) model, XGBoost, which is essentially unchanged (0.309 -> 0.306); it only nudges the close ordering among the trees.

## What the results say about the proposal's questions

### Temporal patterns (the primary signal)

Demand is driven first by time-of-day, and the daily shape differs sharply between working and non-working days: a bimodal morning + evening commuter peak on working days versus a single smoother afternoon peak otherwise. This `hour × workingday` interaction is the crux of the problem:

- A linear model (Ridge) with only first-harmonic cyclic hour features **cannot** represent two peaks in one day, so it originally beat the global mean but lost to the trivial hour-of-day average (RMSLE 0.905 vs 0.755). The feature experiment (`docs/experiments/2026-06-01_leakage-safe-feature-sweep.md`) added a second hour harmonic and workingday-gated cyclic terms — a linear-safe encoding of the `hour × workingday` interaction — and Ridge fell to 0.718, just past the hourly-mean baseline. It is still far behind the trees, but it demonstrates that the bimodal shape, not linearity itself, was the linear model's binding constraint.
- The tree/boosting models capture the interaction non-linearly and cut the error in half without any explicit interaction feature.
- Model-agnostic permutation importance on the holdout (figure 23), scored with count-scale RMSLE, confirms and sharpens the story: `hour` dominates by a wide margin, followed by the engineered cyclic terms — the second harmonic `hour_sin2` and the workingday-gated `hour_sin_workday`/`hour_cos_workday` — together with `hour_cos`, `hour_sin`, and `year`; the environmental inputs (humidity, weather, temp, season) sit a clear tier below. The contrast with the impurity ranking (figure 13) is instructive: the raw `workingday` and `is_weekend` flags fall near the bottom under permutation importance — impurity rates `is_weekend` highly, but shuffling it barely moves held-out error because the workingday-gated cyclic features already carry that signal. The residuals-by-hour figure for the best model (figure 15) hug zero across the whole day — the opposite of Ridge's hour-structured residuals in notebook 03.

### Environmental impact (a real but secondary signal)

Temperature, humidity, weather category, and season form the next tier of importance. They modulate demand around the dominant daily rhythm rather than setting it. The per-condition error analysis (figure 16) shows how the best model's absolute error varies across weather categories and seasons. This matches the EDA: demand rises with temperature, falls with humidity, and declines as weather worsens (categories 1→3). The bivariate EDA (figures 19-20) sharpens the picture: temperature lifts demand mainly when humidity is low — the warm, dry band carries the highest mean count — and weather's drag is not uniform, biting hardest during commute hours and varying by season.

### Where the model errs (demand level and commute hours)

Stratifying the best model's held-out residuals by demand level (figure 21) shows error scales with demand: RMSE climbs from about 10 count units in the lowest-demand quintile to about 79 in the highest, and the mean bias turns negative at the top (about -35), i.e. the model slightly under-predicts the busiest hours — expected given the `log1p` target and the high variance of peak demand. The hour × workingday error heatmap (figure 22) localizes those errors to the working-day commute peaks, while the flat overnight hours are predicted almost perfectly. This is the error-side confirmation of the temporal story above.

That `log1p` retransformation bias has a standard correction — Duan's smearing estimator — and we tested it (`docs/experiments/2026-06-05_peak-underprediction.md`): scaling predictions by θ = mean(exp(training log-residuals)) ≈ 1.02 cut the top-quintile mean bias from about −33 to about −23 (~30%) and was a small Pareto improvement on the holdout (RMSLE 0.306 → 0.305, RMSE 47.5 → 47.1, MAE 28.0 → 27.7). The all-metric gains are small enough to sit within single-split noise, so we keep the headline metrics and the prediction artifact tied to the plain model and ship smearing as a tested, documented correction (`bike_sharing.postprocess.apply_smearing`) rather than the default. The robust takeaway is that the peak underprediction is a correctable retransformation bias, not a modeling failure.

### Sequential-data objective

The proposal raised a sequential-analysis goal (target lags such as `count(t-1)`). We deliberately did **not** add target-lag features. The test set covers later days of each month with no observed counts, so a target lag cannot be constructed at inference time without true future values, and using it in validation would leak. The temporal signal is instead captured through calendar and cyclic features. A leakage-safe sequential experiment would require recursive multi-step prediction at inference plus a matching recursive validation protocol; it is scoped as optional future work and would be documented under `docs/experiments/` before any code, per the repository's experiment-log rules.

## Limitations

- **Hyperparameters are sensible defaults.** A RandomizedSearchCV tuning sweep (`docs/experiments/2026-06-05_xgb-gbm-tuning.md`, scored on count-scale RMSLE with the day-of-month holdout excluded from the search) did not beat them: the tuned params lowered the chronological CV RMSLE (XGBoost 0.46 → 0.41, gradient boosting 0.45 → 0.41), but that gain did not transfer to the held-out days 16-19 (XGBoost flat on RMSLE with slightly worse RMSE/MAE/R², gradient boosting marginally worse on RMSLE), so the defaults were kept. Current values live in `config/models.yaml`.
- **Feature importance is reported two ways.** The impurity-based ranking (figure 13) is a quick in-training diagnostic but is biased toward continuous/high-cardinality features and can split credit between correlated inputs (e.g. `temp`/`atemp`). Holdout permutation importance (figure 23), scored with count-scale RMSLE, is the stronger, model-agnostic view and is now the primary importance evidence. Both share the usual caveat that importance among strongly correlated features (the `hour` family) can be distributed somewhat arbitrarily.
- **`day`-of-month is intentionally excluded** as a feature because train (days 1-19) and test (days 20+) do not overlap on it; a schema-contract test enforces that train and test predictors stay identical.
- **The three trees are statistically close**; the "best model" label should not be over-read.
- **The dual-target stretch was tested and not adopted.** Predicting `casual` and `registered` with separate models and summing them (`docs/experiments/2026-06-05_dual-target.md`) did not beat the direct-`count` model on the headline metric: holdout RMSLE was marginally worse (XGBoost 0.306 → 0.307). It did improve the count-scale errors (XGBoost holdout RMSE 47.5 → 45.8, MAE 28.0 → 27.1) and CV RMSLE (0.463 → 0.448), so the result is metric-dependent rather than a clear loss; but on the right-skew-appropriate RMSLE the simpler single-`count` model holds, so it stays the deployed approach.

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
