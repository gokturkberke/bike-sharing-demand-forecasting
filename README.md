# Bike Sharing Demand Forecasting

Hourly bike rental demand forecasting using the [Kaggle Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand) dataset. Course project for **CMP4336**.

## Goal

Predict the hourly bike rental `count` from temporal, seasonal, and weather-related features. This is a course project: the emphasis is exploratory analysis, feature engineering, modeling, validation, and explainable reporting — not a leaderboard score. Results are reported with RMSLE, RMSE, MAE, and R² together and interpreted jointly.

## Data leakage warning

`casual` and `registered` sum exactly to `count` and are **not** present in `test.csv`. They must **never** be used as features for a model that predicts `count`. This rule is enforced in three places: `config/config.yaml` (`drop_columns`), `src/bike_sharing/preprocessing.py` (`drop_leakage_columns`), and a unit test in `tests/test_preprocessing.py`.

## Project structure

```
.
├── config/        # YAML configs (paths, seed, target, drop_columns, model hyperparams)
├── data/
│   ├── raw/       # Original Kaggle CSVs (gitignored)
│   ├── interim/   # Intermediate parsed/cleaned data
│   └── processed/ # Feature-engineered parquet files
├── docs/          # Project proposal and notes
├── notebooks/     # EDA and experimentation notebooks
├── src/bike_sharing/   # Reusable package: config, data, preprocessing, features, evaluate, models, train, predict
├── scripts/       # Thin orchestrators that call into src/
├── tests/         # Unit tests (including leakage guard)
├── models/        # Saved trained models (gitignored)
└── reports/       # Figures, metrics, and test-set prediction artifacts
```

Layer responsibilities (notebooks vs `src/` vs `scripts/` vs `config/` vs `reports/`) and the rules that govern when each layer accepts new code are documented in AGENTS.md §2 and §6.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Verify the package is importable:

```bash
python -c "import bike_sharing; print(bike_sharing.__version__)"
```

> macOS note: `xgboost` needs the OpenMP runtime, which pip does not bundle. If `import xgboost` fails with a `libomp` error, run `brew install libomp`. Every other model (baselines, Ridge, the scikit-learn trees) works without it, and `xgboost` is the only model that needs it.

## Data setup

The raw CSVs are not tracked in the repo. Download them from the Kaggle competition page and place the three files exactly as follows:

```
data/raw/train.csv
data/raw/test.csv
data/raw/sampleSubmission.csv
```

Source: https://www.kaggle.com/c/bike-sharing-demand/data (requires a Kaggle account). Expected data rows (excluding header): `train.csv` = 10,886; `test.csv` = 6,493; `sampleSubmission.csv` = 6,493. All downstream code in `src/` and `scripts/` reads from these paths via `config/config.yaml`.

## Run tests

The data tests in `tests/test_data.py` and the real-data leakage test in `tests/test_preprocessing.py` read from `data/raw/`, so complete the Data setup step first.

```bash
pytest
```

## End-to-end pipeline (after later phases land)

```bash
python scripts/prepare_data.py
python scripts/train_model.py --model ridge
python scripts/train_model.py --model random_forest
python scripts/train_model.py --model gradient_boosting
python scripts/train_model.py --model xgboost            # Phase 6
python scripts/generate_submission.py --model xgboost    # Phase 6: writes a datetime,count prediction artifact
```

## Results

All four metrics (RMSLE, RMSE, MAE, R²) are reported together and read jointly; none is treated as the single deciding score. RMSLE is shown because the target is right-skewed. Each model is validated two leakage-safe ways: chronological `TimeSeriesSplit(5)` (mean over folds) and a day-of-month holdout (train on days 1-15, validate on the latest labeled days 16-19), which mirrors the dataset's own day-of-month train/test structure and is the more realistic generalization estimate.

Day-of-month holdout (all four metrics):

| Model | RMSLE | RMSE | MAE | R² | CV RMSLE |
|---|---|---|---|---|---|
| XGBoost | 0.306 | 47.5 | 28.0 | 0.93 | 0.463 |
| Gradient Boosting | 0.312 | 51.7 | 31.0 | 0.92 | 0.453 |
| Random Forest | 0.328 | 51.8 | 29.9 | 0.92 | 0.515 |
| Ridge (cyclic + log1p) | 0.718 | 140.8 | 89.1 | 0.41 | 0.806 |
| Hourly-mean baseline | 0.755 | 125.9 | 86.1 | 0.53 | 0.739 |
| Mean baseline | 1.531 | 183.1 | 142.6 | -0.00 | 1.402 |

The tree models change the picture: Random Forest, Gradient Boosting, and XGBoost beat every baseline and Ridge on all metrics and both validation views (holdout R² ~0.92), because they capture the `hour × workingday` interaction non-linearly using the full feature set. XGBoost leads the day-of-month holdout on all four metrics (RMSLE 0.31, R² 0.93), while Gradient Boosting is marginally ahead on chronological CV (RMSLE 0.45 vs 0.46); the three trees are best read as a near-tie, and XGBoost stays the deployed model because it leads the more realistic holdout view. The Ridge row reflects the feature experiment in `docs/experiments/2026-06-01_leakage-safe-feature-sweep.md`: adding second-harmonic and workingday-gated cyclic features cut its holdout RMSLE from 0.91 to 0.72, enough to edge past the hourly-mean baseline — a linear-baseline / explainability gain that leaves the deployed XGBoost and the model ranking unchanged. Feature importance is dominated by the temporal signal (hour, workingday, year), with environmental inputs (temperature, humidity, weather, season) as a real but secondary tier. The test-set `datetime,count` prediction artifact is produced from a trained model by `scripts/generate_submission.py`.

**The full written results report is [`reports/RESULTS.md`](reports/RESULTS.md)** — the project's headline deliverable. It consolidates this comparison with the best model's out-of-sample error analysis (`notebooks/05_results_and_interpretation.ipynb`, figures 14-16) and reads the results against the proposal's environmental, temporal, and sequential-data questions, including why target-lag features were deliberately excluded for leakage safety.
