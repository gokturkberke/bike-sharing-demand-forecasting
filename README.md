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
├── src/bike_sharing/   # Reusable package; current: config, data, preprocessing, features, evaluate, models, train (predict added in later phases)
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
python scripts/train_model.py --model xgb            # Phase 6
python scripts/generate_submission.py --model xgb    # Phase 6: writes a datetime,count prediction artifact
```

## Results

All four metrics (RMSLE, RMSE, MAE, R²) are reported together and read jointly; none is treated as the single deciding score. RMSLE is shown because the target is right-skewed. Each model is validated two leakage-safe ways: chronological `TimeSeriesSplit(5)` (mean over folds) and a day-of-month holdout (train on days 1-15, validate on the latest labeled days 16-19), which mirrors the dataset's own day-of-month train/test structure and is the more realistic generalization estimate.

Day-of-month holdout (all four metrics):

| Model | RMSLE | RMSE | MAE | R² | CV RMSLE |
|---|---|---|---|---|---|
| Mean baseline | 1.531 | 183.1 | 142.6 | -0.00 | 1.402 |
| Hourly-mean baseline | 0.755 | 125.9 | 86.1 | 0.53 | 0.739 |
| Ridge (cyclic + log1p) | 0.906 | 162.2 | 106.2 | 0.21 | 0.987 |
| Random Forest | 0.330 | 51.9 | 30.4 | 0.92 | 0.514 |
| Gradient Boosting | 0.334 | 59.0 | 36.5 | 0.90 | 0.471 |
| XGBoost | — | — | — | — | — |

The tree models change the picture: Random Forest and Gradient Boosting beat every baseline and Ridge on all metrics and both validation views (holdout R² ~0.90), because they capture the `hour × workingday` interaction non-linearly using the full feature set that the linear model had to drop. Ridge still trails the simple hour-of-day average — first-harmonic cyclic features cannot represent the bimodal commuter pattern. Feature importance is dominated by the temporal signal (hour, workingday, year), with environmental inputs (temperature, humidity, weather, season) as a real but secondary tier. The consolidated results-and-interpretation report (Phase 7) ties these numbers back to the environmental and temporal questions in the proposal.
