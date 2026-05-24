# Bike Sharing Demand Forecasting

Hourly bike rental demand forecasting using the [Kaggle Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand) dataset. Course project for **CMP4336**.

## Goal

Predict the hourly bike rental `count` from temporal, seasonal, and weather-related features.

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
├── src/bike_sharing/   # Reusable package: data, features, models, train, evaluate, predict
├── scripts/       # Thin orchestrators that call into src/
├── tests/         # Unit tests (including leakage guard)
├── models/        # Saved trained models (gitignored)
└── reports/       # Figures, metrics, and Kaggle submissions
```

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
python scripts/train_model.py --model rf
python scripts/train_model.py --model xgb
python scripts/evaluate_model.py
python scripts/generate_submission.py --model xgb
```

## Results

| Model | RMSLE | RMSE | MAE | R² |
|---|---|---|---|---|
| Mean baseline | — | — | — | — |
| Hourly-mean baseline | — | — | — | — |
| Ridge (log1p) | — | — | — | — |
| Random Forest | — | — | — | — |
| Gradient Boosting | — | — | — | — |
| XGBoost | — | — | — | — |

_Filled in after Phase 4 onward._
