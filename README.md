# 🏪 Predict RTO — X5 Tech Hackathon

A solution for forecasting **RTO (Retail Turnover)** of X5 store locations across two hackathon stages.

---

## 📋 Problem Statement

The goal was to predict the next month's RTO for each store using historical sales data, store characteristics, and external factors (foot/car traffic, demographics, nearby competitors).

**Target metric:** MAPE (Mean Absolute Percentage Error)

---

## 🗂 Repository Structure

```
├── first stage/          # Stage 1
│   ├── train_boost.py    # CatBoost training
│   ├── grid.py           # Hyperparameter search (GridSearch)
│   ├── create_test.py    # Test set construction with lag features
│   ├── predict.py        # Inference and export
│   └── data_normalize.csv
│
├── second_stage/         # Stage 2
│   ├── edit_data.py      # Feature engineering
│   ├── create_predict.py # March 2025 inference pipeline
│   ├── utils.py          # Shared utilities
│   ├── cat_boost/        # CatBoostRegressor (final model)
│   ├── gradboost/        # GradientBoostingRegressor
│   └── RandomForest/     # Baseline — RandomForestRegressor
│
└── requirements.txt
```

---

## ⚙️ Approach

### Stage 1

- Target `RTO` log-transformed via `log1p` to reduce skewness
- Built **lag features** (lag1–lag6) per store
- Added **rolling mean** (mean3) and **trend** (lag1 − lag3)
- Seasonality encoded with `sin/cos` of month index
- Categorical features (region, city, store area, opening date) passed natively to CatBoost without encoding
- Hyperparameters tuned with `ParameterGrid`

### Stage 2

- Extended lag set: lag1, lag2, lag3, lag6, lag12
- Added `mean3`, `mean6`, `EMA3`, `std3`, `trend`, `trend6`
- Log-normalized numerical features: population, households, RTO
- Feature selection via **CatBoost SHAP** (`RecursiveByShapValues`), tested with 14, 18, 22 and full feature sets
- Trained three models for comparison: CatBoost, GradientBoosting, RandomForest (baseline)
- Separate inference script `create_predict.py` for March 2025 with exact lag reproducibility

---

## 📊 Results

### Stage 1

| Model | MAPE |
|-------|------|
| CatBoostRegressor | 14% |

### Stage 2

| Model | MAPE | R² |
|-------|------|----|
| RandomForest (baseline) | 13.47% | 0.861 |
| GradientBoosting | 0.34% | 0.958 |
| **CatBoost (final)** | **0.27%** | **0.969** |

---

## 🛠 Stack

`Python` · `CatBoost` · `scikit-learn` · `pandas` · `numpy` · `matplotlib`
