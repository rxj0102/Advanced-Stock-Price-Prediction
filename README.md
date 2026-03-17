# Stock Price Prediction — Multi-Model Comparative Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red.svg)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange.svg)](https://jupyter.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous comparative study of **12+ machine learning algorithms** for predicting stock prices 5 trading days ahead. Covers 16 years of historical data (2008–2024) across 5 major stocks, 40+ engineered technical features, and a strict temporal train/test split to prevent lookahead bias.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results](#results)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

| Detail | Value |
|--------|-------|
| **Stocks** | AAPL, MSFT, JPM, JNJ, XOM |
| **Sectors** | Technology, Financial, Healthcare, Energy |
| **Time Period** | 2008-01-01 — 2024-01-01 (16 years) |
| **Data Points** | 19,115 total (4,027 per stock) |
| **Prediction Horizon** | 5 trading days |
| **Features** | 40+ engineered technical indicators |
| **Models Compared** | 12+ (linear → ensembles → neural networks) |
| **Validation** | Temporal 80/20 train/test split |

---

## Project Structure

```
Advanced-Stock-Price-Prediction/
│
├── src/
│   └── stock_prediction/          # Installable Python package
│       ├── __init__.py
│       ├── config.py              # All constants and hyperparameters
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py          # yfinance download & column flattening
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineer.py        # 40+ technical indicator features
│       ├── models/
│       │   ├── __init__.py
│       │   ├── evaluate.py        # ModelMetrics dataclass + comparison table
│       │   └── train.py           # build_models(), train_pipeline(), run_all_stocks()
│       └── visualization/
│           ├── __init__.py
│           └── plots.py           # All matplotlib/seaborn chart functions
│
├── notebooks/
│   └── analysis.ipynb             # Clean end-to-end notebook using src/
│
├── tests/
│   ├── conftest.py                # Synthetic fixtures (no network required)
│   ├── test_features.py           # Feature engineering tests
│   ├── test_evaluate.py           # Metrics and comparison table tests
│   └── test_train.py              # Pipeline and train/test split tests
│
├── adv_model_compare.ipynb        # Original exploratory notebook
├── pyproject.toml                 # Package metadata, pytest, ruff config
├── requirements.txt               # Pinned dependencies
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/rxj0102/Advanced-Stock-Price-Prediction.git
cd Advanced-Stock-Price-Prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3a. Install as an editable package (recommended for development)
pip install -e ".[dev]"

# 3b. Or just install runtime dependencies
pip install -r requirements.txt

# 4. (Optional) install boosting libraries
pip install -e ".[boosting]"
```

---

## Quick Start

```python
from stock_prediction.config      import STOCKS
from stock_prediction.data        import download_stocks
from stock_prediction.models      import build_models, build_ensemble, train_pipeline
from stock_prediction.models.evaluate import build_comparison_table

# Download 16 years of data for 5 stocks
stock_data = download_stocks()

# Build all models (linear, trees, ensembles, optional: XGBoost/LightGBM/CatBoost)
base     = build_models()
ensemble = build_ensemble(base)

# Train & evaluate everything on AAPL with a temporal 80/20 split
result = train_pipeline(stock_data["AAPL"], "AAPL", models={**base, **ensemble})

# Pretty-print ranked comparison table
print(build_comparison_table(result["results"]))
```

Or open the clean notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Results

### Model Rankings (Average Across All Stocks)

| Rank | Model | R² | RMSE | MAE | MAPE | Directional Acc. |
|:----:|-------|----|------|-----|------|:----------------:|
| **1** | **Lasso Regression** | **0.9118** | **$5.99** | **$4.65** | **3.10%** | **53.66%** |
| 2 | ElasticNet | 0.9110 | $6.01 | $4.68 | 3.12% | 50.39% |
| 3 | Voting Ensemble | 0.9060 | $6.18 | $4.82 | 3.22% | 52.88% |
| 4 | Linear Regression | 0.9029 | $6.28 | $4.90 | 3.27% | 52.62% |
| 5 | Ridge Regression | 0.9013 | $6.34 | $4.96 | 3.31% | 52.62% |
| 6 | Stacking Ensemble | 0.8914 | $6.64 | $5.16 | 3.45% | 52.75% |
| 7 | Neural Network (MLP) | -1.8131 | $33.81 | $31.77 | 20.89% | 49.48% |
| 8 | Decision Tree | -3.2603 | $41.61 | $36.28 | 22.40% | 47.64% |
| 9 | Gradient Boosting | -3.9902 | $45.04 | $40.52 | 25.27% | 53.14% |
| 10 | Random Forest | -4.1259 | $45.64 | $41.32 | 25.83% | 51.18% |
| 11 | XGBoost | -5.7477 | $52.37 | $48.53 | 30.65% | 51.96% |
| 12 | SVR | -12.3763 | $73.73 | $65.67 | 40.97% | 49.48% |

> **Key insight:** Linear models with regularization dramatically outperform tree-based and neural network models on this task. Short-lag price autocorrelation rewards linear extrapolation; tree models require return-relative normalisation to compete.

---

### Per-Stock Performance (Lasso Regression)

| Stock | Sector | R² | RMSE | MAE | RMSE/Price | Dir. Acc. |
|:-----:|--------|----|------|-----|:----------:|:---------:|
| XOM | Energy | 0.9769 | $3.42 | $2.66 | 4.54% | 49.74% |
| AAPL | Technology | 0.9152 | $5.87 | $4.56 | 3.84% | 53.14% |
| JPM | Financial | 0.8882 | $4.81 | $3.80 | 3.70% | 51.83% |
| MSFT | Technology | 0.8604 | $15.29 | $11.49 | 5.51% | 51.44% |
| JNJ | Healthcare | 0.7736 | $3.39 | $2.74 | 2.27% | 54.32% |

**Average R² (all stocks):** `0.8829`

---

### Top 10 Features by Lasso Coefficient

| Rank | Feature | Description | Impact |
|:----:|---------|-------------|--------|
| 1 | `Price_Lag_5` | Price 5 days ago | +10.89 |
| 2 | `MA_5` | 5-day moving average | +6.21 |
| 3 | `MA_10` | 10-day moving average | +1.73 |
| 4 | `MA_20` | 20-day moving average | +1.68 |
| 5 | `MA_200` | 200-day moving average | +1.25 |
| 6 | `Momentum_5` | 5-day momentum | +1.07 |
| 7 | `Price_Lag_1` | Previous day's price | +0.98 |
| 8 | `MA_50` | 50-day moving average | +0.67 |
| 9 | `Std_10` | 10-day rolling volatility | -0.45 |
| 10 | `Std_20` | 20-day rolling volatility | +0.34 |

17 of 42 features (40.5%) were zeroed out by Lasso — automatic feature selection.

---

### Sector Predictability

| Sector | Avg. R² | Avg. RMSE | Predictability |
|--------|:-------:|:---------:|:--------------:|
| Energy | 0.9769 | $3.42 | Highest |
| Financial | 0.8882 | $4.81 | High |
| Technology | 0.8878 | $10.58 | Medium |
| Healthcare | 0.7736 | $3.39 | Lowest |

---

## Methodology

### Feature Engineering (`src/stock_prediction/features/engineer.py`)

40+ features engineered from raw OHLCV data — **no future information used**:

```python
features = {
    "Moving Averages":  ["MA_5", "MA_10", "MA_20", "MA_50", "MA_100", "MA_200"],
    "Volatility":       ["Std_5", "Std_10", "Std_20", "Std_50", "Std_100", "Std_200"],
    "Crossovers":       ["MA_5_20_Crossover", "MA_20_50_Crossover", "MA_50_200_Crossover"],
    "Ratios":           ["Price_MA20_Ratio", "Price_MA50_Ratio", "Price_MA200_Ratio"],
    "Intra-day":        ["High_Low_Range", "Close_Open_Gap"],
    "Volume":           ["Volume_Ratio", "Volume_Price_Trend"],
    "Lag Features":     ["Price_Lag_1", ..., "Price_Lag_10"],
    "Momentum":         ["Momentum_5", "Momentum_20", "ROC_5", "ROC_20"],
    "Rolling Stats":    ["Rolling_Max_20", "Rolling_Min_20", "Price_Position"],
}
```

### Validation Strategy

- **Temporal split:** first 80% for training, last 20% for testing
- No shuffling — preserves time-series ordering
- `StandardScaler` fit exclusively on training data, applied to test

### Package API

```
stock_prediction.config          — STOCKS, dates, hyperparameters
stock_prediction.data.loader     — download_stocks()
stock_prediction.features.engineer  — engineer_features(), prepare_xy()
stock_prediction.models.evaluate — evaluate_model(), ModelMetrics, build_comparison_table()
stock_prediction.models.train    — build_models(), train_pipeline(), run_all_stocks()
stock_prediction.visualization.plots — plot_predictions(), plot_model_comparison(), …
```

---

## Key Findings

1. **Regularised linear models win.** Lasso (R²=0.91) and ElasticNet beat all tree-based and neural network models by a wide margin.

2. **Tree-based models fail without return-normalisation.** Negative R² scores indicate predictions worse than the mean baseline.

3. **Lasso performs automatic feature selection.** 17/42 features are zeroed out, producing a sparse, interpretable model.

4. **Energy stocks are most predictable** (XOM R²=0.977). Healthcare is hardest (JNJ R²=0.774).

5. **Directional accuracy hovers near 50–54%** — price level prediction is strong, direction is harder.

6. **Short-term lags dominate.** `Price_Lag_5` and `MA_5` are the two most impactful features, confirming strong short-term autocorrelation.

---

## Running Tests

```bash
# All tests (no network required — uses synthetic data)
pytest

# With coverage report
pytest --cov=src/stock_prediction --cov-report=term-missing

# Single module
pytest tests/test_features.py -v
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow. Quick summary:

1. Fork → feature branch → PR
2. Clear all notebook outputs before committing (`Kernel > Restart & Clear Output`)
3. Run `pytest` and ensure all tests pass
4. Follow PEP 8 / `ruff` style

---

## License

This project is licensed under the [MIT License](LICENSE).
