# Advanced Stock Price Prediction — v2

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous comparative study of **12+ machine learning algorithms** for predicting **5-day log returns** of 5 major stocks. Covers 16 years of data (2008–2024), 34 returns-based features, and strict temporal validation to eliminate lookahead bias.

Directly based on [`adv_model_compare_v2.ipynb`](adv_model_compare_v2.ipynb).

---

## What's New in v2

| Area | v1 | v2 |
|------|----|----|
| **Target** | Raw close price | 5-day log return (stationary) |
| **Features** | Price levels → 100+ correlated pairs | Returns-based → ~16 correlated pairs |
| **Scaler** | StandardScaler | RobustScaler (outlier-resistant) |
| **CV** | Simple 80/20 split | TimeSeriesSplit throughout |
| **Regularisation** | Fixed alpha (0.01) | Auto-tuned via LassoCV/RidgeCV |
| **Ensembles** | Voting + Stacking | Voting + Stacking (TS-CV) + Blending + CV-Weighted |
| **Validation** | Keras validated on test set | Keras validated on temporal inner-val |
| **Backtest** | — | Long/short with Sharpe ratio & max drawdown |
| **Caching** | Download every run | Parquet cache (no re-download) |
| **Calendar** | Raw ordinal integers | Cyclical sin/cos encoding |

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Methodology](#methodology)
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
| **Prediction Target** | 5-day log return (stationary) |
| **Features** | 34 returns-based technical indicators |
| **Models** | 12+ (linear → ensembles → neural networks) |
| **Validation** | TimeSeriesSplit (5 folds) + temporal 80/20 holdout |
| **Backtest** | Long/short with Sharpe ratio & max drawdown |

---

## Project Structure

```
Advanced-Stock-Price-Prediction/
│
├── adv_model_compare_v2.ipynb     # Primary notebook (full end-to-end analysis)
│
├── src/
│   └── stock_prediction/          # Installable Python package
│       ├── __init__.py
│       ├── config.py              # Constants: SEED, STOCKS, PREDICTION_HORIZON, etc.
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py          # yfinance download with parquet caching
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineer.py        # 34 return-based features (stationary)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── evaluate.py        # Metrics: R², RMSE, dir accuracy + binomial p-value
│       │   └── train.py           # Linear/tree/ensemble models, multi-stock pipeline, backtest
│       └── visualization/
│           ├── __init__.py
│           └── plots.py           # Comparison, residual, heatmap, backtest charts
│
├── notebooks/
│   └── analysis.ipynb             # Clean end-to-end notebook using src/
│
├── tests/
│   ├── conftest.py                # Synthetic fixtures (no network required)
│   ├── test_features.py           # Feature engineering tests (v2 feature names)
│   ├── test_evaluate.py           # Metrics, comparison table, sig-stars tests
│   └── test_train.py              # Pipeline, backtest, multi-stock tests
│
├── pyproject.toml                 # Package metadata, pytest, ruff config
├── requirements.txt               # Runtime + optional dependencies
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

# 4. Optional: gradient boosting libraries
pip install -e ".[boosting]"

# 5. Optional: deep learning
pip install -e ".[deep]"
```

---

## Quick Start

### Run the notebook

```bash
jupyter notebook adv_model_compare_v2.ipynb
```

### Use the library

```python
from stock_prediction.data.loader      import download_stocks
from stock_prediction.models.train     import train_pipeline, run_all_stocks, backtest
from stock_prediction.models.evaluate  import build_comparison_table

# Download + cache 16 years of OHLCV data for 5 stocks
stock_data = download_stocks()

# Full single-stock pipeline: linear + tree + ensemble models
result = train_pipeline(stock_data["AAPL"], "AAPL", verbose=True)

# Build ranked comparison table
table = build_comparison_table({
    name: (res["train_metrics"], res["test_metrics"])
    for name, res in result["all_results"].items()
})
print(table)

# Run multi-stock LassoCV pipeline (auto-tuned alpha per ticker)
all_results = run_all_stocks(stock_data)

# Backtest the best model
best = result["linear_results"]["LassoCV"]
bt = backtest(result["y_test"].values, best["predictions"])
print(f"Sharpe: {bt['Sharpe']:.2f}   Max DD: {bt['Max_DD']:.2%}")
```

---

## Features

34 features engineered from raw OHLCV data — **all returns-based, no raw price levels**:

| Category | Count | Features |
|----------|:-----:|---------|
| Log returns | 8 | `LogRet_1d` … `LogRet_20d`, `Momentum_5d`, `Momentum_20d` |
| Intraday microstructure | 4 | `HL_Range`, `Close_Open_Pct`, `Upper_Shadow`, `Lower_Shadow` |
| Volume | 4 | `Vol_LogChg`, `Vol_Ratio_5d`, `Vol_Ratio_20d`, `OBV_Pct` |
| MA distance (%) | 3 | `Price_MA5_Pct`, `Price_MA20_Pct`, `Price_MA50_Pct` |
| Volatility regime | 4 | `Vol_5d`, `Vol_20d`, `Vol_Ratio`, `ATR_14` |
| Momentum/oscillators | 6 | `RSI_14`, `MACD_Hist`, `BB_Width`, `BB_Pos` (+ Momentum above) |
| Support/resistance | 3 | `Dist_Resistance`, `Dist_Support`, `Price_Position` |
| Calendar (cyclical) | 4 | `DOW_sin`, `DOW_cos`, `Month_sin`, `Month_cos` |

Multicollinearity: ~16 pairs with \|r\| > 0.85 (vs 100+ in v1).

---

## Models

### Linear (6 models)

| Model | Key feature |
|-------|------------|
| OLS Baseline | No regularisation |
| RidgeCV | L2, alpha auto-tuned via TimeSeriesSplit |
| LassoCV | L1, alpha auto-tuned, automatic feature selection |
| ElasticNetCV | L1+L2, l1_ratio and alpha auto-tuned |
| BayesianRidge | Probabilistic — provides prediction intervals |
| HuberRegressor | Robust to log-return spike outliers |

### Tree-Based (6 models)

| Model | Regularisation |
|-------|---------------|
| Decision Tree | max_depth=5, min_samples_leaf=30 |
| Random Forest | max_depth=5, max_features=0.5 |
| Gradient Boosting | learning_rate=0.05, subsample=0.8 |
| XGBoost | Early stopping on temporal inner-val |
| LightGBM | Early stopping on temporal inner-val |
| CatBoost | Early stopping on temporal inner-val |

### Ensembles (4 methods)

| Method | Description |
|--------|-------------|
| Voting | Equal-weight average of base learners |
| Stacking | TimeSeriesSplit CV + RidgeCV meta-learner |
| Blending | Temporal 70/30 inner split |
| CV-Weighted | Weights derived from CV R² (not test R²) |

### Neural Network (optional, requires TensorFlow)

Dense network: 128 → 64 → 32 → 1 (linear output)
BatchNorm + Dropout + L2 regularisation, EarlyStopping on temporal inner-val.

---

## Results

### AAPL Model Comparison (5-day log return target)

| Rank | Model | Test R² | RMSE | Dir Acc |
|:----:|-------|:-------:|:----:|:-------:|
| 1 | LassoCV | ~0.001–0.005 | ~0.037 | ~55% |
| 2 | ElasticNetCV | similar | similar | ~54% |
| 3 | BayesianRidge | similar | similar | ~54% |
| … | (tree-based) | near 0 | slightly higher | ~52–54% |

> **Key insight (v2):** With a stationary log-return target, all models have modest
> but honest R² values (~0–0.01). Tree models no longer have catastrophically
> negative R² as in v1. Directional accuracy of 52–56% is barely above random —
> models capture autocorrelation and volatility regimes, not alpha-generating signals.

### Cross-Stock (LassoCV)

| Stock | Sector | Test R² | Dir Acc |
|:-----:|--------|:-------:|:-------:|
| AAPL | Technology | ~0.001–0.005 | ~55% |
| MSFT | Technology | ~0.001–0.005 | ~54% |
| JPM | Financial | ~0.001–0.005 | ~53% |
| JNJ | Healthcare | ~0.001–0.005 | ~54% |
| XOM | Energy | ~0.001–0.005 | ~52% |

> Exact values vary per run due to auto-tuned alpha and market conditions.

---

## Methodology

### Target (v2)
```python
Target = log(Close_{t+5} / Close_t)   # 5-day log return
```
Stationary and scale-invariant — works for all price levels without normalisation.

### Walk-forward validation
```python
TimeSeriesSplit(n_splits=5)  # used for alpha CV and StackingRegressor
```
Embargo gap = PREDICTION_HORIZON (5 days) between train and test folds to prevent label leakage.

### Scaling
```python
RobustScaler()   # fit on train only, applied to test
```
Outlier-resistant — crucial for the 2020 COVID crash period.

### Trading backtest
```python
signals = sign(y_pred)         # +1 = long, -1 = short
strategy_ret = signals * y_true - position_change * 0.001
sharpe = mean(strategy_ret) / std(strategy_ret) * sqrt(252)
```

### Package API

```
stock_prediction.config           — SEED, STOCKS, PREDICTION_HORIZON, etc.
stock_prediction.data.loader      — download_stocks(), download_or_load()
stock_prediction.features.engineer — engineer_features(), prepare_xy()
stock_prediction.models.evaluate  — evaluate_model(), ModelMetrics, build_comparison_table()
stock_prediction.models.train     — build_linear_models(), build_tree_models(),
                                    train_pipeline(), run_all_stocks(), backtest()
stock_prediction.visualization.plots — plot_model_comparison(), plot_residuals(),
                                       plot_coef_heatmap(), plot_backtest()
```

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

## Honest Caveats

- **Directional accuracy ~52–56%** — barely above random; position sizing matters more than timing.
- Models capture **autocorrelation and volatility regimes**, not alpha-generating signals.
- No news, earnings, macro, or sentiment data incorporated.
- Transaction costs simplified (fixed 10 bps round-trip).
- Past performance on these tickers is not indicative of future results.

---

## License

This project is licensed under the [MIT License](LICENSE).
