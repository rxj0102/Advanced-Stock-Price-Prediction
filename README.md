# Advanced Stock Price Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous comparative study of **12+ machine learning algorithms** for predicting **5-day log returns** of 5 major stocks. Covers 16 years of data (2008–2024), 34 returns-based technical features, and strict temporal validation to eliminate lookahead bias.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Models](#models)
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
| **Benchmark** | S&P 500 (^GSPC) |
| **Time Period** | 2008-01-01 — 2024-01-01 (16 years) |
| **Prediction Target** | 5-day log return (stationary) |
| **Features** | 34 returns-based technical indicators |
| **Models** | 12+ (linear → ensembles → neural networks) |
| **Validation** | TimeSeriesSplit (5 folds) + temporal 80/20 holdout |
| **Backtest** | Long/short strategy with Sharpe ratio & max drawdown |
| **Data Cache** | Local parquet cache — no re-download on restart |

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
│       ├── config.py              # Constants: SEED, STOCKS, dates, hyperparameters
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py          # yfinance download with local parquet caching
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineer.py        # 34 returns-based features + log-return target
│       ├── models/
│       │   ├── __init__.py
│       │   ├── evaluate.py        # Metrics: R², RMSE, dir accuracy + binomial p-value
│       │   └── train.py           # Linear/tree/ensemble models, multi-stock pipeline, backtest
│       └── visualization/
│           ├── __init__.py
│           └── plots.py           # Comparison, residual, heatmap, and backtest charts
│
├── notebooks/
│   └── analysis.ipynb             # Clean end-to-end notebook using src/
│
├── tests/
│   ├── conftest.py                # Synthetic OHLCV fixtures (no network required)
│   ├── test_features.py           # Feature engineering tests
│   ├── test_evaluate.py           # Metrics and comparison table tests
│   └── test_train.py              # Pipeline, backtest, and multi-stock tests
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

# 5. Optional: deep learning (Keras neural network)
pip install -e ".[deep]"
```

---

## Quick Start

### Run the full notebook

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

# Full single-stock pipeline: linear + tree + ensemble models on AAPL
result = train_pipeline(stock_data["AAPL"], "AAPL", verbose=True)

# Build ranked comparison table (train R², test R², overfitting gap, dir accuracy)
table = build_comparison_table({
    name: (res["train_metrics"], res["test_metrics"])
    for name, res in result["all_results"].items()
})
print(table)

# Multi-stock LassoCV pipeline (alpha auto-tuned per ticker)
all_results = run_all_stocks(stock_data)

# Trading backtest on best model
bt = backtest(result["y_test"].values,
              result["linear_results"]["LassoCV"]["predictions"])
print(f"Sharpe: {bt['Sharpe']:.2f}   Max DD: {bt['Max_DD']:.2%}")
```

---

## Features

34 features engineered from raw OHLCV — **all returns-based, no raw price levels**:

| Category | Count | Features |
|----------|:-----:|---------|
| Log returns | 8 | `LogRet_1d` … `LogRet_20d`, `Momentum_5d`, `Momentum_20d` |
| Intraday microstructure | 4 | `HL_Range`, `Close_Open_Pct`, `Upper_Shadow`, `Lower_Shadow` |
| Volume | 4 | `Vol_LogChg`, `Vol_Ratio_5d`, `Vol_Ratio_20d`, `OBV_Pct` |
| MA distance (%) | 3 | `Price_MA5_Pct`, `Price_MA20_Pct`, `Price_MA50_Pct` |
| Volatility regime | 4 | `Vol_5d`, `Vol_20d`, `Vol_Ratio`, `ATR_14` |
| Momentum / oscillators | 6 | `Momentum_5d`, `Momentum_20d`, `RSI_14`, `MACD_Hist`, `BB_Width`, `BB_Pos` |
| Support / resistance | 3 | `Dist_Resistance`, `Dist_Support`, `Price_Position` |
| Calendar (cyclical) | 4 | `DOW_sin`, `DOW_cos`, `Month_sin`, `Month_cos` |

Returns-based features are scale-invariant and work across all price levels without normalisation. Cyclical calendar encoding captures weekly and annual periodicity without ordinal bias.

---

## Models

### Linear (6 models)

| Model | Description |
|-------|-------------|
| OLS Baseline | Ordinary least squares — no regularisation |
| RidgeCV | L2 regularisation, alpha auto-tuned via TimeSeriesSplit |
| LassoCV | L1 regularisation, alpha auto-tuned, automatic feature selection |
| ElasticNetCV | L1+L2 blend, l1\_ratio and alpha auto-tuned |
| BayesianRidge | Probabilistic model — provides prediction intervals |
| HuberRegressor | Robust to extreme log-return outliers |

### Tree-Based (up to 6 models)

| Model | Regularisation |
|-------|---------------|
| Decision Tree | `max_depth=5`, `min_samples_leaf=30` |
| Random Forest | `max_depth=5`, `max_features=0.5` |
| Gradient Boosting | `learning_rate=0.05`, `subsample=0.8` |
| XGBoost *(optional)* | Early stopping on temporal inner-val |
| LightGBM *(optional)* | Early stopping on temporal inner-val |
| CatBoost *(optional)* | Early stopping on temporal inner-val |

### Ensembles (4 methods)

| Method | Description |
|--------|-------------|
| Voting | Equal-weight average of diverse base learners |
| Stacking | TimeSeriesSplit CV with RidgeCV meta-learner |
| Blending | Temporal 70/30 inner split, meta-model on OOF predictions |
| CV-Weighted | Weights derived from CV R² — no test-set information used |

### Neural Network *(optional, requires TensorFlow)*

Dense: 128 → BN → Dropout(0.3) → 64 → BN → Dropout(0.2) → 32 → 1 (linear)
L2 regularisation + EarlyStopping + ReduceLROnPlateau, validated on temporal inner-val.

---

## Methodology

### Target

```python
Target = log(Close_{t+5} / Close_t)   # 5-day log return
```

Stationary and scale-invariant — suitable for all price levels without additional normalisation.

### Data splits

```
Full data  ───────────────────────────────────────────
           │            80%             │    20%     │
           │          Training          │   Test     │
           │                            │            │
           │    90%    │   10% (inner)  │            │
           │  Training │  Validation    │            │
                       ↑                ↑
               (inner val for       (final holdout)
               early stopping)
```

### Walk-forward cross-validation

```python
TimeSeriesSplit(n_splits=5)   # with embargo gap = PREDICTION_HORIZON
```

Used for alpha selection (LassoCV/RidgeCV), StackingRegressor, and CV-Weighted ensemble weights.

### Scaling

```python
RobustScaler()   # fit on training data only, applied to test
```

Robust to extreme outliers such as the 2020 COVID volatility spike.

### Evaluation metrics

- **R²**, **RMSE**, **MAE**, **MAPE** — on log-return scale
- **Directional accuracy** — fraction of correct up/down predictions
- **Binomial p-value** — tests H₀: directional accuracy = 50% (random)
- **Overfitting gap** — Train R² − Test R² (explicit for every model)

### Trading backtest

```python
signal       = sign(y_pred)           # +1 long, -1 short
strategy_ret = signal * y_true - position_change * 0.001  # 10 bps cost
sharpe       = mean(strategy_ret) / std(strategy_ret) * sqrt(252)
```

### Package API

```
stock_prediction.config              — SEED, STOCKS, PREDICTION_HORIZON, dates
stock_prediction.data.loader         — download_stocks(), download_or_load()
stock_prediction.features.engineer   — engineer_features(), prepare_xy()
stock_prediction.models.evaluate     — evaluate_model(), ModelMetrics, build_comparison_table()
stock_prediction.models.train        — build_linear_models(), build_tree_models(),
                                       train_pipeline(), run_all_stocks(), backtest()
stock_prediction.visualization.plots — plot_model_comparison(), plot_residuals(),
                                       plot_coef_heatmap(), plot_backtest()
```

---

## Key Findings

- **Directional accuracy 52–56%** — barely above random; models capture autocorrelation and volatility regimes, not alpha-generating signals.
- **LassoCV performs automatic feature selection** — typically zeroes 30–50% of features.
- **Ensemble methods offer marginal improvement** — diversity of base learners matters more than the aggregation method.
- **BayesianRidge** provides calibrated prediction intervals useful for position sizing.
- **Tree-based models** with conservative regularisation achieve honest, near-zero R² rather than severe overfitting.

### Honest caveats

- No news, earnings, macro, or sentiment data incorporated.
- Transaction costs simplified (fixed 10 bps round-trip).
- Past performance on these tickers is not indicative of future results.
- Position sizing matters far more than directional timing at this accuracy level.

### Suggested next steps

1. Add LSTM / TCN for sequence modelling
2. Incorporate VIX, put/call ratio, earnings date proximity
3. Walk-forward re-fit with rolling 252-day training windows
4. Extend to 50+ tickers with cross-sectional ranking
5. Kelly Criterion for position sizing based on predicted return distribution

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
