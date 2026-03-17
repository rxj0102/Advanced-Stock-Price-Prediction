# Stock Price Prediction — Multi-Model Comparative Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red.svg)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous comparative study of **12+ machine learning algorithms** for predicting stock prices 5 trading days ahead. Covers 16 years of historical data (2008–2024) across 5 major stocks, 42 engineered features, and strict temporal validation to prevent lookahead bias.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
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
| **Features** | 42 engineered technical indicators |
| **Models Compared** | 12+ (linear → ensembles → neural networks) |
| **Validation** | Temporal 80/20 train/test split |

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

> **Key insight:** Linear models with regularization dramatically outperform tree-based and ensemble methods on this task. Without feature scaling normalization for tree models, raw price momentum features give linear models an inherent advantage.

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

17 of 42 features (40.5%) were zeroed out by Lasso — a natural feature selection step.

---

### AAPL Prediction Accuracy Bands

| Threshold | % of Predictions Within Range |
|:---------:|:-----------------------------:|
| ±1% | 21.0% |
| ±2% | 42.1% |
| ±3% | 59.2% |
| ±5% | 80.1% |
| ±10% | 98.0% |

---

### Sector Predictability

| Sector | Avg. R² | Avg. RMSE | Predictability |
|--------|:-------:|:---------:|:--------------:|
| Energy | 0.9769 | $3.42 | Highest |
| Financial | 0.8882 | $4.81 | High |
| Technology | 0.8878 | $10.58 | Medium |
| Healthcare | 0.7736 | $3.39 | Lowest |

---

## Project Structure

```
Advanced-Stock-Price-Prediction/
├── adv_model_compare.ipynb   # Main analysis notebook (12 cells)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/rxj0102/Advanced-Stock-Price-Prediction.git
cd Advanced-Stock-Price-Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook adv_model_compare.ipynb
```

---

## Usage

Open `adv_model_compare.ipynb` and run cells in order:

| Cell | Description |
|------|-------------|
| 1 | Import libraries |
| 2 | Download 5 stocks via `yfinance` (2008–2024) |
| 3 | Engineer 42 technical features for AAPL |
| 4 | Prepare feature matrix and temporal train/test split |
| 5 | Define evaluation function; train Linear Regression baseline |
| 6 | Train all 12 models and collect metrics |
| 7 | Train regularized models (Ridge, Lasso, ElasticNet) |
| 8 | Deep-dive on best model; build ensemble methods |
| 9 | Run full pipeline across all 5 stocks |
| 10 | Final results table and comprehensive summary |
| 11 | Advanced models and sophisticated ensembles |
| 12 | Feature engineering experiments and alternative approaches |

---

## Methodology

### Feature Engineering

42 features are engineered from raw OHLCV data:

```python
features = {
    'Moving Averages':  ['MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_100', 'MA_200'],
    'Volatility':       ['Std_5', 'Std_10', 'Std_20', 'Std_50', 'Std_100', 'Std_200'],
    'Crossovers':       ['MA_5_20_Crossover', 'MA_20_50_Crossover', 'MA_50_200_Crossover'],
    'Ratios':           ['Price_MA20_Ratio', 'Price_MA50_Ratio', 'Price_MA200_Ratio'],
    'Price Features':   ['High_Low_Range', 'Close_Open_Gap'],
    'Volume':           ['Volume_Ratio', 'Volume_Price_Trend'],
    'Lag Features':     ['Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_5', 'Price_Lag_10'],
    'Momentum':         ['Momentum_5', 'Momentum_20', 'ROC_5', 'ROC_20'],
    'Rolling Stats':    ['Rolling_Max_20', 'Rolling_Min_20', 'Price_Position'],
}
```

### Validation Strategy

- **Temporal split:** first 80% of data for training, last 20% for testing
- No shuffling — preserves time-series ordering and prevents lookahead bias
- All features are scaled with `StandardScaler` fit on training data only

### Models Evaluated

- **Linear:** Linear Regression, Ridge, Lasso (α=0.01), ElasticNet
- **Tree-based:** Decision Tree, Random Forest, Gradient Boosting, XGBoost
- **Other:** SVR, MLP Neural Network
- **Ensembles:** Voting Regressor, Stacking Regressor

---

## Key Findings

1. **Regularized linear models win.** Lasso (R²=0.91) and ElasticNet beat all tree-based and neural network models by a wide margin. Short-lag price features dominate, rewarding linear extrapolation.

2. **Tree-based models fail without target-relative scaling.** Negative R² scores indicate predictions worse than the mean — these models require raw price prediction without proper normalization.

3. **Lasso provides automatic feature selection.** 17/42 features are zeroed out, leaving a sparse and interpretable model.

4. **Energy stocks are most predictable** (XOM R²=0.977). Healthcare is hardest (JNJ R²=0.774), likely due to idiosyncratic regulatory/clinical news events.

5. **Directional accuracy hovers near 50–54%** across all models — price level prediction is strong, but direction is harder and closer to random.

6. **Short-term lag features dominate** — `Price_Lag_5` and `MA_5` are by far the most impactful features, confirming strong short-term autocorrelation in stock prices.

---

## License

This project is licensed under the [MIT License](LICENSE).
