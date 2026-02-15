# Advanced-Stock-Price-Prediction
Advanced Stock Price Prediction: A Comparative Study of Machine Learning Models with Technical Indicators
📝 Description

This project presents an in-depth comparative analysis of various machine learning algorithms for predicting stock prices 5 days ahead. Using historical data from 5 major stocks across different sectors (Technology, Financial, Healthcare, and Energy), we engineered 42 technical features including moving averages, volatility measures, momentum indicators, and price patterns.

# 📈 Stock Price Prediction with Machine Learning: Multi-Model Comparative Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project presents an **in-depth comparative analysis of 12+ machine learning algorithms** for predicting stock prices 5 days ahead. Using historical data from **5 major stocks** across different sectors (Technology, Financial, Healthcare, and Energy), we engineered 42 technical features and evaluated models ranging from simple linear regression to advanced ensemble methods.

### 🎯 Key Features
- **Multi-Stock Analysis**: AAPL, MSFT, JPM, JNJ, XOM (2008-2024)
- **16 Years of Historical Data**: 4,027 trading days per stock
- **42 Engineered Features**: Technical indicators, lag features, rolling statistics
- **12+ ML Models Compared**: Linear models, tree-based models, ensembles, neural networks
- **Time-Based Validation**: Strict temporal train/test split (80/20) to prevent lookahead bias

---

## 📊 Model Performance Comparison

### Overall Model Rankings (All Stocks Average)

| Rank | Model | R² Score | RMSE | MAE | MAPE | Directional Accuracy |
|:----:|-------|:--------:|:----:|:---:|:----:|:--------------------:|
| **1** | **Lasso Regression** | **0.9118** | **$5.99** | **$4.65** | **3.10%** | **53.66%** |
| 2 | ElasticNet | 0.9110 | $6.01 | $4.68 | 3.12% | 50.39% |
| 3 | Linear Regression | 0.9029 | $6.28 | $4.90 | 3.27% | 52.62% |
| 4 | Ridge Regression | 0.9013 | $6.34 | $4.96 | 3.31% | 52.62% |
| 5 | Voting Ensemble | 0.9060 | $6.18 | $4.82 | 3.22% | 52.88% |
| 6 | Stacking Ensemble | 0.8914 | $6.64 | $5.16 | 3.45% | 52.75% |
| 7 | Neural Network (MLP) | -1.8131 | $33.81 | $31.77 | 20.89% | 49.48% |
| 8 | Decision Tree | -3.2603 | $41.61 | $36.28 | 22.40% | 47.64% |
| 9 | Gradient Boosting | -3.9902 | $45.04 | $40.52 | 25.27% | 53.14% |
| 10 | Random Forest | -4.1259 | $45.64 | $41.32 | 25.83% | 51.18% |
| 11 | XGBoost | -5.7477 | $52.37 | $48.53 | 30.65% | 51.96% |
| 12 | SVR | -12.3763 | $73.73 | $65.67 | 40.97% | 49.48% |

---

### 📈 Cross-Stock Performance (Lasso Model)

| Stock | Sector | R² Score | RMSE | MAE | RMSE/Price | Directional Accuracy |
|:-----:|--------|:--------:|:----:|:---:|:----------:|:--------------------:|
| **XOM** | Energy | **0.9769** | **$3.42** | **$2.66** | **4.54%** | 49.74% |
| **AAPL** | Technology | **0.9152** | **$5.87** | **$4.56** | **3.84%** | **53.14%** |
| **JPM** | Financial | **0.8882** | **$4.81** | **$3.80** | **3.70%** | 51.83% |
| MSFT | Technology | 0.8604 | $15.29 | $11.49 | 5.51% | 51.44% |
| JNJ | Healthcare | 0.7736 | $3.39 | $2.74 | 2.27% | **54.32%** |

**Average R² Across All Stocks:** `0.8829`

---

### 🔍 Top 10 Most Important Features (Lasso Regression)

| Rank | Feature | Description | Coefficient Impact |
|:----:|---------|-------------|:------------------:|
| 1 | `Price_Lag_5` | Price 5 days ago | **Positive (+10.89)** |
| 2 | `MA_5` | 5-day moving average | Positive (+6.21) |
| 3 | `MA_10` | 10-day moving average | Positive (+1.73) |
| 4 | `MA_20` | 20-day moving average | Positive (+1.68) |
| 5 | `MA_200` | 200-day moving average | Positive (+1.25) |
| 6 | `Momentum_5` | 5-day momentum | Positive (+1.07) |
| 7 | `Price_Lag_1` | Previous day's price | Positive (+0.98) |
| 8 | `MA_50` | 50-day moving average | Positive (+0.67) |
| 9 | `Std_10` | 10-day volatility | Negative (-0.45) |
| 10 | `Std_20` | 20-day volatility | Positive (+0.34) |

**Features with zero coefficient (removed by Lasso):** `17` (40.5% of features)

---

### 🎯 Prediction Accuracy Bands (AAPL)

| Accuracy Threshold | % of Predictions Within Range |
|:------------------:|:-----------------------------:|
| ±1% | 21.0% |
| ±2% | 42.1% |
| ±3% | 59.2% |
| ±5% | 80.1% |
| ±10% | 98.0% |

---

### 📊 Sector-wise Predictability

| Sector | Average R² | Average RMSE | Predictability |
|--------|:----------:|:------------:|:--------------:|
| Energy | 0.9769 | $3.42 | 🟢 **Highest** |
| Financial | 0.8882 | $4.81 | 🟡 High |
| Technology | 0.8878 | $10.58 | 🟡 Medium |
| Healthcare | 0.7736 | $3.39 | 🔴 Lowest |

---

### ⚡ Model Training Summary

| Metric | Value |
|--------|-------|
| **Total Stocks Analyzed** | 5 |
| **Time Period** | 2008-01-01 to 2024-01-01 (~16 years) |
| **Total Data Points** | 19,115 |
| **Features per Stock** | 36 |
| **Prediction Horizon** | 5 trading days |
| **Best Algorithm** | Lasso Regression (α=0.01) |
| **Average R²** | 0.8829 |
| **Average RMSE** | $6.56 |
| **Average Directional Accuracy** | 52.09% |

---

## 🛠️ Technical Implementation

### Feature Engineering

# Key engineered features
features = {
    'Moving Averages': ['MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_100', 'MA_200'],
    'Volatility': ['Std_5', 'Std_10', 'Std_20', 'Std_50', 'Std_100', 'Std_200'],
    'Crossovers': ['MA_5_20_Crossover', 'MA_20_50_Crossover', 'MA_50_200_Crossover'],
    'Ratios': ['Price_MA20_Ratio', 'Price_MA50_Ratio', 'Price_MA200_Ratio'],
    'Price Features': ['High_Low_Range', 'Close_Open_Gap'],
    'Volume': ['Volume_Ratio', 'Volume_Price_Trend'],
    'Lag Features': ['Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_5', 'Price_Lag_10'],
    'Momentum': ['Momentum_5', 'Momentum_20', 'ROC_5', 'ROC_20'],
    'Rolling Stats': ['Rolling_Max_20', 'Rolling_Min_20', 'Price_Position']
}
