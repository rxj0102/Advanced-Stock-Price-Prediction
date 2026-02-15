# Advanced-Stock-Price-Prediction
Advanced Stock Price Prediction: A Comparative Study of Machine Learning Models with Technical Indicators
📝 Description

This project presents an in-depth comparative analysis of various machine learning algorithms for predicting stock prices 5 days ahead. Using historical data from 5 major stocks across different sectors (Technology, Financial, Healthcare, and Energy), we engineered 42 technical features including moving averages, volatility measures, momentum indicators, and price patterns.

🔍 Key Features:

Multi-Stock Analysis: AAPL, MSFT, JPM, JNJ, XOM (2008-2024)
16 Years of Historical Data: 4,027 trading days per stock
42 Engineered Features: Technical indicators, lag features, rolling statistics
10+ ML Models Compared: Linear models, tree-based models, ensembles, neural networks
Time-Based Validation: No lookahead bias with strict temporal train/test split (80/20)
🎯 Main Findings:

Lasso Regression emerged as the best model with R² = 0.9118
Linear models significantly outperformed complex tree-based models
Simple regularized linear regression beats sophisticated ensembles for this task
Average directional accuracy: 52-55% (slightly better than random)
Feature importance: Price lags and moving averages dominate predictions
📊 Models Implemented

Category	Models
Linear Models	Linear Regression, Lasso, Ridge, ElasticNet
Tree-Based	Decision Tree, Random Forest, Gradient Boosting, XGBoost
Ensembles	Voting Regressor, Stacking, Blending, Weighted Average
Advanced	LightGBM, CatBoost, Polynomial Regression, Neural Networks
Support Vector	SVR


COMPREHENSIVE MODEL COMPARISON
Models sorted by R² Score:

               Model   RMSE    MAE       R²   MAPE Directional_Acc
    Lasso Regression  $5.99  $4.65   0.9118  3.10%          53.66%
          ElasticNet  $6.01  $4.68   0.9110  3.12%          50.39%
   Linear Regression  $6.28  $4.90   0.9029  3.27%          52.62%
    Ridge Regression  $6.34  $4.96   0.9013  3.31%          52.62%
Neural Network (MLP) $33.81 $31.77  -1.8131 20.89%          49.48%
       Decision Tree $41.61 $36.28  -3.2603 22.40%          47.64%
   Gradient Boosting $45.04 $40.52  -3.9902 25.27%          53.14%
       Random Forest $45.64 $41.32  -4.1259 25.83%          51.18%
             XGBoost $52.37 $48.53  -5.7477 30.65%          51.96%
                 SVR $73.73 $65.67 -12.3763 40.97%          49.48%

🏆 BEST PERFORMING MODEL: Lasso Regression
   R² Score: 0.9118
   RMSE: $5.99

📊 Cross-Stock Performance

Stock	Sector	R²	RMSE	RMSE/Price
XOM	Energy	0.9769	$3.42	4.54%
AAPL	Technology	0.9152	$5.87	3.84%
JPM	Financial	0.8882	$4.81	3.70%
MSFT	Technology	0.8604	$15.29	5.51%
JNJ	Healthcare	0.7736	$3.39	2.27%

🔧 Technical Implementation

Feature Engineering:

Moving Averages: 5, 10, 20, 50, 100, 200-day windows
Volatility Measures: Rolling standard deviations
Price Ratios: High/Low ranges, Close/Open gaps
Momentum Indicators: ROC, price momentum
Lag Features: Price and return lags (1,2,3,5,10 days)
Rolling Statistics: Min/Max over 20-day windows
Key Libraries:

yfinance - Data acquisition
scikit-learn - Machine learning models
xgboost - Gradient boosting
pandas/numpy - Data manipulation
matplotlib/seaborn - Visualization

💡 Key Insights

Why Lasso Won: Regularization effectively handles multicollinearity in technical indicators
Tree Models Overfit: Complex models capture noise, not signal in financial data
Feature Importance: Price lags and moving averages are most predictive
Sector Matters: Energy and Financial stocks more predictable than Tech
Limitations: Models don't account for news, earnings, or macroeconomic events
🎯 Practical Applications

Traders: Use predictions for position sizing and risk management
Quant Analysts: Framework for feature engineering and model comparison
Portfolio Managers: Sector-based performance insights for allocation decisions
