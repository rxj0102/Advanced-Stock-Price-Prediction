"""
stock_prediction — Advanced stock price prediction library (v2).

Based on adv_model_compare_v2.ipynb:
  - Stationary 5-day log return target
  - 34 returns-based features (no raw price levels)
  - RobustScaler + TimeSeriesSplit throughout
  - Auto-tuned LassoCV / RidgeCV / ElasticNetCV
  - Ensemble methods: Voting, Stacking (TS-CV), Blending, CV-Weighted
  - Trading backtest with Sharpe ratio and max drawdown

Modules
-------
config                    : Global constants and configuration
data.loader               : yfinance download with parquet caching
features.engineer         : 34 return-based feature engineering
models.evaluate           : Metrics with directional accuracy + significance tests
models.train              : Model training, ensembles, multi-stock pipeline, backtest
visualization.plots       : Model comparison, residual, and backtest charts
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("stock-prediction")
except PackageNotFoundError:
    __version__ = "0.2.0"
