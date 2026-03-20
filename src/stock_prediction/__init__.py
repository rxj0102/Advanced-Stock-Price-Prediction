"""
stock_prediction — Advanced stock price prediction library.

Modules
-------
config                    : Global constants and configuration
data.loader               : yfinance download with local parquet caching
features.engineer         : 34 returns-based feature engineering
models.evaluate           : Metrics with directional accuracy and significance tests
models.train              : Model training, ensembles, multi-stock pipeline, backtest
visualization.plots       : Model comparison, residual, and backtest charts
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("stock-prediction")
except PackageNotFoundError:
    __version__ = "0.1.0"
