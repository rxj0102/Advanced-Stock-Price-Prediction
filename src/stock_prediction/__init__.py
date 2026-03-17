"""
stock_prediction — Multi-model stock price prediction library.

Modules
-------
config       : Global constants and configuration
data.loader  : yfinance-based data download utilities
features.engineer : Technical indicator feature engineering
models.evaluate   : Evaluation metrics and reporting
models.train      : Model training and cross-stock pipeline
visualization.plots : Prediction and comparison charts
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("stock-prediction")
except PackageNotFoundError:
    __version__ = "0.1.0"
