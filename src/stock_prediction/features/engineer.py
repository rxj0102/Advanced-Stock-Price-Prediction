"""
Technical indicator feature engineering.

All features are derived solely from *past* data so there is no lookahead
bias.  The target column (``Target``) is the closing price
``PREDICTION_HORIZON`` trading days *ahead*, created via a forward shift and
dropped from ``X``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stock_prediction.config import (
    MA_WINDOWS,
    LAG_WINDOWS,
    PREDICTION_HORIZON,
)

logger = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    ticker: str,
    prediction_horizon: int = PREDICTION_HORIZON,
) -> pd.DataFrame:
    """Add all technical indicators and the forward-return target.

    Parameters
    ----------
    df:
        Flat OHLCV DataFrame with columns ``<TICKER>_Close``, etc.
        (as returned by :func:`data.loader.download_stocks`).
    ticker:
        Ticker symbol used to identify the price columns.
    prediction_horizon:
        Number of trading days ahead to predict.

    Returns
    -------
    pd.DataFrame
        Original columns **plus** engineered features and a ``Target``
        column.  Rows with any ``NaN`` are dropped.
    """
    close  = f"{ticker}_Close"
    high   = f"{ticker}_High"
    low    = f"{ticker}_Low"
    open_  = f"{ticker}_Open"
    volume = f"{ticker}_Volume"

    out = df.copy()

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------
    out["Target"] = out[close].shift(-prediction_horizon)

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------
    out["Return"]     = out[close].pct_change()
    out["Log_Return"] = np.log(out[close] / out[close].shift(1))

    # ------------------------------------------------------------------
    # Moving averages & rolling std
    # ------------------------------------------------------------------
    for w in MA_WINDOWS:
        out[f"MA_{w}"]  = out[close].rolling(w).mean()
        out[f"Std_{w}"] = out[close].rolling(w).std()

    # ------------------------------------------------------------------
    # MA crossovers
    # ------------------------------------------------------------------
    out["MA_5_20_Crossover"]  = out["MA_5"]  - out["MA_20"]
    out["MA_20_50_Crossover"] = out["MA_20"] - out["MA_50"]
    out["MA_50_200_Crossover"]= out["MA_50"] - out["MA_200"]

    # ------------------------------------------------------------------
    # Price-to-MA ratios
    # ------------------------------------------------------------------
    for w in [20, 50, 200]:
        out[f"Price_MA{w}_Ratio"] = out[close] / out[f"MA_{w}"]

    # ------------------------------------------------------------------
    # Volatility (return-based)
    # ------------------------------------------------------------------
    out["Volatility_20"] = out["Return"].rolling(20).std()
    out["Volatility_50"] = out["Return"].rolling(50).std()

    # ------------------------------------------------------------------
    # Intra-day features
    # ------------------------------------------------------------------
    out["High_Low_Range"]  = (out[high] - out[low]) / out[close]
    out["Close_Open_Gap"]  = (out[close] - out[open_]) / out[open_]

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------
    vol_ma20 = out[volume].rolling(20).mean()
    out["Volume_Ratio"]       = out[volume] / vol_ma20
    out["Volume_Price_Trend"] = out["Volume_Ratio"] * out["Return"]

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------
    for lag in LAG_WINDOWS:
        out[f"Price_Lag_{lag}"]  = out[close].shift(lag)
        out[f"Return_Lag_{lag}"] = out["Return"].shift(lag)

    # ------------------------------------------------------------------
    # Rolling range statistics
    # ------------------------------------------------------------------
    roll_max = out[close].rolling(20).max()
    roll_min = out[close].rolling(20).min()
    out["Rolling_Max_20"]  = roll_max
    out["Rolling_Min_20"]  = roll_min
    out["Price_Position"]  = (out[close] - roll_min) / (roll_max - roll_min)

    # ------------------------------------------------------------------
    # Momentum & rate-of-change
    # ------------------------------------------------------------------
    out["Momentum_5"]  = out[close] - out[close].shift(5)
    out["Momentum_20"] = out[close] - out[close].shift(20)
    out["ROC_5"]       = (out[close] - out[close].shift(5))  / out[close].shift(5)
    out["ROC_20"]      = (out[close] - out[close].shift(20)) / out[close].shift(20)

    # ------------------------------------------------------------------
    # Drop NaN rows (rolling look-back + target shift)
    # ------------------------------------------------------------------
    before = len(out)
    out = out.dropna()
    logger.debug(
        "%s: %d → %d rows after dropna (removed %d)",
        ticker, before, len(out), before - len(out),
    )

    return out


def prepare_xy(
    df: pd.DataFrame,
    ticker: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split the engineered DataFrame into feature matrix X and target y.

    Raw OHLCV columns and the current-period return are excluded from ``X``
    to prevent lookahead.

    Parameters
    ----------
    df:
        Output of :func:`engineer_features`.
    ticker:
        Used to identify the raw OHLCV column names to exclude.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (no NaN).
    y : pd.Series
        Target series (forward price).
    feature_cols : list[str]
        Ordered list of column names in ``X``.
    """
    exclude = {
        "Target",
        f"{ticker}_Close",
        f"{ticker}_High",
        f"{ticker}_Low",
        f"{ticker}_Open",
        f"{ticker}_Volume",
        "Return",
        "Log_Return",
    }
    feature_cols = [c for c in df.columns if c not in exclude]
    return df[feature_cols], df["Target"], feature_cols
