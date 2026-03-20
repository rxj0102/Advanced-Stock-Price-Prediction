"""
Stationary, returns-based feature engineering.

Target
------
``Target = log(Close_{t+5} / Close_t)`` — 5-day log return.
Stationary and scale-invariant; suitable for all price levels.

Feature design
--------------
All 34 features are returns-based or normalised by price so they are
scale-invariant and generalisable across tickers without additional
transformation. Raw price levels are intentionally excluded to avoid
multicollinearity. Calendar features use cyclical sine/cosine encoding
instead of raw ordinal integers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stock_prediction.config import PREDICTION_HORIZON

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build a stationary, returns-based feature matrix from OHLCV data.

    Parameters
    ----------
    df:
        Flat OHLCV DataFrame with columns Open, High, Low, Close, Volume.
    ticker:
        Ticker symbol used for logging only; all features are generic.

    Returns
    -------
    pd.DataFrame
        ``Target`` column plus 34 engineered features. Rows with any NaN
        (from rolling windows and the forward-shifted target) are dropped.
    """
    out = pd.DataFrame(index=df.index)
    c   = df["Close"]

    # ── Target: 5-day log return (forward-shifted, no leakage) ───────────────
    out["Target"] = np.log(c.shift(-PREDICTION_HORIZON) / c)

    # ── 1. Multi-period log returns ───────────────────────────────────────────
    for n in [1, 2, 3, 5, 10, 20]:
        out[f"LogRet_{n}d"] = np.log(c / c.shift(n))

    # ── 2. Intraday microstructure (price-normalised) ─────────────────────────
    out["HL_Range"]       = (df["High"] - df["Low"]) / c
    out["Close_Open_Pct"] = (c - df["Open"]) / df["Open"]
    out["Upper_Shadow"]   = (df["High"] - np.maximum(c, df["Open"])) / c
    out["Lower_Shadow"]   = (np.minimum(c, df["Open"]) - df["Low"])  / c

    # ── 3. Volume (price- and volume-normalised) ──────────────────────────────
    vol = df["Volume"]
    out["Vol_LogChg"]    = np.log(vol / vol.shift(1))
    out["Vol_Ratio_5d"]  = vol / vol.rolling(5).mean()
    out["Vol_Ratio_20d"] = vol / vol.rolling(20).mean()
    out["OBV_Pct"] = (
        (vol * np.sign(out["LogRet_1d"])).rolling(10).mean()
        / vol.rolling(10).mean()
    )

    # ── 4. Price distance from moving averages (%) ────────────────────────────
    for w in [5, 20, 50]:
        ma = c.rolling(w).mean()
        out[f"Price_MA{w}_Pct"] = (c - ma) / ma

    # ── 5. Volatility regime ──────────────────────────────────────────────────
    ret1 = out["LogRet_1d"]
    out["Vol_5d"]    = ret1.rolling(5).std()
    out["Vol_20d"]   = ret1.rolling(20).std()
    out["Vol_Ratio"] = out["Vol_5d"] / (out["Vol_20d"] + 1e-12)
    out["ATR_14"] = (
        pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - c.shift(1)).abs(),
                (df["Low"]  - c.shift(1)).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(14)
        .mean()
        / c
    )

    # ── 6. Momentum and oscillators ───────────────────────────────────────────
    out["Momentum_5d"]  = np.log(c / c.shift(5))
    out["Momentum_20d"] = np.log(c / c.shift(20))

    # RSI (14-day) — hand-rolled
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    out["RSI_14"] = 100 - (100 / (1 + gain / (loss + 1e-12)))

    # MACD histogram (12/26/9 EMA, price-normalised)
    ema12       = c.ewm(span=12, adjust=False).mean()
    ema26       = c.ewm(span=26, adjust=False).mean()
    macd_line   = (ema12 - ema26) / c
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = macd_line - signal_line

    # Bollinger Bands (20-day, price-normalised)
    bb_mean     = c.rolling(20).mean()
    bb_std      = c.rolling(20).std()
    out["BB_Width"] = (2 * bb_std) / bb_mean
    out["BB_Pos"]   = (c - (bb_mean - 2 * bb_std)) / (4 * bb_std + 1e-12)

    # ── 7. Support and resistance (price-normalised) ──────────────────────────
    high20 = df["High"].rolling(20).max()
    low20  = df["Low"].rolling(20).min()
    out["Dist_Resistance"] = (high20 - c) / c
    out["Dist_Support"]    = (c - low20)  / c
    out["Price_Position"]  = (c - low20)  / (high20 - low20 + 1e-12)

    # ── 8. Calendar (cyclical sin/cos encoding) ───────────────────────────────
    dow = out.index.dayofweek
    mth = out.index.month
    out["DOW_sin"]   = np.sin(2 * np.pi * dow / 5)
    out["DOW_cos"]   = np.cos(2 * np.pi * dow / 5)
    out["Month_sin"] = np.sin(2 * np.pi * mth / 12)
    out["Month_cos"] = np.cos(2 * np.pi * mth / 12)

    before = len(out)
    out    = out.dropna().copy()
    logger.debug(
        "%s: %d → %d rows after dropna (removed %d)",
        ticker, before, len(out), before - len(out),
    )
    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (all columns except ``Target``)."""
    return [c for c in df.columns if c != "Target"]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split an engineered DataFrame into feature matrix X and target y.

    Parameters
    ----------
    df:
        Output of :func:`engineer_features`.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (no NaN).
    y : pd.Series
        Target series (5-day log return).
    feature_cols : list[str]
        Ordered list of feature column names.
    """
    feature_cols = get_feature_cols(df)
    return df[feature_cols], df["Target"], feature_cols
