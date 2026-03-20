"""
Stationary, returns-based feature engineering.

Directly based on adv_model_compare_v2.ipynb Cell 3.

Key design choices (vs v1):
  - Target  = 5-day log return (stationary, scale-invariant)
  - Features = returns-based only — no raw price levels → no multicollinearity
  - Calendar = cyclical sine/cosine encoding (not raw ordinal integers)
  - 34 features across 8 categories
  - Multicollinearity reduced from 100+ pairs (v1) to ~16 pairs
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
        Flat-column OHLCV DataFrame with columns Open, High, Low, Close, Volume.
    ticker:
        Ticker symbol (used for logging only; all features are generic).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Target`` column + 34 engineered features.
        Rows with any NaN are dropped.
    """
    out = pd.DataFrame(index=df.index)
    c = df["Close"]

    # ── Target: log return over the prediction horizon (STATIONARY) ──────────
    # log(C_{t+h} / C_t) — forward-shifted so there is no leakage
    out["Target"] = np.log(c.shift(-PREDICTION_HORIZON) / c)

    # ── 1. Multi-period log returns (all lagged, no current-day leakage) ─────
    for n in [1, 2, 3, 5, 10, 20]:
        out[f"LogRet_{n}d"] = np.log(c / c.shift(n))

    # ── 2. Intraday microstructure (normalised — scale-invariant) ────────────
    out["HL_Range"]       = (df["High"] - df["Low"]) / c
    out["Close_Open_Pct"] = (c - df["Open"]) / df["Open"]
    out["Upper_Shadow"]   = (df["High"] - np.maximum(c, df["Open"])) / c
    out["Lower_Shadow"]   = (np.minimum(c, df["Open"]) - df["Low"]) / c

    # ── 3. Volume (normalised) ────────────────────────────────────────────────
    vol = df["Volume"]
    out["Vol_LogChg"]    = np.log(vol / vol.shift(1))
    out["Vol_Ratio_5d"]  = vol / vol.rolling(5).mean()
    out["Vol_Ratio_20d"] = vol / vol.rolling(20).mean()
    # Signed volume: positive on up days, negative on down days
    out["OBV_Pct"] = (
        (vol * np.sign(out["LogRet_1d"])).rolling(10).mean()
        / vol.rolling(10).mean()
    )

    # ── 4. Trend: price distance from MAs (percent — scale-invariant) ────────
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

    # ── 6. Momentum / oscillators ─────────────────────────────────────────────
    out["Momentum_5d"]  = np.log(c / c.shift(5))
    out["Momentum_20d"] = np.log(c / c.shift(20))

    # RSI (14-day) — hand-rolled, no ta-lib dependency
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-12)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD histogram (12/26/9 EMA, price-normalised)
    ema12       = c.ewm(span=12, adjust=False).mean()
    ema26       = c.ewm(span=26, adjust=False).mean()
    macd_line   = (ema12 - ema26) / c
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = macd_line - signal_line

    # Bollinger Bands (normalised)
    bb_mean     = c.rolling(20).mean()
    bb_std      = c.rolling(20).std()
    out["BB_Width"] = (2 * bb_std) / bb_mean
    out["BB_Pos"]   = (c - (bb_mean - 2 * bb_std)) / (4 * bb_std + 1e-12)

    # ── 7. Support / resistance (distance, normalised) ───────────────────────
    high20 = df["High"].rolling(20).max()
    low20  = df["Low"].rolling(20).min()
    out["Dist_Resistance"] = (high20 - c) / c
    out["Dist_Support"]    = (c - low20) / c
    out["Price_Position"]  = (c - low20) / (high20 - low20 + 1e-12)

    # ── 8. Calendar features — CYCLICAL encoding ─────────────────────────────
    dow = out.index.dayofweek
    mth = out.index.month
    out["DOW_sin"]   = np.sin(2 * np.pi * dow / 5)
    out["DOW_cos"]   = np.cos(2 * np.pi * dow / 5)
    out["Month_sin"] = np.sin(2 * np.pi * mth / 12)
    out["Month_cos"] = np.cos(2 * np.pi * mth / 12)

    before = len(out)
    out = out.dropna().copy()
    logger.debug(
        "%s: %d → %d rows after dropna (removed %d)",
        ticker,
        before,
        len(out),
        before - len(out),
    )

    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (all columns except ``Target``)."""
    return [c for c in df.columns if c != "Target"]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split engineered DataFrame into feature matrix X and target y.

    Parameters
    ----------
    df:
        Output of :func:`engineer_features`.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target series (5-day log return).
    feature_cols : list[str]
        Ordered list of feature column names.
    """
    feature_cols = get_feature_cols(df)
    return df[feature_cols], df["Target"], feature_cols
