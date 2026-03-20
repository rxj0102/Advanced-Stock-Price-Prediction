"""
Download historical OHLCV data from Yahoo Finance via yfinance.

Features from adv_model_compare_v2.ipynb:
  - Local parquet caching to avoid re-downloading on every run
  - Flat single-level columns: Open, High, Low, Close, Volume
  - Data quality check for long NaN runs
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
import yfinance as yf

from stock_prediction.config import (
    BENCHMARK,
    CACHE_DIR,
    END_DATE,
    START_DATE,
    STOCKS,
)

logger = logging.getLogger(__name__)


def download_or_load(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """Load from local parquet cache if available; otherwise download and cache.

    Parameters
    ----------
    ticker:
        Ticker symbol (e.g. ``"AAPL"`` or ``"^GSPC"``).
    start, end:
        Date strings accepted by yfinance (``"YYYY-MM-DD"``).
    cache_dir:
        Directory for parquet files.

    Returns
    -------
    pd.DataFrame
        Flat-column OHLCV DataFrame (Open, High, Low, Close, Volume).
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = ticker.replace("^", "")
    cache_path = os.path.join(cache_dir, f"{safe_name}.parquet")

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        covers_start = df.index.min() <= pd.Timestamp(start)
        covers_end = df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=10)
        if covers_start and covers_end:
            logger.info("%s: loaded from cache (%d rows)", ticker, len(df))
            return df

    logger.info("%s: downloading from Yahoo Finance...", ticker)
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns produced by yfinance >= 0.2
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw.to_parquet(cache_path)
    logger.info("%s: downloaded & cached (%d rows)", ticker, len(raw))
    return raw


def download_stocks(
    tickers: Optional[dict[str, str]] = None,
    start: str = START_DATE,
    end: str = END_DATE,
    cache_dir: str = CACHE_DIR,
) -> dict[str, pd.DataFrame]:
    """Download (or load from cache) OHLCV data for each ticker.

    Parameters
    ----------
    tickers:
        Mapping of ``{ticker: sector}``.  Defaults to :data:`config.STOCKS`.
    start, end:
        Date strings accepted by yfinance.
    cache_dir:
        Directory for parquet cache files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are ticker symbols; values are flat OHLCV DataFrames.
    """
    if tickers is None:
        tickers = STOCKS

    logger.info(
        "Fetching %d stocks [%s → %s]", len(tickers), start, end
    )

    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        try:
            df = download_or_load(ticker, start=start, end=end, cache_dir=cache_dir)
            # Data quality check: flag runs of >= 5 consecutive NaN close prices
            nan_runs = (
                df["Close"]
                .isna()
                .astype(int)
                .groupby((df["Close"].notna()).cumsum())
                .sum()
            )
            if nan_runs.max() >= 5:
                logger.warning(
                    "%s: long NaN run detected (max %d) — check source data",
                    ticker,
                    nan_runs.max(),
                )
            result[ticker] = df
        except Exception as exc:
            logger.error("Error fetching %s: %s", ticker, exc)
            failed.append(ticker)

    if failed:
        logger.warning("Failed downloads: %s", failed)

    logger.info("Done. %d/%d tickers ready.", len(result), len(tickers))
    return result


def download_benchmark(
    start: str = START_DATE,
    end: str = END_DATE,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame | None:
    """Download or load the S&P 500 benchmark (^GSPC)."""
    try:
        return download_or_load(BENCHMARK, start=start, end=end, cache_dir=cache_dir)
    except Exception as exc:
        logger.warning("Benchmark unavailable: %s", exc)
        return None
