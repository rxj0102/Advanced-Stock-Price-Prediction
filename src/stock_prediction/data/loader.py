"""
Download historical OHLCV data from Yahoo Finance via yfinance.

The returned DataFrames use a flat single-level column index of the form
``<TICKER>_<Field>`` (e.g. ``AAPL_Close``), which is compatible with the
rest of the pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from stock_prediction.config import START_DATE, END_DATE, STOCKS

logger = logging.getLogger(__name__)


def download_stocks(
    tickers: Optional[dict[str, str]] = None,
    start: str = START_DATE,
    end: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for each ticker and return flat DataFrames.

    Parameters
    ----------
    tickers:
        Mapping of ``{ticker: sector}``.  Defaults to :data:`config.STOCKS`.
    start, end:
        Date strings accepted by ``yfinance`` (``"YYYY-MM-DD"``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are ticker symbols; values are DataFrames with columns
        ``<TICKER>_Close``, ``<TICKER>_High``, etc.
    """
    if tickers is None:
        tickers = STOCKS

    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)

    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False)
            if raw.empty:
                logger.warning("No data returned for %s — skipping", ticker)
                failed.append(ticker)
                continue

            # Flatten MultiIndex columns  →  AAPL_Close, AAPL_High, …
            flat = raw.copy()
            flat.columns = [f"{col[1]}_{col[0]}" for col in flat.columns]
            result[ticker] = flat
            logger.info("  %s: %d trading days", ticker, len(flat))

        except Exception as exc:  # noqa: BLE001
            logger.error("Error downloading %s: %s", ticker, exc)
            failed.append(ticker)

    if failed:
        logger.warning("Failed downloads: %s", failed)

    logger.info("Download complete. %d/%d tickers successful.", len(result), len(tickers))
    return result
