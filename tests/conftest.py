"""
Shared pytest fixtures.

All fixtures create lightweight synthetic data so tests run quickly
without hitting the network or requiring a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


TICKER = "TEST"
N_ROWS = 400   # enough for 200-day MA + target shift + train/test


@pytest.fixture(scope="session")
def raw_ohlcv() -> pd.DataFrame:
    """Synthetic flat OHLCV DataFrame mimicking yfinance output."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2015-01-01", periods=N_ROWS)

    # Random-walk close price starting at $100
    log_returns = rng.normal(0.0003, 0.012, size=N_ROWS)
    close = 100 * np.exp(np.cumsum(log_returns))

    noise = lambda scale: rng.uniform(1, 1 + scale, N_ROWS)

    df = pd.DataFrame(
        {
            f"{TICKER}_Close":  close,
            f"{TICKER}_High":   close * noise(0.02),
            f"{TICKER}_Low":    close / noise(0.02),
            f"{TICKER}_Open":   close * rng.uniform(0.99, 1.01, N_ROWS),
            f"{TICKER}_Volume": rng.integers(1_000_000, 10_000_000, size=N_ROWS).astype(float),
        },
        index=dates,
    )
    return df


@pytest.fixture(scope="session")
def engineered_df(raw_ohlcv) -> pd.DataFrame:
    """Pre-engineered DataFrame (expensive rolling — computed once)."""
    import sys
    sys.path.insert(0, "src")
    from stock_prediction.features.engineer import engineer_features
    return engineer_features(raw_ohlcv, TICKER)


@pytest.fixture(scope="session")
def xy(engineered_df) -> tuple:
    """(X, y, feature_cols) ready for modelling."""
    import sys
    sys.path.insert(0, "src")
    from stock_prediction.features.engineer import prepare_xy
    return prepare_xy(engineered_df, TICKER)
