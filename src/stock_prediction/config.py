"""
Central configuration for the stock prediction project (v2).

Based on adv_model_compare_v2.ipynb — stationary log-return target,
returns-based features, RobustScaler, TimeSeriesSplit throughout.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED: int = 42

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

STOCKS: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM":  "Financial",
    "JNJ":  "Healthcare",
    "XOM":  "Energy",
}

BENCHMARK: str = "^GSPC"  # S&P 500

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

START_DATE: str = "2008-01-01"
END_DATE:   str = "2024-01-01"
CACHE_DIR:  str = "data_cache"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

PREDICTION_HORIZON: int = 5   # trading days ahead — target = 5-day log return

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

TRAIN_RATIO:     float = 0.80   # 80 / 20 temporal split
INNER_VAL_RATIO: float = 0.90   # last 10% of training used as temporal inner-val
TS_CV_SPLITS:    int   = 5      # TimeSeriesSplit folds

# Transaction cost and risk-free rate for backtesting
TRANSACTION_COST: float = 0.001  # 10 bps round-trip
ANNUAL_RF:        float = 0.04   # 4% (2023 context)
