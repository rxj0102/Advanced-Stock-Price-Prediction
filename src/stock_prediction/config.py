"""
Central configuration for the stock prediction project.

All tuneable constants live here so that notebooks and scripts
only need a single import to share the same values.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

START_DATE: str = "2008-01-01"
END_DATE:   str = "2024-01-01"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

PREDICTION_HORIZON: int = 5          # trading days ahead to predict
MA_WINDOWS: list[int] = [5, 10, 20, 50, 100, 200]
LAG_WINDOWS: list[int] = [1, 2, 3, 5, 10]

# Raw columns excluded from the feature matrix (avoid lookahead / redundancy)
RAW_COLUMNS: list[str] = [
    "Target",
    "{ticker}_Close",
    "{ticker}_High",
    "{ticker}_Low",
    "{ticker}_Open",
    "{ticker}_Volume",
    "Return",
]

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

TRAIN_RATIO: float = 0.80            # 80 / 20 temporal split

# Hyperparameters for each model used in the comparison
MODEL_PARAMS: dict[str, dict] = {
    "Linear Regression": {},
    "Ridge Regression":  {"alpha": 1.0, "random_state": 42},
    "Lasso Regression":  {"alpha": 0.01, "random_state": 42, "max_iter": 10_000},
    "ElasticNet":        {"alpha": 0.01, "l1_ratio": 0.5, "random_state": 42, "max_iter": 10_000},
    "Decision Tree":     {"max_depth": 10, "random_state": 42},
    "Random Forest":     {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1},
    "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "random_state": 42},
    "XGBoost":           {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5,
                          "random_state": 42, "n_jobs": -1, "verbosity": 0},
    "SVR":               {"kernel": "rbf", "C": 100, "gamma": 0.01, "epsilon": 0.1},
    "MLP":               {"hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "adam",
                          "alpha": 0.001, "max_iter": 500, "random_state": 42, "early_stopping": True},
    "LightGBM":          {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5,
                          "num_leaves": 31, "random_state": 42, "n_jobs": -1, "verbose": -1},
    "CatBoost":          {"iterations": 200, "learning_rate": 0.05, "depth": 6,
                          "random_state": 42, "verbose": False},
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
