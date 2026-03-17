"""
Model construction, training, and the full multi-stock pipeline.

Typical usage
-------------
>>> from stock_prediction.models.train import run_all_stocks
>>> results = run_all_stocks(stock_data)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from stock_prediction.config import MODEL_PARAMS, TRAIN_RATIO
from stock_prediction.features.engineer import engineer_features, prepare_xy
from stock_prediction.models.evaluate import ModelMetrics, evaluate_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_models() -> dict[str, Any]:
    """Instantiate all comparison models with their configured hyperparameters.

    Returns
    -------
    dict[str, estimator]
        Keys match :data:`config.MODEL_PARAMS`.
    """
    p = MODEL_PARAMS
    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(**p["Linear Regression"]),
        "Ridge Regression":  Ridge(**p["Ridge Regression"]),
        "Lasso Regression":  Lasso(**p["Lasso Regression"]),
        "ElasticNet":        ElasticNet(**p["ElasticNet"]),
        "Decision Tree":     DecisionTreeRegressor(**p["Decision Tree"]),
        "Random Forest":     RandomForestRegressor(**p["Random Forest"]),
        "Gradient Boosting": GradientBoostingRegressor(**p["Gradient Boosting"]),
        "SVR":               SVR(**p["SVR"]),
        "MLP":               MLPRegressor(**p["MLP"]),
    }

    # Optional heavy dependencies
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(**p["XGBoost"])
    except ImportError:
        logger.warning("xgboost not installed — XGBoost skipped")

    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(**p["LightGBM"])
    except ImportError:
        logger.warning("lightgbm not installed — LightGBM skipped")

    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(**p["CatBoost"])
    except ImportError:
        logger.warning("catboost not installed — CatBoost skipped")

    return models


def build_ensemble(base_models: dict[str, Any]) -> dict[str, Any]:
    """Build Voting and Stacking ensembles from a subset of base estimators.

    Parameters
    ----------
    base_models:
        Dictionary of already-instantiated (but **not** fitted) estimators.

    Returns
    -------
    dict[str, estimator]
    """
    estimator_list = [
        (name, model)
        for name, model in base_models.items()
        if name in {"Lasso Regression", "Ridge Regression", "Linear Regression", "ElasticNet"}
    ]
    if len(estimator_list) < 2:
        return {}

    return {
        "Voting Ensemble":   VotingRegressor(estimators=estimator_list, n_jobs=-1),
        "Stacking Ensemble": StackingRegressor(
            estimators=estimator_list,
            final_estimator=LinearRegression(),
            cv=5,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Single-stock pipeline
# ---------------------------------------------------------------------------

def train_pipeline(
    df_raw: pd.DataFrame,
    ticker: str,
    *,
    models: dict[str, Any] | None = None,
    train_ratio: float = TRAIN_RATIO,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full train/evaluate pipeline for one ticker.

    Parameters
    ----------
    df_raw:
        Flat OHLCV DataFrame as returned by :func:`data.loader.download_stocks`.
    ticker:
        Ticker symbol (used for column look-ups).
    models:
        Pre-built model dictionary.  Defaults to :func:`build_models`.
    train_ratio:
        Fraction of data used for training (temporal split).
    verbose:
        Print evaluation reports.

    Returns
    -------
    dict with keys:
        ``ticker``, ``X_train``, ``X_test``, ``y_train``, ``y_test``,
        ``scaler``, ``feature_cols``, ``results`` (dict[name → ModelMetrics]),
        ``predictions`` (dict[name → np.ndarray]).
    """
    if models is None:
        models = {**build_models(), **build_ensemble(build_models())}

    # Feature engineering
    df_feat = engineer_features(df_raw, ticker)
    X, y, feature_cols = prepare_xy(df_feat, ticker)

    # Temporal split
    split = int(len(X) * train_ratio)
    X_train_raw, X_test_raw = X.iloc[:split], X.iloc[split:]
    y_train, y_test           = y.iloc[:split], y.iloc[split:]

    # Scaling (fit on train only)
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_raw)
    X_test_np  = scaler.transform(X_test_raw)
    X_train    = pd.DataFrame(X_train_np, columns=feature_cols, index=X_train_raw.index)
    X_test     = pd.DataFrame(X_test_np,  columns=feature_cols, index=X_test_raw.index)

    results:     dict[str, ModelMetrics] = {}
    predictions: dict[str, np.ndarray]  = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred, name, verbose=verbose)
            results[name]     = metrics
            predictions[name] = y_pred
            logger.info("[%s] %s  R²=%.4f  RMSE=$%.2f", ticker, name, metrics.r2, metrics.rmse)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] %s failed: %s", ticker, name, exc)

    return {
        "ticker":       ticker,
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_test":       y_test,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "results":      results,
        "predictions":  predictions,
    }


# ---------------------------------------------------------------------------
# Multi-stock pipeline
# ---------------------------------------------------------------------------

def run_all_stocks(
    stock_data: dict[str, pd.DataFrame],
    *,
    best_model_name: str = "Lasso Regression",
    train_ratio: float = TRAIN_RATIO,
    verbose: bool = False,
) -> dict[str, dict]:
    """Run the best-performing model across all tickers.

    Parameters
    ----------
    stock_data:
        Output of :func:`data.loader.download_stocks`.
    best_model_name:
        Which model to use for the cross-stock comparison.
    train_ratio:
        Temporal split fraction.
    verbose:
        Print evaluation per stock.

    Returns
    -------
    dict[ticker, pipeline_output]
        Same structure as :func:`train_pipeline`, limited to
        ``best_model_name``.
    """
    all_results: dict[str, dict] = {}

    for ticker, df in stock_data.items():
        logger.info("Processing %s …", ticker)
        try:
            best = {best_model_name: Lasso(**MODEL_PARAMS["Lasso Regression"])}
            output = train_pipeline(df, ticker, models=best,
                                    train_ratio=train_ratio, verbose=verbose)
            all_results[ticker] = output
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline failed for %s: %s", ticker, exc)

    return all_results
