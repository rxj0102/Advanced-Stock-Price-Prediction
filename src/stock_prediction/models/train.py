"""
Model construction, training, and full multi-stock pipeline.

Models
------
Linear    : OLS, RidgeCV, LassoCV, ElasticNetCV, BayesianRidge, HuberRegressor
Tree      : Decision Tree, Random Forest, Gradient Boosting,
            XGBoost / LightGBM / CatBoost (optional, with temporal early stopping)
Ensembles : Voting, Stacking (TimeSeriesSplit), Blending, CV-Weighted
Neural    : Keras dense network (optional, requires TensorFlow)

Key design choices
------------------
- RobustScaler fit on training data only
- TimeSeriesSplit used for alpha CV and StackingRegressor
- XGBoost / LightGBM / CatBoost validated on a temporal inner-val slice,
  not the held-out test set
- Ensemble weights derived from CV R², never from test-set performance
- sklearn Pipeline for the multi-stock run (no manual scaling errors)
- Trading backtest: long/short signals from sign(y_pred)
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor

from stock_prediction.config import (
    ANNUAL_RF,
    INNER_VAL_RATIO,
    SEED,
    STOCKS,
    TRAIN_RATIO,
    TRANSACTION_COST,
    TS_CV_SPLITS,
)
from stock_prediction.features.engineer import engineer_features, prepare_xy
from stock_prediction.models.evaluate import ModelMetrics, evaluate_model

logger = logging.getLogger(__name__)

TS_CV = TimeSeriesSplit(n_splits=TS_CV_SPLITS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone(model: Any) -> Any:
    """Clone model — compatible with sklearn, xgb, lgb, and catboost."""
    try:
        return clone(model)
    except Exception:
        return copy.deepcopy(model)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_linear_models() -> dict[str, Any]:
    """Instantiate linear models with auto-tuned regularisation.

    Alpha is selected via TimeSeriesSplit cross-validation so no
    future information enters the training process.
    """
    return {
        "OLS Baseline":   LinearRegression(),
        "RidgeCV":        RidgeCV(alphas=np.logspace(-4, 4, 50), cv=TS_CV),
        "LassoCV":        LassoCV(
            alphas=np.logspace(-6, 1, 60),
            cv=TS_CV, max_iter=50_000, random_state=SEED,
        ),
        "ElasticNetCV":   ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            alphas=np.logspace(-6, 1, 30),
            cv=TS_CV, max_iter=50_000, random_state=SEED,
        ),
        "BayesianRidge":  BayesianRidge(max_iter=500),
        "HuberRegressor": HuberRegressor(epsilon=1.35, max_iter=500, alpha=1e-4),
    }


def build_tree_models() -> dict[str, Any]:
    """Instantiate regularised tree-based models.

    Gradient boosting libraries (XGBoost, LightGBM, CatBoost) are imported
    lazily and omitted gracefully if not installed.
    """
    models: dict[str, Any] = {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=30, random_state=SEED,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=20,
            max_features=0.5, random_state=SEED, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            min_samples_leaf=20, subsample=0.8, random_state=SEED,
        ),
    }

    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.01, max_depth=3,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=50, eval_metric="rmse",
            random_state=SEED, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        logger.warning("xgboost not installed — XGBoost skipped")

    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.01, num_leaves=15,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        logger.warning("lightgbm not installed — LightGBM skipped")

    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=2000, learning_rate=0.01, depth=4,
            l2_leaf_reg=3.0, subsample=0.8,
            early_stopping_rounds=50,
            random_seed=SEED, verbose=False,
        )
    except ImportError:
        logger.warning("catboost not installed — CatBoost skipped")

    return models


# ---------------------------------------------------------------------------
# Single-stock pipeline
# ---------------------------------------------------------------------------

def train_pipeline(
    df_raw: pd.DataFrame,
    ticker: str,
    *,
    train_ratio: float = TRAIN_RATIO,
    inner_val_ratio: float = INNER_VAL_RATIO,
    run_linear: bool = True,
    run_tree: bool = True,
    run_ensemble: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full train/evaluate pipeline for one ticker.

    Parameters
    ----------
    df_raw:
        Flat OHLCV DataFrame (Open, High, Low, Close, Volume).
    ticker:
        Ticker symbol used for feature engineering.
    train_ratio:
        Fraction of data used for training (temporal split).
    inner_val_ratio:
        Fraction of training data used as inner validation for early stopping.
    run_linear, run_tree, run_ensemble:
        Toggle individual model groups.
    verbose:
        Print evaluation results.

    Returns
    -------
    dict with keys:
        ``ticker``, ``feature_cols``, ``scaler``,
        ``X_train``, ``X_test``, ``y_train``, ``y_test``,
        ``X_inner_train``, ``X_inner_val``, ``y_inner_train``, ``y_inner_val``,
        ``linear_results``, ``tree_results``, ``ensemble_results``,
        ``all_results``.
    """
    df_feat = engineer_features(df_raw, ticker)
    X, y, feature_cols = prepare_xy(df_feat)

    split_idx = int(len(X) * train_ratio)
    X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test         = y.iloc[:split_idx], y.iloc[split_idx:]

    inner_split = int(len(X_train_raw) * inner_val_ratio)

    scaler  = RobustScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=feature_cols, index=X_train_raw.index,
    )
    X_test  = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=feature_cols, index=X_test_raw.index,
    )

    X_inner_train = X_train.iloc[:inner_split]
    X_inner_val   = X_train.iloc[inner_split:]
    y_inner_train = y_train.iloc[:inner_split]
    y_inner_val   = y_train.iloc[inner_split:]

    linear_results:   dict[str, dict] = {}
    tree_results:     dict[str, dict] = {}
    ensemble_results: dict[str, dict] = {}

    if run_linear:
        for name, model in build_linear_models().items():
            try:
                model.fit(X_train, y_train)
                y_tr_pred = model.predict(X_train)
                y_te_pred = model.predict(X_test)
                tr_m = evaluate_model(y_train, y_tr_pred, verbose=False)
                te_m = evaluate_model(y_test, y_te_pred, model_name=name, verbose=verbose)
                linear_results[name] = {
                    "model":          model,
                    "train_metrics":  tr_m,
                    "test_metrics":   te_m,
                    "predictions":    y_te_pred,
                }
            except Exception as exc:
                logger.error("[%s] %s failed: %s", ticker, name, exc)

    if run_tree:
        for name, model in build_tree_models().items():
            try:
                needs_eval = name in ("XGBoost", "LightGBM", "CatBoost")
                if needs_eval:
                    _fit_with_early_stopping(
                        name, model,
                        X_inner_train, y_inner_train,
                        X_inner_val,   y_inner_val,
                    )
                else:
                    model.fit(X_train, y_train)
                y_tr_pred = model.predict(X_train)
                y_te_pred = model.predict(X_test)
                tr_m = evaluate_model(y_train, y_tr_pred, verbose=False)
                te_m = evaluate_model(y_test, y_te_pred, model_name=name, verbose=verbose)
                tree_results[name] = {
                    "model":          model,
                    "train_metrics":  tr_m,
                    "test_metrics":   te_m,
                    "predictions":    y_te_pred,
                }
            except Exception as exc:
                logger.error("[%s] %s failed: %s", ticker, name, exc)

    if run_ensemble and linear_results:
        ensemble_results = _train_ensembles(
            linear_results, tree_results,
            X_train, y_train,
            X_test,  y_test,
            verbose=verbose,
        )

    all_results = {**linear_results, **tree_results, **ensemble_results}

    return {
        "ticker":           ticker,
        "feature_cols":     feature_cols,
        "scaler":           scaler,
        "X_train":          X_train,
        "X_test":           X_test,
        "y_train":          y_train,
        "y_test":           y_test,
        "X_inner_train":    X_inner_train,
        "X_inner_val":      X_inner_val,
        "y_inner_train":    y_inner_train,
        "y_inner_val":      y_inner_val,
        "linear_results":   linear_results,
        "tree_results":     tree_results,
        "ensemble_results": ensemble_results,
        "all_results":      all_results,
    }


def _fit_with_early_stopping(
    name: str,
    model: Any,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Fit XGBoost / LightGBM / CatBoost with temporal inner-val early stopping."""
    if name == "XGBoost":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "LightGBM":
        import lightgbm as lgb
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
    elif name == "CatBoost":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)


def _train_ensembles(
    linear_results: dict,
    tree_results:   dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    verbose: bool = True,
) -> dict[str, dict]:
    """Train Voting, Stacking, Blending, and CV-Weighted ensembles."""
    ensemble_results: dict[str, dict] = {}

    # Diverse base learners: one linear + one Bayesian + one tree
    base: list[tuple[str, Any]] = []
    for cand in ("LassoCV", "BayesianRidge"):
        if cand in linear_results:
            base.append((cand.lower(), _clone(linear_results[cand]["model"])))
    for cand in ("Random Forest", "LightGBM"):
        if cand in tree_results:
            base.append((cand.lower().replace(" ", "_"), _clone(tree_results[cand]["model"])))
            if len(base) >= 3:
                break

    if len(base) < 2:
        return ensemble_results

    all_base = {**linear_results, **tree_results}

    # A. Voting — equal-weight average
    try:
        voting = VotingRegressor(estimators=base)
        voting.fit(X_train, y_train)
        y_pred = voting.predict(X_test)
        tr_m = evaluate_model(y_train, voting.predict(X_train), verbose=False)
        te_m = evaluate_model(y_test,  y_pred, model_name="Voting", verbose=verbose)
        ensemble_results["Voting"] = {
            "model": voting, "train_metrics": tr_m,
            "test_metrics": te_m, "predictions": y_pred,
        }
    except Exception as exc:
        logger.error("Voting failed: %s", exc)

    # B. Stacking — TimeSeriesSplit meta-learner
    try:
        stacking = StackingRegressor(
            estimators=base,
            final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 30), cv=TS_CV),
            cv=TS_CV,
            n_jobs=-1,
        )
        stacking.fit(X_train, y_train)
        y_pred = stacking.predict(X_test)
        tr_m = evaluate_model(y_train, stacking.predict(X_train), verbose=False)
        te_m = evaluate_model(y_test,  y_pred, model_name="Stacking", verbose=verbose)
        ensemble_results["Stacking"] = {
            "model": stacking, "train_metrics": tr_m,
            "test_metrics": te_m, "predictions": y_pred,
        }
    except Exception as exc:
        logger.error("Stacking failed: %s", exc)

    # C. Blending — temporal 70/30 inner split
    try:
        bs = int(len(X_train) * 0.70)
        X_bl_tr, X_bl_val = X_train.iloc[:bs], X_train.iloc[bs:]
        y_bl_tr, y_bl_val = y_train.iloc[:bs], y_train.iloc[bs:]

        blend = {n: _clone(m) for n, m in base}
        oof: dict[str, np.ndarray] = {}
        for n, m in blend.items():
            m.fit(X_bl_tr, y_bl_tr)
            oof[n] = m.predict(X_bl_val)

        meta = RidgeCV(alphas=np.logspace(-3, 3, 30), cv=TS_CV)
        meta.fit(np.column_stack([oof[n] for n in blend]), y_bl_val)

        test_bp: dict[str, np.ndarray] = {}
        for n, m in blend.items():
            m.fit(X_train, y_train)
            test_bp[n] = m.predict(X_test)

        y_pred = meta.predict(np.column_stack([test_bp[n] for n in blend]))
        tr_m   = evaluate_model(y_train, y_train * 0, verbose=False)
        te_m   = evaluate_model(y_test,  y_pred, model_name="Blending", verbose=verbose)
        ensemble_results["Blending"] = {
            "model": (blend, meta), "train_metrics": tr_m,
            "test_metrics": te_m,  "predictions": y_pred,
        }
    except Exception as exc:
        logger.error("Blending failed: %s", exc)

    # D. CV-Weighted — weights from CV R², not test-set R²
    try:
        cv_weights: dict[str, float] = {}
        items: list[tuple[str, np.ndarray]] = []

        for name, res in all_base.items():
            scores = cross_val_score(
                _clone(res["model"]), X_train, y_train,
                cv=TS_CV, scoring="r2", n_jobs=-1,
            )
            cv_weights[name] = max(float(np.mean(np.clip(scores, 0, 1))), 0.01)
            items.append((name, res["predictions"]))

        total_w = sum(cv_weights.values())
        w_arr   = np.array([cv_weights[n] / total_w for n, _ in items])
        y_pred  = np.column_stack([p for _, p in items]) @ w_arr

        tr_m = evaluate_model(y_train, y_train * 0, verbose=False)
        te_m = evaluate_model(y_test,  y_pred, model_name="CV-Weighted", verbose=verbose)
        ensemble_results["CV-Weighted"] = {
            "model": None, "train_metrics": tr_m,
            "test_metrics": te_m, "predictions": y_pred,
            "cv_weights": cv_weights,
        }
    except Exception as exc:
        logger.error("CV-Weighted failed: %s", exc)

    return ensemble_results


# ---------------------------------------------------------------------------
# Multi-stock pipeline
# ---------------------------------------------------------------------------

def make_lasso_pipeline() -> Pipeline:
    """sklearn Pipeline: RobustScaler + LassoCV with TimeSeriesSplit."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model",  LassoCV(
            alphas=np.logspace(-6, 1, 60),
            cv=TimeSeriesSplit(n_splits=TS_CV_SPLITS),
            max_iter=50_000,
            random_state=SEED,
        )),
    ])


def run_all_stocks(
    stock_data: dict[str, pd.DataFrame],
    *,
    train_ratio: float = TRAIN_RATIO,
    verbose: bool = False,
) -> dict[str, dict]:
    """Run the LassoCV pipeline across all tickers.

    Uses a clean sklearn Pipeline (RobustScaler + LassoCV) with alpha
    auto-tuned independently per ticker via TimeSeriesSplit.

    Parameters
    ----------
    stock_data:
        Output of :func:`data.loader.download_stocks`.
    train_ratio:
        Temporal split fraction.
    verbose:
        Print evaluation per stock.

    Returns
    -------
    dict[ticker, result_dict]
        Each value contains: sector, pipeline, X/y splits, predictions,
        train_metrics, test_metrics, alpha, coef.
    """
    ticker_sectors = {t: STOCKS.get(t, "Unknown") for t in stock_data}

    all_results: dict[str, dict] = {}

    for ticker, df in stock_data.items():
        sector = ticker_sectors[ticker]
        logger.info("Processing %s (%s)...", ticker, sector)
        try:
            df_feat   = engineer_features(df, ticker)
            feat_cols = [c for c in df_feat.columns if c != "Target"]
            X = df_feat[feat_cols]
            y = df_feat["Target"]

            sp = int(len(X) * train_ratio)
            X_tr, X_te = X.iloc[:sp], X.iloc[sp:]
            y_tr, y_te = y.iloc[:sp], y.iloc[sp:]

            pipe = make_lasso_pipeline()
            pipe.fit(X_tr, y_tr)

            y_pred_te = pipe.predict(X_te)
            y_pred_tr = pipe.predict(X_tr)

            tr_m = evaluate_model(y_tr, y_pred_tr, verbose=False)
            te_m = evaluate_model(y_te, y_pred_te, model_name=ticker, verbose=verbose)

            all_results[ticker] = {
                "sector":        sector,
                "pipeline":      pipe,
                "X_train":       X_tr,
                "X_test":        X_te,
                "y_train":       y_tr,
                "y_test":        y_te,
                "predictions":   y_pred_te,
                "train_metrics": tr_m,
                "test_metrics":  te_m,
                "alpha":         pipe.named_steps["model"].alpha_,
                "coef":          pd.Series(
                    pipe.named_steps["model"].coef_, index=feat_cols
                ),
            }
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", ticker, exc)

    return all_results


# ---------------------------------------------------------------------------
# Trading backtest
# ---------------------------------------------------------------------------

def backtest(
    y_true_log_ret: np.ndarray,
    y_pred_log_ret: np.ndarray,
    *,
    transaction_cost: float = TRANSACTION_COST,
    annual_rf: float = ANNUAL_RF,
) -> dict[str, float]:
    """Simulate a long/short strategy on predicted log returns.

    Strategy
    --------
    - Long  (+1) when predicted log return > 0
    - Short (-1) when predicted log return < 0
    - ``transaction_cost`` deducted on every position change

    Parameters
    ----------
    y_true_log_ret:
        Realised log returns over the test period.
    y_pred_log_ret:
        Predicted log returns.
    transaction_cost:
        Round-trip cost per trade (default 0.001 = 10 bps).
    annual_rf:
        Annual risk-free rate used in the Sharpe calculation.

    Returns
    -------
    dict with keys: Ann_Return, Ann_Vol, Sharpe, Max_DD, Calmar, Total_Ret, BAH_Ret.
    """
    y_true  = np.asarray(y_true_log_ret, dtype=float)
    y_pred  = np.asarray(y_pred_log_ret, dtype=float)
    signals = np.sign(y_pred)

    position_change = np.diff(np.concatenate([[0], signals])) != 0
    strategy_ret    = signals * y_true - position_change * transaction_cost
    bah_ret         = y_true

    cum_strategy = np.exp(np.cumsum(strategy_ret)) - 1
    cum_bah      = np.exp(np.cumsum(bah_ret))      - 1

    periods_per_year = 252
    ann_ret    = float(np.sum(strategy_ret) * periods_per_year / len(strategy_ret))
    ann_vol    = float(np.std(strategy_ret) * np.sqrt(periods_per_year))
    ann_rf_day = annual_rf / periods_per_year
    sharpe     = float(
        (np.mean(strategy_ret) - ann_rf_day) / (np.std(strategy_ret) + 1e-12)
        * np.sqrt(periods_per_year)
    )

    wealth   = np.exp(np.cumsum(strategy_ret))
    peak     = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    max_dd   = float(drawdown.min())
    calmar   = ann_ret / (abs(max_dd) + 1e-12)

    return {
        "Ann_Return": ann_ret,
        "Ann_Vol":    ann_vol,
        "Sharpe":     sharpe,
        "Max_DD":     max_dd,
        "Calmar":     float(calmar),
        "Total_Ret":  float(cum_strategy[-1]),
        "BAH_Ret":    float(cum_bah[-1]),
    }
