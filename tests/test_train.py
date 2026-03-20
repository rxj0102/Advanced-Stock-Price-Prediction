"""
Tests for stock_prediction.models.train (v2)

Uses lightweight synthetic data and only fast linear models to stay snappy.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, Ridge

from stock_prediction.models.train import (
    backtest,
    build_linear_models,
    build_tree_models,
    make_lasso_pipeline,
    run_all_stocks,
    train_pipeline,
)
from stock_prediction.models.evaluate import ModelMetrics

TICKER = "TEST"


class TestBuildLinearModels:
    def test_returns_dict(self):
        models = build_linear_models()
        assert isinstance(models, dict)
        assert len(models) >= 5

    def test_all_have_fit(self):
        for name, m in build_linear_models().items():
            assert hasattr(m, "fit"), f"{name} missing .fit"

    def test_expected_model_names(self):
        models = build_linear_models()
        expected = {"OLS Baseline", "RidgeCV", "LassoCV", "ElasticNetCV",
                    "BayesianRidge", "HuberRegressor"}
        assert set(models.keys()) == expected


class TestBuildTreeModels:
    def test_returns_dict(self):
        models = build_tree_models()
        assert isinstance(models, dict)
        assert len(models) >= 3

    def test_all_have_fit(self):
        for name, m in build_tree_models().items():
            assert hasattr(m, "fit"), f"{name} missing .fit"

    def test_core_models_present(self):
        models = build_tree_models()
        assert "Decision Tree"     in models
        assert "Random Forest"     in models
        assert "Gradient Boosting" in models


class TestTrainPipeline:
    def test_output_keys(self, raw_ohlcv):
        out = train_pipeline(
            raw_ohlcv, TICKER,
            run_linear=True, run_tree=False, run_ensemble=False,
            verbose=False,
        )
        for key in ("ticker", "feature_cols", "X_train", "X_test",
                    "y_train", "y_test", "all_results"):
            assert key in out, f"Missing key: {key}"

    def test_all_results_contain_metrics(self, raw_ohlcv):
        out = train_pipeline(
            raw_ohlcv, TICKER,
            run_linear=True, run_tree=False, run_ensemble=False,
            verbose=False,
        )
        for name, res in out["all_results"].items():
            assert isinstance(res["test_metrics"], ModelMetrics), \
                f"{name}: test_metrics is not ModelMetrics"
            assert isinstance(res["train_metrics"], ModelMetrics), \
                f"{name}: train_metrics is not ModelMetrics"

    def test_predictions_length(self, raw_ohlcv):
        out = train_pipeline(
            raw_ohlcv, TICKER,
            run_linear=True, run_tree=False, run_ensemble=False,
            verbose=False,
        )
        n_test = len(out["y_test"])
        for name, res in out["all_results"].items():
            assert len(res["predictions"]) == n_test, \
                f"{name} predictions length mismatch"

    def test_no_data_leakage(self, raw_ohlcv):
        """Train index must strictly precede test index."""
        out = train_pipeline(
            raw_ohlcv, TICKER,
            run_linear=True, run_tree=False, run_ensemble=False,
            verbose=False,
        )
        assert out["X_train"].index.max() < out["X_test"].index.min()

    def test_robust_scaler_median_near_zero(self, raw_ohlcv):
        """RobustScaler: median of each scaled training feature should be ~0."""
        out = train_pipeline(
            raw_ohlcv, TICKER,
            run_linear=True, run_tree=False, run_ensemble=False,
            verbose=False,
        )
        median_vals = out["X_train"].median()
        assert (median_vals.abs() < 0.1).all(), \
            "RobustScaler: some training medians are far from zero"

    def test_custom_train_ratio(self, raw_ohlcv):
        out = train_pipeline(
            raw_ohlcv, TICKER, train_ratio=0.7,
            run_linear=True, run_tree=False, run_ensemble=False,
            verbose=False,
        )
        total = len(out["X_train"]) + len(out["X_test"])
        ratio = len(out["X_train"]) / total
        assert abs(ratio - 0.7) < 0.03


class TestMakeLassoPipeline:
    def test_pipeline_has_two_steps(self):
        pipe = make_lasso_pipeline()
        assert len(pipe.steps) == 2

    def test_pipeline_step_names(self):
        pipe = make_lasso_pipeline()
        assert pipe.steps[0][0] == "scaler"
        assert pipe.steps[1][0] == "model"

    def test_pipeline_fits_and_predicts(self, xy):
        X, y, _ = xy
        sp = int(len(X) * 0.8)
        X_tr, X_te = X.iloc[:sp], X.iloc[sp:]
        y_tr, y_te = y.iloc[:sp], y.iloc[sp:]
        pipe = make_lasso_pipeline()
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        assert len(preds) == len(y_te)


class TestRunAllStocks:
    def test_runs_on_multiple_tickers(self, raw_ohlcv):
        rng = np.random.default_rng(1)
        df2 = raw_ohlcv.copy()
        # Slight perturbation so it's a "different" stock
        df2["Close"] = df2["Close"] * rng.uniform(0.95, 1.05, len(df2))

        stock_data = {"TEST": raw_ohlcv, "TEST2": df2}
        results = run_all_stocks(stock_data, verbose=False)
        assert set(results.keys()) == {"TEST", "TEST2"}

    def test_each_result_has_metrics(self, raw_ohlcv):
        results = run_all_stocks({"TEST": raw_ohlcv}, verbose=False)
        res = results["TEST"]
        assert isinstance(res["test_metrics"], ModelMetrics)
        assert isinstance(res["train_metrics"], ModelMetrics)

    def test_each_result_has_coef(self, raw_ohlcv):
        results = run_all_stocks({"TEST": raw_ohlcv}, verbose=False)
        assert "coef" in results["TEST"]

    def test_temporal_ordering(self, raw_ohlcv):
        results = run_all_stocks({"TEST": raw_ohlcv}, verbose=False)
        res = results["TEST"]
        assert res["X_train"].index.max() < res["X_test"].index.min()


class TestBacktest:
    def _log_rets(self, n: int = 200, seed: int = 42):
        rng = np.random.default_rng(seed)
        y_true = rng.normal(0.0, 0.02, n)
        y_pred = y_true + rng.normal(0.0, 0.005, n)
        return y_true, y_pred

    def test_returns_expected_keys(self):
        y_true, y_pred = self._log_rets()
        bt = backtest(y_true, y_pred)
        for key in ("Ann_Return", "Ann_Vol", "Sharpe", "Max_DD", "Calmar",
                    "Total_Ret", "BAH_Ret"):
            assert key in bt, f"Missing key: {key}"

    def test_max_dd_non_positive(self):
        y_true, y_pred = self._log_rets()
        bt = backtest(y_true, y_pred)
        assert bt["Max_DD"] <= 0.0

    def test_ann_vol_positive(self):
        y_true, y_pred = self._log_rets()
        bt = backtest(y_true, y_pred)
        assert bt["Ann_Vol"] >= 0.0

    def test_sharpe_finite(self):
        import math
        y_true, y_pred = self._log_rets()
        bt = backtest(y_true, y_pred)
        assert math.isfinite(bt["Sharpe"])
