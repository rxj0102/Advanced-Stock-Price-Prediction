"""
Tests for stock_prediction.models.train

These tests use lightweight synthetic data and only a subset of fast
models to keep the test suite snappy (< 30 s total).
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import Lasso, LinearRegression, Ridge

from stock_prediction.models.train import (
    build_ensemble,
    build_models,
    run_all_stocks,
    train_pipeline,
)
from stock_prediction.models.evaluate import ModelMetrics

TICKER = "TEST"
FAST_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Lasso Regression":  Lasso(alpha=0.01, max_iter=5_000),
}


class TestBuildModels:
    def test_returns_dict(self):
        models = build_models()
        assert isinstance(models, dict)
        assert len(models) >= 9   # at least the sklearn models

    def test_all_have_fit(self):
        for name, m in build_models().items():
            assert hasattr(m, "fit"), f"{name} missing .fit"

    def test_ensemble_requires_linear_bases(self):
        ensembles = build_ensemble(FAST_MODELS)
        assert "Voting Ensemble" in ensembles
        assert "Stacking Ensemble" in ensembles

    def test_ensemble_empty_with_too_few_bases(self):
        ensembles = build_ensemble({"Only One": LinearRegression()})
        assert ensembles == {}


class TestTrainPipeline:
    def test_output_keys(self, raw_ohlcv):
        out = train_pipeline(raw_ohlcv, TICKER, models=FAST_MODELS, verbose=False)
        for key in ("ticker", "X_train", "X_test", "y_train", "y_test",
                    "scaler", "feature_cols", "results", "predictions"):
            assert key in out, f"Missing key: {key}"

    def test_results_are_model_metrics(self, raw_ohlcv):
        out = train_pipeline(raw_ohlcv, TICKER, models=FAST_MODELS, verbose=False)
        for name, m in out["results"].items():
            assert isinstance(m, ModelMetrics), f"{name} result is not ModelMetrics"

    def test_predictions_length(self, raw_ohlcv):
        out = train_pipeline(raw_ohlcv, TICKER, models=FAST_MODELS, verbose=False)
        n_test = len(out["y_test"])
        for name, preds in out["predictions"].items():
            assert len(preds) == n_test, f"{name} predictions length mismatch"

    def test_no_data_leakage(self, raw_ohlcv):
        """Train index must precede test index."""
        out = train_pipeline(raw_ohlcv, TICKER, models=FAST_MODELS, verbose=False)
        assert out["X_train"].index.max() < out["X_test"].index.min()

    def test_scaler_zero_mean(self, raw_ohlcv):
        out = train_pipeline(raw_ohlcv, TICKER, models=FAST_MODELS, verbose=False)
        col_means = out["X_train"].mean()
        assert (col_means.abs() < 1e-6).all(), "Scaled training data should have ~zero mean"

    def test_custom_train_ratio(self, raw_ohlcv):
        out = train_pipeline(raw_ohlcv, TICKER, models=FAST_MODELS,
                             train_ratio=0.7, verbose=False)
        total = len(out["X_train"]) + len(out["X_test"])
        ratio = len(out["X_train"]) / total
        assert abs(ratio - 0.7) < 0.02   # allow small rounding


class TestRunAllStocks:
    def test_runs_on_multiple_tickers(self, raw_ohlcv):
        # Create a second synthetic stock with a different ticker
        df2 = raw_ohlcv.copy()
        df2.columns = [c.replace("TEST", "TEST2") for c in df2.columns]

        stock_data = {"TEST": raw_ohlcv, "TEST2": df2}
        results = run_all_stocks(stock_data, verbose=False)

        assert set(results.keys()) == {"TEST", "TEST2"}

    def test_each_result_has_metrics(self, raw_ohlcv):
        results = run_all_stocks({"TEST": raw_ohlcv}, verbose=False)
        assert "results" in results["TEST"]
        assert len(results["TEST"]["results"]) == 1   # only Lasso by default
