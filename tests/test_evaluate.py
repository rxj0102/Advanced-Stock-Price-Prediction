"""
Tests for stock_prediction.models.evaluate
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import math

import numpy as np
import pytest

from stock_prediction.models.evaluate import (
    ModelMetrics,
    build_comparison_table,
    directional_accuracy,
    evaluate_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect(n: int = 100):
    y = np.linspace(10, 110, n)
    return y, y.copy()


def _constant_pred(n: int = 100):
    y_true = np.linspace(10, 110, n)
    y_pred = np.full(n, y_true.mean())
    return y_true, y_pred


# ---------------------------------------------------------------------------
# directional_accuracy
# ---------------------------------------------------------------------------

class TestDirectionalAccuracy:
    def test_perfect_direction(self):
        y = np.arange(1, 11, dtype=float)
        assert directional_accuracy(y, y) == 1.0

    def test_inverted_direction(self):
        y = np.arange(1, 11, dtype=float)
        assert directional_accuracy(y, y[::-1]) == 0.0

    def test_short_array_returns_nan(self):
        assert math.isnan(directional_accuracy(np.array([1.0]), np.array([2.0])))

    def test_empty_array_returns_nan(self):
        assert math.isnan(directional_accuracy(np.array([]), np.array([])))

    def test_random_is_near_half(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=10_000)
        y_pred = rng.normal(size=10_000)
        acc = directional_accuracy(y_true, y_pred)
        assert 0.45 < acc < 0.55


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_perfect_predictions(self):
        y_true, y_pred = _perfect()
        m = evaluate_model(y_true, y_pred, verbose=False)
        assert m.r2 == pytest.approx(1.0, abs=1e-6)
        assert m.rmse == pytest.approx(0.0, abs=1e-6)
        assert m.mape == pytest.approx(0.0, abs=1e-4)

    def test_constant_pred_r2_zero(self):
        y_true, y_pred = _constant_pred()
        m = evaluate_model(y_true, y_pred, verbose=False)
        assert m.r2 == pytest.approx(0.0, abs=1e-6)

    def test_returns_model_metrics(self):
        y_true, y_pred = _perfect()
        m = evaluate_model(y_true, y_pred, "TestModel", verbose=False)
        assert isinstance(m, ModelMetrics)
        assert m.model_name == "TestModel"

    def test_all_fields_finite(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(100, 10, 200)
        y_pred = y_true + rng.normal(0, 2, 200)
        m = evaluate_model(y_true, y_pred, verbose=False)
        for val in (m.mse, m.rmse, m.mae, m.r2, m.mape, m.directional_accuracy):
            assert math.isfinite(val)

    def test_mape_is_percentage(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 220.0, 330.0])  # 10% error each
        m = evaluate_model(y_true, y_pred, verbose=False)
        assert m.mape == pytest.approx(10.0, abs=0.1)

    def test_summary_dict_keys(self):
        y, yp = _perfect()
        m = evaluate_model(y, yp, verbose=False)
        assert set(m.summary.keys()) == {"MSE", "RMSE", "MAE", "R2", "MAPE", "Directional_Accuracy"}


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    def _make_metrics(self, names):
        rng = np.random.default_rng(7)
        results = {}
        for name in names:
            y = rng.normal(100, 10, 200)
            yp = y + rng.normal(0, rng.uniform(1, 5))
            results[name] = evaluate_model(y, yp, name, verbose=False)
        return results

    def test_sorted_descending_r2(self):
        results = self._make_metrics(["A", "B", "C", "D"])
        df = build_comparison_table(results)
        r2_vals = df["R²"].tolist()
        assert r2_vals == sorted(r2_vals, reverse=True)

    def test_all_models_present(self):
        names = ["ModelX", "ModelY", "ModelZ"]
        results = self._make_metrics(names)
        df = build_comparison_table(results)
        assert set(df["Model"]) == set(names)

    def test_index_starts_at_one(self):
        results = self._make_metrics(["Alpha", "Beta"])
        df = build_comparison_table(results)
        assert df.index[0] == 1
