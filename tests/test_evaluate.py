"""
Tests for stock_prediction.models.evaluate (v2)
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
    evaluate_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_returns(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic log returns for testing (centred near 0)."""
    rng = np.random.default_rng(seed)
    y_true = rng.normal(0.0, 0.02, n)
    y_pred = y_true + rng.normal(0.0, 0.005, n)
    return y_true, y_pred


def _perfect(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(-0.05, 0.05, n)
    return y, y.copy()


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_perfect_predictions_r2(self):
        y_true, y_pred = _perfect()
        m = evaluate_model(y_true, y_pred, verbose=False)
        assert m.r2 == pytest.approx(1.0, abs=1e-6)

    def test_perfect_predictions_rmse(self):
        y_true, y_pred = _perfect()
        m = evaluate_model(y_true, y_pred, verbose=False)
        assert m.rmse == pytest.approx(0.0, abs=1e-8)

    def test_returns_model_metrics_instance(self):
        y_true, y_pred = _log_returns()
        m = evaluate_model(y_true, y_pred, "TestModel", verbose=False)
        assert isinstance(m, ModelMetrics)
        assert m.model_name == "TestModel"

    def test_all_fields_finite(self):
        y_true, y_pred = _log_returns()
        m = evaluate_model(y_true, y_pred, verbose=False)
        for val in (m.mse, m.rmse, m.mae, m.r2, m.mape, m.dir_acc, m.dir_pval):
            assert math.isfinite(val), f"Non-finite value: {val}"

    def test_dir_acc_in_unit_interval(self):
        y_true, y_pred = _log_returns()
        m = evaluate_model(y_true, y_pred, verbose=False)
        assert 0.0 <= m.dir_acc <= 1.0

    def test_dir_pval_in_unit_interval(self):
        y_true, y_pred = _log_returns()
        m = evaluate_model(y_true, y_pred, verbose=False)
        assert 0.0 <= m.dir_pval <= 1.0

    def test_summary_dict_keys(self):
        y_true, y_pred = _perfect()
        m = evaluate_model(y_true, y_pred, verbose=False)
        assert set(m.summary.keys()) == {"MSE", "RMSE", "MAE", "R2", "MAPE", "Dir_Acc", "Dir_p"}

    def test_sig_stars_perfect_dir(self):
        """All-correct direction should give significant p-value."""
        y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 40)
        y_pred = y_true * 0.9  # same sign
        m = evaluate_model(y_true, y_pred, verbose=False)
        assert m.sig_stars in ("*", "**", "***")

    def test_constant_pred_r2_zero(self):
        y_true = np.linspace(-0.05, 0.05, 100)
        y_pred = np.zeros_like(y_true)
        m = evaluate_model(y_true, y_pred, verbose=False)
        assert m.r2 == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    def _make_results(self, names: list[str]) -> dict:
        rng = np.random.default_rng(7)
        results = {}
        for name in names:
            y = rng.normal(0.0, 0.02, 200)
            noise = rng.uniform(0.002, 0.008)
            yp = y + rng.normal(0.0, noise, 200)
            tr_m = evaluate_model(y, yp * 1.1, verbose=False)
            te_m = evaluate_model(y, yp, name, verbose=False)
            results[name] = (tr_m, te_m)
        return results

    def test_sorted_descending_test_r2(self):
        results = self._make_results(["A", "B", "C", "D"])
        df = build_comparison_table(results)
        r2_vals = df["Test R²"].astype(float).tolist()
        assert r2_vals == sorted(r2_vals, reverse=True)

    def test_all_models_present(self):
        names = ["ModelX", "ModelY", "ModelZ"]
        results = self._make_results(names)
        df = build_comparison_table(results)
        assert set(df["Model"]) == set(names)

    def test_index_starts_at_one(self):
        results = self._make_results(["Alpha", "Beta"])
        df = build_comparison_table(results)
        assert df.index[0] == 1

    def test_gap_column_present(self):
        results = self._make_results(["X", "Y"])
        df = build_comparison_table(results)
        assert "Gap" in df.columns

    def test_dir_acc_column_present(self):
        results = self._make_results(["X"])
        df = build_comparison_table(results)
        assert "Dir Acc" in df.columns
