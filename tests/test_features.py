"""
Tests for stock_prediction.features.engineer (v2)
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from stock_prediction.features.engineer import engineer_features, prepare_xy
from stock_prediction.config import PREDICTION_HORIZON

TICKER = "TEST"


class TestEngineerFeatures:
    def test_returns_dataframe(self, raw_ohlcv):
        out = engineer_features(raw_ohlcv, TICKER)
        assert isinstance(out, pd.DataFrame)

    def test_no_nans(self, engineered_df):
        assert engineered_df.isna().sum().sum() == 0, "Unexpected NaN after engineer_features"

    def test_target_column_present(self, engineered_df):
        assert "Target" in engineered_df.columns

    def test_log_return_features_present(self, engineered_df):
        for n in [1, 2, 3, 5, 10, 20]:
            assert f"LogRet_{n}d" in engineered_df.columns, f"LogRet_{n}d missing"

    def test_intraday_features_present(self, engineered_df):
        for col in ["HL_Range", "Close_Open_Pct", "Upper_Shadow", "Lower_Shadow"]:
            assert col in engineered_df.columns, f"{col} missing"

    def test_volume_features_present(self, engineered_df):
        for col in ["Vol_LogChg", "Vol_Ratio_5d", "Vol_Ratio_20d", "OBV_Pct"]:
            assert col in engineered_df.columns, f"{col} missing"

    def test_ma_distance_features_present(self, engineered_df):
        for w in [5, 20, 50]:
            assert f"Price_MA{w}_Pct" in engineered_df.columns, f"Price_MA{w}_Pct missing"

    def test_volatility_features_present(self, engineered_df):
        for col in ["Vol_5d", "Vol_20d", "Vol_Ratio", "ATR_14"]:
            assert col in engineered_df.columns, f"{col} missing"

    def test_oscillator_features_present(self, engineered_df):
        for col in ["RSI_14", "MACD_Hist", "BB_Width", "BB_Pos"]:
            assert col in engineered_df.columns, f"{col} missing"

    def test_support_resistance_features_present(self, engineered_df):
        for col in ["Dist_Resistance", "Dist_Support", "Price_Position"]:
            assert col in engineered_df.columns, f"{col} missing"

    def test_calendar_features_present(self, engineered_df):
        for col in ["DOW_sin", "DOW_cos", "Month_sin", "Month_cos"]:
            assert col in engineered_df.columns, f"{col} missing"

    def test_row_count_reduced_by_dropna(self, raw_ohlcv, engineered_df):
        # Must lose rows due to rolling windows (max=50) + forward target shift (5)
        assert len(engineered_df) < len(raw_ohlcv) - 50

    def test_target_is_log_return(self, raw_ohlcv):
        """Target should equal log(Close_{t+5} / Close_t)."""
        df = engineer_features(raw_ohlcv, TICKER)
        idx = df.index[50]
        pos = raw_ohlcv.index.get_loc(idx)
        expected_idx = raw_ohlcv.index[pos + PREDICTION_HORIZON]
        expected = np.log(
            raw_ohlcv.loc[expected_idx, "Close"] / raw_ohlcv.loc[idx, "Close"]
        )
        assert abs(df.loc[idx, "Target"] - expected) < 1e-10

    def test_target_mean_near_zero(self, engineered_df):
        """5-day log return is stationary — mean should be near 0."""
        assert abs(engineered_df["Target"].mean()) < 0.01

    def test_hl_range_non_negative(self, engineered_df):
        assert (engineered_df["HL_Range"] >= 0).all()

    def test_price_position_in_unit_interval(self, engineered_df):
        pp = engineered_df["Price_Position"]
        assert (pp >= -0.01).all() and (pp <= 1.01).all()

    def test_rsi_in_valid_range(self, engineered_df):
        rsi = engineered_df["RSI_14"]
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_feature_count(self, engineered_df):
        feat_cols = [c for c in engineered_df.columns if c != "Target"]
        assert len(feat_cols) >= 30, f"Expected >= 30 features, got {len(feat_cols)}"


class TestPrepareXY:
    def test_shapes_match(self, xy):
        X, y, feat_cols = xy
        assert len(X) == len(y)
        assert X.shape[1] == len(feat_cols)

    def test_target_not_in_X(self, xy):
        X, _, _ = xy
        assert "Target" not in X.columns

    def test_no_nans_in_X(self, xy):
        X, y, _ = xy
        assert X.isna().sum().sum() == 0
        assert y.isna().sum() == 0

    def test_feature_count_reasonable(self, xy):
        _, _, feat_cols = xy
        assert len(feat_cols) >= 30
