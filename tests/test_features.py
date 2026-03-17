"""
Tests for stock_prediction.features.engineer
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from stock_prediction.features.engineer import engineer_features, prepare_xy
from stock_prediction.config import MA_WINDOWS, LAG_WINDOWS, PREDICTION_HORIZON

TICKER = "TEST"


class TestEngineerFeatures:
    def test_returns_dataframe(self, raw_ohlcv):
        out = engineer_features(raw_ohlcv, TICKER)
        assert isinstance(out, pd.DataFrame)

    def test_no_nans(self, engineered_df):
        assert engineered_df.isna().sum().sum() == 0, "Unexpected NaN values after engineer_features"

    def test_target_column_present(self, engineered_df):
        assert "Target" in engineered_df.columns

    def test_moving_averages_present(self, engineered_df):
        for w in MA_WINDOWS:
            assert f"MA_{w}" in engineered_df.columns, f"MA_{w} missing"
            assert f"Std_{w}" in engineered_df.columns, f"Std_{w} missing"

    def test_lag_features_present(self, engineered_df):
        for lag in LAG_WINDOWS:
            assert f"Price_Lag_{lag}" in engineered_df.columns, f"Price_Lag_{lag} missing"

    def test_crossover_features(self, engineered_df):
        for col in ["MA_5_20_Crossover", "MA_20_50_Crossover", "MA_50_200_Crossover"]:
            assert col in engineered_df.columns

    def test_momentum_features(self, engineered_df):
        for col in ["Momentum_5", "Momentum_20", "ROC_5", "ROC_20"]:
            assert col in engineered_df.columns

    def test_row_count_reduced_by_dropna(self, raw_ohlcv, engineered_df):
        # Must lose at least `max(MA_WINDOWS) + PREDICTION_HORIZON` rows
        min_dropped = max(MA_WINDOWS) + PREDICTION_HORIZON
        assert len(engineered_df) <= len(raw_ohlcv) - min_dropped

    def test_target_is_future_close(self, raw_ohlcv):
        """Target should equal Close shifted back by PREDICTION_HORIZON."""
        df = engineer_features(raw_ohlcv, TICKER)
        close_col = f"{TICKER}_Close"
        # Pick a middle row and verify
        idx = df.index[50]
        pos_in_raw = raw_ohlcv.index.get_loc(idx)
        expected_target_idx = raw_ohlcv.index[pos_in_raw + PREDICTION_HORIZON]
        expected_price = raw_ohlcv.loc[expected_target_idx, close_col]
        assert abs(df.loc[idx, "Target"] - expected_price) < 1e-6

    def test_high_low_range_non_negative(self, engineered_df):
        assert (engineered_df["High_Low_Range"] >= 0).all()

    def test_price_position_in_unit_interval(self, engineered_df):
        pp = engineered_df["Price_Position"]
        assert (pp >= 0).all() and (pp <= 1).all()


class TestPrepareXY:
    def test_shapes_match(self, xy):
        X, y, feat_cols = xy
        assert len(X) == len(y)
        assert X.shape[1] == len(feat_cols)

    def test_target_not_in_X(self, xy):
        X, _, _ = xy
        assert "Target" not in X.columns

    def test_raw_ohlcv_not_in_X(self, xy):
        X, _, _ = xy
        for col in [f"{TICKER}_Close", f"{TICKER}_High", f"{TICKER}_Low",
                    f"{TICKER}_Open", f"{TICKER}_Volume"]:
            assert col not in X.columns

    def test_no_nans_in_X(self, xy):
        X, y, _ = xy
        assert X.isna().sum().sum() == 0
        assert y.isna().sum() == 0

    def test_feature_count_reasonable(self, xy):
        _, _, feat_cols = xy
        # We expect at least 30 engineered features
        assert len(feat_cols) >= 30
