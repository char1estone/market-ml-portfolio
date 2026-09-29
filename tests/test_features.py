"""
Tests for the feature engineering module, with particular focus on the
thing most likely to quietly break a project like this: lookahead bias.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    FEATURE_COLUMNS,
    add_moving_averages,
    add_returns,
    add_target,
    build_feature_frame,
)


@pytest.fixture
def sample_ohlcv():
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame(
        {
            "open": price * (1 + rng.normal(0, 0.001, n)),
            "high": price * (1 + np.abs(rng.normal(0, 0.003, n))),
            "low": price * (1 - np.abs(rng.normal(0, 0.003, n))),
            "close": price,
            "adj_close": price,
            "volume": rng.integers(1000, 10000, n),
        },
        index=dates,
    )
    return df


def test_target_is_shifted_forward(sample_ohlcv):
    """target_return at time t must equal the return realized *after* t,
    not the return that already happened by t (that would be lookahead)."""
    df = add_target(sample_ohlcv, horizon=1)
    expected = df["adj_close"].shift(-1) / df["adj_close"] - 1
    pd.testing.assert_series_equal(df["target_return"], expected, check_names=False)


def test_features_do_not_use_target_column(sample_ohlcv):
    """The feature columns used for modeling must never include the target."""
    assert "target_return" not in FEATURE_COLUMNS
    assert "target_direction" not in FEATURE_COLUMNS


def test_moving_average_only_uses_past_and_present(sample_ohlcv):
    """A rolling mean at row i must be computable from rows <= i alone --
    changing a future row's price must not change past SMA values."""
    df1 = add_moving_averages(sample_ohlcv.copy())
    modified = sample_ohlcv.copy()
    modified.iloc[-1, modified.columns.get_loc("adj_close")] *= 2
    df2 = add_moving_averages(modified)

    # all rows except ones whose rolling window includes the last row
    # should be unaffected
    pd.testing.assert_series_equal(
        df1["sma_5"].iloc[:-5], df2["sma_5"].iloc[:-5], check_names=False
    )


def test_build_feature_frame_has_no_nans(sample_ohlcv):
    feat = build_feature_frame(sample_ohlcv)
    assert not feat[FEATURE_COLUMNS + ["target_direction"]].isna().any().any()


def test_build_feature_frame_drops_warmup_rows(sample_ohlcv):
    """The longest rolling window (200) should cost us the first ~200 rows."""
    feat = build_feature_frame(sample_ohlcv)
    assert len(feat) < len(sample_ohlcv)
    assert len(feat) > len(sample_ohlcv) - 210  # sanity bound, not exact
