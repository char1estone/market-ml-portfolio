"""
Technical indicators and target labels, built on top of stored OHLCV data.

Kept deliberately as pure functions of a DataFrame -> DataFrame so they're
easy to unit test and easy to reuse in both the dashboard and the model
training pipeline.

A note on lookahead bias, since it's the single easiest way to accidentally
make a market-prediction project look better than it is: every indicator
here is computed using only data up to and including the current row's
`close`. The *target* column is explicitly shifted into the future
(next-day return), and callers are responsible for never using the target
as a feature. See tests/test_features.py for a regression test on this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def add_returns(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    df = df.copy()
    df["return_1d"] = df[price_col].pct_change()
    df["log_return_1d"] = np.log(df[price_col]).diff()
    return df


def add_moving_averages(df: pd.DataFrame, price_col: str = "adj_close",
                         windows=(5, 10, 20, 50, 200)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df[price_col].rolling(w).mean()
        df[f"ema_{w}"] = df[price_col].ewm(span=w, adjust=False).mean()
        # price relative to the average -- more useful as an ML feature
        # than the raw average, which is not stationary
        df[f"close_to_sma_{w}"] = df[price_col] / df[f"sma_{w}"] - 1
    return df


def add_rsi(df: pd.DataFrame, price_col: str = "adj_close", window: int = 14) -> pd.DataFrame:
    df = df.copy()
    df[f"rsi_{window}"] = ta.momentum.RSIIndicator(df[price_col], window=window).rsi()
    return df


def add_macd(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    df = df.copy()
    macd = ta.trend.MACD(df[price_col])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    return df


def add_bollinger_bands(df: pd.DataFrame, price_col: str = "adj_close", window: int = 20) -> pd.DataFrame:
    df = df.copy()
    bb = ta.volatility.BollingerBands(df[price_col], window=window)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()  # position within the bands, 0-1
    df["bb_width"] = bb.bollinger_wband()
    return df


def add_volatility(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"volatility_{w}"] = df["log_return_1d"].rolling(w).std() * np.sqrt(252)
    return df


def add_target(df: pd.DataFrame, price_col: str = "adj_close", horizon: int = 1) -> pd.DataFrame:
    """
    Add the prediction target: next-`horizon`-day forward return and its
    sign. This is the ONLY forward-looking column in the pipeline --
    everything else uses only data available at time t.
    """
    df = df.copy()
    fwd_return = df[price_col].shift(-horizon) / df[price_col] - 1
    df["target_return"] = fwd_return
    df["target_direction"] = (fwd_return > 0).astype(int)
    return df


def build_feature_frame(raw: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Full pipeline: raw OHLCV -> features + target, ready for modeling.
    Drops rows with NaNs introduced by rolling windows / the shifted target.
    """
    df = raw.sort_index()
    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volatility(df)
    df = add_target(df, horizon=horizon)
    df = df.dropna()
    return df


FEATURE_COLUMNS = [
    "return_1d", "log_return_1d",
    "close_to_sma_5", "close_to_sma_10", "close_to_sma_20", "close_to_sma_50", "close_to_sma_200",
    "rsi_14", "macd", "macd_signal", "macd_diff",
    "bb_pct", "bb_width",
    "volatility_5", "volatility_10", "volatility_20",
]
