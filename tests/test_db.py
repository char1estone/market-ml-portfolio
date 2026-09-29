"""Tests for the SQLite storage layer, using a temp on-disk DB per test."""

import pandas as pd

from src.data.db import get_engine, latest_date, load_prices, upsert_prices


def _sample_df(ticker="TEST", n=5, start="2024-01-01"):
    dates = pd.date_range(start, periods=n, freq="D").date
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "open": range(n),
        "high": range(n),
        "low": range(n),
        "close": range(n),
        "adj_close": range(n),
        "volume": range(n),
    })


def test_upsert_and_load_roundtrip(tmp_path):
    engine = get_engine(str(tmp_path / "test.db"))
    df = _sample_df()
    n_written = upsert_prices(engine, df)
    assert n_written == 5

    loaded = load_prices(engine, "TEST")
    assert len(loaded) == 5
    assert list(loaded["close"]) == [0, 1, 2, 3, 4]


def test_upsert_is_idempotent(tmp_path):
    engine = get_engine(str(tmp_path / "test.db"))
    df = _sample_df()
    upsert_prices(engine, df)
    upsert_prices(engine, df)  # re-run the same data

    loaded = load_prices(engine, "TEST")
    assert len(loaded) == 5  # no duplicates


def test_upsert_updates_existing_row(tmp_path):
    engine = get_engine(str(tmp_path / "test.db"))
    df = _sample_df()
    upsert_prices(engine, df)

    revised = df.copy()
    revised.loc[0, "close"] = 999
    upsert_prices(engine, revised)

    loaded = load_prices(engine, "TEST")
    assert loaded.iloc[0]["close"] == 999
    assert len(loaded) == 5


def test_latest_date_returns_none_for_empty(tmp_path):
    engine = get_engine(str(tmp_path / "test.db"))
    assert latest_date(engine, "NOPE") is None


def test_latest_date_returns_most_recent(tmp_path):
    engine = get_engine(str(tmp_path / "test.db"))
    df = _sample_df(n=5, start="2024-01-01")
    upsert_prices(engine, df)
    result = latest_date(engine, "TEST")
    assert result == pd.Timestamp("2024-01-05")
