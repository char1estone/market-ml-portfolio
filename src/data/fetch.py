"""
Download OHLCV data from Yahoo Finance and store it via db.py.

Run directly for a one-off full backfill:
    python -m src.data.fetch --full

Or import `update_all()` / `update_ticker()` for incremental updates
(used by the dashboard and can be wired into a cron/scheduled task).
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import yfinance as yf

from src.data.config import DB_PATH, DEFAULT_START, INSTRUMENTS
from src.data.db import get_engine, latest_date, upsert_prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _download_one(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """
    Pull raw OHLCV for one ticker from Yahoo Finance and reshape it into
    the schema db.py expects.

    This is the one function to swap out if you move to a paid vendor
    (Polygon, Databento, IBKR, etc.) later -- everything downstream is
    written against the `prices` table, not against yfinance's shape.
    """
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    # yfinance sometimes names it "adj_close", sometimes "adj close"
    rename_map = {"adj_close": "adj_close", "date": "date"}
    raw = raw.rename(columns=rename_map)

    out = pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(raw["date"]).dt.date,
            "open": raw.get("open"),
            "high": raw.get("high"),
            "low": raw.get("low"),
            "close": raw.get("close"),
            "adj_close": raw.get("adj_close", raw.get("close")),
            "volume": raw.get("volume"),
        }
    )
    return out


def update_ticker(engine, ticker: str, full: bool = False) -> int:
    """
    Update stored data for a single ticker. Incremental by default: only
    fetches from the day after the last stored date. Pass full=True to
    re-pull everything from DEFAULT_START (useful if you suspect stale
    or corrected data).
    """
    if full:
        start = DEFAULT_START
    else:
        last = latest_date(engine, ticker)
        if last is None:
            start = DEFAULT_START
        else:
            start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Fetching %s from %s", ticker, start)
    df = _download_one(ticker, start=start)
    n = upsert_prices(engine, df)
    logger.info("  -> %d rows written for %s", n, ticker)
    return n


def update_all(full: bool = False) -> None:
    engine = get_engine(DB_PATH)
    for inst in INSTRUMENTS:
        try:
            update_ticker(engine, inst.ticker, full=full)
        except Exception:
            logger.exception("Failed to update %s", inst.ticker)
        time.sleep(1)  # be polite to the free endpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch/update market data")
    parser.add_argument(
        "--full", action="store_true", help="Re-pull full history instead of incremental update"
    )
    args = parser.parse_args()
    update_all(full=args.full)
