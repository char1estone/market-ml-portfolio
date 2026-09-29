"""
Thin SQLite storage layer for OHLCV data.

Design choices, and why:
- SQLite, not CSVs: one file, queryable, avoids "which CSV is the latest"
  drift, and every other module reads through this module rather than
  touching files directly.
- One `prices` table with a `ticker` column rather than one table per
  ticker: simpler schema, trivial to add instruments, and joins/filters
  are just WHERE clauses.
- Upsert on (ticker, date): re-running the fetch is always safe and
  idempotent -- it will not create duplicate rows, and it will silently
  correct a row if Yahoo revises a value (which does happen for the
  most recent 1-2 days as data settles).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    Float,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

metadata = MetaData()

prices_table = Table(
    "prices",
    metadata,
    Column("ticker", String, primary_key=True),
    Column("date", Date, primary_key=True),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("adj_close", Float),
    Column("volume", Float),
)


def get_engine(db_path: str) -> Engine:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    metadata.create_all(engine)
    return engine


def upsert_prices(engine: Engine, df: pd.DataFrame, batch_size: int = 400) -> int:
    """
    Upsert a DataFrame of OHLCV rows.

    Expects columns: ticker, date, open, high, low, close, adj_close, volume.
    Returns the number of rows written.

    Rows are written in batches rather than one single statement. SQLite
    caps the number of "?" placeholders allowed in a single statement
    (historically 999, higher on newer builds) -- with 8 columns per row,
    a full multi-year backfill (thousands of rows) blows past that limit
    in one shot. batch_size=400 rows * 8 columns = 3200 placeholders per
    statement, comfortably under the limit regardless of SQLite version.
    """
    if df.empty:
        return 0

    records = df.to_dict(orient="records")
    total_written = 0

    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = sqlite_insert(prices_table).values(batch)
            update_cols = {
                c.name: getattr(stmt.excluded, c.name)
                for c in prices_table.columns
                if c.name not in ("ticker", "date")
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "date"], set_=update_cols
            )
            conn.execute(stmt)
            total_written += len(batch)

    return total_written


def load_prices(engine: Engine, ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load stored OHLCV rows for a ticker as a date-indexed DataFrame."""
    query = "SELECT * FROM prices WHERE ticker = :ticker"
    params: dict = {"ticker": ticker}
    if start:
        query += " AND date >= :start"
        params["start"] = start
    if end:
        query += " AND date <= :end"
        params["end"] = end
    query += " ORDER BY date"

    df = pd.read_sql(query, engine, params=params, parse_dates=["date"])
    df = df.set_index("date")
    return df


def latest_date(engine: Engine, ticker: str) -> pd.Timestamp | None:
    """Return the most recent stored date for a ticker, or None if empty."""
    query = "SELECT MAX(date) as max_date FROM prices WHERE ticker = :ticker"
    with engine.connect() as conn:
        result = conn.exec_driver_sql(
            "SELECT MAX(date) as max_date FROM prices WHERE ticker = ?", (ticker,)
        ).fetchone()
    if result is None or result[0] is None:
        return None
    return pd.Timestamp(result[0])
