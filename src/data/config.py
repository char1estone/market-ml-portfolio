"""
Central place for which instruments this project tracks.

Futures data on Yahoo Finance is continuous-contract data (it splices
front-month contracts together), which is good enough for research and
portfolio purposes but is NOT what a trading desk would use for live
trading (roll adjustments differ from what a broker/exchange feed gives
you). This is called out explicitly in the README's limitations section.

If you have access to a paid data vendor (Polygon, Databento, Interactive
Brokers, etc.) later, the only file that needs to change is fetch.py's
`_download_one` function — everything downstream (storage, features,
models, backtest, dashboard) is vendor-agnostic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    ticker: str          # Yahoo Finance symbol
    name: str             # human-readable name
    kind: str             # "future", "index", "rate"


# Core instruments we care about
INSTRUMENTS = [
    Instrument("ES=F", "S&P 500 E-mini Futures", "future"),
    Instrument("NQ=F", "Nasdaq 100 E-mini Futures", "future"),
    # ETF proxies -- useful as a sanity check / longer history than futures
    Instrument("SPY", "S&P 500 ETF", "index"),
    Instrument("QQQ", "Nasdaq 100 ETF", "index"),
    # Macro / cross-asset features
    Instrument("^VIX", "CBOE Volatility Index", "index"),
    Instrument("^TNX", "10-Year Treasury Yield", "rate"),
]

# The instrument we build the primary classifier for.
PRIMARY_TICKER = "ES=F"

DB_PATH = "data/market.db"

# Earliest date to attempt to pull. Yahoo's ES=F/NQ=F history starts
# roughly around 2000; earlier requests simply return whatever exists.
DEFAULT_START = "2000-01-01"
