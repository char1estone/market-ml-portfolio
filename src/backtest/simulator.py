"""
A deliberately simple long/flat backtest simulator.

Strategy: on each day, if the model predicts "up", hold a long position
for the next day's return; if it predicts "down", stay flat (cash). No
shorting, no leverage, no position sizing -- keeping the mechanics simple
makes it easy to reason about, and this is a portfolio project about
demonstrating sound methodology, not a claim of a tradeable edge.

Costs matter more than people expect: a strategy that "predicts direction
well" in isolation can still lose to buy-and-hold once realistic
transaction costs are included, because switching in and out of a
position every time the prediction flips racks up costs fast. This
module reports results with and without costs specifically to make that
visible, per the project's "honest evaluation" ethos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    buy_hold_curve: pd.Series
    total_return: float
    buy_hold_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def run_backtest(
    dates: pd.DatetimeIndex,
    actual_returns: np.ndarray,
    predicted_direction: np.ndarray,
    cost_per_trade_bps: float = 2.0,
) -> BacktestResult:
    """
    actual_returns[t] = realized next-day return earned by being long on day t
    predicted_direction[t] = 1 (go long) or 0 (stay flat) for day t

    cost_per_trade_bps: round-trip cost in basis points charged whenever
    the position changes (entering or exiting). 2bps is a reasonable
    starting assumption for liquid index futures; the dashboard lets you
    vary it.
    """
    position = pd.Series(predicted_direction, index=dates).astype(float)
    returns = pd.Series(actual_returns, index=dates)

    position_change = position.diff().fillna(position.iloc[0]).abs()
    n_trades = int((position_change > 0).sum())

    cost = position_change * (cost_per_trade_bps / 10_000)
    strategy_returns = position * returns - cost

    equity = (1 + strategy_returns).cumprod()
    buy_hold = (1 + returns).cumprod()

    ann_return = equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / len(equity)) - 1
    ann_vol = strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")

    return BacktestResult(
        equity_curve=equity,
        buy_hold_curve=buy_hold,
        total_return=equity.iloc[-1] - 1,
        buy_hold_return=buy_hold.iloc[-1] - 1,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=_max_drawdown(equity),
        n_trades=n_trades,
    )


def result_summary(r: BacktestResult) -> dict:
    return {
        "total_return": f"{r.total_return:.2%}",
        "buy_hold_return": f"{r.buy_hold_return:.2%}",
        "annualized_return": f"{r.annualized_return:.2%}",
        "annualized_vol": f"{r.annualized_vol:.2%}",
        "sharpe_ratio": f"{r.sharpe_ratio:.2f}",
        "max_drawdown": f"{r.max_drawdown:.2%}",
        "n_trades": r.n_trades,
    }
