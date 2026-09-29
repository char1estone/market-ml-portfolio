"""Tests for the backtest simulator."""

import numpy as np
import pandas as pd

from src.backtest.simulator import run_backtest


def test_always_long_matches_buy_and_hold_when_costs_zero():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0005, 0.01, 100)
    always_long = np.ones(100)

    result = run_backtest(dates, returns, always_long, cost_per_trade_bps=0)
    assert abs(result.total_return - result.buy_hold_return) < 1e-9


def test_always_flat_has_zero_return():
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    rng = np.random.default_rng(2)
    returns = rng.normal(0.001, 0.01, 50)
    always_flat = np.zeros(50)

    result = run_backtest(dates, returns, always_flat, cost_per_trade_bps=2)
    assert abs(result.total_return) < 1e-9
    # entering flat from flat should register at most 1 "trade" (the initial state)
    assert result.n_trades <= 1


def test_costs_reduce_returns_for_a_flip_flopping_strategy():
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    returns = np.full(20, 0.0)  # flat market -- any activity should only cost money
    flip_flop = np.array([i % 2 for i in range(20)])

    no_cost = run_backtest(dates, returns, flip_flop, cost_per_trade_bps=0)
    with_cost = run_backtest(dates, returns, flip_flop, cost_per_trade_bps=10)
    assert with_cost.total_return < no_cost.total_return


def test_max_drawdown_is_non_positive():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.02, 100)
    preds = rng.integers(0, 2, 100)

    result = run_backtest(dates, returns, preds)
    assert result.max_drawdown <= 0
