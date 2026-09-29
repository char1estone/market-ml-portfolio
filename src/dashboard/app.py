"""
Streamlit dashboard: price/indicator exploration, walk-forward model
results, and an interactive backtest.

Run:
    streamlit run src/dashboard/app.py

Deploy for free on Streamlit Community Cloud by pointing it at this file
in your GitHub repo -- gives you a live link to put in your CV/portfolio
rather than "clone this repo to see it work".
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow running as `streamlit run src/dashboard/app.py` from repo root
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.backtest.simulator import result_summary, run_backtest
from src.data.config import DB_PATH, INSTRUMENTS, PRIMARY_TICKER
from src.data.db import get_engine, load_prices
from src.features.indicators import FEATURE_COLUMNS, build_feature_frame
from src.models.walk_forward import walk_forward_splits

st.set_page_config(page_title="Futures ML Explorer", layout="wide")

TICKER_OPTIONS = {inst.name: inst.ticker for inst in INSTRUMENTS}


@st.cache_resource
def _engine():
    return get_engine(DB_PATH)


@st.cache_data(ttl=3600)
def _load(ticker: str) -> pd.DataFrame:
    return load_prices(_engine(), ticker)


st.title("Futures & Index ML Explorer")
st.caption(
    "Historical data, technical indicators, walk-forward-validated direction "
    "models, and a cost-aware backtest for S&P/Nasdaq futures."
)

with st.sidebar:
    st.header("Settings")
    name = st.selectbox("Instrument", list(TICKER_OPTIONS.keys()),
                         index=list(TICKER_OPTIONS.values()).index(PRIMARY_TICKER))
    ticker = TICKER_OPTIONS[name]
    st.markdown("---")
    cost_bps = st.slider("Round-trip cost per trade (bps)", 0, 20, 2)
    n_splits = st.slider("Walk-forward folds", 3, 8, 5)
    st.markdown("---")
    st.caption(
        "Data source: Yahoo Finance continuous futures / ETF proxies. "
        "Not adjusted the way a broker feed would be -- see README for details."
    )

raw = _load(ticker)

if raw.empty:
    st.error(
        f"No data stored for {ticker} yet. Run `python -m src.data.fetch --full` "
        "from the project root first, then reload this page."
    )
    st.stop()

tab_prices, tab_model, tab_backtest = st.tabs(["Prices & Indicators", "Model Performance", "Backtest"])

# ---------------------------------------------------------------- Prices
with tab_prices:
    st.subheader(f"{name} ({ticker})")
    feat = build_feature_frame(raw)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=feat.index, open=feat["open"], high=feat["high"],
        low=feat["low"], close=feat["close"], name="Price",
    ))
    fig.add_trace(go.Scatter(x=feat.index, y=feat["sma_20"], name="SMA 20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=feat.index, y=feat["sma_50"], name="SMA 50", line=dict(width=1)))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**RSI (14)**")
        st.line_chart(feat["rsi_14"])
    with col2:
        st.markdown("**MACD**")
        st.line_chart(feat[["macd", "macd_signal"]])

    st.markdown("**20-day annualized volatility**")
    st.line_chart(feat["volatility_20"])

# ---------------------------------------------------------------- Model
with tab_model:
    st.subheader("Walk-forward direction classification")
    st.caption(
        "Time-ordered expanding-window validation (never trains on future data). "
        "The bar to beat is the persistence and majority-class baselines, not 50%."
    )

    feat = build_feature_frame(raw)
    X = feat[FEATURE_COLUMNS].values
    y = feat["target_direction"].values
    folds = walk_forward_splits(feat.index, n_splits=n_splits)

    from src.models.train import evaluate, majority_class_baseline, naive_last_direction_baseline

    rows = []
    for fold in folds:
        X_train, X_test = X[fold.train_idx], X[fold.test_idx]
        y_train, y_test = y[fold.train_idx], y[fold.test_idx]
        scaler = StandardScaler().fit(X_train)

        clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
        clf.fit(scaler.transform(X_train), y_train)
        preds = clf.predict(scaler.transform(X_test))

        for label, p in [
            ("random_forest", preds),
            ("baseline_persistence", naive_last_direction_baseline(y_train, y_test)),
            ("baseline_majority", majority_class_baseline(y_train, y_test)),
        ]:
            m = evaluate(y_test, p)
            rows.append({"model": label, "test_end": fold.test_end_date.date(), **m})

    results_df = pd.DataFrame(rows)
    summary = results_df.groupby("model")[["accuracy", "precision", "recall", "f1"]].mean().sort_values(
        "accuracy", ascending=False
    )
    st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)

    st.markdown("**Accuracy by fold**")
    pivot = results_df.pivot(index="test_end", columns="model", values="accuracy")
    st.line_chart(pivot)

# ---------------------------------------------------------------- Backtest
with tab_backtest:
    st.subheader("Strategy backtest (long/flat on model signal)")

    last_fold = folds[-1]
    X_train, X_test = X[last_fold.train_idx], X[last_fold.test_idx]
    y_train = y[last_fold.train_idx]
    scaler = StandardScaler().fit(X_train)
    clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
    clf.fit(scaler.transform(X_train), y_train)
    test_preds = clf.predict(scaler.transform(X_test))

    test_dates = feat.index[last_fold.test_idx]
    test_actual_returns = feat["target_return"].values[last_fold.test_idx]

    result = run_backtest(test_dates, test_actual_returns, test_preds, cost_per_trade_bps=cost_bps)

    metrics = result_summary(result)
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k.replace("_", " ").title(), v)

    curve_df = pd.DataFrame({
        "Strategy": result.equity_curve,
        "Buy & Hold": result.buy_hold_curve,
    })
    st.markdown("**Equity curve (final walk-forward test block, out-of-sample)**")
    st.line_chart(curve_df)

    st.caption(
        "This backtest covers only the final out-of-sample fold, on one instrument, "
        "with a simple long/flat rule -- treat it as a methodology demo, not a "
        "performance claim. See README's limitations section."
    )
