# Futures & Index ML Explorer

A small, end-to-end project on S&P 500 / Nasdaq futures: a data pipeline,
technical-indicator features, a direction-classification model validated
the way time series actually needs to be validated, and a cost-aware
backtest — all surfaced in an interactive dashboard.

This project is not a claim that markets are predictable with this
approach. It's a demonstration of doing the *methodology* correctly:
proper time-ordered validation, honest baselines, and reporting results
including where the model doesn't add value. See
[Honest limitations](#honest-limitations-please-read) below.

## Architecture

```
src/
  data/         Fetch (Yahoo Finance) + SQLite storage, upsert-based, idempotent
  features/     Technical indicators (RSI, MACD, Bollinger, moving averages, volatility)
  models/       Walk-forward (expanding-window) validation + classifiers vs. baselines
  backtest/     Long/flat simulator with transaction costs, Sharpe/drawdown reporting
  dashboard/    Streamlit app tying it all together
tests/          pytest suite, incl. a regression test against lookahead bias
.github/workflows/ci.yml   Runs the test suite on every push/PR
```

Data flows one direction: `fetch → SQLite → features → model → backtest → dashboard`.
Each stage reads from the previous stage's output, not from raw files, so
any stage can be re-run independently.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Pull full history for all configured instruments (ES=F, NQ=F, SPY, QQQ, ^VIX, ^TNX)
python -m src.data.fetch --full

# Run the walk-forward model evaluation from the command line
python -m src.models.train

# Run the test suite
pytest -v

# Launch the interactive dashboard
streamlit run src/dashboard/app.py
```

Re-running `python -m src.data.fetch` (without `--full`) only pulls data
since the last stored date — safe to run daily/on a schedule.

## What's actually interesting here (for a reviewer)

- **Time-based validation, not random splits.** `src/models/walk_forward.py`
  implements expanding-window walk-forward validation and is unit-tested
  (`tests/test_walk_forward.py`) to guarantee no test fold's index ever
  precedes data it was trained on. Shuffled k-fold cross-validation on
  time series data is a very common and very serious mistake in
  market-prediction projects — it leaks future information into training
  and produces accuracy numbers that look great and mean nothing live.
- **Baselines, always.** Every model is compared against a persistence
  baseline (predict tomorrow repeats today) and a majority-class baseline,
  in the same walk-forward folds. A model that doesn't beat both isn't
  adding signal, whatever its raw accuracy looks like.
- **Costs are not an afterthought.** The backtest (`src/backtest/simulator.py`)
  charges a configurable per-trade cost and reports strategy vs. buy-and-hold
  side by side — a common failure mode in these projects is a strategy
  that "predicts direction well" but loses to buy-and-hold once realistic
  costs are included, and this project is built to make that visible
  rather than hide it.
- **No lookahead bias.** `tests/test_features.py` includes a regression
  test asserting the target column is strictly forward-shifted and that
  changing a future price cannot alter a historical feature value.

## Honest limitations (please read)

This section exists on purpose. A project that predicts markets with
suspiciously high accuracy is a red flag to anyone who has worked with
financial data before; being explicit about where this falls short is
more credible than pretending it doesn't.

### What running this on real data actually showed

Running `python -m src.models.train` on real ES=F data (2000-2026, 5
walk-forward folds) produced this mean accuracy across folds:

| model | accuracy | precision | recall | f1 |
|---|---|---|---|---|
| baseline_majority_class | 0.541 | 0.541 | 1.000 | 0.702 |
| logistic_regression | 0.541 | 0.544 | 0.941 | 0.689 |
| random_forest | 0.514 | 0.535 | 0.810 | 0.639 |
| baseline_persistence | 0.492 | 0.528 | 0.528 | 0.528 |

Neither real model beats the majority-class baseline ("always predict
up") on this data. This is the expected, honest result the section above
warned about, not a bug — it means the technical indicators used here
don't carry predictive signal beyond the market's long-run upward drift,
at daily granularity, over this period.

**The backtest result is also highly sensitive to an arbitrary
configuration choice.** The dashboard's Backtest tab only evaluates the
*final* walk-forward fold, and which calendar period ends up as "the
final fold" depends entirely on how many folds you choose. On NQ=F, with
otherwise identical settings:

- `n_splits=3` → strategy total return **105.11%** vs. buy-and-hold **101.04%** (looks like it wins)
- `n_splits=5` → strategy total return **75.68%** vs. buy-and-hold **102.74%** (loses)
- `n_splits=8` → strategy total return **28.90%** vs. buy-and-hold **50.98%** (loses badly)

Same model, same instrument, same code — three very different verdicts,
purely from how the timeline happened to get sliced. This is a genuine
limitation of evaluating on a single held-out period rather than
averaging across all folds, and it's the clearest concrete argument in
this whole project for never trusting a single backtest number.

### Other limitations

- **Direction accuracy is close to a coin flip.** Realistically, expect
  the classifiers here to land around 51-56% next-day direction accuracy
  — only a few points above the baselines, if at all, and not
  consistently across folds or instruments (see table above). That's in
  line with what efficient-markets research would predict for a model
  using only price-derived features at daily granularity. Treat any
  single run that shows much higher accuracy as a sign to check for a bug
  (most likely lookahead), not as a discovery.
- **Yahoo Finance futures data is continuous-contract data**, spliced
  across front-month contracts as they roll. This is fine for research
  and portfolio purposes but is not the same as a broker/exchange feed,
  and roll-adjustment methodology differs from what you'd get from a
  real trading desk. Not suitable for anything beyond research use.
- **The backtest is intentionally simple** — long/flat only, no shorting,
  no position sizing, no slippage model beyond a flat per-trade cost, and
  (as shown above) it evaluates only the final walk-forward fold on one
  instrument at a time, which makes any single result sensitive to fold
  count. It's built to demonstrate sound backtesting *methodology*, not
  to produce a number you could put in front of an investor.
- **Survivorship / instrument scope is narrow.** Only a handful of
  instruments are tracked; there's no cross-sectional or multi-asset
  portfolio construction here.
- **This is not investment advice**, and nothing in this repo should be
  used to make real trading decisions.

## Extending this project

Ideas if you want to take it further:
- Add regime detection (e.g. an HMM on volatility) and condition the
  model on regime.
- Add cross-asset features (bond yields, currency futures, credit
  spreads) as additional inputs.
- Compare against a gradient-boosted tree model (LightGBM/XGBoost) —
  scaffolding for it is straightforward to add alongside the existing
  logistic regression / random forest in `src/models/train.py`.
- Swap the data source for a paid vendor (Polygon, Databento, Interactive
  Brokers) for cleaner futures data — only `src/data/fetch.py`'s
  `_download_one` needs to change; storage and everything downstream is
  vendor-agnostic.

## Deploying the dashboard

The dashboard is a single Streamlit app (`src/dashboard/app.py`) with no
external services beyond the SQLite file it reads from. Free options to
get a live link for a CV/portfolio:
1. Push this repo to GitHub.
2. Sign up at [share.streamlit.io](https://share.streamlit.io) (Streamlit
   Community Cloud) and point it at `src/dashboard/app.py`.
3. Note: the deployed app needs the SQLite DB populated — either commit a
   small pre-fetched `data/market.db` (fine for a demo; the `.gitignore`
   currently excludes it, so remove that line if you want to commit it),
   or add a startup step that runs `python -m src.data.fetch --full`.
