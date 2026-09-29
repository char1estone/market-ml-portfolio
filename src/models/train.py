"""
Train and evaluate direction-classification models using walk-forward
validation, benchmarked against naive baselines.

Framing: predicting next-day *direction* (up/down) as a binary
classification problem, not predicting price levels. This is deliberate --
see README.md "Honest limitations" for why price-level prediction is a
much weaker signal to chase, and why this project treats ~55-58% direction
accuracy as a realistic ceiling rather than a bug.

Run:
    python -m src.models.train
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.data.config import DB_PATH, PRIMARY_TICKER
from src.data.db import get_engine, load_prices
from src.features.indicators import FEATURE_COLUMNS, build_feature_frame
from src.models.walk_forward import walk_forward_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    model_name: str
    train_end: str
    test_end: str
    n_train: int
    n_test: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def naive_last_direction_baseline(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Predict tomorrow repeats today's realized direction (persistence baseline)."""
    # shift by one within the test block; first test-row falls back to the
    # last training label
    preds = np.empty_like(y_test)
    preds[0] = y_train[-1]
    preds[1:] = y_test[:-1]
    return preds


def majority_class_baseline(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Predict the majority class observed in training (accounts for market's long-run up-bias)."""
    majority = int(np.round(y_train.mean()))
    return np.full_like(y_test, fill_value=majority)


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_walk_forward(df: pd.DataFrame, n_splits: int = 5) -> list[FoldResult]:
    X = df[FEATURE_COLUMNS].values
    y = df["target_direction"].values
    folds = walk_forward_splits(df.index, n_splits=n_splits)

    results: list[FoldResult] = []

    models = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42
        ),
    }

    for fold in folds:
        X_train, X_test = X[fold.train_idx], X[fold.test_idx]
        y_train, y_test = y[fold.train_idx], y[fold.test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        # baselines
        for name, preds in [
            ("baseline_persistence", naive_last_direction_baseline(y_train, y_test)),
            ("baseline_majority_class", majority_class_baseline(y_train, y_test)),
        ]:
            m = evaluate(y_test, preds)
            results.append(FoldResult(
                model_name=name,
                train_end=str(fold.train_end_date.date()),
                test_end=str(fold.test_end_date.date()),
                n_train=len(y_train), n_test=len(y_test),
                **m,
            ))

        # real models
        for name, make_model in models.items():
            clf = make_model()
            clf.fit(X_train_s, y_train)
            preds = clf.predict(X_test_s)
            m = evaluate(y_test, preds)
            results.append(FoldResult(
                model_name=name,
                train_end=str(fold.train_end_date.date()),
                test_end=str(fold.test_end_date.date()),
                n_train=len(y_train), n_test=len(y_test),
                **m,
            ))

    return results


def summarize(results: list[FoldResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in results])
    summary = df.groupby("model_name")[["accuracy", "precision", "recall", "f1"]].mean()
    return summary.sort_values("accuracy", ascending=False)


def main():
    engine = get_engine(DB_PATH)
    raw = load_prices(engine, PRIMARY_TICKER)
    if raw.empty:
        raise SystemExit(
            f"No data for {PRIMARY_TICKER} found in {DB_PATH}. "
            "Run `python -m src.data.fetch --full` first."
        )

    feat = build_feature_frame(raw)
    logger.info("Feature frame: %d rows, %d features", len(feat), len(FEATURE_COLUMNS))

    results = run_walk_forward(feat, n_splits=5)

    out_path = "data/walk_forward_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("Wrote per-fold results to %s", out_path)

    summary = summarize(results)
    print("\nMean metrics across folds (higher accuracy than both baselines is the bar):\n")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
