"""
Walk-forward (expanding-window) time-series cross-validation.

Why not sklearn's KFold or a random train/test split: shuffling time series
rows lets the model "see the future" during training (e.g. a row from 2015
sits in the training fold while a row from 2010 sits in the test fold),
which inflates accuracy in a way that will not reproduce live. This is one
of the most common mistakes in market-prediction projects and the first
thing an experienced reviewer will check for.

Walk-forward validation instead trains on all data up to time T, tests on
the next block after T, then slides forward -- the same order information
would actually arrive in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end_date: pd.Timestamp
    test_end_date: pd.Timestamp


def walk_forward_splits(index: pd.DatetimeIndex, n_splits: int = 5, min_train_fraction: float = 0.4):
    """
    Yield `n_splits` expanding-window folds over a date-sorted index.

    The first `min_train_fraction` of the data is always in the initial
    training set (a model trained on 3 rows is not meaningful), and the
    remainder is split into n_splits equal-sized test blocks, each
    preceded by all data before it.
    """
    n = len(index)
    min_train = int(n * min_train_fraction)
    remaining = n - min_train
    if remaining < n_splits:
        raise ValueError("Not enough data for the requested number of splits")

    block_size = remaining // n_splits
    folds = []
    for i in range(n_splits):
        test_start = min_train + i * block_size
        test_end = n if i == n_splits - 1 else min_train + (i + 1) * block_size
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        folds.append(
            Fold(
                train_idx=train_idx,
                test_idx=test_idx,
                train_end_date=index[train_idx[-1]],
                test_end_date=index[test_idx[-1]],
            )
        )
    return folds
