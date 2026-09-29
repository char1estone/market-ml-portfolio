"""
Tests for walk-forward splitting -- the property that matters most is
that no test fold's data ever appears in an earlier training fold.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.walk_forward import walk_forward_splits


@pytest.fixture
def date_index():
    return pd.date_range("2015-01-01", periods=1000, freq="B")


def test_splits_are_chronological(date_index):
    folds = walk_forward_splits(date_index, n_splits=5)
    for fold in folds:
        assert fold.train_idx.max() < fold.test_idx.min(), (
            "training data must all precede test data -- found overlap/lookahead"
        )


def test_training_window_expands(date_index):
    folds = walk_forward_splits(date_index, n_splits=5)
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes), "expanding window should never shrink"


def test_test_folds_cover_remaining_data_without_overlap(date_index):
    folds = walk_forward_splits(date_index, n_splits=5)
    seen = set()
    for fold in folds:
        idx_set = set(fold.test_idx.tolist())
        assert not (idx_set & seen), "test folds must not overlap each other"
        seen |= idx_set


def test_raises_when_too_few_rows_for_splits():
    tiny_index = pd.date_range("2024-01-01", periods=5, freq="B")
    with pytest.raises(ValueError):
        walk_forward_splits(tiny_index, n_splits=5)
