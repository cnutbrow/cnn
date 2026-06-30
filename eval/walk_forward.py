"""
Purged walk-forward time-series cross-validation.

No random shuffling, no k-fold across overlapping windows: each fold's
train/val/test blocks are contiguous in time and ordered train < val < test,
with a purge gap removed at each boundary so that no label (which looks
`horizon_sec` seconds into the future) leaks across the split.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    fold_id: int


def make_folds(bars: pd.DataFrame, train_days: int, val_days: int, test_days: int,
                step_days: int, purge_sec: int) -> list[Fold]:
    """bars must be indexed by a sorted DatetimeIndex (as produced by resample_to_bars)."""
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must be indexed by DatetimeIndex")
    bars = bars.sort_index()
    start = bars.index.min()
    end = bars.index.max()
    purge = pd.Timedelta(seconds=purge_sec)

    folds = []
    fold_id = 0
    train_start = start
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        val_start = train_end + purge
        val_end = val_start + pd.Timedelta(days=val_days)
        test_start = val_end + purge
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > end:
            break

        train = bars.loc[train_start:train_end - purge]
        val = bars.loc[val_start:val_end - purge]
        test = bars.loc[test_start:test_end]

        if len(train) and len(val) and len(test):
            folds.append(Fold(train=train, val=val, test=test, fold_id=fold_id))
            fold_id += 1

        train_start = train_start + pd.Timedelta(days=step_days)

    if not folds:
        raise ValueError(
            f"No folds produced: data span {(end - start).days} days is too short for "
            f"train={train_days}+val={val_days}+test={test_days} days per fold. "
            f"Shrink the walk_forward config or widen data.start_date/end_date."
        )
    return folds
