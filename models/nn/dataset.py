"""
Builds raw/lightly-processed LOB snapshot sequences for the NN, as opposed to
the heavily hand-engineered tabular features the GBM uses. Each snapshot is
just normalized best-quote state (log return, relative spread, queue ratio,
log sizes) on a fine fixed grid; the model has to learn temporal structure
itself instead of being handed VPIN/OFI/realized-vol.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SNAPSHOT_GRID_SEC = 1  # fine grid the raw sequence is sampled on

RAW_COLS = ["log_ret", "rel_spread", "queue_depth_ratio", "log_bid_qty", "log_ask_qty"]


def build_snapshot_grid(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.set_index("ts")
    grid = df[["best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty"]].resample(
        f"{SNAPSHOT_GRID_SEC}s"
    ).last().ffill()
    mid = (grid["best_bid_price"] + grid["best_ask_price"]) / 2
    grid["log_ret"] = np.log(mid).diff().fillna(0.0)
    grid["rel_spread"] = (grid["best_ask_price"] - grid["best_bid_price"]) / mid
    denom = grid["best_bid_qty"] + grid["best_ask_qty"]
    grid["queue_depth_ratio"] = grid["best_bid_qty"] / denom.replace(0, np.nan)
    grid["log_bid_qty"] = np.log1p(grid["best_bid_qty"])
    grid["log_ask_qty"] = np.log1p(grid["best_ask_qty"])
    return grid[RAW_COLS].fillna(0.0)


class LOBSequenceDataset(Dataset):
    """Aligns `lookback` consecutive fine-grid snapshots ending at each bar
    timestamp with that bar's forward-direction label."""

    def __init__(self, snapshot_grid: pd.DataFrame, bars: pd.DataFrame, lookback: int):
        self.lookback = lookback
        grid_idx = snapshot_grid.index
        values = snapshot_grid.values.astype(np.float32)

        # normalize using only this dataset's own stats (fold-local, fit on train, applied to val/test by caller)
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0) + 1e-8

        labels = bars["label_direction"].dropna()
        labels = labels[labels != 0]

        samples = []
        targets = []
        bar_index = []
        pos = grid_idx.searchsorted(labels.index)
        for i, (ts, label) in enumerate(labels.items()):
            end = pos[i]
            start = end - lookback
            if start < 0:
                continue
            samples.append((start, end))
            targets.append(1 if label == 1 else 0)
            bar_index.append(ts)

        self.values = values
        self.samples = samples
        self.targets = np.array(targets, dtype=np.int64)
        self.bar_index = pd.DatetimeIndex(bar_index)

    def set_norm_stats(self, mean, std):
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start, end = self.samples[idx]
        seq = (self.values[start:end] - self.mean) / self.std
        return torch.from_numpy(seq), torch.tensor(self.targets[idx], dtype=torch.long)
