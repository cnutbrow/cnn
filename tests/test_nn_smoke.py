import numpy as np
import torch

from models.nn.dataset import LOBSequenceDataset, build_snapshot_grid
from models.nn.model import build_model
from tests.test_pipeline_smoke import make_synthetic_panel
from features.microstructure import compute_all_features
from features.resample import resample_to_bars, add_labels


def test_nn_dataset_and_model_smoke():
    panel = make_synthetic_panel()
    grid = build_snapshot_grid(panel)
    assert len(grid) > 1000

    feat_cfg = {
        "vol_windows_sec": [10, 60], "vpin_bucket_volume": 5000, "vpin_num_buckets": 10,
        "spread_rolling_windows_sec": [30], "ofi_lookback_ticks": 20,
    }
    feat = compute_all_features(panel, feat_cfg)
    bars = resample_to_bars(feat, horizon_sec=30, vol_windows=[10, 60], spread_windows=[30])
    bars = add_labels(bars, horizon_bars=1)

    ds = LOBSequenceDataset(grid, bars, lookback=20)
    assert len(ds) > 100
    x, y = ds[0]
    assert x.shape == (20, 5)
    assert y.item() in (0, 1)

    for arch in ["gru", "cnn", "transformer"]:
        model = build_model(arch, n_features=5, hidden_dim=8, num_layers=1, dropout=0.1)
        batch = torch.stack([ds[i][0] for i in range(4)])
        out = model(batch)
        assert out.shape == (4, 2)


if __name__ == "__main__":
    test_nn_dataset_and_model_smoke()
    print("nn smoke test passed")
