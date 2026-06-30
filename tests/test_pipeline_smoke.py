"""
Smoke test on synthetic data: exercises the full feature -> resample -> fold ->
backtest path without needing network access, to catch wiring bugs.
"""
import numpy as np
import pandas as pd

from backtest.simulator import run_backtest
from eval.walk_forward import make_folds
from features.microstructure import compute_all_features
from features.resample import resample_to_bars, add_labels


def make_synthetic_panel(n_quotes=20000, n_trades=8000, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01", tz="UTC")

    q_ts = start + pd.to_timedelta(np.sort(rng.uniform(0, 30 * 86400, n_quotes)), unit="s")
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n_quotes))
    spread = np.abs(rng.normal(0.05, 0.01, n_quotes)) + 0.01
    bid_qty = np.abs(rng.normal(5, 2, n_quotes)) + 0.1
    ask_qty = np.abs(rng.normal(5, 2, n_quotes)) + 0.1
    quotes = pd.DataFrame({
        "ts": q_ts, "best_bid_price": mid - spread / 2, "best_bid_qty": bid_qty,
        "best_ask_price": mid + spread / 2, "best_ask_qty": ask_qty, "source": "quote",
    })

    t_ts = start + pd.to_timedelta(np.sort(rng.uniform(0, 30 * 86400, n_trades)), unit="s")
    price = 100 + np.cumsum(rng.normal(0, 0.01, n_trades))
    qty = np.abs(rng.normal(1, 0.5, n_trades)) + 0.01
    is_buyer_maker = rng.random(n_trades) > 0.5
    trades = pd.DataFrame({
        "ts": t_ts, "price": price, "quantity": qty, "is_buyer_maker": is_buyer_maker,
        "signed_qty": qty * np.where(is_buyer_maker, -1, 1), "source": "trade",
    })

    panel = pd.concat([quotes, trades], ignore_index=True).sort_values("ts").reset_index(drop=True)
    panel[["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]] = panel[
        ["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]
    ].ffill()
    return panel.dropna(subset=["best_bid_price", "best_ask_price"]).reset_index(drop=True)


def test_full_pipeline_smoke():
    panel = make_synthetic_panel()
    feat_cfg = {
        "vol_windows_sec": [10, 60],
        "vpin_bucket_volume": 5000,
        "vpin_num_buckets": 10,
        "spread_rolling_windows_sec": [30],
        "ofi_lookback_ticks": 20,
    }
    feat = compute_all_features(panel, feat_cfg)
    assert "vpin" in feat.columns and "ofi" in feat.columns

    bars = resample_to_bars(feat, horizon_sec=30, vol_windows=[10, 60], spread_windows=[30])
    bars = add_labels(bars, horizon_bars=1)
    assert len(bars) > 1000
    assert set(bars["label_direction"].dropna().unique()) <= {-1, 0, 1}

    folds = make_folds(bars, train_days=14, val_days=3, test_days=3, step_days=5, purge_sec=30)
    assert len(folds) >= 1

    fold = folds[0]
    pred = pd.Series(np.sign(fold.test["fwd_return"].fillna(0)).replace(0, 1), index=fold.test.index)
    bt = run_backtest(fold.test, pred, {
        "taker_fee_bps": 10, "maker_fee_bps": 10, "slippage_bps": 2, "position_size_usd": 1000,
    })
    assert "net_pnl_bps" in bt and np.isfinite(bt["net_pnl_bps"])


if __name__ == "__main__":
    test_full_pipeline_smoke()
    print("smoke test passed")
