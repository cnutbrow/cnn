"""
Resample the tick-level feature panel onto a fixed-horizon bar grid and attach
forward-looking labels for that horizon. Horizon is a parameter, not hardcoded,
so the same panel can produce a 30s grid and a 5min grid for comparison.

Usage:
    python3 -m features.resample --config configs/config.yaml --symbol BTCUSDT --horizon-sec 30
"""
import argparse

import numpy as np
import pandas as pd

from common.config import load_config, resolve
from features.microstructure import compute_all_features


def resample_to_bars(df: pd.DataFrame, horizon_sec: int, vol_windows: list[int],
                      spread_windows: list[int]) -> pd.DataFrame:
    df = df.set_index("ts")
    rule = f"{horizon_sec}s"

    last_cols = (
        ["mid", "microprice", "spread", "rel_spread", "queue_depth_ratio", "ofi",
         "vpin", "time_since_last_trade"]
        + [f"realized_vol_{w}s" for w in vol_windows]
        + [f"spread_mean_{w}s" for w in spread_windows]
        + [f"spread_std_{w}s" for w in spread_windows]
    )
    last_bars = df[last_cols].resample(rule).last()

    is_trade = df["source"] == "trade"
    trade_through_rate = df["trade_through"].where(is_trade).resample(rule).mean()
    signed_vol = df["trade_signed_qty"].resample(rule).sum()
    n_trades = is_trade.resample(rule).sum()
    notional_vol = (df["price"] * df["quantity"]).where(is_trade).resample(rule).sum()

    bars = last_bars.copy()
    bars["trade_through_rate"] = trade_through_rate
    bars["signed_volume"] = signed_vol
    bars["n_trades"] = n_trades
    bars["notional_volume"] = notional_vol
    bars["bar_realized_vol"] = np.log(bars["mid"]).diff().pow(2)  # within-bar move, complements rolling RV features

    bars = bars.ffill().dropna(subset=["mid"])
    return bars


def add_labels(bars: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    """Forward return / direction over one bar (the bar interval IS the horizon)."""
    bars = bars.copy()
    fwd_mid = bars["mid"].shift(-horizon_bars)
    bars["fwd_return"] = (fwd_mid - bars["mid"]) / bars["mid"]
    bars["label_direction"] = np.sign(bars["fwd_return"]).astype("Int64")
    bars.loc[bars["fwd_return"] == 0, "label_direction"] = 0
    return bars.iloc[:-horizon_bars] if horizon_bars > 0 else bars


def build_bars_for_symbol(cfg: dict, symbol: str, horizon_sec: int) -> pd.DataFrame:
    processed_dir = resolve(cfg["data"]["processed_dir"])
    panel = pd.read_parquet(processed_dir / "panel" / f"{symbol}.parquet")
    feat = compute_all_features(panel, cfg["features"])
    bars = resample_to_bars(
        feat, horizon_sec, cfg["features"]["vol_windows_sec"], cfg["features"]["spread_rolling_windows_sec"]
    )
    bars = add_labels(bars, horizon_bars=1)
    return bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(resolve("configs/config.yaml")))
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--horizon-sec", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    bars = build_bars_for_symbol(cfg, args.symbol, args.horizon_sec)
    out_dir = resolve(cfg["data"]["processed_dir"]) / "bars" / f"{args.horizon_sec}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.symbol}.parquet"
    bars.to_parquet(out_path)
    print(f"{args.symbol} @ {args.horizon_sec}s: {len(bars):,} bars -> {out_path}")


if __name__ == "__main__":
    main()
