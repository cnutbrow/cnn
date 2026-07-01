"""
Merge per-symbol bookTicker (best bid/ask) and aggTrades parquet shards into a
single time-ordered panel per symbol, ready for feature engineering.

We do NOT have a full depth book (see download_binance.py docstring), so
"queue depth" here means best-bid/best-ask quantity only, not 10 levels.

Usage:
    python3 -m data_pipeline.build_quote_trade_panel --config configs/config.yaml
"""
import argparse

import pandas as pd

from common.config import load_config, resolve


def load_stream(raw_dir, market, stream, symbol) -> pd.DataFrame:
    paths = sorted((raw_dir / market / stream / symbol).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {stream} data for {symbol} under {raw_dir}/{market}/{stream}/{symbol}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def build_panel(raw_dir, market, symbol) -> pd.DataFrame:
    bt = load_stream(raw_dir, market, "bookTicker", symbol)
    bt["ts"] = pd.to_datetime(bt["transaction_time"], unit="ms", utc=True)
    bt = bt.sort_values("ts")[
        ["ts", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]
    ]
    bt["source"] = "quote"

    tr = load_stream(raw_dir, market, "aggTrades", symbol)
    tr["ts"] = pd.to_datetime(tr["transact_time"], unit="ms", utc=True)
    tr = tr.sort_values("ts")[["ts", "price", "quantity", "is_buyer_maker"]]
    # convention: is_buyer_maker True => the aggressor was a SELL (hit the bid)
    tr["signed_qty"] = tr["quantity"] * tr["is_buyer_maker"].map({True: -1, False: 1})
    tr["source"] = "trade"

    panel = pd.concat([bt, tr], ignore_index=True).sort_values("ts").reset_index(drop=True)
    # forward-fill quotes onto trade rows so every trade has a contemporaneous best bid/ask
    panel[["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]] = panel[
        ["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]
    ].ffill()
    panel = panel.dropna(subset=["best_bid_price", "best_ask_price"]).reset_index(drop=True)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(resolve("configs/config.yaml")))
    args = ap.parse_args()
    cfg = load_config(args.config)

    raw_dir = resolve(cfg["data"]["raw_dir"])
    processed_dir = resolve(cfg["data"]["processed_dir"])
    market = cfg["data"]["market"]

    for symbol in cfg["symbols"]:
        panel = build_panel(raw_dir, market, symbol)
        out_path = processed_dir / "panel" / f"{symbol}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out_path, index=False)
        print(f"{symbol}: {len(panel):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
