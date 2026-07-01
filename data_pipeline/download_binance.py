"""
Download historical market data from Binance Vision (data.binance.vision).

Data scope (see README): Binance Vision has no historical full L2 order book
for any market. What it does have:
  - spot:          aggTrades, trades, klines                  (no bookTicker)
  - futures/um:     aggTrades, trades, klines, bookTicker, bookDepth

bookTicker (best bid/ask price+qty, updated on every change) is only published
historically for USDT-M futures, not spot. If config.data.market == "spot" we
can still get trades/aggTrades, but best-bid/ask must come from the futures
bookTicker dump instead, or be skipped. The downloader will tell you which
streams are actually available for the market you chose and fail loudly
rather than silently producing an empty feature set.

Usage:
    python3 -m data_pipeline.download_binance --config configs/config.yaml
"""
import argparse
import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from common.config import load_config, resolve

BASE_URL = "https://data.binance.vision/data"

# column schemas + dtypes per stream, per Binance Vision docs (no header row in CSVs)
COLUMNS = {
    "bookTicker": [
        "update_id", "best_bid_price", "best_bid_qty",
        "best_ask_price", "best_ask_qty", "transaction_time", "event_time",
    ],
    "aggTrades": [
        "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
        "timestamp", "is_buyer_maker", "is_best_match",
    ],
}

DTYPES = {
    "bookTicker": {
        "update_id": "int64", "best_bid_price": "float64", "best_bid_qty": "float64",
        "best_ask_price": "float64", "best_ask_qty": "float64",
        "transaction_time": "int64", "event_time": "int64",
    },
    "aggTrades": {
        "agg_trade_id": "int64", "price": "float64", "quantity": "float64",
        "first_trade_id": "int64", "last_trade_id": "int64",
        "timestamp": "int64", "is_buyer_maker": "bool", "is_best_match": "bool",
    },
}

# which (market, stream) combos actually exist on Binance Vision
AVAILABLE = {
    ("spot", "aggTrades"): True,
    ("spot", "bookTicker"): False,
    ("futures/um", "aggTrades"): True,
    ("futures/um", "bookTicker"): True,
}


def daterange(start: str, end: str):
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    d = d0
    while d <= d1:
        yield d.isoformat()
        d += timedelta(days=1)


def stream_url(market: str, stream: str, symbol: str, day: str) -> str:
    return f"{BASE_URL}/{market}/daily/{stream}/{symbol}/{symbol}-{stream}-{day}.zip"


def download_one(market: str, stream: str, symbol: str, day: str, out_dir: Path) -> Path | None:
    out_path = out_dir / market / stream / symbol / f"{symbol}-{stream}-{day}.parquet"
    if out_path.exists():
        return out_path
    url = stream_url(market, stream, symbol, day)
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return None  # day may not exist yet (e.g. today) or symbol didn't trade that day
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS[stream],
                             dtype=DTYPES[stream], low_memory=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(resolve("configs/config.yaml")))
    args = ap.parse_args()
    cfg = load_config(args.config)

    market = cfg["data"]["market"]
    raw_dir = resolve(cfg["data"]["raw_dir"])
    streams = cfg["data"]["streams"]

    for stream in streams:
        if not AVAILABLE.get((market, stream), False):
            alt = [m for (m, s) in AVAILABLE if s == stream and AVAILABLE[(m, s)]]
            raise SystemExit(
                f"Stream '{stream}' is not published on Binance Vision for market "
                f"'{market}'. Available for: {alt or 'no market'}. "
                f"Set data.market accordingly in config.yaml, or drop this stream."
            )

    days = list(daterange(cfg["data"]["start_date"], cfg["data"]["end_date"]))
    for symbol in cfg["symbols"]:
        for stream in streams:
            missing = []
            for day in tqdm(days, desc=f"{symbol}/{stream}"):
                path = download_one(market, stream, symbol, day, raw_dir)
                if path is None:
                    missing.append(day)
            if missing:
                print(f"  [{symbol}/{stream}] {len(missing)} days unavailable (e.g. not yet published): "
                      f"{missing[:3]}{'...' if len(missing) > 3 else ''}")


if __name__ == "__main__":
    main()
