"""
End-to-end data pipeline: download -> merge panel -> resample bars for every
configured horizon. Run this once before training either model.

Usage:
    python3 scripts/run_pipeline.py --config configs/config.yaml
"""
import argparse
import subprocess
import sys

from common.config import load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(resolve("configs/config.yaml")))
    args = ap.parse_args()
    cfg = load_config(args.config)

    run = lambda mod, *extra: subprocess.run(
        [sys.executable, "-m", mod, "--config", args.config, *extra], check=True
    )

    run("data_pipeline.download_binance")
    run("data_pipeline.build_quote_trade_panel")
    for symbol in cfg["symbols"]:
        for horizon_sec in cfg["horizons_sec"]:
            run("features.resample", "--symbol", symbol, "--horizon-sec", str(horizon_sec))


if __name__ == "__main__":
    main()
