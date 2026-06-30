# Intraday crypto signal research: NN vs GBM on microstructure features

Goal: honest comparison of a temporal neural net against a tuned LightGBM
baseline for short-horizon directional prediction on BTC/ETH/SOL, with
walk-forward evaluation and a realistic backtest. This is a research harness
for signal discovery, not a trading bot.

## Data scope (read this first)

Binance Vision (`data.binance.vision`) — the only data source in scope —
**does not provide historical full L2/L3 order book depth** for any market.
It has:

- **spot**: `klines`, `trades`, `aggTrades` only.
- **futures/um**: `klines`, `trades`, `aggTrades`, plus `bookTicker` (best
  bid/ask price+qty) and `bookDepth` (a coarse percentage-banded depth
  summary, not raw levels).

There is no historical 10-level order book replay available from this
source at any price. Genuine multi-level LOB reconstruction requires either
recording your own websocket diff-depth stream going forward, or a paid
vendor (e.g. Tardis.dev).

**This repo is scoped to best-bid/ask + trades** (`data.market: futures/um`
in `configs/config.yaml`, since spot has no historical `bookTicker` dump).
Features below are described accordingly — "queue depth ratio" means
best-bid-qty vs best-ask-qty, not a 10-level book. If you later get access
to real L2 data, the feature/dataset code is the only layer that needs to
change; the eval harness, models, and backtest are agnostic to feature
count.

## Pipeline

```
data_pipeline/download_binance.py        # pull bookTicker + aggTrades zips, cache as parquet
data_pipeline/build_quote_trade_panel.py # merge into one time-ordered event panel per symbol
features/microstructure.py               # tick-level features: OFI, VPIN, realized vol, spread dynamics, trade-through rate, time since last trade
features/resample.py                     # resample onto fixed-horizon bars (config: horizons_sec), attach forward-direction labels
```

Run the whole thing: `python3 scripts/run_pipeline.py --config configs/config.yaml`

## Models

- **GBM baseline** (`models/gbm/train_gbm.py`): LightGBM binary classifier on
  the engineered tabular features, grid-searched over learning rate /
  num_leaves / max_depth with early stopping, per walk-forward fold. This is
  the bar the NN has to clear.
- **NN** (`models/nn/`): GRU by default (`nn.architecture` in config also
  supports `cnn` / `transformer`), trained on raw/lightly-processed best-quote
  snapshot sequences (`models/nn/dataset.py`) — log return, relative spread,
  queue ratio, log sizes on a 1s grid, `nn.lookback` snapshots per sample —
  rather than the GBM's hand-engineered features, to test whether the network
  extracts structure the tabular features miss.

## Evaluation

`eval/walk_forward.py` builds **purged, contiguous, time-ordered**
train/val/test folds (`configs/config.yaml: walk_forward`) — no shuffling,
no k-fold over overlapping windows, and a purge gap (>= max horizon) at each
boundary so labels can't leak across splits.

Results are reported **per symbol x per horizon**, never averaged together,
since the NN/GBM gap is expected to differ between 30s and 5min. Two naive
baselines (persistence, momentum) run alongside both models on identical
folds so "beats doing nothing" is always checkable.

`backtest/simulator.py` charges round-trip taker fee + slippage
(`configs/config.yaml: backtest`, defaults to Binance spot VIP0: 10bps
taker each side + 2bps slippage each side = 24bps round trip) against
predicted-direction trades, one trade per bar (no carrying position across
bars — conservative for a fast-decaying microstructure signal).

## Running

```bash
pip install -r requirements.txt
python3 scripts/run_pipeline.py --config configs/config.yaml
python3 -m models.gbm.train_gbm --config configs/config.yaml      # local/CPU
python3 -m models.nn.train_nn --config configs/config.yaml        # local CPU or, for real runs:
bash runpod/run_nn_job.sh configs/config.yaml                     # on a RunPod GPU pod, backgrounded
python3 scripts/build_results_table.py                            # final symbol x horizon x model table + edge flag
```

Final output: `results/comparison_table.csv` — directional accuracy and net
backtest PnL (bps) per symbol x horizon x model, with a `genuine_edge` column
flagging combinations that beat both naive baselines net of costs vs. ones
that don't (noise).

## Tests

```bash
python3 -m pytest tests/test_pipeline_smoke.py tests/test_gbm_smoke.py -q
python3 -m pytest tests/test_nn_smoke.py -q
```

Run the NN test separately from the GBM/pipeline tests. PyTorch and LightGBM
ship conflicting bundled OpenMP runtimes that segfault when both are imported
in the same process on macOS — this only bites `pytest` running multiple test
files together; `models/gbm/train_gbm.py` and `models/nn/train_nn.py` are
always invoked as separate processes in actual use, so it does not affect the
pipeline.

## Config knobs you'll likely want to revisit

- `walk_forward.train_days/val_days/test_days/step_days`: sized for the
  default 3-month `data.start_date`/`end_date` window — widen the date range
  if you want more folds.
- `nn.lookback`: 100 snapshots (~100s of history on the 1s grid) by default.
- `backtest.taker_fee_bps` / `slippage_bps`: Binance spot VIP0, no BNB
  discount, 2bps slippage on top of spread-crossing. Tighten if you trade
  enough volume for a lower fee tier.
