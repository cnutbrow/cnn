"""
Tick-level microstructure feature computation on the merged quote+trade panel
produced by data_pipeline.build_quote_trade_panel.

All features are computed on the irregular event stream (one row per quote
update or trade) so that volume/tick-based features like VPIN and OFI use
their natural units before being resampled onto a fixed time grid.
"""
import numpy as np
import pandas as pd


def add_quote_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["mid"] = (df["best_bid_price"] + df["best_ask_price"]) / 2
    df["spread"] = df["best_ask_price"] - df["best_bid_price"]
    df["rel_spread"] = df["spread"] / df["mid"]
    denom = df["best_bid_qty"] + df["best_ask_qty"]
    df["microprice"] = (
        df["best_bid_price"] * df["best_ask_qty"] + df["best_ask_price"] * df["best_bid_qty"]
    ) / denom.replace(0, np.nan)
    df["queue_depth_ratio"] = df["best_bid_qty"] / denom.replace(0, np.nan)
    return df


def add_order_flow_imbalance(df: pd.DataFrame, lookback_ticks: int) -> pd.DataFrame:
    """Cont-Kukanov-Stoikov style OFI: signed change in best-quote queue sizes,
    only counting quote-update rows (trades don't move the OFI accumulator directly,
    they already show up via queue depletion on the next quote update)."""
    is_quote = df["source"] == "quote"
    bid_p, bid_q = df["best_bid_price"], df["best_bid_qty"]
    ask_p, ask_q = df["best_ask_price"], df["best_ask_qty"]

    bid_p_prev, bid_q_prev = bid_p.shift(1), bid_q.shift(1)
    ask_p_prev, ask_q_prev = ask_p.shift(1), ask_q.shift(1)

    e_bid = np.where(bid_p > bid_p_prev, bid_q,
             np.where(bid_p == bid_p_prev, bid_q - bid_q_prev, -bid_q_prev))
    e_ask = np.where(ask_p < ask_p_prev, ask_q,
             np.where(ask_p == ask_p_prev, ask_q - ask_q_prev, -ask_q_prev))

    ofi_tick = np.where(is_quote, e_bid - e_ask, 0.0)
    df = df.copy()
    df["ofi_tick"] = ofi_tick
    df["ofi"] = df["ofi_tick"].rolling(lookback_ticks, min_periods=1).sum()
    return df


def add_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_trade = df["source"] == "trade"
    df["trade_signed_qty"] = np.where(is_trade, df["signed_qty"], 0.0)

    # trade-through: trade price strictly beyond the prevailing best quote on its side
    prev_bid = df["best_bid_price"].shift(1)
    prev_ask = df["best_ask_price"].shift(1)
    buy_through = is_trade & (df["signed_qty"] > 0) & (df["price"] > prev_ask)
    sell_through = is_trade & (df["signed_qty"] < 0) & (df["price"] < prev_bid)
    df["trade_through"] = (buy_through | sell_through).astype(float)

    ts_seconds = df["ts"].astype("int64") / 1e9
    last_trade_ts = pd.Series(np.where(is_trade, ts_seconds, np.nan), index=df.index).ffill()
    df["time_since_last_trade"] = ts_seconds - last_trade_ts
    return df


def add_vpin(df: pd.DataFrame, bucket_volume: float, num_buckets: int) -> pd.DataFrame:
    """Volume-Synchronized Probability of Informed Trading (Easley et al.).
    Buckets are fixed *quote* volume (price*qty) of trades; VPIN is the rolling
    average of |buy_vol - sell_vol| / total_vol over the last `num_buckets` buckets."""
    df = df.copy()
    is_trade = df["source"] == "trade"
    notional = (df["price"] * df["quantity"]).where(is_trade, 0.0).fillna(0.0)
    buy_notional = notional.where(df["signed_qty"] > 0, 0.0)
    sell_notional = notional.where(df["signed_qty"] < 0, 0.0)

    cum_vol = notional.cumsum()
    bucket_id = (cum_vol // bucket_volume).astype("Int64")

    bucket_buy = buy_notional.groupby(bucket_id).sum()
    bucket_sell = sell_notional.groupby(bucket_id).sum()
    bucket_total = bucket_buy + bucket_sell
    bucket_imbalance = (bucket_buy - bucket_sell).abs() / bucket_total.replace(0, np.nan)
    vpin_by_bucket = bucket_imbalance.rolling(num_buckets, min_periods=1).mean()

    df["_bucket_id"] = bucket_id
    df["vpin"] = df["_bucket_id"].map(vpin_by_bucket)
    df["vpin"] = df["vpin"].ffill()
    df.drop(columns="_bucket_id", inplace=True)
    return df


def add_realized_vol(df: pd.DataFrame, windows_sec: list[int]) -> pd.DataFrame:
    df = df.copy()
    log_mid = np.log(df["mid"])
    ret = log_mid.diff()
    ts = df["ts"]
    idx = pd.DatetimeIndex(ts)
    for w in windows_sec:
        rv = (
            pd.Series(ret.values, index=idx)
            .rolling(f"{w}s", min_periods=2)
            .apply(lambda x: np.sqrt(np.sum(x**2)), raw=True)
        )
        df[f"realized_vol_{w}s"] = rv.values
    return df


def add_spread_dynamics(df: pd.DataFrame, windows_sec: list[int]) -> pd.DataFrame:
    df = df.copy()
    idx = pd.DatetimeIndex(df["ts"])
    spread = pd.Series(df["spread"].values, index=idx)
    for w in windows_sec:
        df[f"spread_mean_{w}s"] = spread.rolling(f"{w}s", min_periods=1).mean().values
        df[f"spread_std_{w}s"] = spread.rolling(f"{w}s", min_periods=1).std().values
    return df


def compute_all_features(panel: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    df = add_quote_features(panel)
    df = add_order_flow_imbalance(df, feat_cfg["ofi_lookback_ticks"])
    df = add_trade_features(df)
    df = add_vpin(df, feat_cfg["vpin_bucket_volume"], feat_cfg["vpin_num_buckets"])
    df = add_realized_vol(df, feat_cfg["vol_windows_sec"])
    df = add_spread_dynamics(df, feat_cfg["spread_rolling_windows_sec"])
    return df
