"""
Minimal event-driven-ish backtest: at each bar, if the model's predicted
direction is nonzero, take a position sized at `position_size_usd` for one
bar, marked-to-market on `fwd_return`, charged taker fee + slippage on entry
and exit (we assume each bar's signal is a fresh round-trip trade — no
position carrying across bars, which is conservative for a 30s/5min horizon
signal that decays).
"""
import numpy as np
import pandas as pd


def run_backtest(bars: pd.DataFrame, pred_direction: pd.Series, backtest_cfg: dict) -> dict:
    fee_bps = (backtest_cfg["taker_fee_bps"]) * 2  # round trip: enter + exit
    slip_bps = backtest_cfg["slippage_bps"] * 2
    cost_bps = fee_bps + slip_bps
    cost_frac = cost_bps / 1e4

    pred = pred_direction.reindex(bars.index).fillna(0)
    fwd_return = bars["fwd_return"]

    gross_return = pred * fwd_return  # only nonzero where we took a position
    traded = pred != 0
    net_return = np.where(traded, gross_return - cost_frac, 0.0)

    position_usd = backtest_cfg["position_size_usd"]
    pnl_usd = net_return * position_usd

    n_trades = int(traded.sum())
    total_net_pnl_bps = float(np.nansum(net_return) * 1e4)
    mean_net_return = np.nanmean(net_return[traded]) if n_trades else float("nan")
    std_net_return = np.nanstd(net_return[traded]) if n_trades else float("nan")
    sharpe = float(mean_net_return / std_net_return * np.sqrt(n_trades)) if std_net_return else float("nan")
    hit_rate = float((net_return[traded] > 0).mean()) if n_trades else float("nan")

    return {
        "n_trades": n_trades,
        "net_pnl_bps": total_net_pnl_bps,
        "net_pnl_usd": float(np.nansum(pnl_usd)),
        "mean_net_return_bps": float(mean_net_return * 1e4) if n_trades else float("nan"),
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "cost_bps_per_trade": cost_bps,
    }
