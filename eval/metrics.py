import numpy as np
import pandas as pd


def directional_accuracy(y_true_sign: pd.Series, y_pred_sign: pd.Series) -> float:
    mask = (y_true_sign != 0) & y_pred_sign.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((y_true_sign[mask] == y_pred_sign[mask]).mean())


def naive_persistence_baseline(bars: pd.DataFrame) -> pd.Series:
    """Predict the sign of the *previous* realized return as the forecast for the next one."""
    prev_return_sign = np.sign(bars["mid"].diff())
    return prev_return_sign


def naive_momentum_baseline(bars: pd.DataFrame, lookback_bars: int = 5) -> pd.Series:
    """Predict continuation of the trailing N-bar return."""
    trailing_return = bars["mid"].pct_change(lookback_bars)
    return np.sign(trailing_return)


def summarize_fold_metrics(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    agg = df.groupby(["symbol", "horizon_sec", "model"]).agg(
        directional_accuracy=("directional_accuracy", "mean"),
        n_folds=("fold_id", "nunique"),
        total_test_bars=("n_test_bars", "sum"),
        net_pnl_bps=("net_pnl_bps", "sum"),
        sharpe=("sharpe", "mean"),
    ).reset_index()
    return agg
