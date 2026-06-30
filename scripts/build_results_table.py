"""
Combine GBM + NN (+ naive baseline) walk-forward results into the final
symbol x horizon x model comparison table, and flag which combinations show
genuine edge (net positive PnL after costs, beating both naive baselines)
vs which are noise.

Usage:
    python3 scripts/build_results_table.py
"""
import pandas as pd

from common.config import resolve


def main():
    results_dir = resolve("results")
    gbm = pd.read_csv(results_dir / "gbm_fold_results.csv")
    nn = pd.read_csv(results_dir / "nn_fold_results.csv")
    all_records = pd.concat([gbm, nn], ignore_index=True)

    summary = all_records.groupby(["symbol", "horizon_sec", "model"]).agg(
        mean_directional_accuracy=("directional_accuracy", "mean"),
        n_folds=("fold_id", "nunique"),
        total_test_bars=("n_test_bars", "sum"),
        total_net_pnl_bps=("net_pnl_bps", "sum"),
        mean_sharpe=("sharpe", "mean"),
        mean_hit_rate=("hit_rate", "mean"),
    ).reset_index()

    pivot = summary.pivot_table(
        index=["symbol", "horizon_sec"], columns="model",
        values=["mean_directional_accuracy", "total_net_pnl_bps"],
    )

    def has_edge(row):
        model_pnl = row.get(("total_net_pnl_bps", "gbm"), float("nan"))
        nn_pnl = row.get(("total_net_pnl_bps", "nn"), float("nan"))
        persistence_pnl = row.get(("total_net_pnl_bps", "naive_persistence"), float("nan"))
        momentum_pnl = row.get(("total_net_pnl_bps", "naive_momentum"), float("nan"))
        naive_best = max(
            x for x in [persistence_pnl, momentum_pnl, 0.0] if pd.notna(x)
        )
        flags = []
        if pd.notna(model_pnl) and model_pnl > 0 and model_pnl > naive_best:
            flags.append("gbm")
        if pd.notna(nn_pnl) and nn_pnl > 0 and nn_pnl > naive_best:
            flags.append("nn")
        return ",".join(flags) if flags else "none (noise)"

    pivot["genuine_edge"] = pivot.apply(has_edge, axis=1)

    out_path = results_dir / "comparison_table.csv"
    pivot.to_csv(out_path)
    print(pivot)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
