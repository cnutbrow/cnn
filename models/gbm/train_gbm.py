"""
LightGBM baseline: binary direction classifier on engineered microstructure
features, tuned with a small grid search + early stopping per walk-forward
fold (this is the model the NN has to beat, so it gets a real tuning loop,
not default params).

Usage:
    python3 -m models.gbm.train_gbm --config configs/config.yaml
"""
import argparse
import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest.simulator import run_backtest
from common.config import load_config, resolve
from eval.metrics import (directional_accuracy, naive_momentum_baseline,
                           naive_persistence_baseline, summarize_fold_metrics)
from eval.walk_forward import make_folds

NON_FEATURE_COLS = {"mid", "microprice", "fwd_return", "label_direction"}


def feature_columns(bars: pd.DataFrame) -> list[str]:
    return [c for c in bars.columns if c not in NON_FEATURE_COLS]


def prep_xy(df: pd.DataFrame, feat_cols: list[str]):
    df = df.dropna(subset=feat_cols + ["label_direction"])
    df = df[df["label_direction"] != 0]  # drop flat bars, this is a directional classifier
    X = df[feat_cols]
    y = (df["label_direction"] == 1).astype(int)
    return X, y, df


def tune_and_fit(train_df, val_df, feat_cols, gbm_cfg, seed):
    X_train, y_train, _ = prep_xy(train_df, feat_cols)
    X_val, y_val, _ = prep_xy(val_df, feat_cols)
    if len(X_train) < 50 or len(X_val) < 20:
        return None

    best_model, best_score = None, np.inf
    grid = itertools.product(
        gbm_cfg["learning_rate_grid"], gbm_cfg["num_leaves_grid"], gbm_cfg["max_depth_grid"]
    )
    for lr, num_leaves, max_depth in grid:
        model = lgb.LGBMClassifier(
            n_estimators=gbm_cfg["n_estimators"],
            learning_rate=lr,
            num_leaves=num_leaves,
            max_depth=max_depth,
            objective="binary",
            random_state=seed,
            verbosity=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(gbm_cfg["early_stopping_rounds"], verbose=False)],
        )
        score = model.best_score_["valid_0"]["binary_logloss"]
        if score < best_score:
            best_score, best_model = score, model
    return best_model


def evaluate_fold(model, test_df, feat_cols, backtest_cfg):
    X_test, y_test, df_test = prep_xy(test_df, feat_cols)
    if model is None or len(X_test) == 0:
        return None
    proba = model.predict_proba(X_test)[:, 1]
    pred_sign = pd.Series(np.where(proba > 0.5, 1, -1), index=df_test.index)

    acc = directional_accuracy(np.sign(df_test["fwd_return"]), pred_sign)
    bt = run_backtest(test_df, pred_sign, backtest_cfg)
    return {"directional_accuracy": acc, "n_test_bars": len(X_test), **bt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(resolve("configs/config.yaml")))
    args = ap.parse_args()
    cfg = load_config(args.config)

    records = []
    for symbol in cfg["symbols"]:
        for horizon_sec in cfg["horizons_sec"]:
            bars_path = resolve(cfg["data"]["processed_dir"]) / "bars" / f"{horizon_sec}s" / f"{symbol}.parquet"
            bars = pd.read_parquet(bars_path)
            feat_cols = feature_columns(bars)

            folds = make_folds(bars, **cfg["walk_forward"])
            for fold in folds:
                model = tune_and_fit(fold.train, fold.val, feat_cols, cfg["gbm"], cfg["seed"])
                result = evaluate_fold(model, fold.test, feat_cols, cfg["backtest"])
                if result is None:
                    continue
                records.append({
                    "symbol": symbol, "horizon_sec": horizon_sec, "model": "gbm",
                    "fold_id": fold.fold_id, **result,
                })

                # naive baselines on the same test fold, for reference
                for name, fn in [("naive_persistence", naive_persistence_baseline),
                                  ("naive_momentum", naive_momentum_baseline)]:
                    pred_sign = fn(fold.test)
                    acc = directional_accuracy(np.sign(fold.test["fwd_return"]), pred_sign)
                    bt = run_backtest(fold.test, pred_sign, cfg["backtest"])
                    records.append({
                        "symbol": symbol, "horizon_sec": horizon_sec, "model": name,
                        "fold_id": fold.fold_id, "directional_accuracy": acc,
                        "n_test_bars": len(fold.test), **bt,
                    })

    results_df = pd.DataFrame(records)
    out_dir = resolve("results")
    out_dir.mkdir(exist_ok=True)
    results_df.to_csv(out_dir / "gbm_fold_results.csv", index=False)
    summary = summarize_fold_metrics(records)
    summary.to_csv(out_dir / "gbm_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
