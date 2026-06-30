from models.gbm.train_gbm import tune_and_fit, evaluate_fold, feature_columns
from tests.test_pipeline_smoke import make_synthetic_panel
from features.microstructure import compute_all_features
from features.resample import resample_to_bars, add_labels
from eval.walk_forward import make_folds


def test_gbm_train_and_eval_smoke():
    panel = make_synthetic_panel()
    feat_cfg = {
        "vol_windows_sec": [10, 60], "vpin_bucket_volume": 5000, "vpin_num_buckets": 10,
        "spread_rolling_windows_sec": [30], "ofi_lookback_ticks": 20,
    }
    feat = compute_all_features(panel, feat_cfg)
    bars = resample_to_bars(feat, horizon_sec=30, vol_windows=[10, 60], spread_windows=[30])
    bars = add_labels(bars, horizon_bars=1)

    folds = make_folds(bars, train_days=14, val_days=3, test_days=3, step_days=10, purge_sec=30)
    fold = folds[0]
    feat_cols = feature_columns(bars)

    gbm_cfg = {
        "n_estimators": 50, "early_stopping_rounds": 10,
        "learning_rate_grid": [0.05], "num_leaves_grid": [15], "max_depth_grid": [-1],
    }
    model = tune_and_fit(fold.train, fold.val, feat_cols, gbm_cfg, seed=42)
    assert model is not None

    backtest_cfg = {"taker_fee_bps": 10, "maker_fee_bps": 10, "slippage_bps": 2, "position_size_usd": 1000}
    result = evaluate_fold(model, fold.test, feat_cols, backtest_cfg)
    assert result is not None
    assert "directional_accuracy" in result and "net_pnl_bps" in result


if __name__ == "__main__":
    test_gbm_train_and_eval_smoke()
    print("gbm smoke test passed")
