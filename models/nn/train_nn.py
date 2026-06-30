"""
Train the temporal NN (GRU/CNN/transformer, set via nn.architecture in config)
on raw LOB snapshot sequences, walk-forward fold by fold. Designed to run as a
single non-interactive job on RunPod: `python3 -m models.nn.train_nn --config ...`
with no notebook/manual steps required.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from backtest.simulator import run_backtest
from common.config import load_config, resolve
from eval.metrics import directional_accuracy, summarize_fold_metrics
from eval.walk_forward import make_folds
from models.nn.dataset import RAW_COLS, LOBSequenceDataset, build_snapshot_grid
from models.nn.model import build_model


def train_one_fold(train_bars, val_bars, test_bars, snapshot_grid, nn_cfg, device):
    train_ds = LOBSequenceDataset(snapshot_grid, train_bars, nn_cfg["lookback"])
    val_ds = LOBSequenceDataset(snapshot_grid, val_bars, nn_cfg["lookback"])
    test_ds = LOBSequenceDataset(snapshot_grid, test_bars, nn_cfg["lookback"])
    if len(train_ds) < 50 or len(val_ds) < 20 or len(test_ds) == 0:
        return None, None

    val_ds.set_norm_stats(train_ds.mean, train_ds.std)
    test_ds.set_norm_stats(train_ds.mean, train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=nn_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=nn_cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=nn_cfg["batch_size"])

    model = build_model(
        nn_cfg["architecture"], len(RAW_COLS), nn_cfg["hidden_dim"], nn_cfg["num_layers"], nn_cfg["dropout"]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=nn_cfg["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss, best_state, patience_left = np.inf, None, nn_cfg["patience"]
    for epoch in range(nn_cfg["max_epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss, best_state, patience_left = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, nn_cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds, bar_index = [], []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            preds.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    proba = np.concatenate(preds) if preds else np.array([])
    pred_sign = pd.Series(np.where(proba > 0.5, 1, -1), index=test_ds.bar_index)
    return pred_sign, test_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(resolve("configs/config.yaml")))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])

    records = []
    for symbol in cfg["symbols"]:
        panel_path = resolve(cfg["data"]["processed_dir"]) / "panel" / f"{symbol}.parquet"
        panel = pd.read_parquet(panel_path)
        snapshot_grid = build_snapshot_grid(panel)

        for horizon_sec in cfg["horizons_sec"]:
            bars_path = resolve(cfg["data"]["processed_dir"]) / "bars" / f"{horizon_sec}s" / f"{symbol}.parquet"
            bars = pd.read_parquet(bars_path)
            folds = make_folds(bars, **cfg["walk_forward"])

            for fold in folds:
                pred_sign, test_ds = train_one_fold(
                    fold.train, fold.val, fold.test, snapshot_grid, cfg["nn"], args.device
                )
                if pred_sign is None:
                    continue
                test_aligned = fold.test.loc[fold.test.index.intersection(test_ds.bar_index)]
                acc = directional_accuracy(np.sign(test_aligned["fwd_return"]), pred_sign)
                bt = run_backtest(fold.test, pred_sign, cfg["backtest"])
                records.append({
                    "symbol": symbol, "horizon_sec": horizon_sec, "model": "nn",
                    "fold_id": fold.fold_id, "directional_accuracy": acc,
                    "n_test_bars": len(test_ds), **bt,
                })
                print(f"[{symbol} {horizon_sec}s fold {fold.fold_id}] acc={acc:.3f} net_pnl_bps={bt['net_pnl_bps']:.1f}")

    results_df = pd.DataFrame(records)
    out_dir = resolve("results")
    out_dir.mkdir(exist_ok=True)
    results_df.to_csv(out_dir / "nn_fold_results.csv", index=False)
    summary = summarize_fold_metrics(records)
    summary.to_csv(out_dir / "nn_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
