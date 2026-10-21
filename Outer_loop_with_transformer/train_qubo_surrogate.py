"""Train a leakage-free Transformer surrogate for inner-QUBO local costs."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (
    CANDIDATES_PER_TASK,
    CATEGORICAL_COLS,
    QUBO_SURROGATE_NUMERIC_COLS,
    QuboCostSurrogateDataset,
    build_feature_matrix,
    compute_feature_stats,
    load_raw,
    scenario_split,
)
from model import OuterLoopSetTransformer, count_parameters


def make_loader(df, stats, batch_size, shuffle):
    return DataLoader(
        QuboCostSurrogateDataset(df, stats, max_candidates=CANDIDATES_PER_TASK),
        batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False,
    )


@torch.no_grad()
def evaluate(model, loader, device, target_mean, target_std):
    model.eval()
    absolute_error = squared_error = 0.0
    top1 = top3 = n_items = n_groups = 0
    for batch in loader:
        candidates = batch["candidates"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        _, predicted_norm, _ = model(candidates, mask)
        predicted = predicted_norm * target_std + target_mean
        valid = mask.bool()
        error = predicted[valid] - target[valid]
        absolute_error += error.abs().sum().item()
        squared_error += error.square().sum().item()
        n_items += valid.sum().item()

        predicted = predicted.masked_fill(~valid, float("inf"))
        target = target.masked_fill(~valid, float("inf"))
        truth = target.argmin(dim=-1)
        ranking = predicted.argsort(dim=-1)
        top1 += (ranking[:, 0] == truth).sum().item()
        top3 += (ranking[:, : min(3, ranking.shape[1])] == truth.unsqueeze(1)).any(dim=1).sum().item()
        n_groups += target.shape[0]
    return {
        "mae": absolute_error / max(n_items, 1),
        "rmse": (squared_error / max(n_items, 1)) ** 0.5,
        "rank_top1": top1 / max(n_groups, 1),
        "rank_top3": top3 / max(n_groups, 1),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")
    t0 = time.time()
    df = load_raw(args.data)
    print(f"[info] loaded {len(df):,} rows in {time.time() - t0:.1f}s")
    train_df, val_df, test_df = scenario_split(df, args.val_frac, args.test_frac, args.seed)
    print(f"[info] scenarios -> train {train_df.scenario_id.nunique()}, val {val_df.scenario_id.nunique()}, test {test_df.scenario_id.nunique()}")

    raw_train_features = build_feature_matrix(
        train_df, numeric_cols=QUBO_SURROGATE_NUMERIC_COLS, categorical_cols=CATEGORICAL_COLS
    )
    stats = compute_feature_stats(raw_train_features, len(QUBO_SURROGATE_NUMERIC_COLS))
    target_mean = float(train_df.candidate_local_cost.mean())
    target_std = float(train_df.candidate_local_cost.std() + 1e-6)
    train_loader = make_loader(train_df, stats, args.batch_size, True)
    val_loader = make_loader(val_df, stats, args.batch_size, False)
    test_loader = make_loader(test_df, stats, args.batch_size, False)

    model = OuterLoopSetTransformer(
        feature_dim=stats.feature_dim, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    print(f"[info] model parameters: {count_parameters(model):,}; leakage-free features: {len(QUBO_SURROGATE_NUMERIC_COLS)} numeric + categorical")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse = nn.MSELoss(reduction="none")
    best_val = float("inf")
    out_path = Path(args.out)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = batches = 0
        for batch in train_loader:
            candidates = batch["candidates"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            _, predicted_norm, _ = model(candidates, mask)
            target_norm = (target - target_mean) / target_std
            loss = (mse(predicted_norm, target_norm) * mask).sum() / mask.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
        scheduler.step()
        metrics = evaluate(model, val_loader, device, target_mean, target_std)
        print(f"[epoch {epoch:03d}] train_mse={loss_sum / max(batches, 1):.5f} val_mae={metrics['mae']:.4f} val_rmse={metrics['rmse']:.4f} val_rank1={metrics['rank_top1']:.4f} val_rank3={metrics['rank_top3']:.4f}")
        if metrics["mae"] < best_val:
            best_val = metrics["mae"]
            torch.save({
                "checkpoint_kind": "transformer_qubo_cost_surrogate_v1",
                "model_state": model.state_dict(),
                "model_args": {"feature_dim": stats.feature_dim, "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers, "d_ff": args.d_ff, "dropout": args.dropout},
                "feature_stats": {"mean": stats.mean, "std": stats.std, "numeric_cols": stats.numeric_cols, "categorical_cols": stats.categorical_cols},
                "target_col": "candidate_local_cost", "target_mean": target_mean, "target_std": target_std,
                "val_mae": best_val,
            }, out_path)
            print(f"           -> saved {out_path} (val_mae={best_val:.4f})")

    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, test_loader, device, target_mean, target_std)
    print(f"[test] mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} rank_top1={metrics['rank_top1']:.4f} rank_top3={metrics['rank_top3']:.4f}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Train the Transformer-to-QUBO cost surrogate")
    parser.add_argument("--data", required=True, help="candidate CSV, xlsx, or parquet with candidate_local_cost targets")
    parser.add_argument("--out", default="transformer_qubo_surrogate.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
