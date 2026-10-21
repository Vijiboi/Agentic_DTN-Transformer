"""
train.py
--------
Trains the mini Set-Transformer outer-loop policy on
`pso_training_dataset_extended.xlsx`.

Usage:
    python train.py --data /path/to/pso_training_dataset_extended.xlsx --epochs 50 --batch-size 128 --out outer_loop_ckpt.pt

The model learns to reproduce the selected destination in the configured label
column (for example, ``pso_selected`` or ``policy_selected``)
for each task, given the same candidate features the inner-loop QUBO
surrogate already exposes (twin fidelity/age/uncertainty, trust, coordination
duals, delay/energy/risk/fidelity costs, etc). This is the "outer loop"
because it plays the role of the regional coordinator's projection step
(Sec. V-D, Eq. 102-103 of the paper) rather than the per-AP QUBO itself.

Reported metrics:
    top-1 accuracy      : does the model pick the exact PSO-selected AP?
    top-3 accuracy      : is the PSO choice within the model's top 3?
    cost regret         : (predicted-choice local_cost) - (PSO-choice local_cost),
                           averaged over the eval set. 0 = model matches or
                           beats PSO's objective; >0 = model picks a costlier AP.
"""
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
    NUMERIC_COLS,
    OuterLoopCandidateDataset,
    build_feature_matrix,
    compute_feature_stats,
    load_raw,
    scenario_split,
)
from model import OuterLoopSetTransformer, count_parameters


def make_loader(df, stats, batch_size, shuffle, label_col):
    ds = OuterLoopCandidateDataset(
        df, stats, max_candidates=CANDIDATES_PER_TASK, label_col=label_col
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0
    top1 = 0
    top3 = 0
    regret_sum = 0.0
    for batch in loader:
        cand = batch["candidates"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        aux = batch["aux"].to(device)

        logits, cost_pred, _ = model(cand, mask)
        pred = logits.argmax(dim=-1)
        top1 += (pred == label).sum().item()

        k = min(3, logits.shape[1])
        top3_idx = logits.topk(k, dim=-1).indices
        top3 += (top3_idx == label.unsqueeze(1)).any(dim=1).sum().item()

        true_cost = aux.gather(1, label.unsqueeze(1)).squeeze(1)
        pred_cost = aux.gather(1, pred.unsqueeze(1)).squeeze(1)
        regret_sum += (pred_cost - true_cost).sum().item()

        n += label.shape[0]

    return {
        "top1_acc": top1 / n,
        "top3_acc": top3 / n,
        "avg_cost_regret": regret_sum / n,
        "n": n,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    print(f"[info] loading dataset from {args.data} ...")
    t0 = time.time()
    df = load_raw(args.data)
    print(f"[info] loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    train_df, val_df, test_df = scenario_split(df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    print(f"[info] scenarios -> train {train_df.scenario_id.nunique()}, "
          f"val {val_df.scenario_id.nunique()}, test {test_df.scenario_id.nunique()}")

    # Fit standardization stats on TRAIN ONLY to avoid leakage.
    train_feat = build_feature_matrix(train_df)
    stats = compute_feature_stats(train_feat, n_numeric=len(NUMERIC_COLS))

    # candidate_local_cost is heavy-tailed (mean ~60, max ~1300), so the
    # auxiliary regression target is standardized separately for a
    # well-scaled loss; the raw values are still used for the (interpretable)
    # cost-regret metric below.
    aux_mean = float(train_df["candidate_local_cost"].mean())
    aux_std = float(train_df["candidate_local_cost"].std() + 1e-6)

    train_loader = make_loader(train_df, stats, args.batch_size, shuffle=True, label_col=args.label_col)
    val_loader = make_loader(val_df, stats, args.batch_size, shuffle=False, label_col=args.label_col)
    test_loader = make_loader(test_df, stats, args.batch_size, shuffle=False, label_col=args.label_col)

    model = OuterLoopSetTransformer(
        feature_dim=stats.feature_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    print(f"[info] model parameters: {count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss(reduction="none")

    best_val_acc = -1.0
    out_path = Path(args.out)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            cand = batch["candidates"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            aux = batch["aux"].to(device)

            logits, cost_pred, _ = model(cand, mask)
            loss_select = ce(logits, label)

            aux_norm = (aux - aux_mean) / aux_std
            cost_err = mse(cost_pred, aux_norm) * mask
            loss_cost = cost_err.sum() / mask.sum().clamp(min=1)

            loss = loss_select + args.aux_weight * loss_cost

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_loss += loss.item()
            n_batches += 1
        sched.step()

        val_metrics = evaluate(model, val_loader, device)
        print(f"[epoch {epoch:03d}] train_loss={running_loss / n_batches:.4f} "
              f"val_top1={val_metrics['top1_acc']:.4f} val_top3={val_metrics['top3_acc']:.4f} "
              f"val_regret={val_metrics['avg_cost_regret']:.4f}")

        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            torch.save({
                "model_state": model.state_dict(),
                "model_args": dict(feature_dim=stats.feature_dim, d_model=args.d_model,
                                    n_heads=args.n_heads, n_layers=args.n_layers,
                                    d_ff=args.d_ff, dropout=args.dropout),
                "feature_stats": {"mean": stats.mean, "std": stats.std,
                                   "numeric_cols": stats.numeric_cols,
                                   "categorical_cols": stats.categorical_cols},
                "aux_mean": aux_mean, "aux_std": aux_std,
                "val_top1_acc": best_val_acc,
                "label_col": args.label_col,
            }, out_path)
            print(f"           -> saved new best checkpoint to {out_path} (val_top1={best_val_acc:.4f})")

    print("[info] training complete. Evaluating best checkpoint on TEST set...")
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"[test] top1_acc={test_metrics['top1_acc']:.4f} top3_acc={test_metrics['top3_acc']:.4f} "
          f"avg_cost_regret={test_metrics['avg_cost_regret']:.4f} (n={test_metrics['n']})")


def build_argparser():
    p = argparse.ArgumentParser(description="Train the mini Transformer outer-loop DTN policy")
    p.add_argument("--data", type=str, required=True,
                    help="path to a PSO training CSV, xlsx file, or cached .parquet")
    p.add_argument("--out", type=str, default="outer_loop_ckpt.pt")
    p.add_argument(
        "--label-col", type=str, default="pso_selected",
        help="binary selection-label column to imitate (for option 1 use policy_selected)",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--aux-weight", type=float, default=0.2,
                    help="weight of the auxiliary local-cost regression loss")
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--d-ff", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
