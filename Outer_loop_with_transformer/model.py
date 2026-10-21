"""
model.py
--------
Mini Transformer for the closed-loop DTN "outer loop": a permutation-aware
set-encoder that looks at the K candidate destinations (produced by the
inner-loop QUBO/BQM surrogate) for one task and reproduces the regional
coordinator's projection decision -- i.e. it plays the role of Sec. V-D /
Eq. (102)-(103) in the paper, learned end-to-end from the PSO-labelled data
instead of the hand-written arg-min rule.

Design notes:
- No positional encoding: the candidates are a *set*, not a sequence, so we
  rely purely on self-attention over the K tokens (permutation-invariant
  input projection + attention == set transformer).
- Padding mask supported (defaults to 10 real candidates in this dataset,
  but the network generalizes to fewer/more candidates per task).
- Two heads:
    * selection head -> per-candidate logit -> softmax over valid candidates
      (this is what predicts x*_{u,m})
    * auxiliary regression head -> predicts candidate_local_cost, i.e. it
      also learns to imitate the QUBO/PSO local surrogate cost itself, which
      regularizes the representation and gives an interpretable score you
      can compare directly against the inner-loop QUBO objective.
- Kept intentionally small ("mini"): 2-4 layers, d_model 64-128. This is a
  per-task-group decision (10 tokens), not a long-sequence LM, so a small
  model trains fast on CPU and is cheap enough to run once per slot inside
  the closed control loop.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class OuterLoopSetTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # A learned "regional coordinator" query token that attends over all
        # candidates -- gives the model a natural summary vector representing
        # the network-level coordination context, on top of per-candidate
        # scores. (Loosely analogous to the paper's regional mission state
        # xi^reg(t) / dual price aggregation.)
        self.coord_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.select_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.cost_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, candidates: torch.Tensor, mask: torch.Tensor):
        """
        candidates : (B, K, F)
        mask       : (B, K)  1 = real candidate, 0 = padding
        returns:
            logits      : (B, K)  masked selection logits (padding -> -inf)
            cost_pred   : (B, K)  predicted local cost per candidate
            coord_embed : (B, d_model) pooled regional-coordination embedding
        """
        B, K, _ = candidates.shape
        x = self.input_proj(candidates)  # (B, K, D)

        coord_tok = self.coord_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([coord_tok, x], dim=1)  # (B, K+1, D)

        coord_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
        full_mask = torch.cat([coord_mask, mask], dim=1)  # (B, K+1)
        key_padding_mask = full_mask == 0  # True where PyTorch should ignore

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, K+1, D)

        coord_embed = h[:, 0, :]
        cand_embed = h[:, 1:, :]

        logits = self.select_head(cand_embed).squeeze(-1)  # (B, K)
        logits = logits.masked_fill(mask == 0, float("-inf"))

        cost_pred = self.cost_head(cand_embed).squeeze(-1)  # (B, K)

        return logits, cost_pred, coord_embed

    @torch.no_grad()
    def select(self, candidates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Convenience for inference: returns the argmax candidate index per row."""
        logits, _, _ = self.forward(candidates, mask)
        return logits.argmax(dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
