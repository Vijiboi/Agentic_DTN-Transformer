"""
infer.py
--------
Loads a trained outer-loop checkpoint and turns it into a drop-in policy you
can call once per (task, candidate-AP-set) at simulation time -- i.e. it
plays the same role as `RegionalController.project()` /
`_rank_scores_with_projected_load()` in the Sagin_DT-DQO repo, but the
ranking is learned instead of hand-derived.

Two ways to use it:

1. From a pandas row-group (same schema as the training xlsx) -- e.g. if you
   log live candidate features in that format:

       policy = OuterLoopPolicy.load("outer_loop_ckpt.pt")
       dest_id = policy.select_from_dataframe(candidate_rows_df)

2. From the repo's live objects (Task / APNode / CandidateScore in
   env/models.py) via `features_from_repo_objects`, which maps the fields the
   simulator already tracks (twin_state.*, coordination_state.*, trust,
   candidate delay/energy/mission/fidelity costs, ...) onto the same feature
   schema the model was trained on. Use this inside RegionalController to
   replace/augment `_rank_scores_with_projected_load`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from data import CANDIDATES_PER_TASK, CATEGORICAL_COLS, NUMERIC_COLS, build_feature_matrix
from model import OuterLoopSetTransformer


class OuterLoopPolicy:
    def __init__(self, model: OuterLoopSetTransformer, mean: np.ndarray, std: np.ndarray, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.mean = mean
        self.std = std
        self.device = device

    @staticmethod
    def load(ckpt_path: str | Path, device: str | None = None) -> "OuterLoopPolicy":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = OuterLoopSetTransformer(**ckpt["model_args"])
        model.load_state_dict(ckpt["model_state"])
        fs = ckpt["feature_stats"]
        return OuterLoopPolicy(model, mean=fs["mean"], std=fs["std"], device=device)

    def _standardize(self, feat: np.ndarray) -> np.ndarray:
        return (feat.astype(np.float32) - self.mean) / self.std

    def select_from_dataframe(self, candidates_df: pd.DataFrame) -> dict:
        """
        candidates_df: rows for ONE task, one row per candidate AP, with the
        same columns used in training (NUMERIC_COLS + CATEGORICAL_COLS keys +
        'destination_id'). Returns the chosen destination_id plus the full
        ranked score list for logging/debugging.
        """
        feat = build_feature_matrix(candidates_df)
        feat = self._standardize(feat)
        k = feat.shape[0]
        pad = CANDIDATES_PER_TASK - k
        if pad > 0:
            feat = np.vstack([feat, np.zeros((pad, feat.shape[1]), dtype=np.float32)])
        mask = np.zeros((CANDIDATES_PER_TASK,), dtype=np.float32)
        mask[:k] = 1.0

        cand_t = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, cost_pred, _ = self.model(cand_t, mask_t)
        probs = torch.softmax(logits[0, :k], dim=-1).cpu().numpy()
        order = np.argsort(-probs)

        dest_ids = candidates_df["destination_id"].to_numpy()
        ranked = [
            {"destination_id": dest_ids[i], "prob": float(probs[i]),
             "predicted_local_cost": float(cost_pred[0, i].item())}
            for i in order
        ]
        return {"selected_destination_id": dest_ids[order[0]], "ranked": ranked}


def features_from_repo_objects(task, ap_lookup: dict, candidate_scores: Sequence) -> pd.DataFrame:
    """
    Bridges Sagin_DT-DQO-main's live `Task` / `APNode` / `CandidateScore`
    objects (env/models.py) into a DataFrame with the training schema, so the
    same `OuterLoopPolicy` can be called inside `RegionalController`.

    NOTE: a handful of columns used at training time (e.g. coord_quantized_dual,
    twin_predicted_bandwidth) come straight from ap.twin_state /
    ap.coordination_state; fields the live sim doesn't track 1:1
    (bandwidth_demand, cpu_demand, power_demand, candidate_count) are filled
    with the closest available proxy. Adjust the mapping below if your
    simulator's field names differ.
    """
    rows = []
    n = len(candidate_scores)
    for rank, score in enumerate(candidate_scores, start=1):
        ap = ap_lookup[score.destination_id]
        rows.append({
            "L_u_mbit": task.L_u, "D_u_cycles_per_bit": task.D_u,
            "omega_u": task.omega_u, "psi_u": task.psi_u, "AoI": task.AoI,
            "candidate_rank": rank, "candidate_count": n,
            "distance_km": float(np.linalg.norm([task.x - ap.x, task.y - ap.y, task.z - ap.z])) / 1000.0,
            "ap_bandwidth": ap.bandwidth, "ap_cpu_capacity": ap.cpu_capacity,
            "ap_communication_budget": ap.communication_budget, "ap_power_budget": ap.power_budget,
            "ap_trust": ap.trust, "ap_intrinsic_reliability": ap.trust,
            "ap_current_task_load": ap.current_task_load, "ap_current_cpu_load": ap.current_cpu_load,
            "twin_predicted_load": ap.twin_state.predicted_load,
            "twin_predicted_bandwidth": ap.twin_state.predicted_bandwidth,
            "twin_predicted_cpu_ratio": ap.twin_state.predicted_cpu_ratio,
            "twin_age": ap.twin_state.age, "twin_uncertainty": ap.twin_state.uncertainty,
            "twin_mismatch": ap.twin_state.mismatch, "twin_fidelity": ap.twin_state.fidelity,
            "coord_dual_price": ap.coordination_state.dual_price,
            "coord_quantized_dual": ap.coordination_state.quantized_dual,
            "coord_coupling_penalty": ap.coordination_state.coupling_penalty,
            "predicted_rate_mbps": ap.twin_state.predicted_rate,
            "sync_active": int(ap.twin_state.age <= 1), "coord_active": int(ap.coordination_state.consensus_drift != 0),
            "candidate_delay_s": score.delay_cost, "candidate_energy_j": score.energy_cost,
            "candidate_freshness_risk": score.mission_cost, "candidate_fidelity_penalty": score.fidelity_cost,
            "candidate_local_cost": score.local_cost,
            "xi_u": task.xi_u, "destination_tier": ap.tier,
            "destination_id": score.destination_id,
        })
    return pd.DataFrame(rows)
