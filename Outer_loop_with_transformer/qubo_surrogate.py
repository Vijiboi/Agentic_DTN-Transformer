"""Runtime bridge: Transformer-predicted candidate costs -> QUBO coefficients."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

try:  # Supports both ``python train_qubo_surrogate.py`` and package imports.
    from .data import CANDIDATES_PER_TASK, build_feature_matrix
    from .model import OuterLoopSetTransformer
except ImportError:
    from data import CANDIDATES_PER_TASK, build_feature_matrix
    from model import OuterLoopSetTransformer


class TransformerQuboCostSurrogate:
    """Predicts base local candidate costs without receiving cost-derived inputs."""

    def __init__(self, model, feature_stats, target_mean: float, target_std: float, device: str):
        self.model = model.to(device).eval()
        self.feature_stats = feature_stats
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("checkpoint_kind") != "transformer_qubo_cost_surrogate_v1":
            raise ValueError("Checkpoint is not a Transformer-to-QUBO cost-surrogate checkpoint.")
        model = OuterLoopSetTransformer(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["model_state"])
        return cls(
            model, checkpoint["feature_stats"], checkpoint["target_mean"],
            checkpoint["target_std"], device,
        )

    def predict_task_costs(self, task, ap_lookup: dict, candidate_scores: Sequence, config) -> dict[str, float]:
        """Return one predicted *base local cost* for each candidate AP of a task."""
        frame = raw_features_from_repo_objects(task, ap_lookup, candidate_scores, config)
        stats = self.feature_stats
        feature = build_feature_matrix(
            frame, numeric_cols=list(stats["numeric_cols"]), categorical_cols=stats["categorical_cols"]
        )
        feature = (feature.astype(np.float32) - stats["mean"]) / stats["std"]
        k = len(frame)
        if k > CANDIDATES_PER_TASK:
            raise ValueError(f"Model supports at most {CANDIDATES_PER_TASK} candidate APs, received {k}.")
        if k < CANDIDATES_PER_TASK:
            feature = np.vstack([feature, np.zeros((CANDIDATES_PER_TASK - k, feature.shape[1]), dtype=np.float32)])
        mask = np.zeros((CANDIDATES_PER_TASK,), dtype=np.float32)
        mask[:k] = 1.0
        with torch.no_grad():
            _, normalized_cost, _ = self.model(
                torch.from_numpy(feature).unsqueeze(0).to(self.device),
                torch.from_numpy(mask).unsqueeze(0).to(self.device),
            )
        predicted = normalized_cost[0, :k].cpu().numpy() * self.target_std + self.target_mean
        # Costs are non-negative by construction. Clipping prevents a rare
        # extrapolation from introducing an unphysical negative QUBO cost.
        predicted = np.maximum(predicted, 0.0)
        return {score.destination_id: float(predicted[index]) for index, score in enumerate(candidate_scores)}


def raw_features_from_repo_objects(task, ap_lookup: dict, candidate_scores: Sequence, config) -> pd.DataFrame:
    """Map live state to the leakage-free cost-surrogate feature schema.

    Candidate-score cost fields are intentionally not present.  The only
    values used are task requirements, AP resources, twin/coordination state,
    and physically computed link rate available before coefficient generation.
    """
    from env.link_model import distance_km, predicted_uplink_rate

    rows = []
    total = len(candidate_scores)
    for rank, score in enumerate(candidate_scores, start=1):
        ap = ap_lookup[score.destination_id]
        n_ref = max(ap.current_task_load + 1, 1)
        rows.append({
            "L_u_mbit": task.L_u, "D_u_cycles_per_bit": task.D_u,
            "omega_u": task.omega_u, "psi_u": task.psi_u, "AoI": task.AoI,
            "candidate_rank": rank, "candidate_count": total,
            "distance_km": distance_km(task, ap),
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
            "predicted_rate_mbps": predicted_uplink_rate(task, ap, config, n_ref),
            "sync_active": int(ap.twin_state.age == 1),
            "coord_active": int(ap.coordination_state.consensus_drift != 0.0),
            "xi_u": task.xi_u, "destination_tier": ap.tier,
            "destination_id": score.destination_id,
        })
    return pd.DataFrame(rows)
