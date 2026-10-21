"""Streaming writer for PSO-labelled outer-loop training data."""
from __future__ import annotations

import csv
from pathlib import Path

from env.config import SimulationConfig
from env.link_model import distance_km, predicted_uplink_rate
from env.models import APNode, CandidateScore, Task
from opt.pso_teacher import PsoResult


INPUT_FEATURE_COLUMNS = [
    "scenario_id", "scenario_profile", "seed", "slot", "task_id", "source_uav", "owner_ap_id",
    "task_x", "task_y", "task_z", "L_u_mbit", "D_u_cycles_per_bit", "omega_u", "psi_u", "xi_u", "AoI",
    "bandwidth_demand", "cpu_demand", "power_demand", "candidate_rank", "candidate_count",
    "destination_id", "destination_tier", "ap_x", "ap_y", "ap_z", "ap_bandwidth", "ap_cpu_capacity",
    "ap_communication_budget", "ap_power_budget", "ap_trust", "ap_intrinsic_reliability",
    "ap_current_task_load", "ap_current_cpu_load", "twin_predicted_load", "twin_predicted_bandwidth",
    "twin_predicted_cpu_ratio", "twin_age", "twin_uncertainty", "twin_mismatch", "twin_fidelity",
    "coord_dual_price", "coord_quantized_dual", "coord_coupling_penalty", "distance_km", "predicted_rate_mbps",
    "sync_active", "coord_active", "candidate_delay_s", "candidate_energy_j", "candidate_freshness_risk",
    "candidate_fidelity_penalty", "candidate_local_cost",
]
LABEL_METADATA_COLUMNS = [
    "pso_selected", "pso_selected_destination_id", "pso_objective_value", "pso_runtime_seconds", "pso_feasible",
]
FIELDNAMES = INPUT_FEATURE_COLUMNS + LABEL_METADATA_COLUMNS


class PsoDatasetWriter:
    """Writes rows incrementally, so a million-row run stays memory bounded."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.rows_written = 0
        self._handle = None
        self._writer = None

    def __enter__(self) -> "PsoDatasetWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle:
            self._handle.close()

    def flush(self) -> None:
        if self._handle:
            self._handle.flush()

    def write_task_candidate_rows(
        self,
        *,
        scenario_id: int,
        scenario_profile: str,
        seed: int,
        slot: int,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        candidate_scores: list[CandidateScore],
        sync_ap_ids: set[str],
        coord_ap_ids: set[str],
        pso_result: PsoResult,
        config: SimulationConfig,
    ) -> None:
        if self._writer is None:
            raise RuntimeError("Use PsoDatasetWriter as a context manager.")
        scores_by_task: dict[str, dict[str, CandidateScore]] = {}
        for score in candidate_scores:
            scores_by_task.setdefault(score.task_id, {})[score.destination_id] = score

        for task in tasks:
            selected_destination = pso_result.assignment.get(task.task_id, "LOCAL")
            for rank, destination_id in enumerate(task.A_u_t, start=1):
                ap = ap_lookup[destination_id]
                score = scores_by_task[task.task_id][destination_id]
                raw_energy = score.energy_cost / max(config.energy_weight, 1e-12)
                self._writer.writerow({
                    "scenario_id": scenario_id, "scenario_profile": scenario_profile, "seed": seed,
                    "slot": slot, "task_id": task.task_id, "source_uav": task.source_uav,
                    "owner_ap_id": task.owner_ap_id, "task_x": task.x, "task_y": task.y, "task_z": task.z,
                    "L_u_mbit": task.L_u, "D_u_cycles_per_bit": task.D_u, "omega_u": task.omega_u,
                    "psi_u": task.psi_u, "xi_u": task.xi_u, "AoI": task.AoI,
                    "bandwidth_demand": task.bandwidth_demand, "cpu_demand": task.cpu_demand,
                    "power_demand": task.power_demand, "candidate_rank": rank, "candidate_count": len(task.A_u_t),
                    "destination_id": destination_id, "destination_tier": ap.tier, "ap_x": ap.x, "ap_y": ap.y,
                    "ap_z": ap.z, "ap_bandwidth": ap.bandwidth, "ap_cpu_capacity": ap.cpu_capacity,
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
                    "coord_coupling_penalty": score.coupling_penalty,
                    "distance_km": distance_km(task, ap),
                    "predicted_rate_mbps": predicted_uplink_rate(task, ap, config, max(ap.current_task_load + 1, 1)),
                    "sync_active": int(destination_id in sync_ap_ids), "coord_active": int(destination_id in coord_ap_ids),
                    "candidate_delay_s": score.delay_cost / max(config.delay_weight, 1e-12),
                    "candidate_energy_j": raw_energy, "candidate_freshness_risk": score.mission_cost,
                    "candidate_fidelity_penalty": score.fidelity_cost, "candidate_local_cost": score.local_cost,
                    "pso_selected": int(destination_id == selected_destination),
                    "pso_selected_destination_id": selected_destination,
                    "pso_objective_value": pso_result.objective_value,
                    "pso_runtime_seconds": pso_result.runtime_seconds,
                    "pso_feasible": int(pso_result.feasible),
                })
                self.rows_written += 1
