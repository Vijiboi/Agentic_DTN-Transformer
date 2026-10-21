"""Offline particle-swarm teacher for producing Transformer training labels."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from env.config import SimulationConfig
from env.feasibility import FeasibilityManager
from env.link_model import predicted_uplink_rate
from env.models import APNode, Task


@dataclass(slots=True)
class PsoResult:
    assignment: dict[str, str]
    objective_value: float
    runtime_seconds: float
    feasible: bool


class OfflinePsoTeacher:
    """A reproducible discrete PSO over one destination index per task.

    Positions are integer candidate indices.  The objective accounts for the
    equal-share load induced by the full particle, so labels reflect a global
    multi-task decision rather than independent per-candidate argmins.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.feasibility = FeasibilityManager(config)

    def optimize(
        self,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        *,
        sync_ap_ids: set[str] | None = None,
        coord_ap_ids: set[str] | None = None,
        seed: int = 7,
    ) -> PsoResult:
        started = perf_counter()
        if not tasks:
            return PsoResult({}, 0.0, 0.0, True)

        rng = np.random.default_rng(seed)
        candidate_counts = np.asarray([len(task.A_u_t) for task in tasks], dtype=int)
        if np.any(candidate_counts == 0):
            assignment = {
                task.task_id: (task.A_u_t[0] if task.A_u_t else "LOCAL")
                for task in tasks
            }
            return PsoResult(assignment, float("inf"), perf_counter() - started, False)

        n_tasks = len(tasks)
        swarm_size = max(2, int(self.config.pso_swarm_size))
        iterations = max(1, int(self.config.pso_iterations))
        positions = np.vstack([
            rng.integers(0, candidate_counts, endpoint=False) for _ in range(swarm_size)
        ]).astype(float)
        velocities = rng.uniform(-1.0, 1.0, size=(swarm_size, n_tasks))

        # Seed one particle with each task's lowest standalone objective.
        positions[0] = self._greedy_position(tasks, ap_lookup)
        costs = np.asarray([
            self._objective(pos.astype(int), tasks, ap_lookup, sync_ap_ids, coord_ap_ids)
            for pos in positions
        ])
        personal_best = positions.copy()
        personal_costs = costs.copy()
        best_index = int(np.argmin(costs))
        global_best = positions[best_index].copy()
        global_cost = float(costs[best_index])

        for iteration in range(iterations):
            inertia = 0.90 - 0.50 * (iteration / max(iterations - 1, 1))
            r1 = rng.random((swarm_size, n_tasks))
            r2 = rng.random((swarm_size, n_tasks))
            velocities = (
                inertia * velocities
                + 1.35 * r1 * (personal_best - positions)
                + 1.35 * r2 * (global_best - positions)
            )
            positions = np.rint(positions + velocities)
            positions = np.minimum(np.maximum(positions, 0), candidate_counts - 1)

            # A small categorical mutation avoids treating candidate order as a
            # permanent constraint in the integer PSO encoding.
            mutation = rng.random((swarm_size, n_tasks)) < 0.03
            for particle, task_idx in np.argwhere(mutation):
                positions[particle, task_idx] = rng.integers(candidate_counts[task_idx])

            costs = np.asarray([
                self._objective(pos.astype(int), tasks, ap_lookup, sync_ap_ids, coord_ap_ids)
                for pos in positions
            ])
            improved = costs < personal_costs
            personal_best[improved] = positions[improved]
            personal_costs[improved] = costs[improved]
            best_index = int(np.argmin(personal_costs))
            if personal_costs[best_index] < global_cost:
                global_best = personal_best[best_index].copy()
                global_cost = float(personal_costs[best_index])

        assignment = {
            task.task_id: task.A_u_t[int(global_best[index])]
            for index, task in enumerate(tasks)
        }
        feasible = self.feasibility.assignment_is_feasible(assignment, tasks, ap_lookup)
        return PsoResult(assignment, float(global_cost), perf_counter() - started, feasible)

    def _greedy_position(self, tasks: list[Task], ap_lookup: dict[str, APNode]) -> np.ndarray:
        choices = []
        for task in tasks:
            choice = min(
                range(len(task.A_u_t)),
                key=lambda index: self._single_cost(task, ap_lookup[task.A_u_t[index]], 1.0, False, False),
            )
            choices.append(choice)
        return np.asarray(choices, dtype=float)

    def _objective(
        self, position: np.ndarray, tasks: list[Task], ap_lookup: dict[str, APNode],
        sync_ap_ids: set[str] | None, coord_ap_ids: set[str] | None,
    ) -> float:
        assignment = {
            task.task_id: task.A_u_t[int(position[index])]
            for index, task in enumerate(tasks)
        }
        return self._objective_from_assignment(assignment, tasks, ap_lookup, sync_ap_ids, coord_ap_ids)

    def _objective_from_assignment(
        self, assignment: dict[str, str], tasks: list[Task], ap_lookup: dict[str, APNode],
        sync_ap_ids: set[str] | None, coord_ap_ids: set[str] | None,
    ) -> float:
        sync_ap_ids = sync_ap_ids or set()
        coord_ap_ids = coord_ap_ids or set()
        by_id = {task.task_id: task for task in tasks}
        counts = Counter(destination for destination in assignment.values() if destination != "LOCAL")
        usage: dict[str, dict[str, float]] = {}
        for task_id, destination in assignment.items():
            if destination == "LOCAL":
                continue
            task = by_id[task_id]
            value = usage.setdefault(destination, {"cpu": 0.0, "bandwidth": 0.0, "power": 0.0})
            value["cpu"] += task.cpu_demand
            value["bandwidth"] += task.bandwidth_demand
            value["power"] += task.power_demand

        cost = 0.0
        for task_id, destination in assignment.items():
            task = by_id[task_id]
            if destination == "LOCAL":
                cost += self._local_cost(task)
                continue
            ap = ap_lookup[destination]
            value = usage[destination]
            overflow = (
                max(0.0, ap.current_cpu_load + value["cpu"] - ap.cpu_capacity)
                + max(0.0, value["bandwidth"] - ap.communication_budget)
                + max(0.0, value["power"] - ap.power_budget)
            )
            cost += self._single_cost(
                task, ap, float(counts[destination]), destination in sync_ap_ids, destination in coord_ap_ids
            ) + 1e6 * overflow
        return float(cost)

    def _single_cost(self, task: Task, ap: APNode, load: float, sync_active: bool, coord_active: bool) -> float:
        rate = predicted_uplink_rate(task, ap, self.config, load)
        delay = task.L_u / rate + task.L_u * task.D_u * load / max(ap.cpu_capacity, 1e-9)
        energy = self.config.uav_transmit_power * task.L_u / rate
        if sync_active:
            delay += self.config.sync_delay_by_tier.get(ap.tier, 0.0)
            energy += self.config.sync_energy_by_tier.get(ap.tier, 0.0)
        if coord_active:
            delay += self.config.coord_signal_delay_by_tier.get(ap.tier, 0.0)
        freshness = 1.0 - np.exp(-self.config.eta_u * task.AoI)
        mission = task.psi_u * (1.0 - ap.trust) * freshness
        fidelity_penalty = 1e6 if ap.twin_state.fidelity < task.F_u_min else 0.0
        return (
            self.config.delay_weight * delay + self.config.energy_weight * energy
            + self.config.mission_weight * mission
            + self.config.fidelity_weight * (1.0 - ap.twin_state.fidelity)
            + self.config.sync_cost_weight * float(sync_active)
            + ap.coordination_state.coupling_penalty + fidelity_penalty
        )

    def _local_cost(self, task: Task) -> float:
        delay = task.L_u * task.D_u / max(self.config.uav_local_cpu_capacity, 1e-9)
        freshness = 1.0 - np.exp(-self.config.eta_u * task.AoI)
        return self.config.delay_weight * delay + task.psi_u * (1.0 - self.config.local_execution_trust) * freshness
