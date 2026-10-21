from __future__ import annotations
from collections import defaultdict
import numpy as np
from env.baseline import greedy_one_hot_assignment
from env.config import SimulationConfig
from env.link_model import predicted_uplink_rate
from env.models import CandidateScore, FinalAssignment, LocalSummary, APNode, Task

class RegionalController:
    def __init__(self, strategy: str = "hierarchical", config: SimulationConfig | None = None):
        self.strategy = strategy
        self.config = config or SimulationConfig()

    def project(
        self,
        local_summaries: list[LocalSummary],
        ap_lookup: dict[str, APNode],
        tasks: list[Task]
    ) -> list[FinalAssignment]:
        """
        Enforces global one-hot task-assignment across all APs and local execution [cite: 1505-1509].
        Selection Rule: m* = arg min { J_hat + lambda_f * I(F < F_min) }[cite: 430].
        """
        all_scores: list[CandidateScore] = []
        # Group scores by task_id to evaluate competition for the same task 
        all_scores_by_task: dict[str, list[CandidateScore]] = defaultdict(list)
        
        for summary in local_summaries:
            all_scores.extend(summary.candidate_scores)
            for score in summary.candidate_scores:
                all_scores_by_task[score.task_id].append(score)
        sync_ap_ids = {summary.ap_id for summary in local_summaries if summary.sync_triggered}
        coord_ap_ids = {summary.ap_id for summary in local_summaries if summary.coordination_triggered}

        final_assignments: list[FinalAssignment] = []
        # Track selected load for reporting. The paper proof-of-concept uses
        # equal sharing, so resource pressure is reflected in delay/energy and
        # coordination cost rather than rejecting AP candidates by synthetic
        # per-task demand fields.
        resource_usage = {
            ap_id: {"cpu": 0.0, "bandwidth": 0.0, "power": 0.0, "tasks": 0.0}
            for ap_id in ap_lookup
        }

        # Match tasks to generated scores
        task_map = {t.task_id: t for t in tasks}
        if self.strategy == "greedy":
            greedy_assignments = greedy_one_hot_assignment(all_scores, ap_lookup, task_map)
            assigned_task_ids = {assignment.task_id for assignment in greedy_assignments}
            for task in tasks:
                if task.task_id not in assigned_task_ids:
                    greedy_assignments.append(self._local_execution_assignment(task))
            return self._reprice_with_equal_sharing(
                greedy_assignments,
                task_map,
                ap_lookup,
                sync_ap_ids,
                coord_ap_ids,
            )

        remaining_tasks = set(all_scores_by_task.keys())

        # Enforce Hierarchical Projection Rule [cite: 437, 1527-1550]
        while remaining_tasks:
            ranked_tasks = []
            for task_id in list(remaining_tasks):
                task_obj = task_map.get(task_id)
                if not task_obj: continue
                
                # Rank candidates for this task based on cost + fidelity penalty [cite: 1530-1533]
                ranked_scores = self._rank_scores_with_projected_load(
                    all_scores_by_task[task_id], 
                    task_obj, 
                    ap_lookup,
                    resource_usage,
                    sync_ap_ids,
                    coord_ap_ids,
                )
                
                # Fidelity is handled in the ranking penalty. Equal-share AP
                # resources stay admissible; extra load increases objective cost.
                feasible_scores = [
                    s for s in ranked_scores 
                    if self._is_feasible(s, ap_lookup, resource_usage)
                ]
                
                if feasible_scores:
                    best_s = feasible_scores[0]
                    # Priority: favor tasks with fewer feasible options or higher cost [cite: 1540-1549]
                    ranked_tasks.append((
                        len(feasible_scores), 
                        self._projected_objective(
                            best_s,
                            task_obj,
                            ap_lookup,
                            resource_usage,
                            sync_ap_ids,
                            coord_ap_ids,
                        ) + best_s.coupling_penalty,
                        task_id, 
                        feasible_scores
                    ))

            if not ranked_tasks: break
            
            # Select the most constrained task to assign next [cite: 1550]
            _, _, winner_task_id, winner_feasible = min(ranked_tasks)
            winner_score = winner_feasible[0]
            
            # Update physical resource consumption [cite: 1552-1555]
            usage = resource_usage[winner_score.destination_id]
            usage["cpu"] += winner_score.required_cpu
            usage["bandwidth"] += winner_score.required_bandwidth
            usage["power"] += winner_score.required_power
            usage["tasks"] += 1.0
            
            # Build final one-hot result [cite: 1556-1567]
            final_assignments.append(FinalAssignment(
                task_id=winner_score.task_id,
                owner_ap_id=winner_score.owner_ap_id,
                destination_id=winner_score.destination_id,
                local_cost=winner_score.local_cost,
                delay_cost=winner_score.delay_cost,
                energy_cost=winner_score.energy_cost,
                mission_cost=winner_score.mission_cost,
                fidelity_cost=winner_score.fidelity_cost
            ))
            remaining_tasks.remove(winner_task_id)

        # Fallback to greedy if projection failed to assign all [cite: 1569-1570]
        if not final_assignments and all_scores:
            greedy_assignments = greedy_one_hot_assignment(all_scores, ap_lookup, task_map)
            assigned_task_ids = {assignment.task_id for assignment in greedy_assignments}
            for task in tasks:
                if task.task_id not in assigned_task_ids:
                    greedy_assignments.append(self._local_execution_assignment(task))
            return self._reprice_with_equal_sharing(
                greedy_assignments,
                task_map,
                ap_lookup,
                sync_ap_ids,
                coord_ap_ids,
            )

        # Paper one-hot set includes m=0 local UAV execution. Use it as the
        # feasibility-preserving destination for tasks that no AP can accept.
        for task_id in sorted(remaining_tasks):
            task = task_map.get(task_id)
            if task:
                final_assignments.append(self._local_execution_assignment(task))
            
        return self._reprice_with_equal_sharing(
            final_assignments,
            task_map,
            ap_lookup,
            sync_ap_ids,
            coord_ap_ids,
        )

    def _local_execution_assignment(self, task: Task) -> FinalAssignment:
        delay = (task.L_u * task.D_u) / max(self.config.uav_local_cpu_capacity, 1e-9)
        energy = self.config.kappa_local * task.L_u * task.D_u * (self.config.uav_local_cpu_capacity**2)
        freshness = 1.0 - np.exp(-self.config.eta_u * task.AoI)
        mission = task.psi_u * (1.0 - self.config.local_execution_trust) * freshness
        return FinalAssignment(
            task_id=task.task_id,
            owner_ap_id=task.owner_ap_id,
            destination_id="LOCAL",
            local_cost=(
                (self.config.delay_weight * delay)
                + (self.config.energy_weight * energy)
                + (self.config.mission_weight * mission)
            ),
            delay_cost=self.config.delay_weight * delay,
            energy_cost=self.config.energy_weight * energy,
            mission_cost=float(mission),
            fidelity_cost=0.0,
        )

    def _reprice_with_equal_sharing(
        self,
        assignments: list[FinalAssignment],
        task_map: dict[str, Task],
        ap_lookup: dict[str, APNode],
        sync_ap_ids: set[str] | None = None,
        coord_ap_ids: set[str] | None = None,
    ) -> list[FinalAssignment]:
        sync_ap_ids = sync_ap_ids or set()
        coord_ap_ids = coord_ap_ids or set()
        assigned_per_ap: dict[str, int] = defaultdict(int)
        for assignment in assignments:
            if assignment.destination_id != "LOCAL":
                assigned_per_ap[assignment.destination_id] += 1

        repriced: list[FinalAssignment] = []
        for assignment in assignments:
            task = task_map[assignment.task_id]
            if assignment.destination_id == "LOCAL":
                repriced.append(self._local_execution_assignment(task))
                continue

            ap = ap_lookup[assignment.destination_id]
            load_count = max(float(assigned_per_ap[assignment.destination_id]), 1.0)
            rate = predicted_uplink_rate(task, ap, self.config, load_count)
            sync_active = assignment.destination_id in sync_ap_ids
            coord_active = assignment.destination_id in coord_ap_ids
            delay = (
                (task.L_u / rate)
                + (task.L_u * task.D_u * load_count) / max(ap.cpu_capacity, 1e-9)
                + self._sync_delay(ap, sync_active)
                + self._coord_delay(ap, coord_active)
            )
            energy = (
                self.config.uav_transmit_power * (task.L_u / rate)
                + self.config.kappa_m * task.L_u * task.D_u * (ap.cpu_capacity / load_count) ** 2
                + self._sync_energy(ap, sync_active)
            )
            freshness = 1.0 - np.exp(-self.config.eta_u * task.AoI)
            mission = task.psi_u * (1.0 - ap.trust) * freshness
            fidelity = 1.0 - ap.twin_state.fidelity
            sync = 1.0 if sync_active else 0.0
            delay_cost = self.config.delay_weight * delay
            energy_cost = self.config.energy_weight * energy
            local_cost = (
                delay_cost
                + energy_cost
                + self.config.mission_weight * mission
                + self.config.fidelity_weight * fidelity
                + self.config.sync_cost_weight * sync
            )
            repriced.append(FinalAssignment(
                task_id=assignment.task_id,
                owner_ap_id=assignment.owner_ap_id,
                destination_id=assignment.destination_id,
                local_cost=float(local_cost),
                delay_cost=float(delay_cost),
                energy_cost=float(energy_cost),
                mission_cost=float(mission),
                fidelity_cost=float(fidelity),
            ))

        return repriced

    def _rank_scores_with_fidelity(self, scores: list[CandidateScore], task: Task, ap_lookup: dict):
        """Applies lambda_f penalty for fidelity violations (Eq. 102) ."""
        lambda_f = 1e6 # Large infeasibility penalty [cite: 432]
        return sorted(
            scores,
            key=lambda s: s.local_cost + s.coupling_penalty + 
            (lambda_f if ap_lookup[s.destination_id].twin_state.fidelity < task.F_u_min else 0)
        )

    def _rank_scores_with_projected_load(
        self,
        scores: list[CandidateScore],
        task: Task,
        ap_lookup: dict[str, APNode],
        resource_usage: dict[str, dict[str, float]],
        sync_ap_ids: set[str],
        coord_ap_ids: set[str],
    ) -> list[CandidateScore]:
        lambda_f = 1e6
        return sorted(
            scores,
            key=lambda score: (
                self._projected_objective(
                    score,
                    task,
                    ap_lookup,
                    resource_usage,
                    sync_ap_ids,
                    coord_ap_ids,
                )
                + score.coupling_penalty
                + (lambda_f if ap_lookup[score.destination_id].twin_state.fidelity < task.F_u_min else 0.0)
            ),
        )

    def _projected_objective(
        self,
        score: CandidateScore,
        task: Task,
        ap_lookup: dict[str, APNode],
        resource_usage: dict[str, dict[str, float]],
        sync_ap_ids: set[str] | None = None,
        coord_ap_ids: set[str] | None = None,
    ) -> float:
        sync_ap_ids = sync_ap_ids or set()
        coord_ap_ids = coord_ap_ids or set()
        ap = ap_lookup[score.destination_id]
        load_count = max(resource_usage[score.destination_id]["tasks"] + 1.0, 1.0)
        rate = predicted_uplink_rate(task, ap, self.config, load_count)
        sync_active = score.destination_id in sync_ap_ids
        coord_active = score.destination_id in coord_ap_ids
        delay = (
            (task.L_u / rate)
            + (task.L_u * task.D_u * load_count) / max(ap.cpu_capacity, 1e-9)
            + self._sync_delay(ap, sync_active)
            + self._coord_delay(ap, coord_active)
        )
        energy = (
            self.config.uav_transmit_power * (task.L_u / rate)
            + self.config.kappa_m * task.L_u * task.D_u * (ap.cpu_capacity / load_count) ** 2
            + self._sync_energy(ap, sync_active)
        )
        mission = self._decision_mission_cost(ap, task)
        fidelity = self._decision_fidelity_cost(ap)
        sync = 1.0 if sync_active else 0.0
        return float(
            self.config.delay_weight * delay
            + self.config.energy_weight * energy
            + self.config.mission_weight * mission
            + self.config.fidelity_weight * fidelity
            + self._decision_sync_weight() * sync
        )

    @staticmethod
    def _is_feasible(score: CandidateScore, ap_lookup, resource_usage):
        """AP candidates remain feasible under the paper's equal-share model."""
        return True

    def _sync_delay(self, ap: APNode, active: bool) -> float:
        if not active:
            return 0.0
        return float(self.config.sync_delay_by_tier.get(ap.tier, 0.0))

    def _coord_delay(self, ap: APNode, active: bool) -> float:
        if not active:
            return 0.0
        return float(self.config.coord_signal_delay_by_tier.get(ap.tier, 0.0))

    def _sync_energy(self, ap: APNode, active: bool) -> float:
        if not active:
            return 0.0
        return float(self.config.sync_energy_by_tier.get(ap.tier, 0.0))

    def _decision_profile(self) -> str:
        return self.config.decision_profile.lower()

    def _decision_risk_gap(self, ap: APNode) -> float:
        profile = self._decision_profile()
        if profile in {"channel_only", "dt_free"}:
            return 0.0
        if profile == "trust_unaware":
            return 0.15
        return 1.0 - ap.trust

    def _decision_freshness(self, task: Task) -> float:
        if self._decision_profile() == "aoi_unaware":
            midpoint_aoi = 0.5 * (self.config.task_aoi_min + self.config.task_aoi_max)
            return float(1.0 - np.exp(-self.config.eta_u * midpoint_aoi))
        return float(1.0 - np.exp(-self.config.eta_u * task.AoI))

    def _decision_mission_cost(self, ap: APNode, task: Task) -> float:
        if self._decision_profile() in {"channel_only", "dt_free"}:
            return 0.0
        return float(task.psi_u * self._decision_risk_gap(ap) * self._decision_freshness(task))

    def _decision_fidelity_cost(self, ap: APNode) -> float:
        if self._decision_profile() in {"channel_only", "dt_free"}:
            return 0.0
        return float(1.0 - ap.twin_state.fidelity)

    def _decision_sync_weight(self) -> float:
        if self._decision_profile() in {"channel_only", "dt_free"}:
            return 0.0
        return self.config.sync_cost_weight
