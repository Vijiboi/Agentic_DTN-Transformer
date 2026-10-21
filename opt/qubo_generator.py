from __future__ import annotations
import numpy as np
from env.config import SimulationConfig
from env.link_model import predicted_uplink_rate
from env.models import APNode, Task, QuboProblem, CandidateScore

class LocalQuboBuilder: 
    def __init__(self, config: SimulationConfig):
        self.config = config

    def build(self, ap: APNode, tasks: list[Task], ap_lookup: dict[str, APNode], slot: int) -> QuboProblem:
        """
        Constructs the backend-compatible local quadratic surrogate (Eq. 93).
        """
        n_ref = self._compute_local_load(ap, tasks)  
        
        linear_mu = {}
        quadratic_Q = {}
        candidate_scores = []
        rho_m = self._decision_risk_gap(ap)
        nu = self.config.coord_overhead_weight  
        qubo_penalty = self.config.qubo_penalty

        for i, u in enumerate(tasks):
            mu_u, b_u = self._calculate_mu(ap, u, n_ref)
            
            var_u = (u.task_id, ap.ap_id)
            
            linear_mu[var_u] = mu_u - qubo_penalty

            for j in range(i + 1, len(tasks)):
                v = tasks[j]
                b_v = self._calculate_b_coefficient(ap, v, n_ref)
                var_v = (v.task_id, ap.ap_id)
                q_uv = 0.5 * (b_u + b_v) + nu + (nu * (rho_m**2) * u.psi_u * v.psi_u)
                quadratic_Q[(var_u, var_v)] = q_uv

            # Pass individual cost components to the score generator [cite: 323]
            candidate_scores.append(self._generate_candidate_score(ap, u, mu_u, n_ref))

        return QuboProblem(
            ap_id=ap.ap_id,
            slot=slot,
            local_load=n_ref,
            variables=[(t.task_id, ap.ap_id) for t in tasks],
            linear=linear_mu,
            quadratic=quadratic_Q,
            penalty_mu=qubo_penalty,
            penalty_by_task={t.task_id: t.psi_u for t in tasks},
            candidate_scores=candidate_scores
        )

    def score_candidates(
        self,
        owner_ap: APNode,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        preferred_local_tasks: set[str] | None = None,
    ) -> list[CandidateScore]:
        candidate_scores: list[CandidateScore] = []

        for task in tasks:
            for destination_id in task.A_u_t:
                destination_ap = ap_lookup[destination_id]
                if (
                    preferred_local_tasks is not None
                    and destination_id == owner_ap.ap_id
                    and task.task_id not in preferred_local_tasks
                ):
                    continue

                n_ref = self._compute_local_load(destination_ap, [task])
                mu_u, _ = self._calculate_mu(destination_ap, task, n_ref)
                candidate_scores.append(self._generate_candidate_score(destination_ap, task, mu_u, n_ref))

        return candidate_scores

    def score_candidates_for_tasks(
        self,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
    ) -> list[CandidateScore]:
        """Score every task against its complete candidate AP set.

        Dataset generation needs this global view without solving a separate
        owner-local QUBO first.
        """
        candidate_scores: list[CandidateScore] = []
        for task in tasks:
            for destination_id in task.A_u_t:
                ap = ap_lookup[destination_id]
                n_ref = self._compute_local_load(ap, [task])
                mu_u, _ = self._calculate_mu(ap, task, n_ref)
                candidate_scores.append(self._generate_candidate_score(ap, task, mu_u, n_ref))
        return candidate_scores

    def apply_transformer_costs(
        self,
        problem: QuboProblem,
        ap: APNode,
        tasks: list[Task],
        predicted_local_costs: dict[str, float],
        blend: float = 1.0,
    ) -> int:
        """Replace QUBO linear coefficients with Transformer-predicted costs.

        The Transformer predicts the base local objective ``gamma_n``.  This
        method analytically restores the QUBO's load gradient and coordination
        terms, producing a valid linear coefficient ``mu_u - penalty`` for the
        existing solver.  ``blend=1`` is a fully learned coefficient; smaller
        values support controlled ablations against the analytical coefficient.
        """
        blend = float(np.clip(blend, 0.0, 1.0))
        n_ref = self._compute_local_load(ap, tasks)
        rho_m = self._decision_risk_gap(ap)
        nu = self.config.coord_overhead_weight
        dual_term = (nu / 2.0) - (nu * ap.coordination_state.dual_price)
        applied = 0
        for task in tasks:
            predicted_cost = predicted_local_costs.get(task.task_id)
            if predicted_cost is None:
                continue
            variable = (task.task_id, ap.ap_id)
            if variable not in problem.linear:
                continue
            b_u = self._calculate_b_coefficient(ap, task, n_ref)
            coordination_term = (
                dual_term
                + (nu / 2.0) * (rho_m**2) * (task.psi_u**2)
                - (nu * ap.coordination_state.quantized_dual * rho_m * task.psi_u)
            )
            predicted_mu = float(predicted_cost) - ((n_ref - 1.0) * b_u) + coordination_term
            analytical_mu = problem.linear[variable] + problem.penalty_mu
            blended_mu = blend * predicted_mu + (1.0 - blend) * analytical_mu
            problem.linear[variable] = blended_mu - problem.penalty_mu
            applied += 1
        return applied

    def _compute_local_load(self, ap: APNode, tasks: list[Task]) -> float:
        return max(float(ap.current_task_load + len(tasks)), 1.0)

    def _calculate_mu(self, ap: APNode, task: Task, n_ref: float) -> tuple[float, float]:
        b_u = self._calculate_b_coefficient(ap, task, n_ref)
        a_u = self._calculate_a_coefficient(ap, task, n_ref, b_u)
        rho_m = self._decision_risk_gap(ap)
        nu = self.config.coord_overhead_weight
        z_bar_1 = ap.coordination_state.dual_price
        mu_u = (
            a_u
            + b_u
            + (nu / 2.0)
            - (nu * z_bar_1)
            + (nu / 2.0) * (rho_m**2) * (task.psi_u**2)
            - (nu * ap.coordination_state.quantized_dual * rho_m * task.psi_u)
        )
        return mu_u, b_u

    def _calculate_b_coefficient(self, ap: APNode, task: Task, n_ref: float) -> float:
        # Linearized surrogate gradient evaluated at the reference load n_ref.
        delay_grad = (task.L_u * task.D_u) / max(ap.cpu_capacity, 1.0) 
        energy_grad = -2.0 * self.config.kappa_m * task.L_u * task.D_u * (ap.cpu_capacity**2) / (n_ref**3 if n_ref > 0 else 1.0) 
        return self.config.delay_weight * delay_grad + self.config.energy_weight * energy_grad

    def _calculate_a_coefficient(self, ap: APNode, task: Task, n_ref: float, b_u: float) -> float:
        rate = predicted_uplink_rate(task, ap, self.config, n_ref)
        sync_active = ap.twin_state.age == 1
        delay = (
            (task.L_u / rate)
            + (task.L_u * task.D_u * n_ref) / max(ap.cpu_capacity, 1.0)
            + self._sync_delay(ap, sync_active)
        )
        energy = (
            self.config.uav_transmit_power * (task.L_u / rate)
            + self.config.kappa_m * task.L_u * task.D_u * (ap.cpu_capacity / max(n_ref, 1.0))**2
            + self._sync_energy(ap, sync_active)
        )
        risk = self._decision_mission_cost(ap, task)
        fidelity = self._decision_fidelity_cost(ap)
        sync = 1.0 if sync_active else 0.0
        gamma_n = self.config.delay_weight * delay + self.config.energy_weight * energy
        gamma_n += (
            self.config.mission_weight * risk
            + self.config.fidelity_weight * fidelity
            + self._decision_sync_weight() * sync
        )
        return float(gamma_n - (n_ref * b_u))

    def _generate_candidate_score(self, ap: APNode, task: Task, mu_u: float, n_ref: float) -> CandidateScore:
        """
        Calculates individual physical costs with high precision for analytics [cite: 336-342].
        """
        # 1. Physical cost components from the closed-loop DTN cost model.
        rate = predicted_uplink_rate(task, ap, self.config, n_ref)
        sync_active = ap.twin_state.age == 1
        raw_delay = (
            (task.L_u / rate)
            + (task.L_u * task.D_u * n_ref) / max(ap.cpu_capacity, 1.0)
            + self._sync_delay(ap, sync_active)
        )
        delay_cost = self.config.delay_weight * raw_delay
        
        f_share = ap.cpu_capacity / max(n_ref, 1.0)
        raw_energy = (
            self.config.uav_transmit_power * (task.L_u / rate)
            + self.config.kappa_m * task.L_u * task.D_u * (f_share**2)
            + self._sync_energy(ap, sync_active)
        )
        energy_cost = self.config.energy_weight * raw_energy

        # 2. Decision-model risk/freshness components [cite: 109, 223-227].
        # Ablated baselines use reduced information here; final reported
        # metrics are repriced by the regional controller under the full model.
        mission_cost = self._decision_mission_cost(ap, task)
        fidelity_cost = self._decision_fidelity_cost(ap)
        sync_cost = 1.0 if ap.twin_state.age == 1 else 0.0
        objective_cost = (
            delay_cost
            + energy_cost
            + self.config.mission_weight * mission_cost
            + self.config.fidelity_weight * fidelity_cost
            + self._decision_sync_weight() * sync_cost
        )
        
        return CandidateScore(
            task_id=task.task_id,
            owner_ap_id=task.owner_ap_id,
            destination_id=ap.ap_id,
            local_cost=float(objective_cost),
            delay_cost=float(delay_cost),
            energy_cost=float(energy_cost), # Ensure this isn't getting clipped
            mission_cost=float(mission_cost),
            fidelity_cost=float(fidelity_cost),
            coupling_penalty=float(ap.coordination_state.coupling_penalty),
            projected_load=float(ap.current_cpu_load),
            required_bandwidth=task.bandwidth_demand,
            required_cpu=task.cpu_demand,
            required_power=task.power_demand
        )

    def _sync_delay(self, ap: APNode, active: bool) -> float:
        if not active:
            return 0.0
        return float(self.config.sync_delay_by_tier.get(ap.tier, 0.0))

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
