from __future__ import annotations
from dataclasses import asdict, dataclass, field

@dataclass(slots=True)
class SimulationConfig:
    seed: int = 7
    slots: int = 50  # Simulation Horizon [cite: 561, 719]
    area_width: float = 1000.0  
    area_height: float = 1000.0
    # Paper formulation allows each task to choose from the infrastructure set M.
    # With the default 5 BS + 3 HAP + 2 LEO topology, 10 exposes all APs.
    candidate_ap_limit: int = 10
    
    # Topology [cite: 515, 561]
    num_bs: int = 5
    num_haps: int = 3
    num_leos: int = 2
    num_uavs: int = 20
    
    task_arrival_probability: float = 0.8 
    
    # DTN Synchronization Thresholds [cite: 527, 561]
    sync_age_threshold: int = 3
    sync_uncertainty_threshold: float = 0.15
    sync_mismatch_threshold: float = 0.1
    
    # Regional Coordination Thresholds [cite: 527, 561]
    coord_trust_threshold: float = 0.1
    coord_state_threshold: float = 0.2
    coord_uncertainty_threshold: float = 0.15
    coord_mismatch_threshold: float = 0.1
    coord_app_pressure_threshold: float = 0.1
    
    # Objective Weights (alpha, beta, gamma, omega_s, omega_f, nu) [cite: 523, 561]
    delay_weight: float = 1.0
    energy_weight: float = 0.5
    mission_weight: float = 1.0
    sync_cost_weight: float = 0.1
    fidelity_weight: float = 0.2
    coord_overhead_weight: float = 0.1 
    sync_delay_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 0.03, "HAP": 0.08, "LEO": 0.12}
    )
    coord_signal_delay_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 0.01, "HAP": 0.03, "LEO": 0.05}
    )
    sync_energy_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 0.02, "HAP": 0.05, "LEO": 0.08}
    )

    # Trust Parameters [cite: 523, 561]
    trust_update_factor: float = 0.3
    trust_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    
    # --- Additional Paper Parameters ---
    eta_u: float = 0.1 # Freshness sensitivity (Eq 8) [cite: 521, 561]
    twin_smoothing: float = 0.65
    qubo_penalty: float = 200.0
    qubo_load_coupling: float = 1.0
    consensus_step_size: float = 0.75
    consensus_quantum: float = 0.10
    consensus_epsilon: float = 0.08
    twin_gaussian_std: float = 0.04
    sensor_filter_factor: float = 0.65
    task_aoi_min: int = 1
    task_aoi_max: int = 5
    uav_transmit_power: float = 1.0
    noise_spectral_density: float = 1e-9
    anneal_reads: int = 5
    anneal_sweeps: int = 20
    small_qubo_exact_threshold: int = 8
    use_solver_guidance: bool = True
    output_dir: str = "results"
    randomize_task_weights: bool = True
    ap_trust_init_min: float = 0.70
    ap_trust_init_max: float = 0.95
    local_execution_trust: float = 0.75
    ap_initial_predicted_bandwidth: float = 0.75
    kappa_m: float = 1e-28
    kappa_local: float = 1e-28
    uav_local_cpu_capacity: float = 1.0
    local_solver_backend: str = "classical"
    # Optional learned surrogate: predicts raw candidate costs and lets the
    # existing QUBO solver optimize the resulting coefficients.
    use_transformer_qubo_surrogate: bool = False
    transformer_qubo_checkpoint: str | None = None
    transformer_qubo_blend: float = 1.0
    # Offline PSO-teacher settings used only by scripts/generate_pso_dataset.py.
    # Keeping them here makes a generated dataset reproducible from its CLI args.
    pso_swarm_size: int = 8
    pso_iterations: int = 6
    decision_profile: str = "full"
    sync_trigger_policy: str = "event"
    coord_trigger_policy: str = "event"
    sync_fidelity_trigger: bool = True
    
    # --- Mobility Parameters [cite: 528] ---
    uav_altitude_min: float = 8.0
    uav_altitude_max: float = 20.0
    uav_max_altitude: float = 25.0
    uav_velocity_xy_min: float = -3.0
    uav_velocity_xy_max: float = 3.0
    uav_velocity_z_min: float = -0.8
    uav_velocity_z_max: float = 0.8
    observation_ratio_clip: float | None = 1.5
    
    # Tier Resources [cite: 519, 561]
    cpu_capacity_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 10.0, "HAP": 20.0, "LEO": 30.0}
    )
    bandwidth_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 20.0, "HAP": 100.0, "LEO": 500.0}
    )
    communication_budget_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 24.0, "HAP": 42.0, "LEO": 60.0}
    )
    power_budget_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 18.0, "HAP": 28.0, "LEO": 36.0}
    )
    altitude_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 0.03, "HAP": 20.0, "LEO": 550.0}
    )
    carrier_frequency_ghz_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 2.0, "HAP": 30.0, "LEO": 20.0}
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_report_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "simulation_horizon_slots": self.slots,
            "network_topology": {
                "num_bs": self.num_bs,
                "num_haps": self.num_haps,
                "num_leos": self.num_leos,
                "num_uavs": self.num_uavs,
                "candidate_ap_limit": self.candidate_ap_limit,
                "area_width": self.area_width,
                "area_height": self.area_height,
                "altitude_by_tier": self.altitude_by_tier,
            },
            "task_generation": {
                "task_arrival_probability": self.task_arrival_probability,
                "randomize_task_weights": self.randomize_task_weights,
                "uav_altitude_min": self.uav_altitude_min,
                "uav_altitude_max": self.uav_altitude_max,
                "uav_max_altitude": self.uav_max_altitude,
                "uav_velocity_xy_min": self.uav_velocity_xy_min,
                "uav_velocity_xy_max": self.uav_velocity_xy_max,
                "uav_velocity_z_min": self.uav_velocity_z_min,
                "uav_velocity_z_max": self.uav_velocity_z_max,
                "eta_u": self.eta_u,
            },
            "tier_resources": {
                "bandwidth_by_tier": self.bandwidth_by_tier,
                "communication_budget_by_tier": self.communication_budget_by_tier,
                "cpu_capacity_by_tier": self.cpu_capacity_by_tier,
                "power_budget_by_tier": self.power_budget_by_tier,
                "uav_local_cpu_capacity": self.uav_local_cpu_capacity,
            },
            "twin_and_trust": {
                "sync_age_threshold": self.sync_age_threshold,
                "sync_mismatch_threshold": self.sync_mismatch_threshold,
                "sync_uncertainty_threshold": self.sync_uncertainty_threshold,
                "coord_trust_threshold": self.coord_trust_threshold,
                "coord_state_threshold": self.coord_state_threshold,
                "twin_smoothing": self.twin_smoothing,
                "twin_gaussian_std": self.twin_gaussian_std,
                "sensor_filter_factor": self.sensor_filter_factor,
                "ap_trust_init_min": self.ap_trust_init_min,
                "ap_trust_init_max": self.ap_trust_init_max,
                "local_execution_trust": self.local_execution_trust,
            },
            "objective_weights": {
                "alpha_delay": self.delay_weight,
                "beta_energy": self.energy_weight,
                "gamma_mission": self.mission_weight,
                "omega_sync": self.sync_cost_weight,
                "omega_fidelity": self.fidelity_weight,
                "nu_coord": self.coord_overhead_weight,
            },
            "closed_loop_overheads": {
                "sync_delay_by_tier": self.sync_delay_by_tier,
                "coord_signal_delay_by_tier": self.coord_signal_delay_by_tier,
                "sync_energy_by_tier": self.sync_energy_by_tier,
            },
            "solver": {
                "anneal_reads": self.anneal_reads,
                "anneal_sweeps": self.anneal_sweeps,
                "small_qubo_exact_threshold": self.small_qubo_exact_threshold,
                "use_solver_guidance": self.use_solver_guidance,
                "local_solver_backend": self.local_solver_backend,
                "use_transformer_qubo_surrogate": self.use_transformer_qubo_surrogate,
                "transformer_qubo_checkpoint": self.transformer_qubo_checkpoint,
                "transformer_qubo_blend": self.transformer_qubo_blend,
                "decision_profile": self.decision_profile,
                "sync_trigger_policy": self.sync_trigger_policy,
                "coord_trigger_policy": self.coord_trigger_policy,
                "sync_fidelity_trigger": self.sync_fidelity_trigger,
            },
        }

    def to_paper_parameter_view(self) -> dict[str, object]:
        return {
            "BS Bandwidth": self.bandwidth_by_tier["BS"],
            "HAP Bandwidth": self.bandwidth_by_tier["HAP"],
            "LEO Bandwidth": self.bandwidth_by_tier["LEO"],
            "BS CPU Capacity": self.cpu_capacity_by_tier["BS"],
            "HAP CPU Capacity": self.cpu_capacity_by_tier["HAP"],
            "LEO CPU Capacity": self.cpu_capacity_by_tier["LEO"],
            "Number of BSs": self.num_bs,
            "Number of HAPs": self.num_haps,
            "Number of LEO Satellites": self.num_leos,
            "Number of UAV Tasks": self.num_uavs,
            "UAV Local CPU Capacity": self.uav_local_cpu_capacity,
            "Freshness Sensitivity (eta)": self.eta_u,
            "Task AoI Range": (self.task_aoi_min, self.task_aoi_max),
            "Sync Age Threshold": self.sync_age_threshold,
            "Sync Mismatch Threshold": self.sync_mismatch_threshold,
            "Simulation Horizon": self.slots,
        }
