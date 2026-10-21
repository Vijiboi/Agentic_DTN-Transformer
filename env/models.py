from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(slots=True)
class UAVAgent:
    uav_id: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

@dataclass(slots=True)
class TwinState:
    """Represents the local twin state y_m(t)[cite: 117]."""
    predicted_load: float = 0.0
    predicted_bandwidth: float = 0.0
    predicted_cpu_ratio: float = 0.0
    age: int = 0             # A_m^tw [cite: 117]
    uncertainty: float = 0.0 # Sigma_m [cite: 117]
    mismatch: float = 0.0    # epsilon_m [cite: 141]
    fidelity: float = 1.0    # F_m [cite: 146]
    # Observed state vector components [cite: 115]
    last_observation: tuple[float, float, float] = (0.0, 0.0, 0.0) 
    filtered_observation: tuple[float, float, float, float, float] | None = None
    forecast_state: tuple[float, float, float] = (0.0, 0.0, 0.0)
    predicted_rate: float = 0.0 # R_hat for delay evaluation [cite: 206]

@dataclass(slots=True)
class CoordinationState:
    """Regional coordination indicators for the closed-loop DTN[cite: 175, 230]."""
    dual_price: float = 0.0       # lambda_m [cite: 287]
    quantized_dual: float = 0.0
    coupling_penalty: float = 0.0 # nu/2 term [cite: 290]
    consensus_drift: float = 0.0
    last_broadcast_slot: int = -1

@dataclass(slots=True)
class APNode:
    """Infrastructure processing nodes (BS, HAP, LEO)[cite: 13, 14]."""
    ap_id: str
    tier: str
    x: float
    y: float
    z: float
    bandwidth: float      # B_m [cite: 79]
    cpu_capacity: float   # F_m^cpu [cite: 79]
    communication_budget: float
    power_budget: float
    trust: float          # tau_m [cite: 105]
    sync_threshold: float # delta triggers [cite: 162]
    coord_threshold: float
    twin_state: TwinState = field(default_factory=TwinState)
    coordination_state: CoordinationState = field(default_factory=CoordinationState)
    current_task_load: int = 0
    current_cpu_load: float = 0.0

@dataclass(slots=True)
class Task:
    """UAV-generated emergency tasks [cite: 53-56]."""
    task_id: str
    source_uav: str
    owner_ap_id: str
    x: float
    y: float
    z: float
    L_u: float      # Input size (bits) [cite: 54]
    D_u: float      # Comp density (cycles/bit) [cite: 54]
    omega_u: float  # Priority weight [cite: 56]
    psi_u: float    # Risk sensitivity [cite: 56]
    xi_u: str       # Task class (control, imagery, alert) [cite: 56]
    bandwidth_demand: float
    cpu_demand: float
    power_demand: float
    A_u_t: list[str] # Candidate destination set M_u [cite: 11]
    arrival_slot: int
    AoI: int = 1    # Age of Information A_u(t) 

    @property
    def F_u_min(self) -> float:
        """Application-aware minimum fidelity requirement."""
        # Critical tasks (control) require higher fidelity than delay-tolerant ones
        return {"control": 0.9, "imagery": 0.7, "alert": 0.8}.get(self.xi_u, 0.5)

@dataclass(slots=True)
class Observation:
    """Physical state s_m(t) used for twin synchronization[cite: 114, 115]."""
    ap_id: str
    slot: int
    load_ratio: float
    bandwidth_ratio: float
    cpu_ratio: float
    queue_size: int
    candidate_overlap: float
    mobility_pressure: float

@dataclass(slots=True)
class CandidateScore:
    """Local surrogate evaluation for task assignment[cite: 312, 432]."""
    task_id: str
    owner_ap_id: str
    destination_id: str
    local_cost: float # Combined J_u,m score [cite: 227]
    delay_cost: float
    energy_cost: float
    mission_cost: float
    fidelity_cost: float
    coupling_penalty: float
    projected_load: float
    required_bandwidth: float
    required_cpu: float
    required_power: float

@dataclass(slots=True)
class QuboProblem:
    """Local quadratic surrogate representation (QUBO)[cite: 271, 387]."""
    ap_id: str
    slot: int
    local_load: float
    variables: list[tuple[str, str]]
    linear: dict[tuple[str, str], float]    # mu vector [cite: 397]
    quadratic: dict[tuple[tuple[str, str], tuple[str, str]], float] # Q matrix [cite: 411]
    penalty_mu: float
    penalty_by_task: dict[str, float]
    candidate_scores: list[CandidateScore]

@dataclass(slots=True)
class SolveResult:
    """Result from the hybrid quantum-classical or classical backend[cite: 399]."""
    sample: dict[tuple[str, str], int]
    energy: float
    samples: list[dict[tuple[str, str], int]]
    solver_name: str
    solver_time: float

@dataclass(slots=True)
class LocalSummary:
    """Compact local summary z_m(t) sent for regional coordination[cite: 175, 230]."""
    ap_id: str
    slot: int
    queue_size: int
    local_load: float
    sync_triggered: bool
    coordination_triggered: bool
    trust: float
    twin_age: int
    uncertainty: float
    mismatch: float
    fidelity: float
    qubo_dimension: int
    solver_time: float
    selected_pairs: list[tuple[str, str]]
    candidate_scores: list[CandidateScore]

@dataclass(slots=True)
class FinalAssignment:
    """Final one-hot assignment result x*_u,m[cite: 434]."""
    task_id: str
    owner_ap_id: str
    destination_id: str
    local_cost: float
    delay_cost: float
    energy_cost: float
    mission_cost: float
    fidelity_cost: float

@dataclass(slots=True)
class SlotResult:
    """Aggregated network-wide result for a single time slot[cite: 559]."""
    slot: int
    assignments: list[FinalAssignment]
    local_summaries: list[LocalSummary]
    average_delay: float
    average_energy: float
    total_delay: float
    total_energy: float
    mission_cost: float
    fidelity_cost: float
    sync_trigger_count: int
    coordination_trigger_count: int
    one_hot_valid: bool
    resource_feasible: bool
    tasks_per_ap: dict[str, int]
    local_execution_count: int
