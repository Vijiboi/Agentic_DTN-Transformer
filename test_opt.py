import numpy as np
from env.config import SimulationConfig
from env.models import APNode, Task
from opt.qubo_generator import LocalQuboBuilder
from opt.solver import ClassicalQuboSolver

def test_optimization_logic():
    # 1. Setup
    config = SimulationConfig()
    builder = LocalQuboBuilder(config)
    solver = ClassicalQuboSolver(config, backend="dimod")
    
    # 2. Mock Infrastructure [cite: 515-519]
    ap = APNode(
        ap_id="BS_0", tier="BS", x=0, y=0, z=0.03,
        bandwidth=20.0, cpu_capacity=10.0,
        communication_budget=24.0, power_budget=18.0,
        trust=0.85, sync_threshold=0.1, coord_threshold=0.1
    )
    ap_lookup = {ap.ap_id: ap}
    
    # 3. Mock Emergency Task [cite: 54, 520]
    tasks = [
        Task(
            task_id="T_Emerg_1", source_uav="UAV_0", owner_ap_id="BS_0",
            x=10.0, y=10.0, z=15.0, L_u=1.5, D_u=1000.0,
            omega_u=1.2, psi_u=1.1, xi_u="control",
            bandwidth_demand=3.0, cpu_demand=4.0, power_demand=2.0,
            A_u_t=["BS_0"], arrival_slot=1, AoI=2
        )
    ]

    # 4. Build QUBO [cite: 325-331]
    problem = builder.build(ap, tasks, ap_lookup, slot=1)
    
    print(f"--- Optimization Test: {problem.ap_id} ---")
    print(f"Linear Terms (mu): {problem.linear}")
    print(f"Penalty (One-Hot): {problem.penalty_mu}")
    
    # 5. Solve (Classical for now, testing structure)
    result = solver.solve(problem)
    
    print(f"Solver Used: {result.solver_name}")
    print(f"Task Assignment Sample: {result.sample}")
    print(f"Energy: {result.energy:.4f}")

if __name__ == "__main__":
    test_optimization_logic()