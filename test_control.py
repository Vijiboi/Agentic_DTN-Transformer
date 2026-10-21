from env.config import SimulationConfig
from env.models import Task, CandidateScore, APNode, TwinState, LocalSummary
from control.regional import RegionalController

def test_regional_projection():
    config = SimulationConfig()
    coord = RegionalController()
    
    # 1. Critical task (control) requiring F_min = 0.9 [cite: 158]
    task = Task(task_id="T_Crit", source_uav="U1", owner_ap_id="BS_0", x=0, y=0, z=0,
                L_u=1.0, D_u=1000, omega_u=1.0, psi_u=1.0, xi_u="control",
                bandwidth_demand=1, cpu_demand=1, power_demand=1, 
                A_u_t=["BS_Low", "BS_High"], arrival_slot=1)
    
    # 2. Mock APs: One cheap/stale, one expensive/fresh [cite: 147-149]
    ap_low = APNode(ap_id="BS_Low", tier="BS", x=0, y=0, z=0, bandwidth=20, cpu_capacity=10, 
                    communication_budget=20, power_budget=18, trust=0.8, 
                    sync_threshold=0.1, coord_threshold=0.1,
                    twin_state=TwinState(fidelity=0.5)) # Fails F_min
    ap_high = APNode(ap_id="BS_High", tier="BS", x=0, y=0, z=0, bandwidth=20, cpu_capacity=10, 
                     communication_budget=20, power_budget=18, trust=0.8, 
                     sync_threshold=0.1, coord_threshold=0.1,
                     twin_state=TwinState(fidelity=0.95)) # Passes F_min
    
    ap_map = {ap_low.ap_id: ap_low, ap_high.ap_id: ap_high}
    
    # 3. Mock Summaries with Candidate Scores
    score_low = CandidateScore(task_id="T_Crit", owner_ap_id="BS_0", destination_id="BS_Low", 
                               local_cost=10.0, delay_cost=0, energy_cost=0, mission_cost=0, 
                               fidelity_cost=0, coupling_penalty=0, projected_load=0, 
                               required_bandwidth=1, required_cpu=1, required_power=1)
    score_high = CandidateScore(task_id="T_Crit", owner_ap_id="BS_0", destination_id="BS_High", 
                                local_cost=50.0, delay_cost=0, energy_cost=0, mission_cost=0, 
                                fidelity_cost=0, coupling_penalty=0, projected_load=0, 
                                required_bandwidth=1, required_cpu=1, required_power=1)
    
    summary = LocalSummary(ap_id="BS_0", slot=1, queue_size=1, local_load=0, sync_triggered=False, 
                           coordination_triggered=False, trust=0.8, twin_age=1, uncertainty=0, 
                           mismatch=0, fidelity=1.0, qubo_dimension=2, solver_time=0, 
                           selected_pairs=[], candidate_scores=[score_low, score_high])

    # 4. Run Projection [cite: 508-509]
    assignments = coord.project([summary], ap_map, [task])
    
    print(f"Task Class: {task.xi_u}, Req. Fidelity: {task.F_u_min}")
    print(f"Assigned Destination: {assignments[0].destination_id}")
    print(f"Cost of Selection: {assignments[0].local_cost}")
    print("-" * 30)
    print("Result: Should select 'BS_High' despite higher cost due to hard fidelity constraint.")

if __name__ == "__main__":
    test_regional_projection()