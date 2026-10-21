from env.config import SimulationConfig
from env.generator import SaginEnvironment

def test_sagin_env():
    config = SimulationConfig(num_uavs=5)
    env = SaginEnvironment(config)
    
    print(f"--- SAGIN Env Initialized ---")
    print(f"APs generated: {len(env.aps)} (BS: {config.num_bs}, HAP: {config.num_haps}, LEO: {config.num_leos})")
    
    tasks = env.create_tasks_for_slot(slot=1)
    print(f"Tasks generated for slot 1: {len(tasks)}")
    
    if tasks:
        sample = tasks[0]
        print(f"\nSample Task: {sample.task_id}")
        print(f"Class: {sample.xi_u}, Size: {sample.L_u} Mbits, Weight: {sample.omega_u:.2f}")
        print(f"Min Fidelity Required: {sample.F_u_min}")
        print(f"Candidate APs: {sample.A_u_t}")

    for ap in env.aps:
        print(f"\nAP {ap.ap_id} ({ap.tier}):")
        print(f" Trust: {ap.trust:.2f}, CPU: {ap.cpu_capacity} GHz")
        print(f" Twin Age: {ap.twin_state.age}, Fidelity: {ap.twin_state.fidelity}")

if __name__ == "__main__":
    test_sagin_env()