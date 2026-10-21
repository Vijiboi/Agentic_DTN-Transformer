import numpy as np
from env.config import SimulationConfig
from env.models import APNode, Observation, TwinState
from twin.twin_logic import TwinManager

def test_twin_behavior():
    # 1. Setup Config and Manager
    config = SimulationConfig()
    manager = TwinManager(config)
    
    # 2. Initialize a dummy AP Node [cite: 116-117, 523]
    ap = APNode(
        ap_id="BS_0",
        tier="BS",
        x=100.0, y=100.0, z=0.03,
        bandwidth=20.0,
        cpu_capacity=10.0,
        communication_budget=20.0,
        power_budget=18.0,
        trust=0.9, # High initial trust [cite: 523]
        sync_threshold=config.sync_mismatch_threshold,
        coord_threshold=config.coord_mismatch_threshold
    )
    
    print(f"Initial State: Age={ap.twin_state.age}, Fidelity={ap.twin_state.fidelity:.2f}, Trust={ap.trust:.2f}")
    print("-" * 50)

    # 3. Simulate 5 time slots of observations [cite: 129-132]
    for t in range(1, 6):
        # Create an observation s_m(t) [cite: 115]
        # We simulate a slight "drift" in the load ratio to test mismatch
        simulated_load = 0.4 + (t * 0.05) 
        obs = Observation(
            ap_id=ap.ap_id,
            slot=t,
            load_ratio=simulated_load, 
            bandwidth_ratio=0.5,
            cpu_ratio=0.4,
            queue_size=5,
            candidate_overlap=0.2
        )
        
        # Run the Twin Update logic [cite: 118, 154]
        sync_triggered, coord_triggered = manager.update(ap, obs)
        
        # Display Results
        print(f"Slot {t}:")
        print(f"  Observed Load: {simulated_load:.2f}")
        print(f"  Twin Age: {ap.twin_state.age}")
        print(f"  Mismatch: {ap.twin_state.mismatch:.4f}")
        print(f"  Uncertainty: {ap.twin_state.uncertainty:.4f}")
        print(f"  Fidelity: {ap.twin_state.fidelity:.4f}")
        print(f"  Sync Triggered: {sync_triggered}")
        print(f"  Trust Updated: {ap.trust:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    try:
        test_twin_behavior()
    except ImportError as e:
        print(f"Environment Error: {e}")
        print("Please run this script from the 'Anaconda Prompt' to ensure Numpy is accessible.")