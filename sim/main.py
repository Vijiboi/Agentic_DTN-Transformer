from __future__ import annotations
import argparse
import json
import numpy as np # Added for serialization help
from env.config import SimulationConfig
from sim.runner import SimulationRunner

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAGIN-Quantum Hierarchical Simulator")
    parser.add_argument("--slots", type=int, default=10, help="Number of slots")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--reads", type=int, default=5, help="Annealing reads")
    parser.add_argument("--sweeps", type=int, default=20, help="Annealing sweeps")
    parser.add_argument(
        "--solver-backend",
        type=str,
        default="classical",
        help="Local backend: classical, exact, annealing, dimod, neal, dwave_hybrid, or none",
    )
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        slots=args.slots,
        seed=args.seed,
        anneal_reads=args.reads,
        anneal_sweeps=args.sweeps,
        local_solver_backend=args.solver_backend,
        output_dir=args.output_dir,
    )
    
    runner = SimulationRunner(config)
    slot_results, summary, output_path = runner.run()
    
    # Use a custom encoder to handle 'int64' and 'float64' types [cite: 606-616]
    def numpy_encoder(obj):
        if isinstance(obj, (np.int64, np.int32, np.integer)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.floating)):
            return float(obj)
        return str(obj)

    print("\n--- SAGIN-Quantum Summary ---")
    print(json.dumps(summary, indent=2, default=numpy_encoder))
    print(f"Outputs saved in: {output_path}")

if __name__ == "__main__":
    main()
