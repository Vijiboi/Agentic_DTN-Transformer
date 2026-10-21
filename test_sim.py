from __future__ import annotations
import argparse
import json
import os
from env.config import SimulationConfig
from sim.runner import SimulationRunner

def parse_args() -> argparse.Namespace:
    """Parses SAGIN simulation arguments [cite: 1634-1640]."""
    parser = argparse.ArgumentParser(description="SAGIN-Quantum Closed-Loop DTN Simulator")
    parser.add_argument("--slots", type=int, default=50, help="Number of simulation slots (Paper uses 50)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    parser.add_argument("--reads", type=int, default=5, help="Annealing reads per AP")
    parser.add_argument("--sweeps", type=int, default=20, help="Annealing sweeps per AP")
    parser.add_argument(
        "--solver-backend",
        type=str,
        default="classical",
        help="Local backend: classical, exact, annealing, dimod, neal, dwave_hybrid, or none",
    )
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output JSON")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # Initialize configuration with paper-aligned parameters [cite: 1641-1649]
    config = SimulationConfig(
        slots=args.slots,
        seed=args.seed,
        anneal_reads=args.reads,
        anneal_sweeps=args.sweeps,
        local_solver_backend=args.solver_backend,
        output_dir=args.output_dir,
    )
    
    # Execute the Hierarchical Closed-Loop runner [cite: 1650-1651]
    runner = SimulationRunner(config)
    slot_results, summary, output_path = runner.run()
    
    # Output the Decision-Oriented Summaries [cite: 1652-1653]
    print("\n--- SAGIN-Quantum Simulation Summary ---")
    print(json.dumps(summary, indent=2))
    
    # Ensure results directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"\nDetailed outputs saved in: {output_path}")

if __name__ == "__main__":
    main()
