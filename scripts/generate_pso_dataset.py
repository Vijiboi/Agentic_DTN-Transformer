"""Generate a streaming, PSO-labelled CSV for the outer-loop Transformer.

Example:
    python scripts/generate_pso_dataset.py

The defaults request 500 diverse scenarios and continue with additional mixed
scenarios only if needed to satisfy the one-million-row floor.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control.consensus import QuantizedConsensusCoordinator
from env.config import SimulationConfig
from env.generator import SaginEnvironment
from env.models import APNode, LocalSummary, Task
from opt.pso_teacher import OfflinePsoTeacher
from opt.qubo_generator import LocalQuboBuilder
from results.pso_dataset import PsoDatasetWriter
from twin.twin_logic import TwinManager


SCENARIO_PROFILES = (
    "normal", "high_traffic", "low_traffic", "good_channel", "poor_channel",
    "few_users", "many_users", "low_energy", "high_energy", "emergency_surge",
    "fidelity_stress", "high_mobility",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PSO labels for outer-loop Transformer training.")
    parser.add_argument("--scenarios", type=int, default=500, help="Initial number of diverse scenarios to run.")
    parser.add_argument("--min-rows", type=int, default=1_000_000, help="Keep adding scenarios until this row count is reached.")
    parser.add_argument("--slots", type=int, default=10, help="Slots per scenario.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--profile", choices=("mixed", *SCENARIO_PROFILES), default="mixed")
    parser.add_argument("--output", type=Path, default=Path("pso_training_dataset_1m.csv"))
    parser.add_argument("--arrival-probability", type=float, default=0.9, help="Base per-UAV task probability (0 to 1).")
    parser.add_argument("--swarm-size", type=int, default=8, help="Particles per PSO slot.")
    parser.add_argument("--iterations", type=int, default=6, help="PSO iterations per slot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenarios < 1 or args.slots < 1 or args.min_rows < 1:
        raise SystemExit("--scenarios, --slots, and --min-rows must be positive.")
    if not 0.0 <= args.arrival_probability <= 1.0:
        raise SystemExit("--arrival-probability must be between 0 and 1.")
    if args.swarm_size < 2 or args.iterations < 1:
        raise SystemExit("--swarm-size must be at least 2 and --iterations at least 1.")

    config = SimulationConfig(
        slots=args.slots, seed=args.seed, task_arrival_probability=args.arrival_probability,
        pso_swarm_size=args.swarm_size, pso_iterations=args.iterations,
    )
    output_path = args.output.resolve()
    scenario_summaries: list[dict[str, object]] = []
    tasks_written = 0
    scenario_id = 1

    with PsoDatasetWriter(output_path) as writer:
        while scenario_id <= args.scenarios or writer.rows_written < args.min_rows:
            profile = _profile_for_scenario(args.profile, scenario_id)
            scenario_config = _scenario_config(config, args.seed + scenario_id - 1, profile)
            task_count, objective_sum = run_scenario(
                scenario_id=scenario_id, scenario_profile=profile,
                config=scenario_config, writer=writer, include_objective=True,
            )
            tasks_written += task_count
            scenario_summaries.append({
                "scenario_id": scenario_id, "profile": profile, "seed": scenario_config.seed,
                "tasks": task_count, "rows_after_scenario": writer.rows_written,
                "mean_pso_objective": objective_sum / max(scenario_config.slots, 1),
            })
            writer.flush()
            print(
                f"scenario {scenario_id:>4} ({profile:>15}): {task_count:>4} tasks, "
                f"{writer.rows_written:>10,} rows",
                flush=True,
            )
            scenario_id += 1

    manifest_path = output_path.with_suffix(".summary.json")
    manifest_path.write_text(json.dumps({
        "dataset_path": str(output_path), "rows": writer.rows_written, "tasks": tasks_written,
        "requested_scenarios": args.scenarios, "actual_scenarios": len(scenario_summaries),
        "minimum_rows": args.min_rows, "slots_per_scenario": args.slots,
        "base_seed": args.seed, "pso_swarm_size": args.swarm_size, "pso_iterations": args.iterations,
        "scenarios": scenario_summaries,
        "label_columns": ["pso_selected", "pso_selected_destination_id", "pso_objective_value", "pso_runtime_seconds", "pso_feasible"],
    }, indent=2), encoding="utf-8")
    print(f"\nDataset saved to: {output_path}")
    print(f"Summary saved to: {manifest_path}")
    print(f"Rows: {writer.rows_written:,}; task groups: {tasks_written:,}; scenarios: {len(scenario_summaries)}")


def run_scenario(
    *, scenario_id: int, scenario_profile: str, config: SimulationConfig,
    writer: PsoDatasetWriter, include_objective: bool = False,
) -> int | tuple[int, float]:
    environment = SaginEnvironment(config)
    twin_manager = TwinManager(config)
    consensus = QuantizedConsensusCoordinator(config)
    qubo_builder = LocalQuboBuilder(config)
    pso_teacher = OfflinePsoTeacher(config)
    ap_lookup = environment.ap_by_id()
    tasks_written = 0
    objective_sum = 0.0

    for slot in range(1, config.slots + 1):
        environment.step_mobility()
        tasks = environment.create_tasks_for_slot(slot)
        grouped_tasks = environment.group_tasks_by_owner(tasks)
        local_summaries = _update_twin_state(environment, twin_manager, slot, tasks, grouped_tasks)
        consensus.update(ap_lookup, environment.communication_graph, local_summaries, slot)
        sync_ap_ids = {summary.ap_id for summary in local_summaries if summary.sync_triggered}
        coord_ap_ids = {summary.ap_id for summary in local_summaries if summary.coordination_triggered}
        candidate_scores = qubo_builder.score_candidates_for_tasks(tasks, ap_lookup)
        for score in candidate_scores:
            score.coupling_penalty = ap_lookup[score.destination_id].coordination_state.coupling_penalty
        pso_result = pso_teacher.optimize(
            tasks, ap_lookup, sync_ap_ids=sync_ap_ids, coord_ap_ids=coord_ap_ids,
            seed=scenario_id * 100_000 + slot,
        )
        writer.write_task_candidate_rows(
            scenario_id=scenario_id, scenario_profile=scenario_profile, seed=config.seed, slot=slot,
            tasks=tasks, ap_lookup=ap_lookup, candidate_scores=candidate_scores,
            sync_ap_ids=sync_ap_ids, coord_ap_ids=coord_ap_ids, pso_result=pso_result, config=config,
        )
        _refresh_ap_loads_from_assignment(pso_result.assignment, tasks, ap_lookup)
        tasks_written += len(tasks)
        objective_sum += pso_result.objective_value
    return (tasks_written, objective_sum) if include_objective else tasks_written


def _update_twin_state(environment: SaginEnvironment, twin_manager: TwinManager, slot: int, tasks: list[Task], grouped_tasks: dict[str, list[Task]]) -> list[LocalSummary]:
    summaries = []
    for ap in environment.aps:
        queue = grouped_tasks.get(ap.ap_id, [])
        observation = environment.build_observation(ap, slot, queue, len(tasks))
        required_fidelity = max((task.F_u_min for task in queue), default=0.5)
        sync_triggered, coord_triggered = twin_manager.update(ap, observation, required_fidelity)
        summaries.append(LocalSummary(
            ap_id=ap.ap_id, slot=slot, queue_size=len(queue),
            local_load=float(ap.current_cpu_load / max(ap.cpu_capacity, 1.0)),
            sync_triggered=bool(sync_triggered), coordination_triggered=bool(coord_triggered),
            trust=float(ap.trust), twin_age=int(ap.twin_state.age), uncertainty=float(ap.twin_state.uncertainty),
            mismatch=float(ap.twin_state.mismatch), fidelity=float(ap.twin_state.fidelity),
            qubo_dimension=0, solver_time=0.0, selected_pairs=[], candidate_scores=[],
        ))
    return summaries


def _refresh_ap_loads_from_assignment(assignment: dict[str, str], tasks: list[Task], ap_lookup: dict[str, APNode]) -> None:
    task_lookup = {task.task_id: task for task in tasks}
    task_counter: dict[str, int] = defaultdict(int)
    cpu_load_counter: dict[str, float] = defaultdict(float)
    for task_id, destination_id in assignment.items():
        if destination_id in ap_lookup:
            task_counter[destination_id] += 1
            cpu_load_counter[destination_id] += task_lookup[task_id].cpu_demand
    for ap_id, ap in ap_lookup.items():
        ap.current_task_load = task_counter[ap_id]
        ap.current_cpu_load = cpu_load_counter[ap_id]


def _profile_for_scenario(requested_profile: str, scenario_id: int) -> str:
    return SCENARIO_PROFILES[(scenario_id - 1) % len(SCENARIO_PROFILES)] if requested_profile == "mixed" else requested_profile


def _scenario_config(base: SimulationConfig, seed: int, profile: str) -> SimulationConfig:
    config = replace(base, seed=seed)
    # ``replace`` preserves mutable dictionaries, so every scenario receives its own copy.
    config.bandwidth_by_tier = copy.deepcopy(base.bandwidth_by_tier)
    config.communication_budget_by_tier = copy.deepcopy(base.communication_budget_by_tier)
    config.power_budget_by_tier = copy.deepcopy(base.power_budget_by_tier)
    if profile == "high_traffic":
        config.num_uavs, config.task_arrival_probability = max(config.num_uavs, 30), 0.95
    elif profile == "low_traffic":
        config.num_uavs, config.task_arrival_probability = max(3, config.num_uavs // 2), 0.45
    elif profile == "good_channel":
        config.bandwidth_by_tier = {tier: value * 1.25 for tier, value in config.bandwidth_by_tier.items()}
        config.communication_budget_by_tier = {tier: value * 1.25 for tier, value in config.communication_budget_by_tier.items()}
        config.noise_spectral_density *= 0.5
    elif profile == "poor_channel":
        config.bandwidth_by_tier = {tier: value * 0.65 for tier, value in config.bandwidth_by_tier.items()}
        config.communication_budget_by_tier = {tier: value * 0.65 for tier, value in config.communication_budget_by_tier.items()}
        config.noise_spectral_density *= 2.0
    elif profile == "few_users":
        config.num_uavs, config.task_arrival_probability = max(3, config.num_uavs // 3), 0.70
    elif profile == "many_users":
        config.num_uavs, config.task_arrival_probability = max(config.num_uavs, 40), 0.90
    elif profile == "low_energy":
        config.power_budget_by_tier = {tier: value * 0.60 for tier, value in config.power_budget_by_tier.items()}
        config.uav_transmit_power *= 0.75
    elif profile == "high_energy":
        config.power_budget_by_tier = {tier: value * 1.30 for tier, value in config.power_budget_by_tier.items()}
        config.uav_transmit_power *= 1.20
    elif profile == "emergency_surge":
        config.num_uavs, config.task_arrival_probability = max(config.num_uavs, 35), 0.98
        config.task_aoi_min, config.task_aoi_max = 3, 8
    elif profile == "fidelity_stress":
        config.sync_age_threshold, config.sync_uncertainty_threshold = 5, 0.25
        config.twin_gaussian_std *= 2.0
    elif profile == "high_mobility":
        config.uav_velocity_xy_min, config.uav_velocity_xy_max = -8.0, 8.0
        config.uav_velocity_z_min, config.uav_velocity_z_max = -2.0, 2.0
    return config


if __name__ == "__main__":
    main()
