from __future__ import annotations
from collections import defaultdict
import numpy as np
import os

from env.config import SimulationConfig
from env.generator import SaginEnvironment
from env.models import CandidateScore, LocalSummary, APNode, Task, SlotResult
from twin.twin_logic import TwinManager
from opt.qubo_generator import LocalQuboBuilder
from opt.solver import ClassicalQuboSolver
from opt.hybrid import DWaveHybridSolver
from control.consensus import QuantizedConsensusCoordinator
from control.regional import RegionalController
from results.io import write_run_outputs 
from results.metrics import build_slot_result 

class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.environment = SaginEnvironment(config)
        self.twin_manager = TwinManager(config)
        self.qubo_builder = LocalQuboBuilder(config)
        self.local_solver = self._build_solver(config)
        self.consensus_coordinator = QuantizedConsensusCoordinator(config)
        self.regional_controller = RegionalController(config=config)
        self.use_solver_guidance = config.use_solver_guidance

    def run(self) -> tuple[list, dict[str, object], str]:
        slot_results = []
        ap_lookup = self.environment.ap_by_id()
        solver_counts = defaultdict(int)
        total_solver_time = 0.0
        
        for slot in range(1, self.config.slots + 1):
            self.environment.step_mobility()
            tasks = self.environment.create_tasks_for_slot(slot)
            grouped_tasks = self.environment.group_tasks_by_owner(tasks)
            local_summaries = []

            for ap in self.environment.aps:
                queue = grouped_tasks.get(ap.ap_id, [])
                observation = self.environment.build_observation(ap, slot, queue, len(tasks))
                required_fidelity = max((task.F_u_min for task in queue), default=0.5)
                sync_triggered, coord_triggered = self.twin_manager.update(ap, observation, required_fidelity)
                
                solver_time = 0.0
                candidate_scores = []
                selected_pairs = []
                if queue:
                    preferred_local_tasks: set[str] | None = None
                    qubo_dimension = 0
                    if self.local_solver is not None:
                        problem = self.qubo_builder.build(ap, queue, ap_lookup, slot)
                        solve_result = self.local_solver.solve(problem)
                        solver_counts[solve_result.solver_name] += 1
                        total_solver_time += solve_result.solver_time
                        solver_time = solve_result.solver_time
                        qubo_dimension = len(problem.variables)
                        selected_pairs = [
                            variable for variable, value in solve_result.sample.items() if value == 1
                        ]
                        if self.use_solver_guidance:
                            preferred_local_tasks = {task_id for task_id, _ in selected_pairs}

                    candidate_scores = self.qubo_builder.score_candidates(
                        ap,
                        queue,
                        ap_lookup,
                        preferred_local_tasks=preferred_local_tasks,
                    )
                else:
                    qubo_dimension = 0
                
                local_summaries.append(LocalSummary(
                    ap_id=ap.ap_id, slot=slot, queue_size=len(queue),
                    local_load=float(ap.current_cpu_load / max(ap.cpu_capacity, 1.0)),
                    sync_triggered=bool(sync_triggered), 
                    coordination_triggered=bool(coord_triggered),
                    trust=float(ap.trust), 
                    twin_age=int(ap.twin_state.age),
                    uncertainty=float(ap.twin_state.uncertainty), 
                    mismatch=float(ap.twin_state.mismatch),
                    fidelity=float(ap.twin_state.fidelity), 
                    qubo_dimension=qubo_dimension,
                    solver_time=float(solver_time), 
                    selected_pairs=selected_pairs, 
                    candidate_scores=candidate_scores
                ))

            self.consensus_coordinator.update(
                ap_lookup,
                self.environment.communication_graph,
                local_summaries,
                slot,
            )
            self._refresh_coordination_penalties(local_summaries, ap_lookup)
            assignments = self.regional_controller.project(local_summaries, ap_lookup, tasks)
            self._refresh_ap_loads(assignments, local_summaries, ap_lookup)
            slot_results.append(build_slot_result(slot, assignments, local_summaries, True))

        summary = self._build_summary(slot_results, solver_counts, total_solver_time)
        # Convert output_path to string and write outputs [cite: 534]
        output_path = str(write_run_outputs(self.config.output_dir, slot_results, summary))
        
        return slot_results, summary, output_path

    @staticmethod
    def _build_solver(config: SimulationConfig):
        backend = config.local_solver_backend.lower()
        if backend in {"dwave", "dwave_hybrid", "hybrid"}:
            return DWaveHybridSolver()
        if backend in {"none", "off", "disabled"}:
            return None
        classical_backend = "auto" if backend == "classical" else backend
        return ClassicalQuboSolver(config, backend=classical_backend)

    @staticmethod
    def _refresh_coordination_penalties(local_summaries, ap_lookup):
        for summary in local_summaries:
            for score in summary.candidate_scores:
                score.coupling_penalty = ap_lookup[score.destination_id].coordination_state.coupling_penalty

    def _refresh_ap_loads(self, assignments, summaries, ap_lookup):
        cpu_load_counter = defaultdict(float)
        task_counter = defaultdict(int)
        score_map = {(s.task_id, s.destination_id): s for summ in summaries for s in summ.candidate_scores}

        for assign in assignments:
            task_counter[assign.destination_id] += 1
            score = score_map.get((assign.task_id, assign.destination_id))
            if score:
                cpu_load_counter[assign.destination_id] += float(score.required_cpu)

        for ap_id, ap in ap_lookup.items():
            ap.current_task_load = int(task_counter.get(ap_id, 0))
            ap.current_cpu_load = float(cpu_load_counter.get(ap_id, 0.0))

    def _build_summary(self, results, solver_counts, total_solver_time: float) -> dict[str, object]:
        """Ensures all metrics are native Python types for JSON serialization [cite: 600-610]."""
        solved_problems = sum(solver_counts.values())
        return {
            "simulation_horizon": int(len(results)),
            "sync_triggers": int(sum(r.sync_trigger_count for r in results)),
            "coord_triggers": int(sum(r.coordination_trigger_count for r in results)),
            "one_hot_validity": bool(all(r.one_hot_valid for r in results)),
            "avg_fidelity": round(float(np.mean([s.fidelity for r in results for s in r.local_summaries])), 4),
            "solver_statistics": {
                "solved_problems": int(solved_problems),
                "solver_counts": dict(solver_counts),
                "total_solver_time": round(float(total_solver_time), 6),
                "avg_solver_time": round(float(total_solver_time / max(solved_problems, 1)), 6),
            },
            "simulation_parameters": self.config.to_report_dict(),
        }
