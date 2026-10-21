from __future__ import annotations

from collections import defaultdict

from .models import CandidateScore, FinalAssignment, APNode, Task

def greedy_one_hot_assignment(
    scores: list[CandidateScore],
    ap_lookup: dict[str, APNode],
    task_map: dict[str, Task] | None = None,
) -> list[FinalAssignment]:
    by_task: dict[str, list[CandidateScore]] = defaultdict(list)
    for score in scores:
        by_task[score.task_id].append(score)
    
    assignments: list[FinalAssignment] = []
    temp_usage = defaultdict(lambda: {"cpu": 0.0, "bandwidth": 0.0, "power": 0.0})
    lambda_f = 1e6

    task_order = sorted(
        by_task,
        key=lambda task_id: min(
            _rank_value(score, task_map.get(task_id) if task_map else None, ap_lookup, lambda_f)
            for score in by_task[task_id]
        ),
    )

    for task_id in task_order:
        task_scores = by_task[task_id]
        task = task_map.get(task_id) if task_map else None
        sorted_candidates = sorted(
            task_scores,
            key=lambda item: _rank_value(item, task, ap_lookup, lambda_f),
        )
        
        for best in sorted_candidates:
            dest = ap_lookup[best.destination_id]
            usage = temp_usage[best.destination_id]
            
            cpu_ok = dest.current_cpu_load + usage["cpu"] + best.required_cpu <= dest.cpu_capacity
            bandwidth_ok = usage["bandwidth"] + best.required_bandwidth <= dest.communication_budget
            power_ok = usage["power"] + best.required_power <= dest.power_budget
            if cpu_ok and bandwidth_ok and power_ok:
                usage["cpu"] += best.required_cpu
                usage["bandwidth"] += best.required_bandwidth
                usage["power"] += best.required_power
                assignments.append(FinalAssignment(
                    task_id=best.task_id,
                    owner_ap_id=best.owner_ap_id,
                    destination_id=best.destination_id,
                    local_cost=best.local_cost,
                    delay_cost=best.delay_cost,
                    energy_cost=best.energy_cost,
                    mission_cost=best.mission_cost,
                    fidelity_cost=best.fidelity_cost
                ))
                break 
    return assignments


def _rank_value(
    score: CandidateScore,
    task: Task | None,
    ap_lookup: dict[str, APNode],
    lambda_f: float,
) -> float:
    fidelity_penalty = 0.0
    if task is not None and ap_lookup[score.destination_id].twin_state.fidelity < task.F_u_min:
        fidelity_penalty = lambda_f
    return score.local_cost + score.coupling_penalty + fidelity_penalty
