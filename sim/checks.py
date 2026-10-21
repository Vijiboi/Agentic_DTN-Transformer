from __future__ import annotations

from env.models import LocalSummary, SlotResult


def validate_pre_quantum_checks(slot_results: list[SlotResult]) -> dict[str, bool]:
    all_summaries: list[LocalSummary] = [summary for slot in slot_results for summary in slot.local_summaries]
    one_hot = all(slot.one_hot_valid for slot in slot_results)
    twin_updates = all(summary.fidelity > 0.0 and summary.uncertainty >= 0.0 for summary in all_summaries)
    trigger_activity = any(summary.sync_triggered or summary.coordination_triggered for summary in all_summaries)
    local_load_tracking = all(summary.local_load >= 0.0 for summary in all_summaries)
    qubo_dimensions = all(
        (summary.queue_size == 0 and summary.qubo_dimension == 0)
        or (summary.queue_size > 0 and summary.qubo_dimension >= summary.queue_size)
        for summary in all_summaries
    )
    regional_projection = one_hot and all(
        len(slot.assignments) == len({item.task_id for item in slot.assignments}) and slot.resource_feasible
        for slot in slot_results
    )
    return {
        "no_multi_assignment": one_hot,
        "twin_variables_update": twin_updates,
        "trigger_rules_active": trigger_activity,
        "local_load_tracking": local_load_tracking,
        "qubo_dimensions_match_candidate_sets": qubo_dimensions,
        "regional_projection_feasible": regional_projection,
    }
