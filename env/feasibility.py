"""Assignment-domain checks for offline reference assignments."""
from __future__ import annotations

from .config import SimulationConfig
from .models import APNode, Task


class FeasibilityManager:
    """Checks legal destinations under the simulator's equal-share model.

    The regional controller deliberately keeps AP candidates admissible and
    expresses resource pressure through delay/energy and the objective.  This
    helper therefore checks the same assignment domain rather than imposing a
    second hard capacity rule that would disagree with generated labels.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def assignment_is_feasible(
        self,
        assignment: dict[str, str],
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
    ) -> bool:
        task_by_id = {task.task_id: task for task in tasks}
        if set(assignment) != set(task_by_id):
            return False

        for task_id, destination_id in assignment.items():
            task = task_by_id[task_id]
            if destination_id == "LOCAL":
                continue
            if destination_id not in ap_lookup or destination_id not in task.A_u_t:
                return False
        return True
