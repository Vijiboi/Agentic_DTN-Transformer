import unittest

from env.config import SimulationConfig
from env.feasibility import FeasibilityManager
from env.generator import SaginEnvironment
from opt.pso_teacher import OfflinePsoTeacher


class PsoTeacherTest(unittest.TestCase):
    def test_pso_teacher_returns_one_reference_assignment_per_task(self):
        config = SimulationConfig(
            num_uavs=6,
            task_arrival_probability=1.0,
            pso_swarm_size=6,
            pso_iterations=4,
        )
        environment = SaginEnvironment(config)
        tasks = environment.create_tasks_for_slot(1)
        ap_lookup = environment.ap_by_id()

        result = OfflinePsoTeacher(config).optimize(
            tasks,
            ap_lookup,
            seed=1,
        )

        self.assertEqual(
            set(result.assignment),
            {task.task_id for task in tasks},
        )
        self.assertEqual(
            result.feasible,
            FeasibilityManager(config).assignment_is_feasible(
                result.assignment,
                tasks,
                ap_lookup,
            ),
        )
        for task in tasks:
            self.assertIn(
                result.assignment[task.task_id],
                set(task.A_u_t) | {"LOCAL"},
            )


if __name__ == "__main__":
    unittest.main()
