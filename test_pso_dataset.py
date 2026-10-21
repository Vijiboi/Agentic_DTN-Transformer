import csv
import tempfile
import unittest

from env.config import SimulationConfig
from results.pso_dataset import INPUT_FEATURE_COLUMNS, LABEL_METADATA_COLUMNS
from results.pso_dataset import PsoDatasetWriter
from scripts.generate_pso_dataset import run_scenario


class PsoDatasetTest(unittest.TestCase):
    def test_pso_dataset_writes_candidate_rows_without_label_leakage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/pso_training_dataset.csv"
            config = SimulationConfig(
                slots=1,
                num_uavs=4,
                task_arrival_probability=1.0,
                pso_swarm_size=4,
                pso_iterations=3,
            )

            with PsoDatasetWriter(output_path) as writer:
                task_count = run_scenario(
                    scenario_id=1,
                    scenario_profile="normal",
                    config=config,
                    writer=writer,
                )

            self.assertGreater(task_count, 0)
            for column in LABEL_METADATA_COLUMNS:
                self.assertNotIn(column, INPUT_FEATURE_COLUMNS)

            with open(output_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertTrue(rows)
            self.assertTrue({"0", "1"} & {row["pso_selected"] for row in rows})
            by_task = {}
            for row in rows:
                key = (row["scenario_id"], row["slot"], row["task_id"])
                by_task.setdefault(key, 0)
                by_task[key] += int(row["pso_selected"])
            self.assertTrue(
                all(selected_count in {0, 1} for selected_count in by_task.values())
            )


if __name__ == "__main__":
    unittest.main()
