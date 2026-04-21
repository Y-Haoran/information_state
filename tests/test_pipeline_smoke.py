from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from information_state import cluster_states, evaluate_observation_robustness, evaluate_phenotypes, extract_embeddings, train_ssl
from information_state.utils import resolve_existing_table
from tests.support import create_synthetic_project


class PipelineSmokeTests(unittest.TestCase):
    def test_synthetic_pipeline_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = create_synthetic_project(root)

            train_ssl.main(
                [
                    "--project-root",
                    str(root),
                    "--window-hours",
                    "4",
                    "--window-stride-hours",
                    "2",
                    "--positive-window-gap-hours",
                    "2",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--d-model",
                    "8",
                    "--num-heads",
                    "2",
                    "--num-layers",
                    "1",
                    "--projection-dim",
                    "8",
                    "--seed",
                    "5",
                    "--device",
                    "cpu",
                ]
            )
            extract_embeddings.main(
                [
                    "--project-root",
                    str(root),
                    "--split",
                    "train",
                    "val",
                    "--window-hours",
                    "4",
                    "--window-stride-hours",
                    "2",
                    "--positive-window-gap-hours",
                    "2",
                    "--batch-size",
                    "4",
                    "--seed",
                    "5",
                    "--device",
                    "cpu",
                ]
            )
            cluster_states.main(["--project-root", str(root), "--split", "train", "--k", "2", "--seed", "5"])
            evaluate_phenotypes.main(["--project-root", str(root)])
            evaluate_observation_robustness.main(
                [
                    "--project-root",
                    str(root),
                    "--split",
                    "val",
                    "--window-hours",
                    "4",
                    "--window-stride-hours",
                    "2",
                    "--positive-window-gap-hours",
                    "2",
                    "--batch-size",
                    "4",
                    "--seed",
                    "5",
                    "--device",
                    "cpu",
                ]
            )

            self.assertTrue(config.observation_checkpoint_path.exists())
            self.assertTrue((config.state_from_observation_dir / "run_config.json").exists())
            self.assertTrue((config.embeddings_dir / "run_config.json").exists())
            self.assertTrue((config.clusters_dir / "run_config.json").exists())
            self.assertTrue((config.evaluation_dir / "run_config.json").exists())
            self.assertTrue((config.robustness_dir / "run_config.json").exists())
            self.assertTrue((config.robustness_dir / "embedding_drift_histogram.png").exists())
            self.assertTrue(resolve_existing_table(config.clusters_dir / "cluster_assignments.parquet").exists())
            self.assertTrue(resolve_existing_table(config.evaluation_dir / "cluster_outcomes.parquet").exists())


if __name__ == "__main__":
    unittest.main()
