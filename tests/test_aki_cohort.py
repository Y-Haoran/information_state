from __future__ import annotations

import unittest

import pandas as pd

from information_state.aki_cohort import detect_aki_from_creatinine, detect_aki_from_urine_output
from information_state.config import ProjectConfig


class AkiCohortTests(unittest.TestCase):
    def test_creatinine_criteria_detect_stage_two_aki(self) -> None:
        stay_start = pd.Timestamp("2100-01-01 00:00:00")
        stay_end = stay_start + pd.Timedelta(hours=72)
        events = pd.DataFrame(
            {
                "charttime": [
                    stay_start - pd.Timedelta(hours=4),
                    stay_start + pd.Timedelta(hours=12),
                    stay_start + pd.Timedelta(hours=36),
                ],
                "creatinine": [1.0, 1.35, 2.1],
            }
        )

        result = detect_aki_from_creatinine(events, stay_start=stay_start, stay_end=stay_end)

        self.assertEqual(result["aki"], 1)
        self.assertEqual(result["aki_creatinine"], 1)
        self.assertEqual(result["aki_stage_creatinine"], 2)
        self.assertAlmostEqual(float(result["aki_onset_hour_creatinine"]), 12.0)
        self.assertAlmostEqual(float(result["aki_baseline_creatinine"]), 1.0)

    def test_urine_output_criteria_detect_stage_three_aki(self) -> None:
        stay_start = pd.Timestamp("2100-01-01 00:00:00")
        stay_end = stay_start + pd.Timedelta(hours=24)
        events = pd.DataFrame(
            {
                "charttime": [stay_start + pd.Timedelta(hours=hour) for hour in range(12)],
                "urine_output_ml": [0.0] * 12,
            }
        )

        result = detect_aki_from_urine_output(
            events,
            weight_kg=70.0,
            stay_start=stay_start,
            stay_end=stay_end,
            bin_hours=1,
        )

        self.assertEqual(result["aki"], 1)
        self.assertEqual(result["aki_urine_output"], 1)
        self.assertEqual(result["aki_stage_urine_output"], 3)
        self.assertAlmostEqual(float(result["aki_onset_hour_urine_output"]), 6.0)
        self.assertAlmostEqual(float(result["aki_weight_kg"]), 70.0)

    def test_aki_config_uses_cohort_specific_artifact_dir(self) -> None:
        config = ProjectConfig(project_root="/tmp/information_state_test", cohort_name="aki")
        self.assertEqual(config.cohort_name, "aki_kdigo")
        self.assertTrue(str(config.state_from_observation_dir).endswith("artifacts/state_from_observation/aki_kdigo"))


if __name__ == "__main__":
    unittest.main()
