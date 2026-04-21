from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from information_state.observation_data import ObservationWindowDataset, compute_stay_deltas, load_window_metadata
from tests.support import create_synthetic_project


class ObservationDataTests(unittest.TestCase):
    def test_compute_stay_deltas_resets_after_observation(self) -> None:
        masks = np.array(
            [
                [0, 1],
                [0, 0],
                [1, 0],
                [0, 0],
            ],
            dtype=np.uint8,
        )
        deltas = compute_stay_deltas(masks, bin_hours=1, delta_cap_hours=4)
        expected = np.array(
            [
                [4.0, 0.0],
                [4.0, 1.0],
                [0.0, 2.0],
                [1.0, 3.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(deltas, expected)

    def test_window_dataset_returns_triplets_with_binary_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = create_synthetic_project(Path(tmp_dir))
            dataset = ObservationWindowDataset(config, split="train")
            window, meta = dataset[0]
            self.assertEqual(tuple(window.shape), (config.window_bins, 3, 3))
            self.assertEqual(int(meta["start_hour"]), 0)
            mask_values = set(window[..., 1].flatten().tolist())
            self.assertTrue(mask_values.issubset({0.0, 1.0}))

    def test_positive_windows_respect_gap_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = create_synthetic_project(Path(tmp_dir))
            windows = load_window_metadata(config)
            positives = windows[windows["positive_flat_start"] >= 0]
            gaps = positives["positive_start_bin"] - positives["start_bin"]
            self.assertTrue((gaps == config.positive_window_gap_bins).all())


if __name__ == "__main__":
    unittest.main()
