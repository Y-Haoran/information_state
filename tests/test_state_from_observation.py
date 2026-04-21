from __future__ import annotations

import unittest

import torch

from information_state.state_from_observation import StateFromObservationModel


class StateFromObservationModelTests(unittest.TestCase):
    def test_encode_returns_expected_shape(self) -> None:
        model = StateFromObservationModel(
            num_variables=3,
            num_time_bins=4,
            d_model=16,
            num_heads=4,
            num_layers=1,
            projection_dim=8,
            dropout=0.0,
        )
        batch = torch.randn(2, 4, 3, 3)
        encoded = model.encode(batch)
        projected, positive = model(batch, batch)
        self.assertEqual(tuple(encoded.shape), (2, 16))
        self.assertEqual(tuple(projected.shape), (2, 8))
        self.assertEqual(tuple(positive.shape), (2, 8))

    def test_missing_heavy_batches_do_not_nan(self) -> None:
        model = StateFromObservationModel(
            num_variables=3,
            num_time_bins=4,
            d_model=12,
            num_heads=3,
            num_layers=1,
            projection_dim=6,
            dropout=0.0,
        )
        batch = torch.zeros(2, 4, 3, 3)
        batch[..., 2] = torch.log1p(torch.full((2, 4, 3), 6.0))
        encoded = model.encode(batch)
        self.assertTrue(torch.isfinite(encoded).all().item())


if __name__ == "__main__":
    unittest.main()
