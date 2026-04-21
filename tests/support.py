from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from information_state.config import ProjectConfig
from information_state.observation_data import build_window_metadata, compute_stay_deltas


def create_synthetic_project(project_root: Path) -> ProjectConfig:
    """Create a tiny synthetic artifact set that exercises the full pipeline."""
    config = ProjectConfig(
        project_root=project_root,
        raw_root=project_root / "raw_unused",
        bin_hours=1,
        window_hours=4,
        window_stride_hours=2,
        positive_window_gap_hours=2,
        delta_cap_hours=6,
        random_seed=11,
    )
    config.state_from_observation_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        ("train", 1001, 2001, 3001, 0.0),
        ("train", 1002, 2002, 3002, 1.0),
        ("val", 1003, 2003, 3003, 0.0),
        ("test", 1004, 2004, 3004, 1.0),
    ]
    num_bins_per_stay = 8
    num_features = 3
    values = np.zeros((len(entries) * num_bins_per_stay, num_features), dtype=np.float32)
    masks = np.zeros_like(values, dtype=np.uint8)
    cohort_rows = []

    for stay_index, (split, subject_id, hadm_id, stay_id, state_offset) in enumerate(entries):
        offset = stay_index * num_bins_per_stay
        stay_start = pd.Timestamp("2100-01-01") + pd.Timedelta(hours=offset)
        full_signal = np.zeros((num_bins_per_stay, num_features), dtype=np.float32)
        stay_masks = np.zeros((num_bins_per_stay, num_features), dtype=np.uint8)

        for time_index in range(num_bins_per_stay):
            full_signal[time_index] = np.array(
                [
                    state_offset + 0.3 * time_index,
                    (1.0 - state_offset) + (0.15 if time_index % 2 == 0 else -0.15),
                    state_offset * 0.8 + 0.05 * (time_index % 3),
                ],
                dtype=np.float32,
            )
            stay_masks[time_index] = np.array(
                [
                    1,
                    1 if (time_index + stay_index) % 2 == 0 else 0,
                    1 if time_index in (0, 3, 7) else 0,
                ],
                dtype=np.uint8,
            )

        stay_values = np.zeros_like(full_signal)
        for feature_index in range(num_features):
            last_value = 0.0
            seen = False
            for time_index in range(num_bins_per_stay):
                if stay_masks[time_index, feature_index]:
                    last_value = full_signal[time_index, feature_index]
                    seen = True
                stay_values[time_index, feature_index] = last_value if seen else 0.0

        values[offset:offset + num_bins_per_stay] = stay_values
        masks[offset:offset + num_bins_per_stay] = stay_masks
        cohort_rows.append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "first_careunit": "MICU",
                "intime": stay_start.isoformat(),
                "outtime": (stay_start + pd.Timedelta(hours=num_bins_per_stay)).isoformat(),
                "admittime": stay_start.isoformat(),
                "dischtime": (stay_start + pd.Timedelta(hours=num_bins_per_stay + 8)).isoformat(),
                "stay_hours": float(num_bins_per_stay),
                "num_bins": num_bins_per_stay,
                "split": split,
                "anchor_age": 60,
                "gender": "F",
                "admission_type": "EMERGENCY",
                "insurance": "OTHER",
                "race": "UNKNOWN",
                "hospital_expire_flag": int(state_offset > 0.5),
                "in_hospital_mortality": int(state_offset > 0.5),
                "icu_los_days": float(num_bins_per_stay / 24.0),
                "stay_index": stay_index,
                "offset": offset,
                "num_windows": 3,
                "num_positive_windows": 2,
            }
        )

    deltas = np.zeros_like(values, dtype=np.float32)
    for stay_index in range(len(entries)):
        offset = stay_index * num_bins_per_stay
        deltas[offset:offset + num_bins_per_stay] = compute_stay_deltas(
            masks[offset:offset + num_bins_per_stay],
            bin_hours=config.bin_hours,
            delta_cap_hours=config.delta_cap_hours,
        )

    np.save(config.observation_hourly_values_path, values)
    np.save(config.observation_hourly_masks_path, masks)
    np.save(config.observation_hourly_deltas_path, deltas)
    pd.DataFrame(cohort_rows).to_csv(config.observation_cohort_path, index=False)
    windows = build_window_metadata(config)
    metadata = {
        "dynamic_feature_names": ["feature_a", "feature_b", "feature_c"],
        "window_hours": config.window_hours,
        "window_stride_hours": config.window_stride_hours,
        "positive_window_gap_hours": config.positive_window_gap_hours,
        "bin_hours": config.bin_hours,
        "delta_cap_hours": config.delta_cap_hours,
        "total_stays": len(entries),
        "total_bins": int(values.shape[0]),
        "total_windows": int(len(windows)),
        "total_positive_windows": int((windows["positive_flat_start"] >= 0).sum()),
    }
    stats = [
        {"name": "feature_a", "lower": -1.0, "upper": 3.0, "mean": 0.0, "std": 1.0},
        {"name": "feature_b", "lower": -1.0, "upper": 3.0, "mean": 0.0, "std": 1.0},
        {"name": "feature_c", "lower": -1.0, "upper": 3.0, "mean": 0.0, "std": 1.0},
    ]
    with open(config.observation_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with open(config.observation_stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return config
