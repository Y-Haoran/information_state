from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .config import ProjectConfig
from .state_from_observation import StateFromObservationModel


def make_project_config(
    *,
    project_root: str | None = None,
    raw_root: str | None = None,
    window_hours: int = 24,
    window_stride_hours: int = 2,
    positive_window_gap_hours: int = 2,
    delta_cap_hours: int = 48,
    max_stays: int | None = None,
    chunk_size: int = 200_000,
    max_chunks: int | None = None,
) -> ProjectConfig:
    root = Path(project_root).resolve() if project_root else ProjectConfig().project_root
    raw = Path(raw_root).resolve() if raw_root else None
    return ProjectConfig(
        project_root=root,
        raw_root=raw,
        bin_hours=1,
        window_hours=window_hours,
        window_stride_hours=window_stride_hours,
        positive_window_gap_hours=positive_window_gap_hours,
        delta_cap_hours=delta_cap_hours,
        max_stays=max_stays,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_dataframe(frame: pd.DataFrame, preferred_path: Path) -> Path:
    preferred_path.parent.mkdir(parents=True, exist_ok=True)
    if preferred_path.suffix == ".parquet":
        try:
            frame.to_parquet(preferred_path, index=False)
            return preferred_path
        except (ImportError, ValueError, ModuleNotFoundError):
            fallback = preferred_path.with_suffix(".csv")
            frame.to_csv(fallback, index=False)
            return fallback
    if preferred_path.suffix == ".csv":
        frame.to_csv(preferred_path, index=False)
        return preferred_path
    raise ValueError(f"Unsupported table suffix: {preferred_path.suffix}")


def read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table suffix: {path.suffix}")


def resolve_existing_table(preferred_path: Path) -> Path:
    if preferred_path.exists():
        return preferred_path
    if preferred_path.suffix == ".parquet":
        fallback = preferred_path.with_suffix(".csv")
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Could not find table at {preferred_path} or CSV fallback.")


def resolve_checkpoint_path(config: ProjectConfig, checkpoint_path: str | None) -> Path:
    return Path(checkpoint_path).resolve() if checkpoint_path else config.observation_checkpoint_path


def ensure_observation_data(config: ProjectConfig, build_data: bool = False) -> None:
    from .observation_data import build_state_from_observation_dataset

    expected_paths = [
        config.observation_cohort_path,
        config.observation_hourly_values_path,
        config.observation_hourly_masks_path,
        config.observation_hourly_deltas_path,
        config.observation_window_metadata_path,
        config.observation_metadata_path,
    ]
    if build_data or not all(path.exists() for path in expected_paths):
        build_state_from_observation_dataset(config)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[StateFromObservationModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint["metadata"]
    checkpoint_config = checkpoint.get("config", {})
    model = StateFromObservationModel(
        num_variables=len(metadata["dynamic_feature_names"]),
        num_time_bins=int(metadata["window_hours"]) // int(metadata["bin_hours"]),
        d_model=int(checkpoint_config.get("d_model", 128)),
        num_heads=int(checkpoint_config.get("num_heads", 4)),
        num_layers=int(checkpoint_config.get("num_layers", 3)),
        projection_dim=int(checkpoint_config.get("projection_dim", 128)),
        dropout=float(checkpoint_config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
