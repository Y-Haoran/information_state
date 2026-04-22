"""Shared helpers for configuration, reproducibility, and artifact I/O."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ProjectConfig
from .state_from_observation import StateFromObservationModel


def make_project_config(
    *,
    project_root: str | None = None,
    raw_root: str | None = None,
    cohort_name: str = "all_adult_icu",
    build_workers: int = 1,
    window_hours: int = 24,
    window_stride_hours: int = 2,
    positive_window_gap_hours: int = 2,
    delta_cap_hours: int = 48,
    max_stays: int | None = None,
    chunk_size: int = 200_000,
    max_chunks: int | None = None,
    random_seed: int = 7,
) -> ProjectConfig:
    """Build a concrete project config from CLI-style arguments."""
    root = Path(project_root).resolve() if project_root else ProjectConfig().project_root
    raw = Path(raw_root).resolve() if raw_root else None
    return ProjectConfig(
        project_root=root,
        raw_root=raw,
        cohort_name=cohort_name,
        build_workers=build_workers,
        bin_hours=1,
        window_hours=window_hours,
        window_stride_hours=window_stride_hours,
        positive_window_gap_hours=positive_window_gap_hours,
        delta_cap_hours=delta_cap_hours,
        max_stays=max_stays,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
        random_seed=random_seed,
    )


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON with stable indentation, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON dictionary from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_dataframe(frame: pd.DataFrame, preferred_path: Path) -> Path:
    """Persist a table, falling back from parquet to CSV when needed."""
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
    """Read a parquet or CSV table."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table suffix: {path.suffix}")


def resolve_existing_table(preferred_path: Path) -> Path:
    """Resolve a preferred table path, allowing parquet-to-CSV fallback."""
    if preferred_path.exists():
        return preferred_path
    if preferred_path.suffix == ".parquet":
        fallback = preferred_path.with_suffix(".csv")
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Could not find table at {preferred_path} or CSV fallback.")


def resolve_checkpoint_path(config: ProjectConfig, checkpoint_path: str | None) -> Path:
    """Resolve an explicit checkpoint path or the default training artifact."""
    return Path(checkpoint_path).resolve() if checkpoint_path else config.observation_checkpoint_path


def ensure_observation_data(config: ProjectConfig, build_data: bool = False) -> None:
    """Build the observation tensor artifacts if they are missing or requested."""
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


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[StateFromObservationModel, dict[str, Any]]:
    """Restore a trained model and its saved checkpoint payload."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint["metadata"]
    checkpoint_config = checkpoint.get("model_config", checkpoint.get("config", {}))
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


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(project_root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def collect_git_provenance(project_root: Path) -> dict[str, Any]:
    """Capture lightweight git provenance for a run manifest."""
    dirty_status = _git_output(project_root, "status", "--porcelain")
    return {
        "commit": _git_output(project_root, "rev-parse", "HEAD"),
        "short_commit": _git_output(project_root, "rev-parse", "--short", "HEAD"),
        "branch": _git_output(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "is_dirty": bool(dirty_status) if dirty_status is not None else None,
    }


def collect_dataset_fingerprint(config: ProjectConfig) -> dict[str, Any]:
    """Fingerprint the dataset artifacts that define an experiment run."""
    payload: dict[str, Any] = {
        "cohort_path": str(config.observation_cohort_path),
        "window_metadata_path": str(config.observation_window_metadata_path),
        "metadata_path": str(config.observation_metadata_path),
        "feature_stats_path": str(config.observation_stats_path),
        "cohort_sha256": _file_sha256(config.observation_cohort_path),
        "window_metadata_sha256": _file_sha256(config.observation_window_metadata_path),
        "metadata_sha256": _file_sha256(config.observation_metadata_path),
        "feature_stats_sha256": _file_sha256(config.observation_stats_path),
    }
    if config.observation_metadata_path.exists():
        payload["metadata"] = load_json(config.observation_metadata_path)
    return payload


def collect_runtime_context() -> dict[str, Any]:
    """Capture runtime details useful for reproducing a run."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cwd": os.getcwd(),
    }


def serialize_project_config(config: ProjectConfig) -> dict[str, Any]:
    """Convert the project config to a JSON-safe payload."""
    return {
        "project_root": str(config.project_root),
        "raw_root": str(config.raw_root),
        "cohort_name": config.cohort_name,
        "build_workers": config.build_workers,
        "bin_hours": config.bin_hours,
        "window_hours": config.window_hours,
        "window_stride_hours": config.window_stride_hours,
        "positive_window_gap_hours": config.positive_window_gap_hours,
        "delta_cap_hours": config.delta_cap_hours,
        "winsor_lower_quantile": config.winsor_lower_quantile,
        "winsor_upper_quantile": config.winsor_upper_quantile,
        "min_age": config.min_age,
        "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
        "random_seed": config.random_seed,
        "chunk_size": config.chunk_size,
        "max_stays": config.max_stays,
        "max_chunks": config.max_chunks,
    }


def write_run_manifest(
    *,
    config: ProjectConfig,
    stage: str,
    cli_args: dict[str, Any],
    output_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write both a stage-local run config and a timestamped manifest record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config.manifests_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, Any] = {
        "stage": stage,
        "runtime": collect_runtime_context(),
        "git": collect_git_provenance(config.project_root),
        "project_config": serialize_project_config(config),
        "cli_args": cli_args,
        "dataset_fingerprint": collect_dataset_fingerprint(config),
    }
    if extra:
        payload.update(extra)

    timestamped_path = config.manifests_dir / f"{stage}_{timestamp}.json"
    latest_path = config.manifests_dir / f"{stage}_latest.json"
    run_config_path = output_dir / "run_config.json"
    write_json(payload, timestamped_path)
    write_json(payload, latest_path)
    write_json(payload, run_config_path)
    return {
        "timestamped_manifest_path": str(timestamped_path),
        "latest_manifest_path": str(latest_path),
        "run_config_path": str(run_config_path),
    }
