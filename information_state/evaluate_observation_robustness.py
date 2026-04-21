from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .observation_data import ObservationWindowDataset
from .utils import (
    ensure_observation_data,
    load_model_from_checkpoint,
    make_project_config,
    read_dataframe,
    resolve_checkpoint_path,
    resolve_existing_table,
    write_dataframe,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure embedding stability under observation perturbations.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--raw-root", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--cluster-model-path", type=str, default=None)
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--window-stride-hours", type=int, default=2)
    parser.add_argument("--positive-window-gap-hours", type=int, default=2)
    parser.add_argument("--delta-cap-hours", type=int, default=48)
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--drop-prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _metadata_batch_to_frame(batch_meta: dict[str, object]) -> pd.DataFrame:
    columns = {}
    for key, value in batch_meta.items():
        if torch.is_tensor(value):
            columns[key] = value.cpu().numpy()
        else:
            columns[key] = list(value)
    return pd.DataFrame(columns)


def _perturb_windows(windows: torch.Tensor, drop_prob: float, delta_cap_hours: float, bin_hours: float, generator: torch.Generator) -> torch.Tensor:
    values = windows[..., 0]
    masks = windows[..., 1] > 0.5
    keep_mask = masks & (torch.rand(masks.shape, generator=generator, device=windows.device) > drop_prob)

    perturbed_values = torch.zeros_like(values)
    perturbed_masks = keep_mask.float()
    perturbed_deltas = torch.zeros_like(values)
    time_steps = windows.size(1)
    num_variables = windows.size(2)

    for batch_index in range(windows.size(0)):
        for feature_index in range(num_variables):
            last_value = values.new_tensor(0.0)
            last_seen = -1
            for time_index in range(time_steps):
                if keep_mask[batch_index, time_index, feature_index]:
                    last_value = values[batch_index, time_index, feature_index]
                    last_seen = time_index
                    perturbed_values[batch_index, time_index, feature_index] = last_value
                    perturbed_deltas[batch_index, time_index, feature_index] = 0.0
                else:
                    perturbed_values[batch_index, time_index, feature_index] = last_value if last_seen >= 0 else 0.0
                    if last_seen < 0:
                        perturbed_deltas[batch_index, time_index, feature_index] = delta_cap_hours
                    else:
                        perturbed_deltas[batch_index, time_index, feature_index] = min(
                            (time_index - last_seen) * bin_hours,
                            delta_cap_hours,
                        )

    return torch.stack([perturbed_values, perturbed_masks, torch.log1p(perturbed_deltas)], dim=-1)


def _assign_to_clusters(embeddings: np.ndarray, model_payload: dict[str, np.ndarray]) -> np.ndarray:
    scaled = (embeddings - model_payload["scaler_mean"]) / model_payload["scaler_scale"]
    centers = model_payload["centers"]
    distances = ((scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return distances.argmin(axis=1)


def main() -> None:
    args = parse_args()
    config = make_project_config(
        project_root=args.project_root,
        raw_root=args.raw_root,
        window_hours=args.window_hours,
        window_stride_hours=args.window_stride_hours,
        positive_window_gap_hours=args.positive_window_gap_hours,
        delta_cap_hours=args.delta_cap_hours,
        max_stays=args.max_stays,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
    )
    ensure_observation_data(config, build_data=args.build_data)

    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint_path)
    model, _ = load_model_from_checkpoint(checkpoint_path, device)

    cluster_model_path = (
        Path(args.cluster_model_path).resolve()
        if args.cluster_model_path
        else (config.clusters_dir / "cluster_model.npz")
    )
    cluster_model = np.load(cluster_model_path) if cluster_model_path.exists() else None

    dataset = ObservationWindowDataset(config, split=args.split, max_windows=args.max_windows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    rng = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    rng.manual_seed(args.seed)

    result_frames = []
    with torch.no_grad():
        for windows, batch_meta in tqdm(loader, desc="Robustness"):
            metadata = _metadata_batch_to_frame(batch_meta)
            original = windows.to(device)
            perturbed = _perturb_windows(original, args.drop_prob, float(config.delta_cap_hours), float(config.bin_hours), rng)

            original_embeddings = model.encode(original).cpu().numpy()
            perturbed_embeddings = model.encode(perturbed).cpu().numpy()

            drift = np.linalg.norm(original_embeddings - perturbed_embeddings, axis=1)
            cosine = (
                (original_embeddings * perturbed_embeddings).sum(axis=1)
                / (np.linalg.norm(original_embeddings, axis=1) * np.linalg.norm(perturbed_embeddings, axis=1) + 1e-8)
            )

            metadata["embedding_drift_l2"] = drift
            metadata["embedding_cosine"] = cosine
            if cluster_model is not None:
                original_cluster = _assign_to_clusters(original_embeddings, cluster_model)
                perturbed_cluster = _assign_to_clusters(perturbed_embeddings, cluster_model)
                metadata["original_cluster"] = original_cluster
                metadata["perturbed_cluster"] = perturbed_cluster
                metadata["cluster_stable"] = (original_cluster == perturbed_cluster).astype(int)
            result_frames.append(metadata)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    config.robustness_dir.mkdir(parents=True, exist_ok=True)
    results_path = write_dataframe(results, config.robustness_dir / "robustness_metrics.parquet")

    summary = {
        "split": args.split,
        "num_windows": int(len(results)),
        "drop_prob": float(args.drop_prob),
        "mean_embedding_drift_l2": float(results["embedding_drift_l2"].mean()) if not results.empty else 0.0,
        "median_embedding_drift_l2": float(results["embedding_drift_l2"].median()) if not results.empty else 0.0,
        "mean_embedding_cosine": float(results["embedding_cosine"].mean()) if not results.empty else 0.0,
        "results_path": str(results_path),
    }
    if not results.empty and "cluster_stable" in results.columns:
        summary["cluster_stability_rate"] = float(results["cluster_stable"].mean())

    write_json(summary, config.robustness_dir / "robustness_summary.json")


if __name__ == "__main__":
    main()
