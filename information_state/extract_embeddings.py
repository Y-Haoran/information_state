"""Export latent state embeddings for one window at a time."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

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
    resolve_checkpoint_path,
    set_global_seed,
    write_dataframe,
    write_json,
    write_run_manifest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for embedding extraction."""
    parser = argparse.ArgumentParser(description="Extract latent state embeddings for individual windows.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--raw-root", type=str, default=None)
    parser.add_argument("--cohort", type=str, default="all_adult_icu")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--split", nargs="+", default=["train", "val"])
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--window-stride-hours", type=int, default=2)
    parser.add_argument("--positive-window-gap-hours", type=int, default=2)
    parser.add_argument("--delta-cap-hours", type=int, default=48)
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _metadata_batch_to_frame(batch_meta: dict[str, object]) -> pd.DataFrame:
    columns = {}
    for key, value in batch_meta.items():
        if torch.is_tensor(value):
            columns[key] = value.cpu().numpy()
        else:
            columns[key] = list(value)
    return pd.DataFrame(columns)


def main(argv: Sequence[str] | None = None) -> None:
    """Extract encoder embeddings and aligned metadata tables."""
    args = parse_args(argv)
    config = make_project_config(
        project_root=args.project_root,
        raw_root=args.raw_root,
        cohort_name=args.cohort,
        window_hours=args.window_hours,
        window_stride_hours=args.window_stride_hours,
        positive_window_gap_hours=args.positive_window_gap_hours,
        delta_cap_hours=args.delta_cap_hours,
        max_stays=args.max_stays,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        random_seed=args.seed,
    )
    set_global_seed(args.seed)
    ensure_observation_data(config, build_data=args.build_data)

    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint_path)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    config.embeddings_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "splits": {},
    }

    for split in args.split:
        dataset = ObservationWindowDataset(config, split=split, max_windows=args.max_windows)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        embeddings = []
        metadata_frames = []
        with torch.no_grad():
            for windows, batch_meta in tqdm(loader, desc=f"Extract {split}"):
                encoded = model.encode(windows.to(device)).cpu().numpy()
                embeddings.append(encoded)
                metadata_frames.append(_metadata_batch_to_frame(batch_meta))

        if embeddings:
            stacked_embeddings = np.concatenate(embeddings, axis=0)
            metadata = pd.concat(metadata_frames, ignore_index=True)
        else:
            hidden_dim = checkpoint["model_state_dict"]["encoder.output_norm.weight"].shape[0]
            stacked_embeddings = np.zeros((0, hidden_dim), dtype=np.float32)
            metadata = pd.DataFrame(columns=list(dataset.window_metadata.columns))

        embedding_path = config.embeddings_dir / f"{split}_embeddings.npy"
        metadata_path = write_dataframe(metadata, config.embeddings_dir / f"{split}_metadata.parquet")
        np.save(embedding_path, stacked_embeddings)

        manifest["splits"][split] = {
            "num_windows": int(len(metadata)),
            "embedding_dim": int(stacked_embeddings.shape[1]) if stacked_embeddings.ndim == 2 else 0,
            "embeddings_path": str(embedding_path),
            "metadata_path": str(metadata_path),
        }

    write_json(manifest, config.embeddings_dir / "embedding_manifest.json")
    write_run_manifest(
        config=config,
        stage="extract_embeddings",
        cli_args=vars(args),
        output_dir=config.embeddings_dir,
        extra={
            "checkpoint_path": str(checkpoint_path),
            "embedding_manifest_path": str(config.embeddings_dir / "embedding_manifest.json"),
            "splits": manifest["splits"],
        },
    )


if __name__ == "__main__":
    main()
