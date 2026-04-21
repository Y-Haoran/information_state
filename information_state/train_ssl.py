"""Training entrypoint for State-from-Observation SSL pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .config import ProjectConfig
from .contrastive import SymmetricInfoNCELoss
from .observation_data import ObservationWindowPairDataset, build_state_from_observation_dataset
from .state_from_observation import StateFromObservationModel
from .utils import load_json, set_global_seed, write_json, write_run_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for SSL training."""
    parser = argparse.ArgumentParser(description="Train the State-from-Observation SSL model on whole MIMIC-IV.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--raw-root", type=str, default=None)
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--window-stride-hours", type=int, default=2)
    parser.add_argument("--positive-window-gap-hours", type=int, default=2)
    parser.add_argument("--delta-cap-hours", type=int, default=48)
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> ProjectConfig:
    """Build the project config for a training run."""
    project_root = Path(args.project_root).resolve() if args.project_root else ProjectConfig().project_root
    raw_root = Path(args.raw_root).resolve() if args.raw_root else None
    return ProjectConfig(
        project_root=project_root,
        raw_root=raw_root,
        bin_hours=1,
        window_hours=args.window_hours,
        window_stride_hours=args.window_stride_hours,
        positive_window_gap_hours=args.positive_window_gap_hours,
        delta_cap_hours=args.delta_cap_hours,
        max_stays=args.max_stays,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        random_seed=args.seed,
    )


def evaluate(model, criterion, loader, device) -> dict[str, float]:
    """Evaluate the contrastive objective on a validation loader."""
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_retrieval = 0.0
    total_cosine = 0.0

    with torch.no_grad():
        for anchor, positive in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            anchor_projection, positive_projection = model(anchor, positive)
            loss, metrics = criterion(anchor_projection, positive_projection)
            batch_size = anchor.size(0)
            total_examples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_retrieval += metrics["retrieval_at_1"] * batch_size
            total_cosine += metrics["positive_cosine"] * batch_size

    denominator = max(total_examples, 1)
    return {
        "loss": total_loss / denominator,
        "retrieval_at_1": total_retrieval / denominator,
        "positive_cosine": total_cosine / denominator,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run SSL training and persist model, history, and reproducibility metadata."""
    args = parse_args(argv)
    config = make_config(args)
    set_global_seed(args.seed)

    expected_files = [
        config.observation_cohort_path,
        config.observation_hourly_values_path,
        config.observation_hourly_masks_path,
        config.observation_hourly_deltas_path,
        config.observation_window_metadata_path,
        config.observation_metadata_path,
    ]
    if args.build_data or not all(path.exists() for path in expected_files):
        build_state_from_observation_dataset(config)

    metadata = load_json(config.observation_metadata_path)

    train_dataset = ObservationWindowPairDataset(config, split="train")
    val_dataset = ObservationWindowPairDataset(config, split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)
    model = StateFromObservationModel(
        num_variables=len(metadata["dynamic_feature_names"]),
        num_time_bins=config.window_bins,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = SymmetricInfoNCELoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_retrieval = float("-inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_examples = 0
        train_loss = 0.0
        train_retrieval = 0.0
        train_cosine = 0.0

        for anchor, positive in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            optimizer.zero_grad()
            anchor_projection, positive_projection = model(anchor, positive)
            loss, metrics = criterion(anchor_projection, positive_projection)
            loss.backward()
            optimizer.step()

            batch_size = anchor.size(0)
            train_examples += batch_size
            train_loss += float(loss.item()) * batch_size
            train_retrieval += metrics["retrieval_at_1"] * batch_size
            train_cosine += metrics["positive_cosine"] * batch_size

        train_metrics = {
            "loss": train_loss / max(train_examples, 1),
            "retrieval_at_1": train_retrieval / max(train_examples, 1),
            "positive_cosine": train_cosine / max(train_examples, 1),
        }
        val_metrics = evaluate(model, criterion, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        if val_metrics["retrieval_at_1"] > best_retrieval:
            best_retrieval = val_metrics["retrieval_at_1"]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_r1={train_metrics['retrieval_at_1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_r1={val_metrics['retrieval_at_1']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model_config = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "projection_dim": args.projection_dim,
        "dropout": args.dropout,
    }
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
        "config": vars(args),
        "model_config": model_config,
        "best_val_retrieval_at_1": best_retrieval,
    }
    torch.save(checkpoint_payload, config.observation_checkpoint_path)
    write_json(history, config.observation_history_path)
    manifest_paths = write_run_manifest(
        config=config,
        stage="train_ssl",
        cli_args=vars(args),
        output_dir=config.state_from_observation_dir,
        extra={
            "artifacts": {
                "checkpoint_path": str(config.observation_checkpoint_path),
                "history_path": str(config.observation_history_path),
            },
            "model_config": model_config,
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "history": history,
            "best_val_retrieval_at_1": best_retrieval,
        },
    )

    checkpoint_payload["manifest_paths"] = manifest_paths
    torch.save(checkpoint_payload, config.observation_checkpoint_path)


if __name__ == "__main__":
    main()
