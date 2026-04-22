"""Generate README-ready figures from a completed State-from-Observation run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "heart_rate": "HR",
    "map": "MAP",
    "creatinine": "Creatinine",
    "bun": "BUN",
    "bicarbonate": "Bicarbonate",
    "hemoglobin": "Hemoglobin",
    "platelets": "Platelets",
    "lactate": "Lactate",
}

HEATMAP_FEATURES = [
    "heart_rate",
    "map",
    "creatinine",
    "bun",
    "bicarbonate",
    "hemoglobin",
    "platelets",
    "lactate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README-ready performance figures.")
    parser.add_argument("--run-root", required=True, type=Path, help="Run root containing artifacts/state_from_observation.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write figure assets into.")
    return parser.parse_args()


def _load_history(history_path: Path) -> list[dict[str, object]]:
    with history_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_training_curves(history: list[dict[str, object]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    val_loss = [row["val"]["loss"] for row in history]
    train_r1 = [row["train"]["retrieval_at_1"] for row in history]
    val_r1 = [row["val"]["retrieval_at_1"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].plot(epochs, train_loss, label="Train loss", color="#1d3557", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val loss", color="#e76f51", linewidth=2)
    axes[0].set_title("SSL Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("InfoNCE loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, train_r1, label="Train R@1", color="#2a9d8f", linewidth=2)
    axes[1].plot(epochs, val_r1, label="Val R@1", color="#f4a261", linewidth=2)
    axes[1].set_title("Retrieval@1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Retrieval@1")
    axes[1].set_ylim(0.9, 1.01)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Subset-Trained SSL Dynamics", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_outcomes(outcomes: pd.DataFrame, output_path: Path) -> None:
    clusters = outcomes["cluster"].astype(str).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].bar(clusters, outcomes["mortality_rate"], color="#bc4749")
    axes[0].set_title("Mortality by Cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Mortality rate")
    axes[0].set_ylim(0, max(outcomes["mortality_rate"]) * 1.2)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(clusters, outcomes["icu_los_days_mean"], color="#577590")
    axes[1].set_title("ICU LOS by Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Mean ICU LOS (days)")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Cluster-Level Clinical Separation", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_heatmap(profiles: pd.DataFrame, output_path: Path) -> None:
    value_columns = [f"value_mean_{feature}" for feature in HEATMAP_FEATURES]
    matrix = profiles[value_columns].to_numpy(dtype=float)
    clusters = profiles["cluster"].astype(int).tolist()
    labels = [FEATURE_LABELS[feature] for feature in HEATMAP_FEATURES]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    vmax = np.abs(matrix).max()
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(clusters)))
    ax.set_yticklabels([f"Cluster {cluster}" for cluster in clusters])
    ax.set_title("Cluster Signal Heatmap")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    colorbar.set_label("Mean normalized value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    state_dir = args.run_root / "artifacts" / "state_from_observation"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _load_history(state_dir / "ssl_history.json")
    outcomes = pd.read_csv(state_dir / "evaluation" / "cluster_outcomes.csv")
    profiles = pd.read_csv(state_dir / "evaluation" / "cluster_feature_profiles.csv")

    plot_training_curves(history, output_dir / "training_curve_subset_train_20260421_gpu_v1.png")
    plot_cluster_outcomes(outcomes, output_dir / "cluster_outcomes_subset_train_20260421_gpu_v1.png")
    plot_cluster_heatmap(profiles, output_dir / "cluster_signal_heatmap_subset_train_20260421_gpu_v1.png")

    robustness_hist = state_dir / "robustness" / "embedding_drift_histogram.png"
    if robustness_hist.exists():
        shutil.copy2(robustness_hist, output_dir / "robustness_subset_train_20260421_gpu_v1.png")


if __name__ == "__main__":
    main()
