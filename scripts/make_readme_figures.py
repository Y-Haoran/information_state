"""Generate README figures from a completed State-from-Observation run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AKI_HEATMAP_FEATURES = [
    "heart_rate",
    "map",
    "creatinine",
    "bun",
    "bicarbonate",
    "potassium",
    "urine_output_ml",
    "vasopressor_event",
]

AKI_FEATURE_LABELS = {
    "heart_rate": "HR",
    "map": "MAP",
    "creatinine": "Creatinine",
    "bun": "BUN",
    "bicarbonate": "Bicarbonate",
    "potassium": "Potassium",
    "urine_output_ml": "Urine output",
    "vasopressor_event": "Vasopressor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README-ready figures.")
    parser.add_argument("--run-root", required=True, type=Path, help="Run root containing artifacts/state_from_observation.")
    parser.add_argument("--cohort", default="all_adult_icu", help="Cohort name. Use all_adult_icu for the default artifact root.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write figure assets into.")
    parser.add_argument("--tag", required=True, help="Tag to append to output filenames.")
    return parser.parse_args()


def state_dir(run_root: Path, cohort: str) -> Path:
    base = run_root / "artifacts" / "state_from_observation"
    if cohort and cohort != "all_adult_icu":
        return base / cohort
    return base


def _load_history(history_path: Path) -> list[dict[str, object]]:
    with history_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_training_curves(history: list[dict[str, object]], output_path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    val_loss = [row["val"]["loss"] for row in history]
    train_r1 = [row["train"]["retrieval_at_1"] for row in history]
    val_r1 = [row["val"]["retrieval_at_1"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].plot(epochs, train_loss, label="Train loss", color="#1d3557", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val loss", color="#e76f51", linewidth=2)
    axes[0].set_title("SSL loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("InfoNCE loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, train_r1, label="Train R@1", color="#2a9d8f", linewidth=2)
    axes[1].plot(epochs, val_r1, label="Val R@1", color="#f4a261", linewidth=2)
    axes[1].set_title("Retrieval@1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Retrieval@1")
    axes[1].set_ylim(min(val_r1) - 0.02, 1.01)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aki_cluster_summary(outcomes: pd.DataFrame, output_path: Path) -> None:
    clusters = outcomes["cluster"].astype(str).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].bar(clusters, outcomes["mortality_rate"], color="#bc4749")
    axes[0].set_title("Mortality")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Rate")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(clusters, outcomes["aki_stage_mean"], color="#577590")
    axes[1].set_title("Mean AKI stage")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Stage")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(
        np.arange(len(clusters)) - 0.18,
        outcomes["creatinine_criterion_rate"],
        width=0.36,
        label="Creatinine criterion",
        color="#6a994e",
    )
    axes[2].bar(
        np.arange(len(clusters)) + 0.18,
        outcomes["urine_output_criterion_rate"],
        width=0.36,
        label="Urine-output criterion",
        color="#4d908e",
    )
    axes[2].set_xticks(np.arange(len(clusters)))
    axes[2].set_xticklabels(clusters)
    axes[2].set_title("KDIGO criterion pattern")
    axes[2].set_xlabel("Cluster")
    axes[2].set_ylabel("Rate")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("AKI phenotype-level separation", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aki_heatmap(profiles: pd.DataFrame, output_path: Path) -> None:
    pivot = (
        profiles[profiles["feature"].isin(AKI_HEATMAP_FEATURES)]
        .pivot(index="cluster", columns="feature", values="value_mean_window")
        .reindex(columns=AKI_HEATMAP_FEATURES)
        .sort_index()
    )
    matrix = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    vmax = np.abs(matrix).max()
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(AKI_HEATMAP_FEATURES)))
    ax.set_xticklabels([AKI_FEATURE_LABELS[name] for name in AKI_HEATMAP_FEATURES], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"Cluster {int(cluster)}" for cluster in pivot.index.tolist()])
    ax.set_title("AKI cluster signal heatmap")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    colorbar.set_label("Mean normalized value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aki_trajectories(trajectories: pd.DataFrame, output_path: Path) -> None:
    feature_order = ["creatinine", "urine_output_ml", "map"]
    feature_titles = {
        "creatinine": "Creatinine trajectory",
        "urine_output_ml": "Urine output trajectory",
        "map": "MAP trajectory",
    }
    colors = ["#1d3557", "#e76f51", "#2a9d8f"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for axis, feature in zip(axes, feature_order):
        subset = trajectories[trajectories["feature"] == feature].sort_values(["cluster", "hour"])
        for color, cluster in zip(colors, sorted(subset["cluster"].unique())):
            curve = subset[subset["cluster"] == cluster]
            axis.plot(curve["hour"], curve["value_mean"], linewidth=2, color=color, label=f"Cluster {cluster}")
        axis.set_title(feature_titles[feature])
        axis.set_xlabel("Hour in window")
        axis.set_ylabel("Mean normalized value")
        axis.grid(alpha=0.25)

    axes[0].legend(frameon=False)
    fig.suptitle("AKI cluster trajectory profiles", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cohort_dir = state_dir(args.run_root, args.cohort)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _load_history(cohort_dir / "ssl_history.json")
    plot_training_curves(history, output_dir / f"training_curve_{args.tag}.png", title=f"{args.cohort} SSL dynamics")

    if args.cohort == "aki_kdigo":
        outcomes = pd.read_csv(cohort_dir / "evaluation" / "aki_cluster_outcomes.csv")
        profiles = pd.read_csv(cohort_dir / "evaluation" / "aki_cluster_feature_profiles.csv")
        trajectories = pd.read_csv(cohort_dir / "evaluation" / "aki_cluster_trajectory_profiles.csv")

        plot_aki_cluster_summary(outcomes, output_dir / f"aki_cluster_summary_{args.tag}.png")
        plot_aki_heatmap(profiles, output_dir / f"aki_cluster_heatmap_{args.tag}.png")
        plot_aki_trajectories(trajectories, output_dir / f"aki_cluster_trajectories_{args.tag}.png")

    robustness_hist = cohort_dir / "robustness" / "embedding_drift_histogram.png"
    if robustness_hist.exists():
        shutil.copy2(robustness_hist, output_dir / f"robustness_{args.tag}.png")


if __name__ == "__main__":
    main()
