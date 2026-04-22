"""Evaluate learned state clusters against outcomes, physiology, and transitions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import make_project_config, read_dataframe, resolve_existing_table, write_dataframe, write_run_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for phenotype evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate learned phenotype clusters against physiology and outcomes.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--cohort", type=str, default="all_adult_icu")
    parser.add_argument("--assignments-path", type=str, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate phenotype summaries from clustered window assignments."""
    args = parse_args(argv)
    config = make_project_config(project_root=args.project_root, cohort_name=args.cohort)
    assignments_path = Path(args.assignments_path).resolve() if args.assignments_path else resolve_existing_table(
        config.clusters_dir / "cluster_assignments.parquet"
    )
    assignments = read_dataframe(assignments_path)
    if args.max_windows is not None:
        assignments = assignments.head(args.max_windows).copy()
    if assignments.empty:
        raise ValueError("Cluster assignments are empty.")

    with open(config.observation_metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    feature_names = list(metadata["dynamic_feature_names"])

    values = np.load(config.observation_hourly_values_path, mmap_mode="r")
    masks = np.load(config.observation_hourly_masks_path, mmap_mode="r")
    deltas = np.load(config.observation_hourly_deltas_path, mmap_mode="r")
    window_bins = config.window_bins

    clusters = sorted(assignments["cluster"].unique().tolist())
    accumulators = {
        cluster: {
            "count": 0,
            "value_sum": np.zeros(len(feature_names), dtype=np.float64),
            "mask_sum": np.zeros(len(feature_names), dtype=np.float64),
            "delta_sum": np.zeros(len(feature_names), dtype=np.float64),
            "overall_mask_density_sum": 0.0,
            "overall_delta_mean_sum": 0.0,
        }
        for cluster in clusters
    }

    for row in tqdm(assignments.itertuples(index=False), total=len(assignments), desc="Profiles"):
        flat_start = int(row.flat_start)
        flat_stop = flat_start + window_bins
        cluster = int(row.cluster)
        window_values = np.asarray(values[flat_start:flat_stop], dtype=np.float32)
        window_masks = np.asarray(masks[flat_start:flat_stop], dtype=np.float32)
        window_deltas = np.asarray(deltas[flat_start:flat_stop], dtype=np.float32)

        stats = accumulators[cluster]
        stats["count"] += 1
        stats["value_sum"] += window_values.mean(axis=0)
        stats["mask_sum"] += window_masks.mean(axis=0)
        stats["delta_sum"] += window_deltas.mean(axis=0)
        stats["overall_mask_density_sum"] += float(window_masks.mean())
        stats["overall_delta_mean_sum"] += float(window_deltas.mean())

    profile_rows = []
    for cluster in clusters:
        stats = accumulators[cluster]
        denominator = max(stats["count"], 1)
        row = {
            "cluster": cluster,
            "n_windows": int(stats["count"]),
            "overall_mask_density": stats["overall_mask_density_sum"] / denominator,
            "overall_delta_mean": stats["overall_delta_mean_sum"] / denominator,
        }
        mean_values = stats["value_sum"] / denominator
        mean_masks = stats["mask_sum"] / denominator
        mean_deltas = stats["delta_sum"] / denominator
        for feature_index, feature_name in enumerate(feature_names):
            row[f"value_mean_{feature_name}"] = mean_values[feature_index]
            row[f"mask_density_{feature_name}"] = mean_masks[feature_index]
            row[f"delta_mean_{feature_name}"] = mean_deltas[feature_index]
        profile_rows.append(row)
    profile_frame = pd.DataFrame(profile_rows).sort_values("cluster").reset_index(drop=True)

    outcome_frame = (
        assignments.groupby("cluster", as_index=False)
        .agg(
            n_windows=("window_id", "count"),
            n_stays=("stay_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            mortality_rate=("in_hospital_mortality", "mean"),
            icu_los_days_mean=("icu_los_days", "mean"),
            start_hour_mean=("start_hour", "mean"),
            end_hour_mean=("end_hour", "mean"),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    transitions = defaultdict(int)
    ordered = assignments.sort_values(["stay_id", "start_bin", "window_id"])
    for _, stay_frame in ordered.groupby("stay_id"):
        labels = stay_frame["cluster"].tolist()
        for current, nxt in zip(labels, labels[1:]):
            transitions[(int(current), int(nxt))] += 1

    transition_rows = []
    by_source = defaultdict(int)
    for (source, target), count in transitions.items():
        by_source[source] += count
    for (source, target), count in sorted(transitions.items()):
        transition_rows.append(
            {
                "source_cluster": source,
                "target_cluster": target,
                "count": count,
                "probability": count / max(by_source[source], 1),
            }
        )
    transition_frame = pd.DataFrame(transition_rows)

    config.evaluation_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = write_dataframe(outcome_frame, config.evaluation_dir / "cluster_outcomes.parquet")
    profiles_path = write_dataframe(profile_frame, config.evaluation_dir / "cluster_feature_profiles.parquet")
    transitions_path = write_dataframe(transition_frame, config.evaluation_dir / "cluster_trajectory_profiles.parquet")

    report_lines = ["# Phenotype Evaluation", ""]
    for row in outcome_frame.itertuples(index=False):
        profile_row = profile_frame[profile_frame["cluster"] == row.cluster].iloc[0]
        value_features = {
            feature_name: float(profile_row[f"value_mean_{feature_name}"])
            for feature_name in feature_names
        }
        top_features = sorted(value_features.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        top_feature_text = ", ".join(f"{name}={value:.2f}" for name, value in top_features)
        report_lines.append(
            f"- Cluster {row.cluster}: n_windows={row.n_windows}, n_stays={row.n_stays}, "
            f"mortality={row.mortality_rate:.3f}, icu_los_days={row.icu_los_days_mean:.2f}, "
            f"top state signals: {top_feature_text}"
        )

    with open(config.evaluation_dir / "evaluation_report.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")
    write_run_manifest(
        config=config,
        stage="evaluate_phenotypes",
        cli_args=vars(args),
        output_dir=config.evaluation_dir,
        extra={
            "assignments_path": str(assignments_path),
            "artifacts": {
                "outcomes_path": str(outcomes_path),
                "profiles_path": str(profiles_path),
                "transitions_path": str(transitions_path),
                "report_path": str(config.evaluation_dir / "evaluation_report.md"),
            },
        },
    )

    print(f"outcomes={outcomes_path}")
    print(f"profiles={profiles_path}")
    print(f"transitions={transitions_path}")


if __name__ == "__main__":
    main()
