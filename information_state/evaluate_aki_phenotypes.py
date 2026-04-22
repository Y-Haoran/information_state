"""Evaluate AKI phenotypes with renal trajectories and KDIGO-specific summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import make_project_config, read_dataframe, resolve_existing_table, write_dataframe, write_run_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for AKI-specific phenotype evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate AKI phenotype clusters against renal trajectories and outcomes.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--cohort", type=str, default="aki_kdigo")
    parser.add_argument("--assignments-path", type=str, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args(argv)


def _preferred_focus_features(feature_names: list[str]) -> list[str]:
    ordered = [
        "creatinine",
        "urine_output_ml",
        "bun",
        "potassium",
        "bicarbonate",
        "map",
        "heart_rate",
        "vasopressor_event",
    ]
    available = [name for name in ordered if name in feature_names]
    if available:
        return available
    return feature_names[: min(5, len(feature_names))]


def main(argv: Sequence[str] | None = None) -> None:
    """Generate AKI-focused phenotype summaries from clustered window assignments."""
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

    missing_columns = {"aki", "aki_stage"} - set(assignments.columns)
    if missing_columns:
        raise ValueError(f"AKI evaluation requires assignment columns {sorted(missing_columns)}.")

    with open(config.observation_metadata_path, "r", encoding="utf-8") as handle:
        observation_metadata = json.load(handle)
    feature_names = list(observation_metadata["dynamic_feature_names"])
    bin_hours = int(observation_metadata["bin_hours"])
    window_bins = int(observation_metadata["window_hours"]) // bin_hours
    focus_features = _preferred_focus_features(feature_names)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    focus_indexes = [feature_index[name] for name in focus_features]

    values = np.load(config.observation_hourly_values_path, mmap_mode="r")
    masks = np.load(config.observation_hourly_masks_path, mmap_mode="r")
    deltas = np.load(config.observation_hourly_deltas_path, mmap_mode="r")

    clusters = sorted(assignments["cluster"].unique().tolist())
    cluster_index = {cluster: idx for idx, cluster in enumerate(clusters)}
    value_sum = np.zeros((len(clusters), len(focus_features), window_bins), dtype=np.float64)
    mask_sum = np.zeros_like(value_sum)
    delta_sum = np.zeros_like(value_sum)
    counts = np.zeros((len(clusters),), dtype=np.int64)

    for row in tqdm(assignments.itertuples(index=False), total=len(assignments), desc="AKI trajectories"):
        cluster = int(row.cluster)
        flat_start = int(row.flat_start)
        flat_stop = flat_start + window_bins
        cluster_pos = cluster_index[cluster]
        window_values = np.asarray(values[flat_start:flat_stop, focus_indexes], dtype=np.float32).T
        window_masks = np.asarray(masks[flat_start:flat_stop, focus_indexes], dtype=np.float32).T
        window_deltas = np.asarray(deltas[flat_start:flat_stop, focus_indexes], dtype=np.float32).T

        value_sum[cluster_pos] += window_values
        mask_sum[cluster_pos] += window_masks
        delta_sum[cluster_pos] += window_deltas
        counts[cluster_pos] += 1

    trajectory_rows = []
    feature_profile_rows = []
    for cluster in clusters:
        cluster_pos = cluster_index[cluster]
        denominator = max(int(counts[cluster_pos]), 1)
        for feature_pos, feature_name in enumerate(focus_features):
            mean_values = value_sum[cluster_pos, feature_pos] / denominator
            mean_masks = mask_sum[cluster_pos, feature_pos] / denominator
            mean_deltas = delta_sum[cluster_pos, feature_pos] / denominator
            feature_profile_rows.append(
                {
                    "cluster": cluster,
                    "feature": feature_name,
                    "value_mean_window": float(mean_values.mean()),
                    "mask_density_window": float(mean_masks.mean()),
                    "delta_mean_window": float(mean_deltas.mean()),
                    "value_peak_window": float(mean_values.max()),
                    "value_trough_window": float(mean_values.min()),
                }
            )
            for time_bin in range(window_bins):
                trajectory_rows.append(
                    {
                        "cluster": cluster,
                        "feature": feature_name,
                        "time_bin": time_bin,
                        "hour": time_bin * bin_hours,
                        "value_mean": float(mean_values[time_bin]),
                        "mask_density": float(mean_masks[time_bin]),
                        "delta_mean": float(mean_deltas[time_bin]),
                    }
                )

    stage_counts = (
        assignments.groupby(["cluster", "aki_stage"], as_index=False)
        .size()
        .rename(columns={"size": "n_windows"})
        .sort_values(["cluster", "aki_stage"])
        .reset_index(drop=True)
    )
    stage_totals = stage_counts.groupby("cluster")["n_windows"].transform("sum").clip(lower=1)
    stage_counts["window_fraction"] = stage_counts["n_windows"] / stage_totals

    outcome_frame = (
        assignments.groupby("cluster", as_index=False)
        .agg(
            n_windows=("window_id", "count"),
            n_stays=("stay_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            mortality_rate=("in_hospital_mortality", "mean"),
            icu_los_days_mean=("icu_los_days", "mean"),
            aki_stage_mean=("aki_stage", "mean"),
            creatinine_criterion_rate=("aki_creatinine", "mean"),
            urine_output_criterion_rate=("aki_urine_output", "mean"),
            onset_window_rate=("window_contains_aki_onset", "mean"),
            mean_hours_since_aki_onset_start=("hours_since_aki_onset_start", "mean"),
            mean_hours_since_aki_onset_end=("hours_since_aki_onset_end", "mean"),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    profile_frame = pd.DataFrame(feature_profile_rows).sort_values(["cluster", "feature"]).reset_index(drop=True)
    trajectory_frame = pd.DataFrame(trajectory_rows).sort_values(["cluster", "feature", "time_bin"]).reset_index(drop=True)

    config.evaluation_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = write_dataframe(outcome_frame, config.evaluation_dir / "aki_cluster_outcomes.parquet")
    profiles_path = write_dataframe(profile_frame, config.evaluation_dir / "aki_cluster_feature_profiles.parquet")
    trajectories_path = write_dataframe(trajectory_frame, config.evaluation_dir / "aki_cluster_trajectory_profiles.parquet")
    stage_counts_path = write_dataframe(stage_counts, config.evaluation_dir / "aki_cluster_stage_distribution.parquet")

    report_lines = ["# AKI Phenotype Evaluation", ""]
    for row in outcome_frame.itertuples(index=False):
        cluster_profiles = profile_frame[profile_frame["cluster"] == row.cluster]
        top_profile = cluster_profiles.sort_values("value_mean_window", ascending=False).head(3)
        feature_text = ", ".join(
            f"{feature}={value_mean_window:.2f}"
            for feature, value_mean_window in zip(top_profile["feature"], top_profile["value_mean_window"])
        )
        report_lines.append(
            f"- Cluster {row.cluster}: n_windows={row.n_windows}, mortality={row.mortality_rate:.3f}, "
            f"mean_aki_stage={row.aki_stage_mean:.2f}, creatinine_criterion={row.creatinine_criterion_rate:.3f}, "
            f"urine_output_criterion={row.urine_output_criterion_rate:.3f}, onset_window_rate={row.onset_window_rate:.3f}, "
            f"dominant signals: {feature_text}"
        )

    report_path = config.evaluation_dir / "aki_evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

    write_run_manifest(
        config=config,
        stage="evaluate_aki_phenotypes",
        cli_args=vars(args),
        output_dir=config.evaluation_dir,
        extra={
            "assignments_path": str(assignments_path),
            "focus_features": focus_features,
            "artifacts": {
                "outcomes_path": str(outcomes_path),
                "profiles_path": str(profiles_path),
                "trajectories_path": str(trajectories_path),
                "stage_distribution_path": str(stage_counts_path),
                "report_path": str(report_path),
            },
        },
    )

    print(f"aki_outcomes={outcomes_path}")
    print(f"aki_profiles={profiles_path}")
    print(f"aki_trajectories={trajectories_path}")
    print(f"aki_stage_distribution={stage_counts_path}")


if __name__ == "__main__":
    main()
