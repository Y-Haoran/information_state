from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mimic_iv_project.blood_culture import (
    assign_provisional_label,
    compute_repeat_features,
    deserialize_organisms,
    load_specimen_from_csv,
    serialize_organisms,
)
from mimic_iv_project.config import ProjectConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build provisional labels for the first Gram-positive blood-culture alert cohort."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root for artifact output.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <project-root>/artifacts/blood_culture.",
    )
    parser.add_argument(
        "--specimen-path",
        type=Path,
        default=None,
        help="Optional override for positive_blood_culture_specimens.csv.",
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=None,
        help="Optional override for first_gp_alert_cohort.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(project_root=args.project_root)
    out_dir = args.out_dir or (config.artifacts_dir / "blood_culture")
    out_dir.mkdir(parents=True, exist_ok=True)

    specimen_path = args.specimen_path or (out_dir / "positive_blood_culture_specimens.csv")
    cohort_path = args.cohort_path or (out_dir / "first_gp_alert_cohort.csv")

    specimen = load_specimen_from_csv(specimen_path)
    cohort = pd.read_csv(cohort_path)
    cohort["alert_time"] = pd.to_datetime(cohort["alert_time"], errors="coerce")
    cohort["organisms"] = cohort["organisms_json"].map(deserialize_organisms)

    labeled = compute_repeat_features(cohort, specimen)
    labeled = assign_provisional_label(labeled)

    labels_csv = out_dir / "first_gp_alert_labels.csv"
    dataset_csv = out_dir / "first_gp_alert_dataset.csv"
    metadata_json = out_dir / "blood_culture_label_metadata.json"

    label_cols = [
        "hadm_id",
        "micro_specimen_id",
        "repeat_any_positive_48h",
        "repeat_gp_positive_48h",
        "repeat_same_organism_48h",
        "repeat_positive_specimen_count_48h",
        "provisional_label",
        "label_source",
        "is_high_confidence_binary",
    ]
    labeled[label_cols].to_csv(labels_csv, index=False)

    dataset = labeled.copy()
    dataset["organisms_json"] = dataset["organisms"].map(serialize_organisms)
    dataset = dataset.drop(columns=["organisms"])
    dataset.to_csv(dataset_csv, index=False)

    label_counts = labeled["provisional_label"].value_counts().to_dict()
    metadata = {
        "label_definition": {
            "likely_contaminant": "contam_like first alert with no repeat positive blood-culture specimen within 48h",
            "likely_true_bsi": "true_like first alert with repeat positive blood culture within 48h",
            "indeterminate": "all other first alerts",
        },
        "counts": {
            "rows": int(len(labeled)),
            "likely_contaminant": int(label_counts.get("likely_contaminant", 0)),
            "likely_true_bsi": int(label_counts.get("likely_true_bsi", 0)),
            "indeterminate": int(label_counts.get("indeterminate", 0)),
            "high_confidence_binary": int(labeled["is_high_confidence_binary"].sum()),
        },
        "files": {
            "labels_csv": str(labels_csv),
            "dataset_csv": str(dataset_csv),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
