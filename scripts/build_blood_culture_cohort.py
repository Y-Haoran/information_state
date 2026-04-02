from __future__ import annotations

import argparse
import json
from pathlib import Path

from mimic_iv_project.blood_culture import (
    build_first_gram_positive_alerts,
    build_specimen_frame,
    flag_icu_at_alert,
    prepare_specimen_for_csv,
    read_positive_blood_cultures,
    serialize_organisms,
)
from mimic_iv_project.config import ProjectConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the first Gram-positive blood-culture alert cohort."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root for artifact output.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="MIMIC-IV root containing hosp/ and icu/. Defaults to MIMIC_IV_ROOT.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <project-root>/artifacts/blood_culture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(project_root=args.project_root, raw_root=args.raw_root)
    out_dir = args.out_dir or (config.artifacts_dir / "blood_culture")
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_rows = read_positive_blood_cultures(config)
    specimen = build_specimen_frame(positive_rows)
    first_alerts = build_first_gram_positive_alerts(specimen)
    first_alerts["in_icu_at_alert"] = flag_icu_at_alert(first_alerts, config)

    specimen_csv = out_dir / "positive_blood_culture_specimens.csv"
    cohort_csv = out_dir / "first_gp_alert_cohort.csv"
    metadata_json = out_dir / "blood_culture_cohort_metadata.json"

    prepare_specimen_for_csv(specimen).to_csv(specimen_csv, index=False)

    cohort_out = first_alerts.copy()
    cohort_out["organisms_json"] = cohort_out["organisms"].map(serialize_organisms)
    cohort_out = cohort_out.drop(columns=["organisms"])
    cohort_out.to_csv(cohort_csv, index=False)

    metadata = {
        "cohort_definition": {
            "unit": "First Gram-positive candidate positive blood-culture alert per hospital admission",
            "alert_time": "Specimen-level storetime with fallback to charttime/storedate/chartdate",
        },
        "overall_positive_blood_culture": {
            "positive_rows": int(len(positive_rows)),
            "positive_specimens": int(specimen["micro_specimen_id"].nunique()),
            "positive_hadm": int(specimen["hadm_id"].nunique()),
        },
        "first_gp_alert_cohort": {
            "rows": int(len(first_alerts)),
            "hadm": int(first_alerts["hadm_id"].nunique()),
            "in_icu_at_alert": int(first_alerts["in_icu_at_alert"].astype(int).sum()),
        },
        "files": {
            "specimen_csv": str(specimen_csv),
            "cohort_csv": str(cohort_csv),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
