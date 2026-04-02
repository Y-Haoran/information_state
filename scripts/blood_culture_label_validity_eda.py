from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from mimic_iv_project.blood_culture import (
    assign_provisional_label,
    build_first_gram_positive_alerts,
    build_specimen_frame,
    compute_repeat_features,
    flag_icu_at_alert,
    is_gram_positive,
    read_positive_blood_cultures,
)
from mimic_iv_project.config import ProjectConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA for Gram-positive blood-culture contamination vs true-BSI label validity."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root where reports are written.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="MIMIC-IV root containing hosp/ and icu/. Defaults to MIMIC_IV_ROOT.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for summary JSON. Defaults to <project-root>/reports/blood_culture_label_validity_summary.json.",
    )
    return parser.parse_args()


def _summarize_first_alerts(first_alerts: pd.DataFrame) -> dict[str, object]:
    category_counts = first_alerts["category"].value_counts().to_dict()
    category_rates = {}
    for category, group in first_alerts.groupby("category"):
        category_rates[str(category)] = {
            "count": int(len(group)),
            "repeat_any_positive_48h": round(100 * group["repeat_any_positive_48h"].mean(), 2),
            "repeat_gp_positive_48h": round(100 * group["repeat_gp_positive_48h"].mean(), 2),
            "repeat_same_organism_48h": round(100 * group["repeat_same_organism_48h"].mean(), 2),
            "in_icu_at_alert": round(100 * group["in_icu_at_alert"].astype(int).mean(), 2),
        }

    labeled = assign_provisional_label(first_alerts)
    label_counts = labeled["provisional_label"].value_counts().to_dict()

    first_org_counts = Counter()
    for orgs in first_alerts["organisms"]:
        for org in orgs:
            if is_gram_positive(org):
                first_org_counts[org] += 1

    summary = {
        "first_alert_count": int(len(first_alerts)),
        "first_alert_hadm": int(first_alerts["hadm_id"].nunique()),
        "first_alert_with_storetime_pct": round(100 * first_alerts["has_storetime"].mean(), 2),
        "first_alert_in_icu_pct": round(100 * first_alerts["in_icu_at_alert"].astype(int).mean(), 2),
        "category_counts": {str(k): int(v) for k, v in category_counts.items()},
        "category_repeat_rates": category_rates,
        "high_confidence_label_candidates": {
            "likely_contaminant": int(label_counts.get("likely_contaminant", 0)),
            "likely_true_bsi": int(label_counts.get("likely_true_bsi", 0)),
            "indeterminate": int(label_counts.get("indeterminate", 0)),
        },
        "top_first_alert_organisms": first_org_counts.most_common(15),
    }
    return summary


def build_summary(config: ProjectConfig) -> dict[str, object]:
    positive_rows = read_positive_blood_cultures(config)
    specimen = build_specimen_frame(positive_rows)

    gp_specimen = specimen[specimen["is_gp_candidate"] == 1].copy()
    first_alerts = build_first_gram_positive_alerts(specimen)
    first_alerts["in_icu_at_alert"] = flag_icu_at_alert(first_alerts, config)
    first_alerts = compute_repeat_features(first_alerts, specimen)

    summary = {
        "cohort_definition": {
            "unit": "First Gram-positive candidate positive blood-culture alert per hospital admission",
            "alert_time": "min(storetime, fallback charttime/storedate/chartdate) at the specimen level",
            "positive_row_definition": "BLOOD CULTURE row with non-empty org_name and org_name != CANCELLED",
        },
        "overall_positive_blood_culture": {
            "positive_rows": int(len(positive_rows)),
            "positive_specimens": int(specimen["micro_specimen_id"].nunique()),
            "positive_hadm": int(specimen["hadm_id"].nunique()),
            "positive_specimens_with_storetime_pct": round(100 * specimen["has_storetime"].mean(), 2),
        },
        "gram_positive_candidate_specimens": {
            "specimens": int(len(gp_specimen)),
            "hadm": int(gp_specimen["hadm_id"].nunique()),
        },
        "first_gram_positive_alerts": _summarize_first_alerts(first_alerts),
    }
    return summary


def main() -> None:
    args = parse_args()
    config = ProjectConfig(project_root=args.project_root, raw_root=args.raw_root)
    json_out = args.json_out or (config.project_root / "reports" / "blood_culture_label_validity_summary.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(config)
    json_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
