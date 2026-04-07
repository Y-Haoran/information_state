from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_iv_project.blood_culture import normalize_org_name, serialize_organisms, specimen_category
from mimic_iv_project.config import ProjectConfig


RAW_USECOLS = [
    "subject_id",
    "hadm_id",
    "micro_specimen_id",
    "order_provider_id",
    "chartdate",
    "charttime",
    "spec_itemid",
    "spec_type_desc",
    "test_seq",
    "storedate",
    "storetime",
    "test_itemid",
    "test_name",
    "org_itemid",
    "org_name",
    "isolate_num",
    "quantity",
    "ab_itemid",
    "ab_name",
    "dilution_text",
    "dilution_value",
    "interpretation",
    "comments",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a specimen-level BLOOD CULTURE subset from microbiologyevents."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=200,
        help="Number of specimen rows to write to the preview CSV.",
    )
    return parser.parse_args()


def _json_sorted_unique(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna() if str(value).strip()})
    return json.dumps(values, separators=(",", ":"))


def _read_blood_culture_rows(config: ProjectConfig, chunksize: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    path = config.hosp_dir / "microbiologyevents.csv"
    dtype_map = {
        "order_provider_id": "string",
        "test_name": "string",
        "org_name": "string",
        "quantity": "string",
        "ab_name": "string",
        "dilution_text": "string",
        "interpretation": "string",
        "comments": "string",
    }
    for chunk in pd.read_csv(path, usecols=RAW_USECOLS, chunksize=chunksize, dtype=dtype_map):
        chunk = chunk[chunk["spec_type_desc"].astype(str) == "BLOOD CULTURE"].copy()
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["micro_specimen_id", "subject_id"]).copy()
        if chunk.empty:
            continue
        frames.append(chunk)

    if not frames:
        raise RuntimeError(f"No BLOOD CULTURE rows found in {path}.")
    rows = pd.concat(frames, ignore_index=True)

    rows["subject_id"] = rows["subject_id"].astype("int64")
    rows["micro_specimen_id"] = rows["micro_specimen_id"].astype("int64")
    rows["hadm_id"] = pd.to_numeric(rows["hadm_id"], errors="coerce").astype("Int64")
    rows["chartdate"] = pd.to_datetime(rows["chartdate"], errors="coerce")
    rows["charttime"] = pd.to_datetime(rows["charttime"], errors="coerce")
    rows["storedate"] = pd.to_datetime(rows["storedate"], errors="coerce")
    rows["storetime"] = pd.to_datetime(rows["storetime"], errors="coerce")

    rows["org_name_clean"] = rows["org_name"].map(normalize_org_name)
    rows["ab_name_clean"] = rows["ab_name"].fillna("").astype(str).str.strip().str.upper()
    rows["has_organism_row"] = rows["org_name_clean"].ne("").astype(int)
    rows["has_susceptibility_row"] = rows["ab_name_clean"].ne("").astype(int)
    rows["specimen_time"] = rows["charttime"].fillna(rows["chartdate"])
    rows["result_time"] = (
        rows["storetime"]
        .fillna(rows["storedate"])
        .fillna(rows["charttime"])
        .fillna(rows["chartdate"])
    )
    rows["positive_result_time_row"] = rows["result_time"].where(rows["has_organism_row"] == 1)
    rows["row_kind"] = np.where(
        rows["has_susceptibility_row"] == 1,
        "susceptibility",
        np.where(rows["has_organism_row"] == 1, "organism", "test_only"),
    )
    return rows


def _build_specimen_subset(rows: pd.DataFrame) -> pd.DataFrame:
    sorted_rows = rows.sort_values(["micro_specimen_id", "test_seq", "result_time", "charttime"])
    grouped = sorted_rows.groupby("micro_specimen_id", as_index=False, sort=False)

    subset = grouped.agg(
        subject_id=("subject_id", "first"),
        hadm_id=("hadm_id", "first"),
        order_provider_id=("order_provider_id", "first"),
        spec_itemid=("spec_itemid", "first"),
        spec_type_desc=("spec_type_desc", "first"),
        specimen_draw_time=("specimen_time", "min"),
        result_time=("result_time", "min"),
        row_count_total=("micro_specimen_id", "size"),
        test_only_row_count=("row_kind", lambda s: int((s == "test_only").sum())),
        organism_row_count=("has_organism_row", "sum"),
        susceptibility_row_count=("has_susceptibility_row", "sum"),
        unique_test_name_count=("test_name", lambda s: int(pd.Series(s.dropna()).nunique())),
        unique_org_count=("org_name_clean", lambda s: int(pd.Series(s[s != ""]).nunique())),
        unique_ab_name_count=("ab_name_clean", lambda s: int(pd.Series(s[s != ""]).nunique())),
        has_charttime=("charttime", lambda s: int(s.notna().any())),
        has_storetime=("storetime", lambda s: int(s.notna().any())),
        first_positive_result_time=("positive_result_time_row", "min"),
        first_test_name=("test_name", "first"),
        last_test_name=("test_name", "last"),
        test_names_json=("test_name", _json_sorted_unique),
        antibiotic_names_json=("ab_name_clean", _json_sorted_unique),
        organisms_json=("org_name_clean", lambda s: serialize_organisms(sorted({x for x in s if x}))),
    )

    subset["is_positive_culture"] = (subset["unique_org_count"] > 0).astype(int)
    subset["label_only_category"] = subset["organisms_json"].map(
        lambda text: specimen_category(json.loads(text)) if text != "[]" else "negative_or_no_growth"
    )
    subset["history_anchor_time"] = subset["first_positive_result_time"].fillna(subset["result_time"])
    subset["hours_draw_to_result"] = (
        (subset["result_time"] - subset["specimen_draw_time"]).dt.total_seconds() / 3600.0
    )
    subset["hours_draw_to_positive_result"] = (
        (subset["first_positive_result_time"] - subset["specimen_draw_time"]).dt.total_seconds() / 3600.0
    )
    return subset.sort_values(["subject_id", "history_anchor_time", "micro_specimen_id"]).reset_index(drop=True)


def _add_prior_history(subset: pd.DataFrame) -> pd.DataFrame:
    out = subset.copy()
    out["prior_blood_culture_specimens_24h"] = np.nan
    out["prior_blood_culture_specimens_7d"] = np.nan
    out["prior_positive_blood_culture_specimens_24h"] = np.nan
    out["prior_positive_blood_culture_specimens_7d"] = np.nan
    out["prior_positive_same_organism_7d"] = np.nan

    with_hadm = out[out["hadm_id"].notna() & out["history_anchor_time"].notna()].copy()
    for hadm_id, group in with_hadm.groupby("hadm_id", sort=False):
        group = group.sort_values(["history_anchor_time", "micro_specimen_id"]).copy()
        times = group["history_anchor_time"].to_numpy(dtype="datetime64[ns]")
        positive_flags = group["is_positive_culture"].to_numpy(dtype=int)
        organism_sets = [
            set(json.loads(text)) if text != "[]" else set()
            for text in group["organisms_json"].tolist()
        ]

        positive_prefix = np.concatenate([[0], np.cumsum(positive_flags)])
        prior_24h = np.zeros(len(group), dtype=int)
        prior_7d = np.zeros(len(group), dtype=int)
        prior_pos_24h = np.zeros(len(group), dtype=int)
        prior_pos_7d = np.zeros(len(group), dtype=int)
        prior_same_7d = np.zeros(len(group), dtype=int)

        left_24h = 0
        left_7d = 0
        time_groups: dict[np.datetime64, list[int]] = {}
        for idx, anchor in enumerate(times):
            time_groups.setdefault(anchor, []).append(idx)

        last_positive_time_by_org: dict[str, np.datetime64] = {}
        for anchor, indices in time_groups.items():
            anchor_24h = anchor - np.timedelta64(24, "h")
            anchor_7d = anchor - np.timedelta64(7, "D")
            while left_24h < len(times) and times[left_24h] < anchor_24h:
                left_24h += 1
            while left_7d < len(times) and times[left_7d] < anchor_7d:
                left_7d += 1

            prior_end = int(np.searchsorted(times, anchor, side="left"))
            all_24h = prior_end - left_24h
            all_7d = prior_end - left_7d
            pos_24h = int(positive_prefix[prior_end] - positive_prefix[left_24h])
            pos_7d = int(positive_prefix[prior_end] - positive_prefix[left_7d])

            for idx in indices:
                prior_24h[idx] = all_24h
                prior_7d[idx] = all_7d
                prior_pos_24h[idx] = pos_24h
                prior_pos_7d[idx] = pos_7d
                if organism_sets[idx]:
                    prior_same_7d[idx] = int(
                        any(
                            org in last_positive_time_by_org
                            and anchor - last_positive_time_by_org[org] <= np.timedelta64(7, "D")
                            for org in organism_sets[idx]
                        )
                    )

            for idx in indices:
                if positive_flags[idx] == 1:
                    for org in organism_sets[idx]:
                        last_positive_time_by_org[org] = anchor

        out.loc[group.index, "prior_blood_culture_specimens_24h"] = prior_24h
        out.loc[group.index, "prior_blood_culture_specimens_7d"] = prior_7d
        out.loc[group.index, "prior_positive_blood_culture_specimens_24h"] = prior_pos_24h
        out.loc[group.index, "prior_positive_blood_culture_specimens_7d"] = prior_pos_7d
        out.loc[group.index, "prior_positive_same_organism_7d"] = prior_same_7d

    return out


def _summary(rows: pd.DataFrame, subset: pd.DataFrame) -> dict[str, object]:
    positive = subset[subset["is_positive_culture"] == 1]
    return {
        "raw_blood_culture_rows": int(len(rows)),
        "distinct_specimens": int(subset["micro_specimen_id"].nunique()),
        "distinct_hadm": int(subset["hadm_id"].dropna().nunique()),
        "specimens_with_hadm_id": int(subset["hadm_id"].notna().sum()),
        "positive_specimens": int((subset["is_positive_culture"] == 1).sum()),
        "negative_or_no_growth_specimens": int((subset["is_positive_culture"] == 0).sum()),
        "specimens_with_susceptibility_rows": int((subset["susceptibility_row_count"] > 0).sum()),
        "median_rows_per_specimen": float(subset["row_count_total"].median()),
        "median_hours_draw_to_result": float(subset["hours_draw_to_result"].dropna().median()),
        "median_hours_draw_to_positive_result": float(positive["hours_draw_to_positive_result"].dropna().median()),
        "top_positive_categories": positive["label_only_category"].value_counts().head(10).to_dict(),
    }


def main() -> None:
    args = parse_args()
    config = ProjectConfig(project_root=args.project_root, raw_root=args.raw_root)
    out_dir = args.out_dir or (config.artifacts_dir / "blood_culture")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_blood_culture_rows(config, chunksize=args.chunksize)
    subset = _build_specimen_subset(rows)
    subset = _add_prior_history(subset)

    raw_csv = out_dir / "blood_culture_rows.csv"
    subset_csv = out_dir / "blood_culture_specimen_subset.csv"
    preview_csv = out_dir / "blood_culture_specimen_subset_preview.csv"
    summary_json = out_dir / "blood_culture_specimen_subset_summary.json"

    rows_out = rows.drop(columns=["org_name_clean", "ab_name_clean", "positive_result_time_row"])
    rows_out.to_csv(raw_csv, index=False)
    subset.to_csv(subset_csv, index=False)
    subset.head(args.preview_rows).to_csv(preview_csv, index=False)

    summary = _summary(rows, subset)
    summary["files"] = {
        "raw_rows_csv": str(raw_csv),
        "specimen_subset_csv": str(subset_csv),
        "preview_csv": str(preview_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
