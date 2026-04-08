from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_iv_project.blood_culture import (
    deserialize_organisms,
    is_gram_positive,
    load_specimen_from_csv,
)
from mimic_iv_project.config import ProjectConfig


LAB_PATTERNS = {
    "wbc": (r"^WBC$", r"^WBC Count$"),
    "hemoglobin": (r"^Hemoglobin$",),
    "platelets": (r"^Platelet Count$",),
    "creatinine": (r"^Creatinine$", r"^Creatinine, Serum$", r"^Creatinine, Blood$"),
    "lactate": (r"^Lactate$",),
    "sodium": (r"^Sodium$", r"^Sodium, Whole Blood$"),
    "potassium": (r"^Potassium$", r"^Potassium, Whole Blood$"),
}

VITAL_ITEM_SPECS = (
    ("heart_rate", (r"^Heart Rate$",), False),
    ("resp_rate", (r"^Respiratory Rate$",), False),
    ("temperature_c", (r"^Temperature Celsius$",), False),
    ("temperature_c", (r"^Temperature Fahrenheit$",), True),
    (
        "map",
        (
            r"^Arterial Blood Pressure mean$",
            r"^Non Invasive Blood Pressure mean$",
        ),
        False,
    ),
    ("spo2", (r"^O2 saturation pulseoxymetry$",), False),
)

VITAL_RANGES = {
    "heart_rate": (0.0, 250.0),
    "resp_rate": (0.0, 80.0),
    "temperature_c": (25.0, 45.0),
    "map": (20.0, 200.0),
    "spo2": (0.0, 100.0),
}

VASOPRESSOR_PATTERNS = {
    "vasopressor": (
        r"^Norepinephrine$",
        r"^Epinephrine$",
        r"^Phenylephrine$",
        r"^Dopamine$",
        r"^Vasopressin$",
        r"^Dobutamine$",
    ),
}

MECH_VENT_PATTERNS = {
    "mechanical_ventilation": (
        r"^Ventilator Mode$",
        r"^Ventilator Mode \(Hamilton\)$",
        r"^PEEP set$",
        r"^Tidal Volume \(set\)$",
        r"^Respiratory Rate \(Set\)$",
        r"^Minute Volume$",
        r"^Ventilator Type$",
    ),
}

ABX_PATTERNS = {
    "vancomycin_iv_like": re.compile(r"VANCOMYCIN", re.I),
    "linezolid": re.compile(r"LINEZOLID", re.I),
    "daptomycin": re.compile(r"DAPTOMYCIN", re.I),
    "broad_gram_negative": re.compile(
        r"PIPERACILLIN|TAZOBACTAM|ZOSYN|CEFEPIME|MEROPENEM|CEFTAZIDIME|IMIPENEM|AZTREONAM",
        re.I,
    ),
}

VANCOMYCIN_EXCLUDE = ("ORAL", "ENEMA", "LOCK", "OPHTH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pre-alert tabular features for first Gram-positive blood-culture alerts."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Lookback window for labs, vitals, antibiotics, and ICU support features.",
    )
    return parser.parse_args()


def _group_race(value: object) -> str:
    text = str(value or "").upper()
    if not text or text == "NAN":
        return "UNKNOWN"
    if "WHITE" in text:
        return "WHITE"
    if "BLACK" in text:
        return "BLACK"
    if "ASIAN" in text:
        return "ASIAN"
    if "HISPANIC" in text or "LATINO" in text:
        return "HISPANIC"
    return "OTHER"


def _group_insurance(value: object) -> str:
    text = str(value or "").upper()
    if not text or text == "NAN":
        return "UNKNOWN"
    if "MEDICARE" in text:
        return "MEDICARE"
    if "MEDICAID" in text:
        return "MEDICAID"
    if "PRIVATE" in text:
        return "PRIVATE"
    if "SELF PAY" in text:
        return "SELF_PAY"
    return "OTHER"


def _contains_any(orgs: list[str], terms: tuple[str, ...]) -> bool:
    return any(any(term in org for term in terms) for org in orgs)


def _organism_family_flags(orgs: list[str]) -> dict[str, int]:
    gp_orgs = [org for org in orgs if is_gram_positive(org)]
    flags = {
        "org_cons": int(
            _contains_any(
                gp_orgs,
                (
                    "COAGULASE NEGATIVE",
                    "STAPHYLOCOCCUS EPIDERMIDIS",
                    "STAPHYLOCOCCUS HOMINIS",
                    "STAPHYLOCOCCUS CAPITIS",
                    "STAPHYLOCOCCUS WARNERI",
                    "STAPHYLOCOCCUS HAEMOLYTICUS",
                    "STAPHYLOCOCCUS PETTENKOFERI",
                ),
            )
        ),
        "org_s_aureus": int(_contains_any(gp_orgs, ("STAPH AUREUS", "STAPHYLOCOCCUS AUREUS"))),
        "org_enterococcus": int(_contains_any(gp_orgs, ("ENTEROCOCCUS",))),
        "org_viridans_strep": int(_contains_any(gp_orgs, ("VIRIDANS STREPTOCOCCI",))),
        "org_beta_or_anginosus_strep": int(
            _contains_any(gp_orgs, ("GROUP B", "ANGINOSUS", "CONSTELLATUS", "INTERMEDIUS"))
        ),
        "org_corynebacterium": int(_contains_any(gp_orgs, ("CORYNEBACTERIUM",))),
        "org_bacillus": int(_contains_any(gp_orgs, ("BACILLUS",))),
        "org_cutibacterium_propionibacterium": int(
            _contains_any(gp_orgs, ("CUTIBACTER", "PROPIONIBACTER"))
        ),
        "org_lactobacillus": int(_contains_any(gp_orgs, ("LACTOBACILLUS",))),
        "org_polymicrobial_gp": int(len(gp_orgs) > 1),
    }
    flags["org_other_gp"] = int(any(gp_orgs) and not any(flags.values()))
    return flags


def _build_prior_culture_features(
    features: pd.DataFrame,
    specimen: pd.DataFrame,
) -> pd.DataFrame:
    specimen = specimen.copy()
    specimen["organisms"] = specimen["organisms_json"].map(deserialize_organisms)
    specimen_by_hadm: dict[int, list[dict[str, object]]] = {}
    for hadm_id, group in specimen.groupby("hadm_id"):
        specimen_by_hadm[int(hadm_id)] = group[
            ["alert_time", "category", "organisms"]
        ].sort_values("alert_time").to_dict("records")

    prior_24h = []
    prior_7d = []
    prior_gp_24h = []
    prior_gp_7d = []
    prior_same_org_7d = []
    for row in features.itertuples(index=False):
        current_orgs = set(deserialize_organisms(row.organisms_json))
        any_24 = 0
        any_7d = 0
        gp_24 = 0
        gp_7d = 0
        same_7d = 0
        for candidate in specimen_by_hadm.get(int(row.hadm_id), []):
            delta = row.alert_time - candidate["alert_time"]
            if delta <= pd.Timedelta(0):
                continue
            if delta <= pd.Timedelta(hours=24):
                any_24 += 1
                if candidate["category"] != "non_gp":
                    gp_24 += 1
            if delta <= pd.Timedelta(days=7):
                any_7d += 1
                if candidate["category"] != "non_gp":
                    gp_7d += 1
                    if current_orgs.intersection(candidate["organisms"]):
                        same_7d = 1
        prior_24h.append(any_24)
        prior_7d.append(any_7d)
        prior_gp_24h.append(gp_24)
        prior_gp_7d.append(gp_7d)
        prior_same_org_7d.append(same_7d)

    out = features.copy()
    out["prior_positive_specimens_24h"] = prior_24h
    out["prior_positive_specimens_7d"] = prior_7d
    out["prior_gp_positive_specimens_24h"] = prior_gp_24h
    out["prior_gp_positive_specimens_7d"] = prior_gp_7d
    out["prior_same_organism_positive_7d"] = prior_same_org_7d
    return out


def _resolve_lab_itemids(config: ProjectConfig) -> dict[int, str]:
    d_lab = pd.read_csv(config.hosp_dir / "d_labitems.csv", usecols=["itemid", "label"])
    item_map: dict[int, str] = {}
    for feature_name, patterns in LAB_PATTERNS.items():
        mask = pd.Series(False, index=d_lab.index)
        for pattern in patterns:
            mask |= d_lab["label"].astype(str).str.contains(pattern, regex=True, na=False)
        for itemid in d_lab.loc[mask, "itemid"].astype(int).tolist():
            item_map[itemid] = feature_name
    return item_map


def _resolve_icu_itemids(
    config: ProjectConfig,
    patterns: dict[str, tuple[str, ...]],
    linksto: str,
) -> dict[int, str]:
    d_items = pd.read_csv(
        config.icu_dir / "d_items.csv.gz",
        usecols=["itemid", "label", "linksto"],
        compression="gzip",
    )
    d_items["linksto"] = d_items["linksto"].astype(str)
    item_map: dict[int, str] = {}
    for feature_name, feature_patterns in patterns.items():
        mask = pd.Series(False, index=d_items.index)
        for pattern in feature_patterns:
            mask |= (
                d_items["linksto"].eq(linksto)
                & d_items["label"].astype(str).str.contains(pattern, regex=True, na=False)
            )
        for itemid in d_items.loc[mask, "itemid"].astype(int).tolist():
            item_map[itemid] = feature_name
    return item_map


def _resolve_vital_itemids(config: ProjectConfig) -> tuple[dict[int, str], set[int]]:
    d_items = pd.read_csv(
        config.icu_dir / "d_items.csv.gz",
        usecols=["itemid", "label", "linksto"],
        compression="gzip",
    )
    d_items = d_items[d_items["linksto"].astype(str) == "chartevents"].copy()
    item_map: dict[int, str] = {}
    fahrenheit_itemids: set[int] = set()
    for feature_name, feature_patterns, convert_f_to_c in VITAL_ITEM_SPECS:
        mask = pd.Series(False, index=d_items.index)
        for pattern in feature_patterns:
            mask |= d_items["label"].astype(str).str.contains(pattern, regex=True, na=False)
        itemids = d_items.loc[mask, "itemid"].astype(int).tolist()
        for itemid in itemids:
            item_map[itemid] = feature_name
        if convert_f_to_c:
            fahrenheit_itemids.update(itemids)
    return item_map, fahrenheit_itemids


def _iter_truncated_gzip_csv(
    path: Path,
    *,
    usecols: list[str],
    chunksize: int,
):
    command = f"gzip -dc {shlex.quote(str(path))} 2>/dev/null || true"
    process = subprocess.Popen(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError(f"Failed to stream {path}")
    try:
        reader = pd.read_csv(process.stdout, usecols=usecols, chunksize=chunksize)
        for chunk in reader:
            yield chunk
    finally:
        process.stdout.close()
        process.wait()


def _build_lab_features(
    features: pd.DataFrame,
    config: ProjectConfig,
    lookback_hours: int,
) -> pd.DataFrame:
    item_map = _resolve_lab_itemids(config)
    if not item_map:
        return features

    cohort_lookup = features[["hadm_id", "alert_time"]].copy()
    hadm_set = set(cohort_lookup["hadm_id"].astype(int).tolist())
    agg: dict[tuple[int, str], dict[str, object]] = {}

    reader = pd.read_csv(
        config.hosp_dir / "labevents.csv",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        chunk = chunk[chunk["itemid"].astype(int).isin(item_map)]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])
        if chunk.empty:
            continue
        chunk = chunk.merge(cohort_lookup, on="hadm_id", how="inner")
        chunk = chunk[
            (chunk["charttime"] <= chunk["alert_time"])
            & (chunk["charttime"] >= chunk["alert_time"] - pd.Timedelta(hours=lookback_hours))
        ].copy()
        if chunk.empty:
            continue
        chunk["feature_name"] = chunk["itemid"].astype(int).map(item_map)
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["valuenum", "feature_name"])
        if chunk.empty:
            continue

        chunk = chunk.sort_values(["hadm_id", "feature_name", "charttime"])
        grouped = chunk.groupby(["hadm_id", "feature_name"], as_index=False)
        summary = grouped.agg(
            last_time=("charttime", "max"),
            last_value=("valuenum", "last"),
            min_value=("valuenum", "min"),
            max_value=("valuenum", "max"),
            mean_value=("valuenum", "mean"),
            count_value=("valuenum", "size"),
        )
        for row in summary.itertuples(index=False):
            key = (int(row.hadm_id), str(row.feature_name))
            record = agg.get(key)
            if record is None:
                agg[key] = {
                    "last_time": row.last_time,
                    "last_value": float(row.last_value),
                    "min_value": float(row.min_value),
                    "max_value": float(row.max_value),
                    "mean_sum": float(row.mean_value) * int(row.count_value),
                    "count_value": int(row.count_value),
                }
            else:
                if row.last_time > record["last_time"]:
                    record["last_time"] = row.last_time
                    record["last_value"] = float(row.last_value)
                record["min_value"] = min(float(row.min_value), float(record["min_value"]))
                record["max_value"] = max(float(row.max_value), float(record["max_value"]))
                record["mean_sum"] += float(row.mean_value) * int(row.count_value)
                record["count_value"] += int(row.count_value)

    out = features.copy()
    for feature_name in LAB_PATTERNS:
        out[f"lab_{feature_name}_last_24h"] = np.nan
        out[f"lab_{feature_name}_min_24h"] = np.nan
        out[f"lab_{feature_name}_max_24h"] = np.nan
        out[f"lab_{feature_name}_mean_24h"] = np.nan
        out[f"lab_{feature_name}_count_24h"] = 0

    for idx, row in out.iterrows():
        hadm_id = int(row["hadm_id"])
        for feature_name in LAB_PATTERNS:
            record = agg.get((hadm_id, feature_name))
            if record is None:
                continue
            out.at[idx, f"lab_{feature_name}_last_24h"] = record["last_value"]
            out.at[idx, f"lab_{feature_name}_min_24h"] = record["min_value"]
            out.at[idx, f"lab_{feature_name}_max_24h"] = record["max_value"]
            out.at[idx, f"lab_{feature_name}_mean_24h"] = record["mean_sum"] / record["count_value"]
            out.at[idx, f"lab_{feature_name}_count_24h"] = record["count_value"]
    return out


def _build_vital_features(
    features: pd.DataFrame,
    config: ProjectConfig,
    lookback_hours: int,
) -> pd.DataFrame:
    item_map, fahrenheit_itemids = _resolve_vital_itemids(config)
    if not item_map:
        return features

    cohort_lookup = features[["hadm_id", "alert_time"]].copy()
    hadm_set = set(cohort_lookup["hadm_id"].astype(int).tolist())
    agg: dict[tuple[int, str], dict[str, object]] = {}

    reader = _iter_truncated_gzip_csv(
        config.icu_dir / "chartevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        chunk = chunk[chunk["itemid"].astype(int).isin(item_map)]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])
        if chunk.empty:
            continue
        chunk = chunk.merge(cohort_lookup, on="hadm_id", how="inner")
        chunk = chunk[
            (chunk["charttime"] <= chunk["alert_time"])
            & (chunk["charttime"] >= chunk["alert_time"] - pd.Timedelta(hours=lookback_hours))
        ].copy()
        if chunk.empty:
            continue
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk["feature_name"] = chunk["itemid"].map(item_map)
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["valuenum", "feature_name"])
        if chunk.empty:
            continue
        if fahrenheit_itemids:
            temp_mask = chunk["itemid"].isin(fahrenheit_itemids)
            chunk.loc[temp_mask, "valuenum"] = (chunk.loc[temp_mask, "valuenum"] - 32.0) * (5.0 / 9.0)
        for feature_name, (min_value, max_value) in VITAL_RANGES.items():
            range_mask = chunk["feature_name"] == feature_name
            if range_mask.any():
                chunk = chunk[
                    (~range_mask)
                    | (
                        (chunk["valuenum"] >= min_value)
                        & (chunk["valuenum"] <= max_value)
                    )
                ]
        if chunk.empty:
            continue

        chunk = chunk.sort_values(["hadm_id", "feature_name", "charttime"])
        grouped = chunk.groupby(["hadm_id", "feature_name"], as_index=False)
        summary = grouped.agg(
            last_time=("charttime", "max"),
            last_value=("valuenum", "last"),
            min_value=("valuenum", "min"),
            max_value=("valuenum", "max"),
            mean_value=("valuenum", "mean"),
            count_value=("valuenum", "size"),
        )
        for row in summary.itertuples(index=False):
            key = (int(row.hadm_id), str(row.feature_name))
            record = agg.get(key)
            if record is None:
                agg[key] = {
                    "last_time": row.last_time,
                    "last_value": float(row.last_value),
                    "min_value": float(row.min_value),
                    "max_value": float(row.max_value),
                    "mean_sum": float(row.mean_value) * int(row.count_value),
                    "count_value": int(row.count_value),
                }
            else:
                if row.last_time > record["last_time"]:
                    record["last_time"] = row.last_time
                    record["last_value"] = float(row.last_value)
                record["min_value"] = min(float(row.min_value), float(record["min_value"]))
                record["max_value"] = max(float(row.max_value), float(record["max_value"]))
                record["mean_sum"] += float(row.mean_value) * int(row.count_value)
                record["count_value"] += int(row.count_value)

    out = features.copy()
    for feature_name in VITAL_RANGES:
        out[f"vital_{feature_name}_last_24h"] = np.nan
        out[f"vital_{feature_name}_min_24h"] = np.nan
        out[f"vital_{feature_name}_max_24h"] = np.nan
        out[f"vital_{feature_name}_mean_24h"] = np.nan
        out[f"vital_{feature_name}_count_24h"] = 0

    for idx, row in out.iterrows():
        hadm_id = int(row["hadm_id"])
        for feature_name in VITAL_RANGES:
            record = agg.get((hadm_id, feature_name))
            if record is None:
                continue
            out.at[idx, f"vital_{feature_name}_last_24h"] = record["last_value"]
            out.at[idx, f"vital_{feature_name}_min_24h"] = record["min_value"]
            out.at[idx, f"vital_{feature_name}_max_24h"] = record["max_value"]
            out.at[idx, f"vital_{feature_name}_mean_24h"] = record["mean_sum"] / record["count_value"]
            out.at[idx, f"vital_{feature_name}_count_24h"] = record["count_value"]
    return out


def _classify_abx_flags(medication: str) -> dict[str, int]:
    med = medication.upper()
    flags = {name: 0 for name in ABX_PATTERNS}
    if ABX_PATTERNS["vancomycin_iv_like"].search(med):
        if not any(token in med for token in VANCOMYCIN_EXCLUDE):
            flags["vancomycin_iv_like"] = 1
    if ABX_PATTERNS["linezolid"].search(med):
        flags["linezolid"] = 1
    if ABX_PATTERNS["daptomycin"].search(med):
        flags["daptomycin"] = 1
    if ABX_PATTERNS["broad_gram_negative"].search(med):
        flags["broad_gram_negative"] = 1
    return flags


def _build_antibiotic_features(
    features: pd.DataFrame,
    config: ProjectConfig,
    lookback_hours: int,
) -> pd.DataFrame:
    cohort_lookup = features[["hadm_id", "alert_time"]].copy()
    hadm_set = set(cohort_lookup["hadm_id"].astype(int).tolist())
    counts = {
        int(hadm): {
            "abx_total_admin_24h": 0,
            "abx_vancomycin_iv_like_24h": 0,
            "abx_linezolid_24h": 0,
            "abx_daptomycin_24h": 0,
            "abx_broad_gram_negative_24h": 0,
        }
        for hadm in hadm_set
    }

    reader = pd.read_csv(
        config.hosp_dir / "emar.csv",
        usecols=["hadm_id", "charttime", "medication", "event_txt"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "charttime", "medication"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        if chunk.empty:
            continue
        chunk["event_txt"] = chunk["event_txt"].fillna("").astype(str).str.lower()
        chunk = chunk[chunk["event_txt"].str.contains("admin", regex=False)]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])
        if chunk.empty:
            continue
        chunk = chunk.merge(cohort_lookup, on="hadm_id", how="inner")
        chunk = chunk[
            (chunk["charttime"] <= chunk["alert_time"])
            & (chunk["charttime"] >= chunk["alert_time"] - pd.Timedelta(hours=lookback_hours))
        ].copy()
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            flags = _classify_abx_flags(str(row.medication))
            if not any(flags.values()):
                continue
            counts[int(row.hadm_id)]["abx_total_admin_24h"] += 1
            for name, value in flags.items():
                if value:
                    counts[int(row.hadm_id)][f"abx_{name}_24h"] += 1

    out = features.copy()
    for column in [
        "abx_total_admin_24h",
        "abx_vancomycin_iv_like_24h",
        "abx_linezolid_24h",
        "abx_daptomycin_24h",
        "abx_broad_gram_negative_24h",
    ]:
        out[column] = out["hadm_id"].astype(int).map(lambda hadm: counts[int(hadm)][column])
        out[f"{column}_flag"] = (out[column] > 0).astype(int)
    out["abx_anti_mrsa_24h_flag"] = (
        (out["abx_vancomycin_iv_like_24h_flag"] == 1)
        | (out["abx_linezolid_24h_flag"] == 1)
        | (out["abx_daptomycin_24h_flag"] == 1)
    ).astype(int)
    return out


def _build_vasopressor_features(
    features: pd.DataFrame,
    config: ProjectConfig,
    lookback_hours: int,
) -> pd.DataFrame:
    item_map = _resolve_icu_itemids(config, VASOPRESSOR_PATTERNS, linksto="inputevents")
    if not item_map:
        return features

    cohort_lookup = features[["hadm_id", "alert_time"]].copy()
    hadm_set = set(cohort_lookup["hadm_id"].astype(int).tolist())
    counts = {
        int(hadm): {
            "vasopressor_event_count_24h": 0,
            "vasopressor_active_24h": 0,
            "vasopressor_on_at_alert": 0,
        }
        for hadm in hadm_set
    }

    reader = _iter_truncated_gzip_csv(
        config.icu_dir / "inputevents.csv.gz",
        usecols=["hadm_id", "itemid", "starttime", "endtime"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "starttime"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        chunk = chunk[chunk["itemid"].astype(int).isin(item_map)]
        if chunk.empty:
            continue
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["endtime"] = pd.to_datetime(chunk["endtime"], errors="coerce")
        chunk = chunk.dropna(subset=["starttime"])
        if chunk.empty:
            continue
        chunk["endtime"] = chunk["endtime"].fillna(chunk["starttime"])
        chunk = chunk.merge(cohort_lookup, on="hadm_id", how="inner")
        window_start = chunk["alert_time"] - pd.Timedelta(hours=lookback_hours)
        chunk = chunk[
            (chunk["starttime"] <= chunk["alert_time"])
            & (chunk["endtime"] >= window_start)
        ].copy()
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            hadm_id = int(row.hadm_id)
            counts[hadm_id]["vasopressor_event_count_24h"] += 1
            counts[hadm_id]["vasopressor_active_24h"] = 1
            if row.starttime <= row.alert_time <= row.endtime:
                counts[hadm_id]["vasopressor_on_at_alert"] = 1

    out = features.copy()
    for column in [
        "vasopressor_event_count_24h",
        "vasopressor_active_24h",
        "vasopressor_on_at_alert",
    ]:
        out[column] = out["hadm_id"].astype(int).map(lambda hadm: counts[int(hadm)][column])
    return out


def _build_mechanical_ventilation_features(
    features: pd.DataFrame,
    config: ProjectConfig,
    lookback_hours: int,
) -> pd.DataFrame:
    item_map = _resolve_icu_itemids(config, MECH_VENT_PATTERNS, linksto="chartevents")
    if not item_map:
        return features

    cohort_lookup = features[["hadm_id", "alert_time"]].copy()
    hadm_set = set(cohort_lookup["hadm_id"].astype(int).tolist())
    counts = {
        int(hadm): {
            "mechanical_ventilation_chart_events_24h": 0,
            "mechanical_ventilation_24h": 0,
        }
        for hadm in hadm_set
    }

    reader = _iter_truncated_gzip_csv(
        config.icu_dir / "chartevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "value", "valuenum"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        chunk = chunk[chunk["itemid"].astype(int).isin(item_map)]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])
        if chunk.empty:
            continue
        has_value = chunk["value"].notna() | chunk["valuenum"].notna()
        chunk = chunk[has_value].copy()
        if chunk.empty:
            continue
        chunk = chunk.merge(cohort_lookup, on="hadm_id", how="inner")
        chunk = chunk[
            (chunk["charttime"] <= chunk["alert_time"])
            & (chunk["charttime"] >= chunk["alert_time"] - pd.Timedelta(hours=lookback_hours))
        ].copy()
        if chunk.empty:
            continue

        grouped = chunk.groupby("hadm_id").size().to_dict()
        for hadm_id, count in grouped.items():
            counts[int(hadm_id)]["mechanical_ventilation_chart_events_24h"] += int(count)
            counts[int(hadm_id)]["mechanical_ventilation_24h"] = 1

    out = features.copy()
    for column in [
        "mechanical_ventilation_chart_events_24h",
        "mechanical_ventilation_24h",
    ]:
        out[column] = out["hadm_id"].astype(int).map(lambda hadm: counts[int(hadm)][column])
    return out


def main() -> None:
    args = parse_args()
    config = ProjectConfig(project_root=args.project_root, raw_root=args.raw_root)
    out_dir = args.out_dir or (config.artifacts_dir / "blood_culture")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "first_gp_alert_dataset.csv"
    specimen_path = out_dir / "positive_blood_culture_specimens.csv"
    features_csv = out_dir / "first_gp_alert_features.csv"
    metadata_json = out_dir / "blood_culture_feature_metadata.json"

    features = pd.read_csv(dataset_path)
    features["alert_time"] = pd.to_datetime(features["alert_time"], errors="coerce")

    patients = pd.read_csv(config.hosp_dir / "patients.csv", usecols=["subject_id", "gender", "anchor_age"])
    admissions = pd.read_csv(
        config.hosp_dir / "admissions.csv",
        usecols=["hadm_id", "admittime", "admission_type", "insurance", "race"],
    )
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")

    features = features.merge(patients, on="subject_id", how="left")
    features = features.merge(admissions, on="hadm_id", how="left")
    features["alert_hours_from_admit"] = (
        (features["alert_time"] - features["admittime"]).dt.total_seconds() / 3600.0
    )
    features["alert_weekend"] = (features["alert_time"].dt.dayofweek >= 5).astype(int)
    features["alert_night"] = (
        (features["alert_time"].dt.hour < 7) | (features["alert_time"].dt.hour >= 19)
    ).astype(int)
    features["race_group"] = features["race"].map(_group_race)
    features["insurance_group"] = features["insurance"].map(_group_insurance)

    org_flags = features["organisms_json"].map(lambda text: _organism_family_flags(deserialize_organisms(text)))
    org_flags_df = pd.DataFrame(org_flags.tolist())
    features = pd.concat([features, org_flags_df], axis=1)

    specimen = load_specimen_from_csv(specimen_path)
    features = _build_prior_culture_features(features, specimen)
    features = _build_lab_features(features, config, lookback_hours=args.lookback_hours)
    features = _build_vital_features(features, config, lookback_hours=args.lookback_hours)
    features = _build_antibiotic_features(features, config, lookback_hours=args.lookback_hours)
    features = _build_vasopressor_features(features, config, lookback_hours=args.lookback_hours)
    features = _build_mechanical_ventilation_features(features, config, lookback_hours=args.lookback_hours)

    features["target_true_bsi"] = np.where(
        features["provisional_label"] == "probable_clinically_significant_bsi_alert",
        1,
        np.where(
            features["provisional_label"] == "probable_contaminant_or_low_significance_alert",
            0,
            np.nan,
        ),
    )

    categorical_cols = ["gender", "admission_type", "insurance_group", "race_group"]
    features = pd.get_dummies(features, columns=categorical_cols, dummy_na=False)

    excluded = {
        "hadm_id",
        "subject_id",
        "micro_specimen_id",
        "alert_time",
        "admittime",
        "insurance",
        "race",
        "category",
        "is_gp_candidate",
        "repeat_any_positive_48h",
        "repeat_gp_positive_48h",
        "repeat_same_organism_48h",
        "repeat_positive_specimen_count_48h",
        "episode_blood_culture_specimens_24h",
        "additional_blood_culture_specimens_24h",
        "multiple_sets_approx_24h",
        "systemic_abx_admin_0_24h",
        "anti_mrsa_admin_0_24h",
        "systemic_abx_admin_24_72h",
        "anti_mrsa_admin_24_72h",
        "continued_systemic_abx_24_72h",
        "continued_anti_mrsa_24_72h",
        "is_high_confidence_binary",
        "organisms_json",
        "provisional_label",
        "label_source",
        "target_true_bsi",
        "row_count",
        "unique_org_count",
        "prior_same_organism_positive_7d",
    }
    feature_columns = [col for col in features.columns if col not in excluded]
    organism_feature_columns = [
        col for col in feature_columns if col.startswith("org_")
    ]

    features.to_csv(features_csv, index=False)
    metadata = {
        "rows": int(len(features)),
        "lookback_hours": int(args.lookback_hours),
        "feature_columns": feature_columns,
        "organism_feature_columns": organism_feature_columns,
        "target_column": "target_true_bsi",
        "high_confidence_rows": int(features["is_high_confidence_binary"].sum()),
        "files": {
            "features_csv": str(features_csv),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
