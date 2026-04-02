from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ProjectConfig


GRAM_POSITIVE_TERMS = (
    "STAPH",
    "STREP",
    "ENTEROCOCC",
    "CORYNEBACTER",
    "BACILLUS",
    "CUTIBACTER",
    "PROPIONIBACTER",
    "MICROCOCC",
    "LISTERIA",
    "ACTINOMYC",
    "LACTOBACILL",
    "PEPTOSTREP",
    "GEMELLA",
    "AEROCOCC",
    "ROTHIA",
    "LEUCONOSTOC",
)

CONTAMINANT_PATTERNS = (
    "COAGULASE NEGATIVE",
    "STAPHYLOCOCCUS EPIDERMIDIS",
    "STAPHYLOCOCCUS HOMINIS",
    "STAPHYLOCOCCUS CAPITIS",
    "STAPHYLOCOCCUS WARNERI",
    "STAPHYLOCOCCUS HAEMOLYTICUS",
    "STAPHYLOCOCCUS PETTENKOFERI",
    "STAPHYLOCOCCUS COHNII",
    "STAPHYLOCOCCUS SIMULANS",
    "STAPHYLOCOCCUS SACCHAROLYTICUS",
    "CORYNEBACTERIUM",
    "BACILLUS",
    "CUTIBACTER",
    "PROPIONIBACTER",
    "MICROCOCCUS",
)

TRUE_PATHOGEN_PATTERNS = (
    "STAPH AUREUS",
    "STAPHYLOCOCCUS AUREUS",
    "STAPHYLOCOCCUS LUGDUNENSIS",
    "ENTEROCOCCUS",
    "STREPTOCOCCUS PNEUMONIAE",
    "STREPTOCOCCUS AGALACTIAE",
    "BETA STREPTOCOCCUS GROUP B",
    "STREPTOCOCCUS ANGINOSUS",
    "STREPTOCOCCUS CONSTELLATUS",
    "STREPTOCOCCUS INTERMEDIUS",
    "LISTERIA",
)


def normalize_org_name(value: object) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    if value in {"", "___", "CANCELLED"}:
        return ""
    return value


def is_gram_positive(org_name: str) -> bool:
    return any(term in org_name for term in GRAM_POSITIVE_TERMS)


def is_contaminant_like(org_name: str) -> bool:
    if "BACILLUS ANTHRACIS" in org_name:
        return False
    return any(term in org_name for term in CONTAMINANT_PATTERNS)


def is_true_pathogen_like(org_name: str) -> bool:
    return any(term in org_name for term in TRUE_PATHOGEN_PATTERNS)


def specimen_category(orgs: list[str]) -> str:
    has_gp = False
    has_contam = False
    has_true = False
    for org in orgs:
        if is_gram_positive(org):
            has_gp = True
        if is_contaminant_like(org):
            has_contam = True
        if is_true_pathogen_like(org):
            has_true = True
    if has_true and has_contam:
        return "mixed_gp"
    if has_true:
        return "true_like"
    if has_contam:
        return "contam_like"
    if has_gp:
        return "ambiguous_gp"
    return "non_gp"


def serialize_organisms(orgs: list[str]) -> str:
    return json.dumps(orgs, separators=(",", ":"))


def deserialize_organisms(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [str(item) for item in decoded]


def read_positive_blood_cultures(config: ProjectConfig, chunksize: int = 500_000) -> pd.DataFrame:
    path = config.hosp_dir / "microbiologyevents.csv"
    frames: list[pd.DataFrame] = []
    usecols = [
        "subject_id",
        "hadm_id",
        "micro_specimen_id",
        "chartdate",
        "charttime",
        "storedate",
        "storetime",
        "spec_type_desc",
        "org_name",
    ]

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["spec_type_desc"] == "BLOOD CULTURE"].copy()
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["hadm_id", "micro_specimen_id"]).copy()
        chunk["org_name"] = chunk["org_name"].map(normalize_org_name)
        chunk = chunk[chunk["org_name"] != ""].copy()
        if chunk.empty:
            continue
        chunk["storetime"] = pd.to_datetime(chunk["storetime"], errors="coerce")
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["storedate"] = pd.to_datetime(chunk["storedate"], errors="coerce")
        chunk["chartdate"] = pd.to_datetime(chunk["chartdate"], errors="coerce")
        chunk["result_time"] = (
            chunk["storetime"]
            .fillna(chunk["charttime"])
            .fillna(chunk["storedate"])
            .fillna(chunk["chartdate"])
        )
        chunk = chunk.dropna(subset=["result_time"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk["micro_specimen_id"] = chunk["micro_specimen_id"].astype("int64")
        chunk["subject_id"] = chunk["subject_id"].astype("int64")
        chunk["has_storetime"] = chunk["storetime"].notna().astype(int)
        chunk["has_charttime"] = chunk["charttime"].notna().astype(int)
        frames.append(
            chunk[
                [
                    "subject_id",
                    "hadm_id",
                    "micro_specimen_id",
                    "org_name",
                    "result_time",
                    "has_storetime",
                    "has_charttime",
                ]
            ]
        )

    if not frames:
        raise RuntimeError(f"No positive blood-culture rows found in {path}.")
    return pd.concat(frames, ignore_index=True)


def build_specimen_frame(positive_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = positive_rows.sort_values(["hadm_id", "micro_specimen_id", "result_time"]).groupby(
        ["hadm_id", "micro_specimen_id"], as_index=False
    )
    specimen = grouped.agg(
        subject_id=("subject_id", "first"),
        alert_time=("result_time", "min"),
        has_storetime=("has_storetime", "max"),
        has_charttime=("has_charttime", "max"),
        row_count=("org_name", "size"),
        unique_org_count=("org_name", lambda s: int(pd.Series(s).nunique())),
        organisms=("org_name", lambda s: sorted(set(s))),
    )
    specimen["category"] = specimen["organisms"].map(specimen_category)
    specimen["is_gp_candidate"] = specimen["category"].ne("non_gp").astype(int)
    return specimen.sort_values(["hadm_id", "alert_time", "micro_specimen_id"]).reset_index(drop=True)


def build_first_gram_positive_alerts(specimen: pd.DataFrame) -> pd.DataFrame:
    gp_specimen = specimen[specimen["is_gp_candidate"] == 1].copy()
    first_alerts = (
        gp_specimen.sort_values(["hadm_id", "alert_time", "micro_specimen_id"])
        .groupby("hadm_id", as_index=False)
        .first()
        .reset_index(drop=True)
    )
    return first_alerts


def flag_icu_at_alert(first_alerts: pd.DataFrame, config: ProjectConfig) -> pd.Series:
    icu = pd.read_csv(
        config.icu_dir / "icustays.csv.gz",
        usecols=["hadm_id", "intime", "outtime"],
        compression="gzip",
    )
    icu = icu.dropna(subset=["hadm_id", "intime", "outtime"]).copy()
    icu["hadm_id"] = icu["hadm_id"].astype("int64")
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = icu.dropna(subset=["intime", "outtime"])

    merged = first_alerts[["hadm_id", "alert_time"]].merge(icu, on="hadm_id", how="left")
    merged["in_icu_at_alert"] = (
        (merged["alert_time"] >= merged["intime"]) & (merged["alert_time"] <= merged["outtime"])
    ).fillna(False)
    flags = merged.groupby(["hadm_id", "alert_time"], as_index=False)["in_icu_at_alert"].max()
    return first_alerts.merge(flags, on=["hadm_id", "alert_time"], how="left")["in_icu_at_alert"].fillna(False)


def compute_repeat_features(first_alerts: pd.DataFrame, specimen: pd.DataFrame) -> pd.DataFrame:
    specimen_by_hadm: dict[int, list[dict[str, object]]] = {}
    for hadm_id, group in specimen.groupby("hadm_id"):
        specimen_by_hadm[int(hadm_id)] = group[
            ["micro_specimen_id", "alert_time", "organisms", "category"]
        ].to_dict("records")

    repeat_any_48h: list[int] = []
    repeat_gp_48h: list[int] = []
    repeat_same_org_48h: list[int] = []
    repeat_count_48h: list[int] = []

    for row in first_alerts.itertuples(index=False):
        later = []
        for candidate in specimen_by_hadm.get(int(row.hadm_id), []):
            if candidate["micro_specimen_id"] == row.micro_specimen_id:
                continue
            delta = candidate["alert_time"] - row.alert_time
            if pd.Timedelta(0) < delta <= pd.Timedelta(hours=48):
                later.append(candidate)

        first_orgs = set(row.organisms)
        repeat_any = int(bool(later))
        repeat_gp = int(any(candidate["category"] != "non_gp" for candidate in later))
        same_org = int(any(first_orgs.intersection(candidate["organisms"]) for candidate in later))

        repeat_any_48h.append(repeat_any)
        repeat_gp_48h.append(repeat_gp)
        repeat_same_org_48h.append(same_org)
        repeat_count_48h.append(len(later))

    enriched = first_alerts.copy()
    enriched["repeat_any_positive_48h"] = repeat_any_48h
    enriched["repeat_gp_positive_48h"] = repeat_gp_48h
    enriched["repeat_same_organism_48h"] = repeat_same_org_48h
    enriched["repeat_positive_specimen_count_48h"] = repeat_count_48h
    return enriched


def assign_provisional_label(first_alerts: pd.DataFrame) -> pd.DataFrame:
    labeled = first_alerts.copy()
    labeled["provisional_label"] = "indeterminate"
    labeled["label_source"] = "fallback_indeterminate"

    contaminant_mask = (
        (labeled["category"] == "contam_like")
        & (labeled["repeat_any_positive_48h"] == 0)
    )
    true_mask = (
        (labeled["category"] == "true_like")
        & (
            (labeled["repeat_any_positive_48h"] == 1)
            | (labeled["repeat_same_organism_48h"] == 1)
        )
    )

    labeled.loc[contaminant_mask, "provisional_label"] = "likely_contaminant"
    labeled.loc[contaminant_mask, "label_source"] = "contam_like_no_repeat_48h"
    labeled.loc[true_mask, "provisional_label"] = "likely_true_bsi"
    labeled.loc[true_mask, "label_source"] = "true_like_with_repeat_48h"
    labeled["is_high_confidence_binary"] = labeled["provisional_label"].isin(
        {"likely_contaminant", "likely_true_bsi"}
    ).astype(int)
    return labeled


def prepare_specimen_for_csv(specimen: pd.DataFrame) -> pd.DataFrame:
    out = specimen.copy()
    out["organisms_json"] = out["organisms"].map(serialize_organisms)
    return out.drop(columns=["organisms"])


def load_specimen_from_csv(path: Path) -> pd.DataFrame:
    specimen = pd.read_csv(path)
    specimen["alert_time"] = pd.to_datetime(specimen["alert_time"], errors="coerce")
    specimen["organisms"] = specimen["organisms_json"].map(deserialize_organisms)
    return specimen

