from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mimic_iv_project.config import ProjectConfig
from mimic_iv_project.metrics import binary_auprc, binary_auroc, binary_brier


WINDOWS = (24, 18)
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

LAB_PATTERNS = {
    "wbc": (r"^WBC$", r"^WBC Count$"),
    "platelets": (r"^Platelet Count$",),
    "creatinine": (r"^Creatinine$", r"^Creatinine, Serum$", r"^Creatinine, Blood$"),
    "lactate": (r"^Lactate$",),
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
    ("spo2", (r"^O2 saturation pulseoxymetry$", r"^SpO2$"), False),
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
        description=(
            "Build a first-blood-culture-event cohort and train 0-24h / 0-18h baselines "
            "for later-confirmed S. aureus."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--windows", type=int, nargs="+", default=[24, 18])
    return parser.parse_args()


def _load_ml_deps():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required. Install with `python3 -m pip install --user scikit-learn xgboost`."
        ) from exc

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is required. Install with `python3 -m pip install --user xgboost`.") from exc

    return {
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "xgb": xgb,
    }


def _iter_truncated_gzip_csv(path: Path, *, usecols: list[str], chunksize: int):
    command = f"gzip -dc {shlex.quote(str(path))} 2>/dev/null || true"
    process = subprocess.Popen(["bash", "-lc", command], stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError(f"Failed to stream {path}")
    try:
        reader = pd.read_csv(process.stdout, usecols=usecols, chunksize=chunksize)
        for chunk in reader:
            yield chunk
    finally:
        process.stdout.close()
        process.wait()


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


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=DATETIME_FORMAT, errors="coerce", cache=True)


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


def _resolve_icu_itemids(config: ProjectConfig, patterns: dict[str, tuple[str, ...]], linksto: str) -> dict[int, str]:
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


def _best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray, deps) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        f1 = deps["f1_score"](y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, deps) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "threshold": float(threshold),
        "f1": float(deps["f1_score"](y_true, y_pred, zero_division=0)),
        "precision": float(deps["precision_score"](y_true, y_pred, zero_division=0)),
        "recall": float(deps["recall_score"](y_true, y_pred, zero_division=0)),
        "accuracy": float(deps["accuracy_score"](y_true, y_pred)),
        "auroc": binary_auroc(y_true, y_prob),
        "auprc": binary_auprc(y_true, y_prob),
        "brier": binary_brier(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
    }


def _subject_split(subjects: np.ndarray, seed: int) -> tuple[set[int], set[int], set[int]]:
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(subjects), dtype=int)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train = set(subjects[:n_train].tolist())
    val = set(subjects[n_train : n_train + n_val].tolist())
    test = set(subjects[n_train + n_val :].tolist())
    return train, val, test


def _build_base_cohort(config: ProjectConfig) -> pd.DataFrame:
    specimen_path = config.artifacts_dir / "blood_culture" / "blood_culture_specimen_subset.csv"
    specimens = pd.read_csv(
        specimen_path,
        usecols=[
            "micro_specimen_id",
            "subject_id",
            "hadm_id",
            "specimen_draw_time",
            "history_anchor_time",
            "organisms_json",
        ],
    )
    specimens = specimens[specimens["hadm_id"].notna()].copy()
    specimens["hadm_id"] = specimens["hadm_id"].astype("int64")
    specimens["subject_id"] = specimens["subject_id"].astype("int64")
    specimens["specimen_draw_time"] = _parse_datetime(specimens["specimen_draw_time"])
    specimens["history_anchor_time"] = _parse_datetime(specimens["history_anchor_time"])
    specimens = specimens.dropna(subset=["specimen_draw_time"]).copy()

    first = (
        specimens.sort_values(["hadm_id", "specimen_draw_time", "micro_specimen_id"])
        .groupby("hadm_id", as_index=False)
        .first()
        .rename(columns={"specimen_draw_time": "anchor_time", "micro_specimen_id": "anchor_micro_specimen_id"})
    )

    specimens["has_s_aureus"] = specimens["organisms_json"].fillna("").astype(str).str.upper().str.contains("STAPH AUREUS")
    sa = specimens[specimens["has_s_aureus"]].copy()
    sa_first = (
        sa.sort_values(["hadm_id", "history_anchor_time", "micro_specimen_id"])
        .groupby("hadm_id", as_index=False)
        .first()[["hadm_id", "history_anchor_time", "micro_specimen_id"]]
        .rename(columns={"history_anchor_time": "first_s_aureus_result_time", "micro_specimen_id": "first_s_aureus_micro_specimen_id"})
    )

    cohort = first.merge(sa_first, on="hadm_id", how="left")
    cohort["target_s_aureus"] = cohort["first_s_aureus_result_time"].notna().astype(int)
    cohort["hours_to_s_aureus_result"] = (
        (cohort["first_s_aureus_result_time"] - cohort["anchor_time"]).dt.total_seconds() / 3600.0
    )

    patients = pd.read_csv(config.hosp_dir / "patients.csv", usecols=["subject_id", "gender", "anchor_age"])
    admissions = pd.read_csv(
        config.hosp_dir / "admissions.csv",
        usecols=["hadm_id", "admittime", "dischtime", "admission_type", "insurance", "race"],
    )
    admissions["admittime"] = _parse_datetime(admissions["admittime"])
    admissions["dischtime"] = _parse_datetime(admissions["dischtime"])
    cohort = cohort.merge(patients, on="subject_id", how="left")
    cohort = cohort.merge(admissions, on="hadm_id", how="left")
    cohort["anchor_hours_from_admit"] = (
        (cohort["anchor_time"] - cohort["admittime"]).dt.total_seconds() / 3600.0
    )
    cohort["anchor_weekend"] = (cohort["anchor_time"].dt.dayofweek >= 5).astype(int)
    cohort["anchor_night"] = (
        (cohort["anchor_time"].dt.hour < 7) | (cohort["anchor_time"].dt.hour >= 19)
    ).astype(int)
    cohort["race_group"] = cohort["race"].map(_group_race)
    cohort["insurance_group"] = cohort["insurance"].map(_group_insurance)

    icustays = pd.read_csv(
        config.icu_dir / "icustays.csv.gz",
        usecols=["hadm_id", "intime", "outtime"],
        compression="gzip",
    )
    icustays["hadm_id"] = pd.to_numeric(icustays["hadm_id"], errors="coerce").astype("Int64")
    icustays = icustays.dropna(subset=["hadm_id"]).copy()
    icustays["hadm_id"] = icustays["hadm_id"].astype("int64")
    icustays["intime"] = _parse_datetime(icustays["intime"])
    icustays["outtime"] = _parse_datetime(icustays["outtime"])
    icu_map: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for hadm_id, group in icustays.groupby("hadm_id", sort=False):
        spans = []
        for row in group.itertuples(index=False):
            if pd.isna(row.intime) or pd.isna(row.outtime):
                continue
            spans.append((row.intime, row.outtime))
        icu_map[int(hadm_id)] = spans

    in_icu = []
    for row in cohort.itertuples(index=False):
        spans = icu_map.get(int(row.hadm_id), [])
        in_icu.append(int(any(start <= row.anchor_time <= end for start, end in spans)))
    cohort["in_icu_at_anchor"] = in_icu
    return cohort


def _init_numeric_summary_dict(keys: set[int], feature_names: list[str]) -> dict[int, dict[str, dict[str, float | int | pd.Timestamp]]]:
    return {
        hadm_id: {feature_name: {} for feature_name in feature_names}
        for hadm_id in keys
    }


def _update_numeric_summary(summary: dict, hadm_id: int, feature_name: str, value: float, charttime: pd.Timestamp) -> None:
    record = summary[hadm_id].setdefault(feature_name, {})
    count = int(record.get("count", 0))
    if count == 0 or charttime >= record["last_time"]:
        record["last_time"] = charttime
        record["last"] = float(value)
    record["min"] = float(value) if count == 0 else min(float(record["min"]), float(value))
    record["max"] = float(value) if count == 0 else max(float(record["max"]), float(value))
    record["sum"] = float(value) if count == 0 else float(record["sum"]) + float(value)
    record["count"] = count + 1


def _apply_numeric_summaries(features: pd.DataFrame, summaries: dict, prefix: str, feature_names: list[str], suffix: str) -> pd.DataFrame:
    out = features.copy()
    for feature_name in feature_names:
        out[f"{prefix}_{feature_name}_last_{suffix}"] = np.nan
        out[f"{prefix}_{feature_name}_min_{suffix}"] = np.nan
        out[f"{prefix}_{feature_name}_max_{suffix}"] = np.nan
        out[f"{prefix}_{feature_name}_mean_{suffix}"] = np.nan
        out[f"{prefix}_{feature_name}_count_{suffix}"] = 0

    for idx, row in out.iterrows():
        hadm_id = int(row["hadm_id"])
        for feature_name in feature_names:
            record = summaries.get(hadm_id, {}).get(feature_name, {})
            if not record:
                continue
            out.at[idx, f"{prefix}_{feature_name}_last_{suffix}"] = record["last"]
            out.at[idx, f"{prefix}_{feature_name}_min_{suffix}"] = record["min"]
            out.at[idx, f"{prefix}_{feature_name}_max_{suffix}"] = record["max"]
            out.at[idx, f"{prefix}_{feature_name}_mean_{suffix}"] = float(record["sum"]) / int(record["count"])
            out.at[idx, f"{prefix}_{feature_name}_count_{suffix}"] = int(record["count"])
    return out


def _build_lab_summaries(base: pd.DataFrame, config: ProjectConfig) -> dict[int, dict]:
    item_map = _resolve_lab_itemids(config)
    cohorts = {
        window: base[base[f"eligible_{window}h"] == 1][["hadm_id", "anchor_time"]].copy()
        for window in WINDOWS
    }
    lookups = {
        window: cohorts[window].set_index("hadm_id")["anchor_time"].to_dict()
        for window in WINDOWS
    }
    hadm_sets = {window: set(cohorts[window]["hadm_id"].astype(int).tolist()) for window in WINDOWS}
    agg = {window: _init_numeric_summary_dict(hadm_sets[window], list(LAB_PATTERNS)) for window in WINDOWS}
    union_hadm = set().union(*hadm_sets.values())

    reader = pd.read_csv(
        config.hosp_dir / "labevents.csv",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(union_hadm)]
        chunk = chunk[chunk["itemid"].astype(int).isin(item_map)]
        if chunk.empty:
            continue
        chunk["charttime"] = _parse_datetime(chunk["charttime"])
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "valuenum"])
        if chunk.empty:
            continue
        chunk["feature_name"] = chunk["itemid"].astype(int).map(item_map)
        chunk = chunk.dropna(subset=["feature_name"])
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            hadm_id = int(row.hadm_id)
            for window in WINDOWS:
                if hadm_id not in hadm_sets[window]:
                    continue
                anchor_time = lookups[window][hadm_id]
                delta = (row.charttime - anchor_time).total_seconds() / 3600.0
                if 0 <= delta <= window:
                    _update_numeric_summary(agg[window], hadm_id, str(row.feature_name), float(row.valuenum), row.charttime)
    return agg


def _build_vital_summaries(base: pd.DataFrame, config: ProjectConfig) -> dict[int, dict]:
    item_map, fahrenheit_itemids = _resolve_vital_itemids(config)
    cohorts = {
        window: base[base[f"eligible_{window}h"] == 1][["hadm_id", "anchor_time"]].copy()
        for window in WINDOWS
    }
    lookups = {
        window: cohorts[window].set_index("hadm_id")["anchor_time"].to_dict()
        for window in WINDOWS
    }
    hadm_sets = {window: set(cohorts[window]["hadm_id"].astype(int).tolist()) for window in WINDOWS}
    agg = {window: _init_numeric_summary_dict(hadm_sets[window], list(VITAL_RANGES)) for window in WINDOWS}
    union_hadm = set().union(*hadm_sets.values())

    reader = _iter_truncated_gzip_csv(
        config.icu_dir / "chartevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(union_hadm)]
        chunk = chunk[chunk["itemid"].astype(int).isin(item_map)]
        if chunk.empty:
            continue
        chunk["charttime"] = _parse_datetime(chunk["charttime"])
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "valuenum"])
        if chunk.empty:
            continue
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk["feature_name"] = chunk["itemid"].map(item_map)
        if fahrenheit_itemids:
            temp_mask = chunk["itemid"].isin(fahrenheit_itemids)
            chunk.loc[temp_mask, "valuenum"] = (chunk.loc[temp_mask, "valuenum"] - 32.0) * (5.0 / 9.0)
        for feature_name, (min_value, max_value) in VITAL_RANGES.items():
            range_mask = chunk["feature_name"] == feature_name
            if range_mask.any():
                chunk = chunk[(~range_mask) | ((chunk["valuenum"] >= min_value) & (chunk["valuenum"] <= max_value))]
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            hadm_id = int(row.hadm_id)
            for window in WINDOWS:
                if hadm_id not in hadm_sets[window]:
                    continue
                anchor_time = lookups[window][hadm_id]
                delta = (row.charttime - anchor_time).total_seconds() / 3600.0
                if 0 <= delta <= window:
                    _update_numeric_summary(agg[window], hadm_id, str(row.feature_name), float(row.valuenum), row.charttime)
    return agg


def _build_antibiotic_summaries(base: pd.DataFrame, config: ProjectConfig) -> dict[int, dict[str, dict[str, int]]]:
    cohorts = {
        window: base[base[f"eligible_{window}h"] == 1][["hadm_id", "anchor_time"]].copy()
        for window in WINDOWS
    }
    lookups = {
        window: cohorts[window].set_index("hadm_id")["anchor_time"].to_dict()
        for window in WINDOWS
    }
    hadm_sets = {window: set(cohorts[window]["hadm_id"].astype(int).tolist()) for window in WINDOWS}
    agg = {
        window: {
            hadm_id: {
                "abx_total_admin": 0,
                "abx_vancomycin_iv_like": 0,
                "abx_linezolid": 0,
                "abx_daptomycin": 0,
                "abx_broad_gram_negative": 0,
            }
            for hadm_id in hadm_sets[window]
        }
        for window in WINDOWS
    }
    union_hadm = set().union(*hadm_sets.values())

    reader = pd.read_csv(
        config.hosp_dir / "emar.csv",
        usecols=["hadm_id", "charttime", "medication", "event_txt"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "charttime", "medication"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(union_hadm)]
        if chunk.empty:
            continue
        chunk["event_txt"] = chunk["event_txt"].fillna("").astype(str).str.lower()
        chunk = chunk[chunk["event_txt"].str.contains("admin", regex=False)]
        if chunk.empty:
            continue
        chunk["charttime"] = _parse_datetime(chunk["charttime"])
        chunk = chunk.dropna(subset=["charttime"])
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            flags = _classify_abx_flags(str(row.medication))
            if not any(flags.values()):
                continue
            hadm_id = int(row.hadm_id)
            for window in WINDOWS:
                if hadm_id not in hadm_sets[window]:
                    continue
                anchor_time = lookups[window][hadm_id]
                delta = (row.charttime - anchor_time).total_seconds() / 3600.0
                if 0 <= delta <= window:
                    agg[window][hadm_id]["abx_total_admin"] += 1
                    for name, value in flags.items():
                        if value:
                            agg[window][hadm_id][f"abx_{name}"] += 1
    return agg


def _build_vasopressor_summaries(base: pd.DataFrame, config: ProjectConfig) -> dict[int, dict[str, dict[str, int]]]:
    item_map = _resolve_icu_itemids(config, VASOPRESSOR_PATTERNS, linksto="inputevents")
    cohorts = {
        window: base[base[f"eligible_{window}h"] == 1][["hadm_id", "anchor_time"]].copy()
        for window in WINDOWS
    }
    lookups = {
        window: cohorts[window].set_index("hadm_id")["anchor_time"].to_dict()
        for window in WINDOWS
    }
    hadm_sets = {window: set(cohorts[window]["hadm_id"].astype(int).tolist()) for window in WINDOWS}
    agg = {
        window: {
            hadm_id: {
                "vasopressor_event_count": 0,
                "vasopressor_active": 0,
                "vasopressor_on_window_end": 0,
            }
            for hadm_id in hadm_sets[window]
        }
        for window in WINDOWS
    }
    union_hadm = set().union(*hadm_sets.values())

    reader = _iter_truncated_gzip_csv(
        config.icu_dir / "inputevents.csv.gz",
        usecols=["hadm_id", "itemid", "starttime", "endtime"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "starttime"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(union_hadm)]
        chunk = chunk[chunk["itemid"].astype(int).isin(item_map)]
        if chunk.empty:
            continue
        chunk["starttime"] = _parse_datetime(chunk["starttime"])
        chunk["endtime"] = _parse_datetime(chunk["endtime"])
        chunk = chunk.dropna(subset=["starttime"])
        if chunk.empty:
            continue
        chunk["endtime"] = chunk["endtime"].fillna(chunk["starttime"])
        for row in chunk.itertuples(index=False):
            hadm_id = int(row.hadm_id)
            for window in WINDOWS:
                if hadm_id not in hadm_sets[window]:
                    continue
                anchor_time = lookups[window][hadm_id]
                window_end = anchor_time + pd.Timedelta(hours=window)
                if row.starttime > window_end or row.endtime < anchor_time:
                    continue
                agg[window][hadm_id]["vasopressor_event_count"] += 1
                agg[window][hadm_id]["vasopressor_active"] = 1
                if row.starttime <= window_end <= row.endtime:
                    agg[window][hadm_id]["vasopressor_on_window_end"] = 1
    return agg


def _build_mech_vent_summaries(base: pd.DataFrame, config: ProjectConfig) -> dict[int, dict[str, dict[str, int]]]:
    item_map = _resolve_icu_itemids(config, MECH_VENT_PATTERNS, linksto="chartevents")
    cohorts = {
        window: base[base[f"eligible_{window}h"] == 1][["hadm_id", "anchor_time"]].copy()
        for window in WINDOWS
    }
    lookups = {
        window: cohorts[window].set_index("hadm_id")["anchor_time"].to_dict()
        for window in WINDOWS
    }
    hadm_sets = {window: set(cohorts[window]["hadm_id"].astype(int).tolist()) for window in WINDOWS}
    agg = {
        window: {
            hadm_id: {
                "mechanical_ventilation_chart_events": 0,
                "mechanical_ventilation": 0,
            }
            for hadm_id in hadm_sets[window]
        }
        for window in WINDOWS
    }
    union_hadm = set().union(*hadm_sets.values())

    reader = _iter_truncated_gzip_csv(
        config.icu_dir / "chartevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "value", "valuenum"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(union_hadm)]
        chunk = chunk[chunk["itemid"].astype(int).isin(item_map)]
        if chunk.empty:
            continue
        chunk["charttime"] = _parse_datetime(chunk["charttime"])
        chunk = chunk.dropna(subset=["charttime"])
        if chunk.empty:
            continue
        has_value = chunk["value"].notna() | chunk["valuenum"].notna()
        chunk = chunk[has_value].copy()
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            hadm_id = int(row.hadm_id)
            for window in WINDOWS:
                if hadm_id not in hadm_sets[window]:
                    continue
                anchor_time = lookups[window][hadm_id]
                delta = (row.charttime - anchor_time).total_seconds() / 3600.0
                if 0 <= delta <= window:
                    agg[window][hadm_id]["mechanical_ventilation_chart_events"] += 1
                    agg[window][hadm_id]["mechanical_ventilation"] = 1
    return agg


def _assemble_window_features(
    base: pd.DataFrame,
    window: int,
    lab_summaries: dict[int, dict],
    vital_summaries: dict[int, dict],
    abx_summaries: dict[int, dict],
    vaso_summaries: dict[int, dict],
    vent_summaries: dict[int, dict],
) -> tuple[pd.DataFrame, list[str]]:
    suffix = f"{window}h"
    out = base[base[f"eligible_{window}h"] == 1].copy().reset_index(drop=True)

    out = _apply_numeric_summaries(out, lab_summaries[window], "lab", list(LAB_PATTERNS), suffix)
    out = _apply_numeric_summaries(out, vital_summaries[window], "vital", list(VITAL_RANGES), suffix)

    for column in [
        "abx_total_admin",
        "abx_vancomycin_iv_like",
        "abx_linezolid",
        "abx_daptomycin",
        "abx_broad_gram_negative",
    ]:
        out[f"{column}_{suffix}"] = out["hadm_id"].astype(int).map(
            lambda hadm: abx_summaries[window][int(hadm)][column]
        )
        out[f"{column}_{suffix}_flag"] = (out[f"{column}_{suffix}"] > 0).astype(int)
    out[f"abx_anti_mrsa_{suffix}_flag"] = (
        (out[f"abx_vancomycin_iv_like_{suffix}_flag"] == 1)
        | (out[f"abx_linezolid_{suffix}_flag"] == 1)
        | (out[f"abx_daptomycin_{suffix}_flag"] == 1)
    ).astype(int)

    for column in ["vasopressor_event_count", "vasopressor_active", "vasopressor_on_window_end"]:
        out[f"{column}_{suffix}"] = out["hadm_id"].astype(int).map(
            lambda hadm: vaso_summaries[window][int(hadm)][column]
        )
    for column in ["mechanical_ventilation_chart_events", "mechanical_ventilation"]:
        out[f"{column}_{suffix}"] = out["hadm_id"].astype(int).map(
            lambda hadm: vent_summaries[window][int(hadm)][column]
        )

    categorical_cols = ["gender", "admission_type", "insurance_group", "race_group"]
    out = pd.get_dummies(out, columns=categorical_cols, dummy_na=False)

    excluded = {
        "hadm_id",
        "subject_id",
        "anchor_time",
        "anchor_micro_specimen_id",
        "history_anchor_time",
        "organisms_json",
        "first_s_aureus_result_time",
        "first_s_aureus_micro_specimen_id",
        "admittime",
        "dischtime",
        "insurance",
        "race",
        "target_s_aureus",
        "hours_to_s_aureus_result",
    }
    excluded |= {f"eligible_{w}h" for w in WINDOWS}
    feature_columns = [col for col in out.columns if col not in excluded]
    return out, feature_columns


def _train_models(features: pd.DataFrame, feature_columns: list[str], target_column: str, random_seed: int, n_estimators: int, max_depth: int):
    deps = _load_ml_deps()
    data = features.copy()
    train_subjects, val_subjects, test_subjects = _subject_split(
        data["subject_id"].drop_duplicates().astype(int).to_numpy(),
        seed=random_seed,
    )
    data["split"] = np.where(
        data["subject_id"].isin(train_subjects),
        "train",
        np.where(data["subject_id"].isin(val_subjects), "val", "test"),
    )

    x_train = data.loc[data["split"] == "train", feature_columns]
    x_val = data.loc[data["split"] == "val", feature_columns]
    x_test = data.loc[data["split"] == "test", feature_columns]
    y_train = data.loc[data["split"] == "train", target_column].to_numpy(dtype=int)
    y_val = data.loc[data["split"] == "val", target_column].to_numpy(dtype=int)
    y_test = data.loc[data["split"] == "test", target_column].to_numpy(dtype=int)

    logistic = deps["Pipeline"](
        steps=[
            (
                "prep",
                deps["ColumnTransformer"](
                    transformers=[
                        (
                            "num",
                            deps["Pipeline"](
                                steps=[
                                    ("imputer", deps["SimpleImputer"](strategy="median")),
                                    ("scaler", deps["StandardScaler"]()),
                                ]
                            ),
                            feature_columns,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", deps["LogisticRegression"](max_iter=2000, class_weight="balanced")),
        ]
    )
    logistic.fit(x_train, y_train)
    val_prob_lr = logistic.predict_proba(x_val)[:, 1]
    threshold_lr, best_val_f1_lr = _best_threshold_by_f1(y_val, val_prob_lr, deps)
    test_prob_lr = logistic.predict_proba(x_test)[:, 1]

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgb_model = deps["xgb"].XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        n_jobs=8,
    )
    xgb_model.fit(x_train, y_train)
    val_prob_xgb = xgb_model.predict_proba(x_val)[:, 1]
    threshold_xgb, best_val_f1_xgb = _best_threshold_by_f1(y_val, val_prob_xgb, deps)
    test_prob_xgb = xgb_model.predict_proba(x_test)[:, 1]

    return {
        "cohort": {
            "rows": int(len(data)),
            "unique_patients": int(data["subject_id"].nunique()),
            "unique_admissions": int(data["hadm_id"].nunique()),
            "positive_rows": int(data[target_column].sum()),
            "positive_prevalence": float(data[target_column].mean()),
            "train_rows": int((data["split"] == "train").sum()),
            "val_rows": int((data["split"] == "val").sum()),
            "test_rows": int((data["split"] == "test").sum()),
            "train_positive": int(y_train.sum()),
            "val_positive": int(y_val.sum()),
            "test_positive": int(y_test.sum()),
        },
        "models": {
            "logistic_regression": {
                "validation": {
                    **_classification_metrics(y_val, val_prob_lr, threshold_lr, deps),
                    "best_f1_scan": best_val_f1_lr,
                },
                "test": _classification_metrics(y_test, test_prob_lr, threshold_lr, deps),
            },
            "xgboost": {
                "validation": {
                    **_classification_metrics(y_val, val_prob_xgb, threshold_xgb, deps),
                    "best_f1_scan": best_val_f1_xgb,
                },
                "test": _classification_metrics(y_test, test_prob_xgb, threshold_xgb, deps),
            },
        },
    }


def _write_report(report_path: Path, metrics: dict, cohort_rows: int) -> None:
    lines = [
        "# `S. aureus` From First Blood-Culture Event",
        "",
        "## Clinical Question",
        "",
        "- using the first 24 hours or 18 hours of clinical data after the first blood-culture event in an admission, can we predict which admissions later have confirmed `Staphylococcus aureus` bacteremia?",
        "",
        "## Cohort",
        "",
        f"- total admissions with a first blood-culture anchor: `{cohort_rows:,}`",
    ]

    for window_key, payload in metrics["windows"].items():
        cohort = payload["cohort"]
        lines.extend(
            [
                f"- eligible for `{window_key}` window: `{cohort['rows']:,}` admissions",
                f"  - later-confirmed `S. aureus`: `{cohort['positive_rows']:,}` ({cohort['positive_prevalence']:.2%})",
            ]
        )

    lines.extend(["", "## Baseline Results", ""])
    for window_key, payload in metrics["windows"].items():
        lr = payload["models"]["logistic_regression"]["test"]
        xgb = payload["models"]["xgboost"]["test"]
        lines.extend(
            [
                f"### `{window_key}`",
                "",
                f"- Logistic Regression: AUROC `{lr['auroc']:.3f}`, AUPRC `{lr['auprc']:.3f}`, F1 `{lr['f1']:.3f}`",
                f"- XGBoost: AUROC `{xgb['auroc']:.3f}`, AUPRC `{xgb['auprc']:.3f}`, F1 `{xgb['f1']:.3f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- this task is much earlier than the first-Gram-positive-alert analysis",
            "- prevalence is much lower, so the class imbalance is harder",
            "- the 0-24h window is the main analysis because it has more complete early data",
            "- the 0-18h window is a sensitivity analysis for an earlier decision point",
            "",
            "## Next Features To Add",
            "",
            "- central-line and device features",
            "- prior MRSA or prior staphylococcal history if available safely",
            "- source clues from diagnoses, procedures, and targeted charting",
            "- richer microbiology context",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    global WINDOWS
    WINDOWS = tuple(dict.fromkeys(int(window) for window in args.windows))
    config = ProjectConfig(project_root=args.project_root, raw_root=args.raw_root)
    artifacts_dir = config.artifacts_dir / "blood_culture"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = config.project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    base = _build_base_cohort(config)
    for window in WINDOWS:
        base[f"eligible_{window}h"] = (
            base["dischtime"].notna()
            & ((base["dischtime"] - base["anchor_time"]).dt.total_seconds() / 3600.0 >= window)
        ).astype(int)

    cohort_csv = artifacts_dir / "s_aureus_first_blood_culture_cohort.csv"
    base.to_csv(cohort_csv, index=False)

    lab_summaries = _build_lab_summaries(base, config)
    vital_summaries = _build_vital_summaries(base, config)
    abx_summaries = _build_antibiotic_summaries(base, config)
    vaso_summaries = _build_vasopressor_summaries(base, config)
    vent_summaries = _build_mech_vent_summaries(base, config)

    window_payloads: dict[str, object] = {}
    for window in WINDOWS:
        features, feature_columns = _assemble_window_features(
            base=base,
            window=window,
            lab_summaries=lab_summaries,
            vital_summaries=vital_summaries,
            abx_summaries=abx_summaries,
            vaso_summaries=vaso_summaries,
            vent_summaries=vent_summaries,
        )
        feature_csv = artifacts_dir / f"s_aureus_first_blood_culture_{window}h_features.csv"
        features.to_csv(feature_csv, index=False)

        training = _train_models(
            features=features,
            feature_columns=feature_columns,
            target_column="target_s_aureus",
            random_seed=args.random_seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        window_payloads[f"{window}h"] = {
            **training,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "files": {
                "features_csv": str(feature_csv),
            },
        }

    metrics = {
        "task_name": "predict_later_confirmed_s_aureus_from_first_blood_culture_event",
        "clinical_question": (
            "Using the first 24 hours or 18 hours of clinical data after the first blood-culture event, "
            "can we predict which admissions later have confirmed Staphylococcus aureus bacteremia?"
        ),
        "cohort_definition": {
            "unit": "first blood-culture event per admission",
            "anchor_time": "earliest BLOOD CULTURE specimen_draw_time within the admission",
            "label": "any later-confirmed S. aureus bacteremia within the same admission",
            "eligibility": {
                "24h": "discharge at least 24h after anchor",
                "18h": "discharge at least 18h after anchor",
            },
        },
        "cohort_rows_total": int(len(base)),
        "windows": window_payloads,
        "files": {
            "cohort_csv": str(cohort_csv),
        },
    }

    metrics_path = reports_dir / "s_aureus_first_blood_culture_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _write_report(reports_dir / "s_aureus_first_blood_culture_report.md", metrics, cohort_rows=len(base))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
