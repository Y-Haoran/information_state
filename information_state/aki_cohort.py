"""AKI-specific cohort annotation utilities.

This module adds a KDIGO-style AKI stay filter without changing the
State-from-Observation model itself. The implementation focuses on
serum-creatinine and urine-output criteria available in MIMIC-IV.
It does not currently model renal replacement therapy as an AKI stage-3 trigger.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ProjectConfig


def _read_csv(path: Path, usecols=None, chunksize=None):
    compression = "gzip" if str(path).endswith(".gz") else None
    return pd.read_csv(path, usecols=usecols, compression=compression, chunksize=chunksize)


def _feature_itemids(catalog: dict, feature_name: str) -> list[int]:
    for feature in catalog["features"]:
        if feature["name"] == feature_name:
            return [int(itemid) for itemid in feature["itemids"]]
    return []


def _empty_aki_result() -> dict[str, float | int | None]:
    return {
        "aki": 0,
        "aki_stage": 0,
        "aki_creatinine": 0,
        "aki_stage_creatinine": 0,
        "aki_urine_output": 0,
        "aki_stage_urine_output": 0,
        "aki_onset_hour": np.nan,
        "aki_onset_bin": np.nan,
        "aki_onset_hour_creatinine": np.nan,
        "aki_onset_hour_urine_output": np.nan,
        "aki_baseline_creatinine": np.nan,
        "aki_weight_kg": np.nan,
    }


def detect_aki_from_creatinine(
    events: pd.DataFrame,
    *,
    stay_start: pd.Timestamp,
    stay_end: pd.Timestamp,
) -> dict[str, float | int | None]:
    """Apply KDIGO-like creatinine criteria to one stay's events.

    Expected columns:
    - ``charttime``
    - ``creatinine``
    """

    result = _empty_aki_result()
    if events.empty:
        return result

    creatinine = events[["charttime", "creatinine"]].copy()
    creatinine["charttime"] = pd.to_datetime(creatinine["charttime"], errors="coerce")
    creatinine["creatinine"] = pd.to_numeric(creatinine["creatinine"], errors="coerce")
    creatinine = creatinine.dropna(subset=["charttime", "creatinine"])
    creatinine = creatinine[creatinine["creatinine"] > 0.0].sort_values("charttime").reset_index(drop=True)
    if creatinine.empty:
        return result

    lookback_baseline = creatinine.loc[creatinine["charttime"] < stay_start, "creatinine"]
    if not lookback_baseline.empty:
        result["aki_baseline_creatinine"] = float(lookback_baseline.min())

    times = creatinine["charttime"].to_numpy()
    values = creatinine["creatinine"].to_numpy(dtype=np.float64)

    onset_hour = np.nan
    max_stage = 0
    for event_index, event_time in enumerate(times):
        event_time = pd.Timestamp(event_time)
        if event_time < stay_start or event_time > stay_end:
            continue

        prior_times = times[:event_index]
        prior_values = values[:event_index]
        stage = 0

        if len(prior_times):
            delta_hours = (event_time - pd.to_datetime(prior_times)).total_seconds() / 3600.0

            baseline_48 = prior_values[delta_hours <= 48.0]
            if len(baseline_48):
                creatinine_delta = float(values[event_index] - baseline_48.min())
                if creatinine_delta >= 0.3:
                    stage = max(stage, 3 if values[event_index] >= 4.0 else 1)

            baseline_7d = prior_values[delta_hours <= (24.0 * 7.0)]
            if len(baseline_7d):
                baseline_value = float(baseline_7d.min())
                if baseline_value > 0.0:
                    ratio = float(values[event_index] / baseline_value)
                    if ratio >= 3.0:
                        stage = max(stage, 3)
                    elif ratio >= 2.0:
                        stage = max(stage, 2)
                    elif ratio >= 1.5:
                        stage = max(stage, 1)
                    if np.isnan(result["aki_baseline_creatinine"]):
                        result["aki_baseline_creatinine"] = baseline_value

        if stage > 0:
            if np.isnan(onset_hour):
                onset_hour = max(0.0, (event_time - stay_start).total_seconds() / 3600.0)
            max_stage = max(max_stage, stage)

    if max_stage == 0:
        return result

    result["aki"] = 1
    result["aki_creatinine"] = 1
    result["aki_stage"] = max_stage
    result["aki_stage_creatinine"] = max_stage
    result["aki_onset_hour"] = onset_hour
    result["aki_onset_hour_creatinine"] = onset_hour
    return result


def resolve_stay_weight(
    events: pd.DataFrame,
    *,
    stay_start: pd.Timestamp,
) -> float | None:
    """Choose one reasonable stay weight for urine-output normalization."""

    if events.empty:
        return None

    weights = events[["charttime", "weight_kg"]].copy()
    weights["charttime"] = pd.to_datetime(weights["charttime"], errors="coerce")
    weights["weight_kg"] = pd.to_numeric(weights["weight_kg"], errors="coerce")
    weights = weights.dropna(subset=["charttime", "weight_kg"])
    weights = weights[(weights["weight_kg"] > 0.0) & (weights["weight_kg"] <= 400.0)]
    if weights.empty:
        return None

    weights = weights.sort_values("charttime").reset_index(drop=True)
    pre_start = weights[weights["charttime"] <= stay_start]
    if not pre_start.empty:
        return float(pre_start.iloc[-1]["weight_kg"])
    return float(weights.iloc[0]["weight_kg"])


def detect_aki_from_urine_output(
    events: pd.DataFrame,
    *,
    weight_kg: float | None,
    stay_start: pd.Timestamp,
    stay_end: pd.Timestamp,
    bin_hours: int,
) -> dict[str, float | int | None]:
    """Apply KDIGO-like urine-output criteria to one stay's events.

    Expected columns:
    - ``charttime``
    - ``urine_output_ml``
    """

    result = _empty_aki_result()
    if events.empty or weight_kg is None or not np.isfinite(weight_kg) or weight_kg <= 0.0:
        return result

    num_bins = int(np.floor((stay_end - stay_start).total_seconds() / 3600.0 / bin_hours))
    if num_bins <= 0:
        return result

    urine = events[["charttime", "urine_output_ml"]].copy()
    urine["charttime"] = pd.to_datetime(urine["charttime"], errors="coerce")
    urine["urine_output_ml"] = pd.to_numeric(urine["urine_output_ml"], errors="coerce")
    urine = urine.dropna(subset=["charttime", "urine_output_ml"])
    urine = urine[(urine["charttime"] >= stay_start) & (urine["charttime"] <= stay_end)]
    urine = urine[urine["urine_output_ml"] >= 0.0]
    if urine.empty:
        return result

    bin_index = np.floor((urine["charttime"] - stay_start).dt.total_seconds() / 3600.0 / bin_hours).astype(int)
    urine = urine[(bin_index >= 0) & (bin_index < num_bins)].copy()
    if urine.empty:
        return result
    urine["bin_index"] = bin_index[(bin_index >= 0) & (bin_index < num_bins)]

    hourly_output = np.zeros(num_bins, dtype=np.float64)
    grouped = urine.groupby("bin_index", as_index=False)["urine_output_ml"].sum()
    hourly_output[grouped["bin_index"].to_numpy(dtype=int)] = grouped["urine_output_ml"].to_numpy(dtype=np.float64)

    def earliest_window(total_window_hours: int, threshold_ml_per_kg_per_hour: float) -> float | None:
        window_bins = total_window_hours // bin_hours
        if window_bins <= 0 or len(hourly_output) < window_bins:
            return None
        rolling_sum = np.convolve(hourly_output, np.ones(window_bins, dtype=np.float64), mode="valid")
        mean_rate = rolling_sum / (weight_kg * total_window_hours)
        hits = np.flatnonzero(mean_rate < threshold_ml_per_kg_per_hour)
        if len(hits) == 0:
            return None
        return float((hits[0] + window_bins) * bin_hours)

    def earliest_anuria(total_window_hours: int) -> float | None:
        window_bins = total_window_hours // bin_hours
        if window_bins <= 0 or len(hourly_output) < window_bins:
            return None
        rolling_sum = np.convolve(hourly_output, np.ones(window_bins, dtype=np.float64), mode="valid")
        hits = np.flatnonzero(rolling_sum <= 1e-6)
        if len(hits) == 0:
            return None
        return float((hits[0] + window_bins) * bin_hours)

    stage_1_hour = earliest_window(6, 0.5)
    stage_2_hour = earliest_window(12, 0.5)
    stage_3_hour = earliest_window(24, 0.3)
    anuria_hour = earliest_anuria(12)

    onset_candidates = [hour for hour in [stage_1_hour, stage_2_hour, stage_3_hour, anuria_hour] if hour is not None]
    if not onset_candidates:
        return result

    stage = 1
    if stage_2_hour is not None:
        stage = max(stage, 2)
    if stage_3_hour is not None or anuria_hour is not None:
        stage = max(stage, 3)

    result["aki"] = 1
    result["aki_stage"] = stage
    result["aki_urine_output"] = 1
    result["aki_stage_urine_output"] = stage
    result["aki_onset_hour"] = float(min(onset_candidates))
    result["aki_onset_hour_urine_output"] = float(min(onset_candidates))
    result["aki_weight_kg"] = float(weight_kg)
    return result


def _collect_creatinine_events(cohort: pd.DataFrame, config: ProjectConfig, catalog: dict) -> pd.DataFrame:
    creatinine_itemids = set(_feature_itemids(catalog, "creatinine"))
    if not creatinine_itemids or cohort.empty:
        return pd.DataFrame(columns=["stay_id", "charttime", "creatinine"])

    stay_lookup = cohort[["stay_id", "hadm_id", "admittime", "intime", "outtime"]].copy()
    stay_lookup["creatinine_lookback_start"] = stay_lookup["intime"] - pd.Timedelta(days=7)
    stay_lookup["creatinine_lookback_start"] = stay_lookup[["creatinine_lookback_start", "admittime"]].max(axis=1)
    hadm_ids = set(stay_lookup["hadm_id"].dropna().astype(int).tolist())

    frames = []
    reader = _read_csv(
        config.source_path("hosp/labevents.csv"),
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=config.chunk_size,
    )
    for chunk_index, chunk in enumerate(reader):
        if config.max_chunks is not None and chunk_index >= config.max_chunks:
            break
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        chunk = chunk[chunk["itemid"].isin(creatinine_itemids)]
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["creatinine"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "creatinine"])
        if chunk.empty:
            continue

        chunk = chunk.merge(
            stay_lookup[["stay_id", "hadm_id", "creatinine_lookback_start", "outtime"]],
            on="hadm_id",
            how="inner",
        )
        chunk = chunk[
            (chunk["charttime"] >= chunk["creatinine_lookback_start"])
            & (chunk["charttime"] <= chunk["outtime"])
        ]
        if chunk.empty:
            continue
        frames.append(chunk[["stay_id", "charttime", "creatinine"]].copy())

    if not frames:
        return pd.DataFrame(columns=["stay_id", "charttime", "creatinine"])
    return pd.concat(frames, ignore_index=True)


def _collect_stay_events(
    *,
    config: ProjectConfig,
    cohort: pd.DataFrame,
    source_relative_path: str,
    usecols: list[str],
    time_column: str,
    value_column: str,
    value_name: str,
    itemids: set[int],
) -> pd.DataFrame:
    if not itemids or cohort.empty:
        return pd.DataFrame(columns=["stay_id", "charttime", value_name])

    stay_lookup = cohort[["stay_id", "intime", "outtime"]].copy()
    stay_ids = set(stay_lookup["stay_id"].dropna().astype(int).tolist())
    frames = []

    reader = _read_csv(config.source_path(source_relative_path), usecols=usecols, chunksize=config.chunk_size)
    for chunk_index, chunk in enumerate(reader):
        if config.max_chunks is not None and chunk_index >= config.max_chunks:
            break
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if chunk.empty:
            continue

        chunk[time_column] = pd.to_datetime(chunk[time_column], errors="coerce")
        chunk[value_name] = pd.to_numeric(chunk[value_column], errors="coerce")
        chunk = chunk.dropna(subset=[time_column, value_name])
        if chunk.empty:
            continue

        chunk = chunk.merge(stay_lookup, on="stay_id", how="inner")
        chunk = chunk[(chunk[time_column] >= chunk["intime"]) & (chunk[time_column] <= chunk["outtime"])]
        if chunk.empty:
            continue

        frames.append(
            chunk[["stay_id", time_column, value_name]]
            .rename(columns={time_column: "charttime"})
            .copy()
        )

    if not frames:
        return pd.DataFrame(columns=["stay_id", "charttime", value_name])
    return pd.concat(frames, ignore_index=True)


def annotate_aki_stays(cohort: pd.DataFrame, config: ProjectConfig, catalog: dict) -> pd.DataFrame:
    """Annotate one ICU cohort with KDIGO-style AKI flags and stages."""

    if cohort.empty:
        return cohort.assign(**_empty_aki_result())

    creatinine_events = _collect_creatinine_events(cohort, config, catalog)
    urine_events = _collect_stay_events(
        config=config,
        cohort=cohort,
        source_relative_path="icu/outputevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "value"],
        time_column="charttime",
        value_column="value",
        value_name="urine_output_ml",
        itemids=set(_feature_itemids(catalog, "urine_output_ml")),
    )
    weight_events = _collect_stay_events(
        config=config,
        cohort=cohort,
        source_relative_path="icu/chartevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        time_column="charttime",
        value_column="valuenum",
        value_name="weight_kg",
        itemids=set(_feature_itemids(catalog, "weight_kg")),
    )

    creatinine_by_stay = {
        int(stay_id): frame[["charttime", "creatinine"]].copy()
        for stay_id, frame in creatinine_events.groupby("stay_id")
    }
    urine_by_stay = {
        int(stay_id): frame[["charttime", "urine_output_ml"]].copy()
        for stay_id, frame in urine_events.groupby("stay_id")
    }
    weight_by_stay = {
        int(stay_id): frame[["charttime", "weight_kg"]].copy()
        for stay_id, frame in weight_events.groupby("stay_id")
    }

    annotations = []
    iter_columns = ["stay_id", "intime", "outtime"]
    for row in tqdm(cohort[iter_columns].itertuples(index=False), total=len(cohort), desc="AKI annotate"):
        creatinine_result = detect_aki_from_creatinine(
            creatinine_by_stay.get(int(row.stay_id), pd.DataFrame(columns=["charttime", "creatinine"])),
            stay_start=pd.Timestamp(row.intime),
            stay_end=pd.Timestamp(row.outtime),
        )
        weight_kg = resolve_stay_weight(
            weight_by_stay.get(int(row.stay_id), pd.DataFrame(columns=["charttime", "weight_kg"])),
            stay_start=pd.Timestamp(row.intime),
        )
        urine_result = detect_aki_from_urine_output(
            urine_by_stay.get(int(row.stay_id), pd.DataFrame(columns=["charttime", "urine_output_ml"])),
            weight_kg=weight_kg,
            stay_start=pd.Timestamp(row.intime),
            stay_end=pd.Timestamp(row.outtime),
            bin_hours=config.bin_hours,
        )

        onset_candidates = [
            value
            for value in [creatinine_result["aki_onset_hour_creatinine"], urine_result["aki_onset_hour_urine_output"]]
            if pd.notna(value)
        ]
        combined = {
            "stay_id": int(row.stay_id),
            "aki": int(creatinine_result["aki_creatinine"] or urine_result["aki_urine_output"]),
            "aki_stage": int(max(creatinine_result["aki_stage_creatinine"], urine_result["aki_stage_urine_output"])),
            "aki_creatinine": int(creatinine_result["aki_creatinine"]),
            "aki_stage_creatinine": int(creatinine_result["aki_stage_creatinine"]),
            "aki_urine_output": int(urine_result["aki_urine_output"]),
            "aki_stage_urine_output": int(urine_result["aki_stage_urine_output"]),
            "aki_onset_hour": float(min(onset_candidates)) if onset_candidates else np.nan,
            "aki_onset_bin": float(min(onset_candidates) // config.bin_hours) if onset_candidates else np.nan,
            "aki_onset_hour_creatinine": creatinine_result["aki_onset_hour_creatinine"],
            "aki_onset_hour_urine_output": urine_result["aki_onset_hour_urine_output"],
            "aki_baseline_creatinine": creatinine_result["aki_baseline_creatinine"],
            "aki_weight_kg": float(weight_kg) if weight_kg is not None else np.nan,
        }
        annotations.append(combined)

    annotations_frame = pd.DataFrame(annotations)
    return cohort.merge(annotations_frame, on="stay_id", how="left")
