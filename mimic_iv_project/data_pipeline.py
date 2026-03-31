from __future__ import annotations

import argparse
import json
from typing import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .config import STATIC_CATEGORICAL_COLUMNS, STATIC_NUMERIC_COLUMNS, TASK_COLUMNS, ProjectConfig
from .feature_catalog import build_catalog, load_catalog


def _read_csv(path, usecols=None, chunksize=None):
    compression = "gzip" if str(path).endswith(".gz") else None
    return pd.read_csv(path, usecols=usecols, compression=compression, chunksize=chunksize)


def build_cohort(config: ProjectConfig) -> pd.DataFrame:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    patients = _read_csv(
        config.hosp_dir / "patients.csv",
        usecols=["subject_id", "gender", "anchor_age"],
    )
    admissions = _read_csv(
        config.hosp_dir / "admissions.csv",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "insurance",
            "race",
            "hospital_expire_flag",
        ],
    )
    icustays = _read_csv(
        config.icu_dir / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"],
    )
    diagnoses = _read_csv(config.hosp_dir / "diagnoses_icd.csv", usecols=["hadm_id"])

    diag_count = diagnoses.groupby("hadm_id").size().rename("diag_count")

    cohort = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    cohort = cohort.merge(patients, on="subject_id", how="left")
    cohort = cohort.merge(diag_count, on="hadm_id", how="left")
    cohort["diag_count"] = cohort["diag_count"].fillna(0).astype(int)

    cohort["anchor_age"] = cohort["anchor_age"].fillna(0)
    cohort = cohort[cohort["anchor_age"] >= config.min_age].copy()

    for column in ["intime", "outtime", "admittime", "dischtime", "deathtime"]:
        cohort[column] = pd.to_datetime(cohort[column], errors="coerce")

    cohort["history_end"] = cohort["intime"] + pd.to_timedelta(config.history_hours, unit="h")
    cohort["label_end"] = cohort["history_end"] + pd.to_timedelta(config.future_hours, unit="h")
    cohort["has_history_window"] = (cohort["outtime"] >= cohort["history_end"]).astype(int)
    cohort["has_label_window"] = (cohort["outtime"] >= cohort["label_end"]).astype(int)
    cohort = cohort[(cohort["has_history_window"] == 1) & (cohort["has_label_window"] == 1)].copy()

    if config.max_stays is not None:
        cohort = cohort.sort_values(["intime", "stay_id"]).head(config.max_stays).copy()

    cohort["in_hospital_mortality"] = cohort["hospital_expire_flag"].fillna(0).astype(int)
    cohort["long_icu_los"] = (cohort["los"] > config.long_los_days).astype(int)
    cohort["vasopressor_next_6h"] = 0

    rng = np.random.default_rng(config.random_seed)
    subjects = cohort["subject_id"].drop_duplicates().to_numpy()
    rng.shuffle(subjects)
    n_subjects = len(subjects)
    n_train = int(n_subjects * config.train_fraction)
    n_val = int(n_subjects * config.val_fraction)
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])

    def assign_split(subject_id: int) -> str:
        if subject_id in train_subjects:
            return "train"
        if subject_id in val_subjects:
            return "val"
        return "test"

    cohort["split"] = cohort["subject_id"].map(assign_split)
    cohort = cohort.sort_values(["split", "subject_id", "intime", "stay_id"]).reset_index(drop=True)
    cohort["row_index"] = np.arange(len(cohort))
    cohort.to_csv(config.cohort_path, index=False)
    return cohort


def _load_or_build_cohort(config: ProjectConfig) -> pd.DataFrame:
    if config.cohort_path.exists():
        cohort = pd.read_csv(config.cohort_path, parse_dates=["intime", "outtime", "history_end", "label_end"])
        return cohort
    return build_cohort(config)


def _static_matrix(cohort: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric = cohort[STATIC_NUMERIC_COLUMNS].fillna(0).astype(float)
    categoricals = pd.get_dummies(cohort[STATIC_CATEGORICAL_COLUMNS].fillna("UNKNOWN"), dummy_na=False)
    static_df = pd.concat([numeric, categoricals], axis=1).astype(float)
    return static_df.to_numpy(dtype=np.float32), static_df.columns.tolist()


def _bin_index(times: pd.Series, start_times: pd.Series, bin_hours: int) -> pd.Series:
    delta = (times - start_times).dt.total_seconds() / 3600.0
    return np.floor(delta / bin_hours).astype("int64")


def _update_last(values, masks, counts, last_time, grouped: pd.DataFrame) -> None:
    for row in grouped.itertuples(index=False):
        i = int(row.row_index)
        b = int(row.bin_index)
        f = int(row.feature_index)
        event_time = int(row.event_time)
        counts[i, b, f] += float(row.obs_count)
        if event_time >= last_time[i, b, f]:
            values[i, b, f] = float(row.value)
            masks[i, b, f] = 1.0
            last_time[i, b, f] = event_time


def _update_sum(values, masks, counts, grouped: pd.DataFrame) -> None:
    for row in grouped.itertuples(index=False):
        i = int(row.row_index)
        b = int(row.bin_index)
        f = int(row.feature_index)
        values[i, b, f] += float(row.value)
        counts[i, b, f] += float(row.obs_count)
        masks[i, b, f] = 1.0


def _update_max(values, masks, counts, grouped: pd.DataFrame) -> None:
    for row in grouped.itertuples(index=False):
        i = int(row.row_index)
        b = int(row.bin_index)
        f = int(row.feature_index)
        if masks[i, b, f] == 0 or float(row.value) > float(values[i, b, f]):
            values[i, b, f] = float(row.value)
        counts[i, b, f] += float(row.obs_count)
        masks[i, b, f] = 1.0


def _apply_grouped_updates(values, masks, counts, last_time, grouped: pd.DataFrame, agg: str) -> None:
    if grouped.empty:
        return
    if agg == "last":
        _update_last(values, masks, counts, last_time, grouped)
        return
    if agg == "sum":
        _update_sum(values, masks, counts, grouped)
        return
    if agg == "max":
        _update_max(values, masks, counts, grouped)
        return
    raise ValueError(f"Unsupported aggregation: {agg}")


def _prepare_feature_maps(catalog: dict) -> tuple[list[dict], dict[int, int], dict[int, str]]:
    features = catalog["features"]
    itemid_to_feature = {}
    feature_agg = {}
    for index, feature in enumerate(features):
        feature_agg[index] = feature["agg"]
        for itemid in feature["itemids"]:
            itemid_to_feature[int(itemid)] = index
    return features, itemid_to_feature, feature_agg


def _process_direct_source(
    path,
    source_name: str,
    cohort: pd.DataFrame,
    features: list[dict],
    itemid_to_feature: dict[int, int],
    feature_agg: dict[int, str],
    values: np.ndarray,
    masks: np.ndarray,
    counts: np.ndarray,
    last_time: np.ndarray,
    config: ProjectConfig,
) -> np.ndarray:
    source_features = [f for f in features if f["source"] == source_name]
    if not source_features:
        return np.zeros(len(cohort), dtype=np.int64)

    keep_itemids = {itemid for feature in source_features for itemid in feature["itemids"]}
    agg_to_features = {}
    for feature_index, agg in feature_agg.items():
        if features[feature_index]["source"] == source_name:
            agg_to_features.setdefault(agg, set()).add(feature_index)

    future_vasopressor = np.zeros(len(cohort), dtype=np.int64)
    lookup = cohort[["stay_id", "row_index", "intime", "history_end", "label_end"]].copy()
    stay_ids = set(lookup["stay_id"].astype(int).tolist())
    time_column = {
        "chart": "charttime",
        "input": "starttime",
        "output": "charttime",
    }[source_name]
    future_label_feature_indexes = {
        index
        for index, feature in enumerate(features)
        if feature["source"] == "input" and feature["name"] == "vasopressor_event"
    }

    reader = _read_csv(path, chunksize=config.chunk_size)
    for chunk_index, chunk in enumerate(reader):
        if config.max_chunks is not None and chunk_index >= config.max_chunks:
            break
        if "stay_id" not in chunk.columns:
            continue
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        chunk = chunk[chunk["itemid"].isin(keep_itemids)]
        if chunk.empty:
            continue

        chunk = chunk.merge(lookup, on="stay_id", how="inner")
        if chunk.empty:
            continue

        chunk[time_column] = pd.to_datetime(chunk[time_column], errors="coerce")
        chunk = chunk.dropna(subset=[time_column])
        chunk["feature_index"] = chunk["itemid"].map(itemid_to_feature)
        chunk = chunk.dropna(subset=["feature_index"])
        chunk["feature_index"] = chunk["feature_index"].astype(int)

        if source_name == "input" and future_label_feature_indexes:
            future_mask = (chunk[time_column] >= chunk["history_end"]) & (chunk[time_column] < chunk["label_end"])
            future_rows = chunk[future_mask & chunk["feature_index"].isin(future_label_feature_indexes)]
            if not future_rows.empty:
                future_vasopressor[future_rows["row_index"].astype(int).to_numpy()] = 1

        chunk = chunk[(chunk[time_column] >= chunk["intime"]) & (chunk[time_column] < chunk["history_end"])]
        if chunk.empty:
            continue

        chunk["bin_index"] = _bin_index(chunk[time_column], chunk["intime"], config.bin_hours)
        chunk = chunk[(chunk["bin_index"] >= 0) & (chunk["bin_index"] < config.history_bins)]
        if chunk.empty:
            continue

        if source_name == "input":
            chunk["_value"] = 1.0
        elif source_name == "output":
            chunk["_value"] = pd.to_numeric(chunk["value"], errors="coerce")
        else:
            chunk["_value"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["_value"])
        if chunk.empty:
            continue

        chunk["event_time"] = (chunk[time_column].astype("int64") // 10**9).astype(np.int64)

        for agg, feature_indexes in agg_to_features.items():
            subset = chunk[chunk["feature_index"].isin(feature_indexes)].copy()
            if subset.empty:
                continue
            if agg == "last":
                grouped = (
                    subset.sort_values("event_time")
                    .groupby(["row_index", "bin_index", "feature_index"], as_index=False)
                    .agg(value=("_value", "last"), event_time=("event_time", "last"), obs_count=("_value", "size"))
                )
            elif agg == "sum":
                grouped = (
                    subset.groupby(["row_index", "bin_index", "feature_index"], as_index=False)
                    .agg(value=("_value", "sum"), obs_count=("_value", "size"))
                )
            else:
                grouped = (
                    subset.groupby(["row_index", "bin_index", "feature_index"], as_index=False)
                    .agg(value=("_value", "max"), obs_count=("_value", "size"))
                )
            _apply_grouped_updates(values, masks, counts, last_time, grouped, agg)

    return future_vasopressor


def _process_lab_source(
    path,
    cohort: pd.DataFrame,
    features: list[dict],
    itemid_to_feature: dict[int, int],
    feature_agg: dict[int, str],
    values: np.ndarray,
    masks: np.ndarray,
    counts: np.ndarray,
    last_time: np.ndarray,
    config: ProjectConfig,
) -> None:
    source_features = [f for f in features if f["source"] == "lab"]
    if not source_features:
        return

    keep_itemids = {itemid for feature in source_features for itemid in feature["itemids"]}
    hadm_lookup = cohort[["hadm_id", "row_index", "intime", "history_end"]].copy()
    hadm_ids = set(hadm_lookup["hadm_id"].astype(int).tolist())

    reader = _read_csv(path, chunksize=config.chunk_size)
    for chunk_index, chunk in enumerate(reader):
        if config.max_chunks is not None and chunk_index >= config.max_chunks:
            break
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        chunk = chunk[chunk["itemid"].isin(keep_itemids)]
        if chunk.empty:
            continue

        chunk = chunk.merge(hadm_lookup, on="hadm_id", how="inner")
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] < chunk["history_end"])]
        if chunk.empty:
            continue

        chunk["feature_index"] = chunk["itemid"].map(itemid_to_feature)
        chunk = chunk.dropna(subset=["feature_index"])
        chunk["feature_index"] = chunk["feature_index"].astype(int)
        chunk["bin_index"] = _bin_index(chunk["charttime"], chunk["intime"], config.bin_hours)
        chunk = chunk[(chunk["bin_index"] >= 0) & (chunk["bin_index"] < config.history_bins)]
        chunk["_value"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["_value"])
        if chunk.empty:
            continue

        chunk["event_time"] = (chunk["charttime"].astype("int64") // 10**9).astype(np.int64)
        grouped = (
            chunk.sort_values("event_time")
            .groupby(["row_index", "bin_index", "feature_index"], as_index=False)
            .agg(value=("_value", "last"), event_time=("event_time", "last"), obs_count=("_value", "size"))
        )
        _apply_grouped_updates(values, masks, counts, last_time, grouped, "last")


def _compute_deltas(masks: np.ndarray, bin_hours: int) -> np.ndarray:
    num_stays, num_bins, num_features = masks.shape
    deltas = np.zeros((num_stays, num_bins, num_features), dtype=np.float32)
    last_seen = np.full((num_stays, num_features), -1, dtype=np.int32)
    for t in range(num_bins):
        delta = np.where(last_seen < 0, t + 1, t - last_seen) * bin_hours
        deltas[:, t, :] = delta.astype(np.float32)
        observed = masks[:, t, :] > 0
        last_seen = np.where(observed, t, last_seen)
    return deltas


def build_sequence_dataset(config: ProjectConfig) -> tuple[pd.DataFrame, dict]:
    cohort = _load_or_build_cohort(config)
    catalog = load_catalog(config) if config.catalog_path.exists() else build_catalog(config)
    features, itemid_to_feature, feature_agg = _prepare_feature_maps(catalog)

    num_stays = len(cohort)
    num_bins = config.history_bins
    num_features = len(features)
    values = np.zeros((num_stays, num_bins, num_features), dtype=np.float32)
    masks = np.zeros((num_stays, num_bins, num_features), dtype=np.float32)
    counts = np.zeros((num_stays, num_bins, num_features), dtype=np.float32)
    last_time = np.full((num_stays, num_bins, num_features), -1, dtype=np.int64)

    future_vasopressor = _process_direct_source(
        config.icu_dir / "inputevents.csv.gz",
        "input",
        cohort,
        features,
        itemid_to_feature,
        feature_agg,
        values,
        masks,
        counts,
        last_time,
        config,
    )
    _process_direct_source(
        config.icu_dir / "chartevents.csv.gz",
        "chart",
        cohort,
        features,
        itemid_to_feature,
        feature_agg,
        values,
        masks,
        counts,
        last_time,
        config,
    )
    _process_direct_source(
        config.icu_dir / "outputevents.csv.gz",
        "output",
        cohort,
        features,
        itemid_to_feature,
        feature_agg,
        values,
        masks,
        counts,
        last_time,
        config,
    )
    _process_lab_source(
        config.hosp_dir / "labevents.csv",
        cohort,
        features,
        itemid_to_feature,
        feature_agg,
        values,
        masks,
        counts,
        last_time,
        config,
    )

    cohort = cohort.copy()
    cohort["vasopressor_next_6h"] = future_vasopressor.astype(int)
    cohort.to_csv(config.cohort_path, index=False)

    static_values, static_feature_names = _static_matrix(cohort)
    deltas = _compute_deltas(masks, config.bin_hours)
    labels = cohort[TASK_COLUMNS].to_numpy(dtype=np.float32)

    np.savez_compressed(
        config.sequence_dataset_path,
        values=values,
        masks=masks,
        counts=counts,
        deltas=deltas,
        static_features=static_values,
        labels=labels,
    )

    metadata = {
        "dynamic_feature_names": [feature["name"] for feature in features],
        "static_feature_names": static_feature_names,
        "task_names": TASK_COLUMNS,
        "history_hours": config.history_hours,
        "future_hours": config.future_hours,
        "bin_hours": config.bin_hours,
    }
    with open(config.sequence_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return cohort, metadata


def _last_observed(feature_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
    last = np.zeros(feature_slice.shape[0], dtype=np.float32)
    for i in range(feature_slice.shape[0]):
        observed = np.where(mask_slice[i] > 0)[0]
        if observed.size > 0:
            last[i] = feature_slice[i, observed[-1]]
    return last


def _safe_nan_stat(feature_slice: np.ndarray, reducer: str) -> np.ndarray:
    output = np.zeros(feature_slice.shape[0], dtype=np.float32)
    observed = ~np.isnan(feature_slice)
    for i in range(feature_slice.shape[0]):
        if not observed[i].any():
            continue
        if reducer == "mean":
            output[i] = float(np.nanmean(feature_slice[i]))
        elif reducer == "min":
            output[i] = float(np.nanmin(feature_slice[i]))
        elif reducer == "max":
            output[i] = float(np.nanmax(feature_slice[i]))
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
    return output


def build_tabular_dataset(config: ProjectConfig) -> pd.DataFrame:
    cohort = _load_or_build_cohort(config)
    data = np.load(config.sequence_dataset_path)
    with open(config.sequence_metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    values = data["values"]
    masks = data["masks"]
    counts = data["counts"]
    static_features = data["static_features"]
    feature_names = metadata["dynamic_feature_names"]
    static_feature_names = metadata["static_feature_names"]

    frame = cohort[["subject_id", "hadm_id", "stay_id", "split"] + TASK_COLUMNS].copy()
    static_df = pd.DataFrame(static_features, columns=static_feature_names)

    observed = np.where(masks > 0, values, np.nan)
    summary = {}
    for feature_index, name in enumerate(feature_names):
        feature_slice = observed[:, :, feature_index]
        mask_slice = masks[:, :, feature_index]
        summary[f"{name}_last"] = _last_observed(values[:, :, feature_index], mask_slice)
        summary[f"{name}_mean"] = _safe_nan_stat(feature_slice, "mean")
        summary[f"{name}_min"] = _safe_nan_stat(feature_slice, "min")
        summary[f"{name}_max"] = _safe_nan_stat(feature_slice, "max")
        summary[f"{name}_count"] = counts[:, :, feature_index].sum(axis=1)

    summary_df = pd.DataFrame(summary)
    tabular = pd.concat([frame, static_df, summary_df], axis=1)
    tabular.to_csv(config.tabular_features_path, index=False)

    feature_columns = [column for column in tabular.columns if column not in {"subject_id", "hadm_id", "stay_id", "split", *TASK_COLUMNS}]
    metadata = {
        "feature_columns": feature_columns,
        "task_names": TASK_COLUMNS,
    }
    with open(config.tabular_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return tabular


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIMIC-IV cohort and training datasets.")
    parser.add_argument("--build-all", action="store_true")
    parser.add_argument("--build-catalog", action="store_true")
    parser.add_argument("--build-cohort", action="store_true")
    parser.add_argument("--build-sequence", action="store_true")
    parser.add_argument("--build-tabular", action="store_true")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--raw-root", type=str, default=None)
    parser.add_argument("--history-hours", type=int, default=24)
    parser.add_argument("--future-hours", type=int, default=6)
    parser.add_argument("--bin-hours", type=int, default=1)
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--max-chunks", type=int, default=None)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> ProjectConfig:
    project_root = Path(args.project_root).resolve() if args.project_root else None
    raw_root = Path(args.raw_root).resolve() if args.raw_root else None
    return ProjectConfig(
        project_root=project_root or ProjectConfig().project_root,
        raw_root=raw_root,
        history_hours=args.history_hours,
        future_hours=args.future_hours,
        bin_hours=args.bin_hours,
        max_stays=args.max_stays,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)

    if args.build_all or args.build_catalog:
        build_catalog(config)
    if args.build_all or args.build_cohort:
        build_cohort(config)
    if args.build_all or args.build_sequence:
        build_sequence_dataset(config)
    if args.build_all or args.build_tabular:
        build_tabular_dataset(config)


if __name__ == "__main__":
    main()
