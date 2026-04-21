from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import ProjectConfig
from .feature_catalog import build_catalog, load_catalog


def _read_csv(path: Path, usecols=None, chunksize=None):
    compression = "gzip" if str(path).endswith(".gz") else None
    return pd.read_csv(path, usecols=usecols, compression=compression, chunksize=chunksize)


def _bin_index(times: pd.Series, start_times: pd.Series, bin_hours: int) -> pd.Series:
    delta = (times - start_times).dt.total_seconds() / 3600.0
    return np.floor(delta / bin_hours).astype("int64")


def _prepare_feature_maps(catalog: dict) -> tuple[list[dict], dict[int, int], dict[int, str]]:
    features = catalog["features"]
    itemid_to_feature = {}
    feature_agg = {}
    for index, feature in enumerate(features):
        feature_agg[index] = feature["agg"]
        for itemid in feature["itemids"]:
            itemid_to_feature[int(itemid)] = index
    return features, itemid_to_feature, feature_agg


def build_observation_cohort(config: ProjectConfig) -> pd.DataFrame:
    config.state_from_observation_dir.mkdir(parents=True, exist_ok=True)

    patients = _read_csv(
        config.source_path("hosp/patients.csv"),
        usecols=["subject_id", "gender", "anchor_age"],
    )
    admissions = _read_csv(
        config.source_path("hosp/admissions.csv"),
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "admission_type",
            "insurance",
            "race",
            "hospital_expire_flag",
        ],
    )
    icustays = _read_csv(
        config.source_path("icu/icustays.csv.gz"),
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"],
    )

    cohort = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    cohort = cohort.merge(patients, on="subject_id", how="left")
    cohort["anchor_age"] = cohort["anchor_age"].fillna(0)
    cohort = cohort[cohort["anchor_age"] >= config.min_age].copy()

    for column in ["intime", "outtime", "admittime", "dischtime"]:
        cohort[column] = pd.to_datetime(cohort[column], errors="coerce")

    cohort = cohort.dropna(subset=["intime", "outtime"]).copy()
    stay_hours = (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
    cohort["stay_hours"] = stay_hours.astype(float)
    cohort["num_bins"] = np.floor(stay_hours / config.bin_hours).astype(int)
    cohort = cohort[cohort["num_bins"] >= config.window_bins].copy()
    cohort["in_hospital_mortality"] = cohort["hospital_expire_flag"].fillna(0).astype(int)
    cohort["icu_los_days"] = pd.to_numeric(cohort["los"], errors="coerce").fillna(0.0).astype(float)

    if config.max_stays is not None:
        cohort = cohort.sort_values(["intime", "stay_id"]).head(config.max_stays).copy()

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
    cohort["stay_index"] = np.arange(len(cohort), dtype=np.int64)

    num_bins = cohort["num_bins"].to_numpy(dtype=np.int64)
    offsets = np.zeros(len(cohort), dtype=np.int64)
    if len(offsets) > 1:
        offsets[1:] = np.cumsum(num_bins[:-1])
    cohort["offset"] = offsets
    cohort["num_windows"] = np.maximum(
        0,
        ((cohort["num_bins"] - config.window_bins) // config.window_stride_bins) + 1,
    )
    cohort["num_positive_windows"] = np.maximum(
        0,
        ((cohort["num_bins"] - config.window_bins - config.positive_window_gap_bins) // config.window_stride_bins) + 1,
    )

    cohort.to_csv(config.observation_cohort_path, index=False)
    return cohort


def load_observation_cohort(config: ProjectConfig) -> pd.DataFrame:
    expected_columns = {"in_hospital_mortality", "icu_los_days", "stay_hours"}
    if config.observation_cohort_path.exists():
        cohort = pd.read_csv(
            config.observation_cohort_path,
            parse_dates=["intime", "outtime", "admittime", "dischtime"],
        )
        if expected_columns.issubset(cohort.columns):
            return cohort
    return build_observation_cohort(config)


def load_window_metadata(config: ProjectConfig) -> pd.DataFrame:
    expected_columns = {"window_index", "end_bin", "start_hour", "end_hour"}
    if config.observation_window_metadata_path.exists():
        windows = pd.read_csv(config.observation_window_metadata_path)
        if expected_columns.issubset(windows.columns):
            return windows
    return build_window_metadata(config)


def _initialize_memmaps(config: ProjectConfig, total_bins: int, num_features: int) -> tuple[np.memmap, np.memmap, np.memmap]:
    values = open_memmap(
        config.observation_hourly_values_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_bins, num_features),
    )
    masks = open_memmap(
        config.observation_hourly_masks_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_bins, num_features),
    )
    last_time = open_memmap(
        config.observation_temp_last_time_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_bins, num_features),
    )
    values[:] = 0.0
    masks[:] = 0
    last_time[:] = -1
    return values, masks, last_time


def _apply_last_updates(values: np.memmap, masks: np.memmap, last_time: np.memmap, grouped: pd.DataFrame) -> None:
    flat_bins = grouped["flat_bin"].to_numpy(dtype=np.int64)
    feature_indexes = grouped["feature_index"].to_numpy(dtype=np.int64)
    update_time = grouped["event_offset_sec"].to_numpy(dtype=np.int32)
    update_value = grouped["value"].to_numpy(dtype=np.float32)
    current_time = last_time[flat_bins, feature_indexes]
    should_update = update_time >= current_time
    if not np.any(should_update):
        return
    flat_bins = flat_bins[should_update]
    feature_indexes = feature_indexes[should_update]
    update_time = update_time[should_update]
    update_value = update_value[should_update]
    values[flat_bins, feature_indexes] = update_value
    masks[flat_bins, feature_indexes] = 1
    last_time[flat_bins, feature_indexes] = update_time


def _apply_sum_updates(values: np.memmap, masks: np.memmap, grouped: pd.DataFrame) -> None:
    flat_bins = grouped["flat_bin"].to_numpy(dtype=np.int64)
    feature_indexes = grouped["feature_index"].to_numpy(dtype=np.int64)
    update_value = grouped["value"].to_numpy(dtype=np.float32)
    values[flat_bins, feature_indexes] = values[flat_bins, feature_indexes] + update_value
    masks[flat_bins, feature_indexes] = 1


def _apply_max_updates(values: np.memmap, masks: np.memmap, grouped: pd.DataFrame) -> None:
    flat_bins = grouped["flat_bin"].to_numpy(dtype=np.int64)
    feature_indexes = grouped["feature_index"].to_numpy(dtype=np.int64)
    update_value = grouped["value"].to_numpy(dtype=np.float32)
    current_value = values[flat_bins, feature_indexes]
    current_mask = masks[flat_bins, feature_indexes]
    should_update = (current_mask == 0) | (update_value > current_value)
    if not np.any(should_update):
        return
    flat_bins = flat_bins[should_update]
    feature_indexes = feature_indexes[should_update]
    update_value = update_value[should_update]
    values[flat_bins, feature_indexes] = update_value
    masks[flat_bins, feature_indexes] = 1


def _process_direct_source(
    path: Path,
    source_name: str,
    cohort: pd.DataFrame,
    features: list[dict],
    itemid_to_feature: dict[int, int],
    feature_agg: dict[int, str],
    values: np.memmap,
    masks: np.memmap,
    last_time: np.memmap,
    config: ProjectConfig,
) -> None:
    source_features = [feature for feature in features if feature["source"] == source_name]
    if not source_features:
        return

    keep_itemids = {itemid for feature in source_features for itemid in feature["itemids"]}
    agg_to_features = {}
    for feature_index, agg in feature_agg.items():
        if features[feature_index]["source"] == source_name:
            agg_to_features.setdefault(agg, set()).add(feature_index)

    lookup = cohort[["stay_id", "offset", "intime", "num_bins"]].copy()
    stay_ids = set(lookup["stay_id"].astype(int).tolist())
    time_column = {
        "chart": "charttime",
        "input": "starttime",
        "output": "charttime",
    }[source_name]

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
        chunk[time_column] = pd.to_datetime(chunk[time_column], errors="coerce")
        chunk = chunk.dropna(subset=[time_column])
        chunk["feature_index"] = chunk["itemid"].map(itemid_to_feature)
        chunk = chunk.dropna(subset=["feature_index"])
        chunk["feature_index"] = chunk["feature_index"].astype(int)

        chunk["bin_index"] = _bin_index(chunk[time_column], chunk["intime"], config.bin_hours)
        chunk = chunk[(chunk["bin_index"] >= 0) & (chunk["bin_index"] < chunk["num_bins"])]
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

        chunk["event_offset_sec"] = (chunk[time_column] - chunk["intime"]).dt.total_seconds().astype(np.int32)
        chunk["flat_bin"] = chunk["offset"].astype(np.int64) + chunk["bin_index"].astype(np.int64)

        for agg, feature_indexes in agg_to_features.items():
            subset = chunk[chunk["feature_index"].isin(feature_indexes)].copy()
            if subset.empty:
                continue
            if agg == "last":
                grouped = (
                    subset.sort_values("event_offset_sec")
                    .groupby(["flat_bin", "feature_index"], as_index=False)
                    .agg(value=("_value", "last"), event_offset_sec=("event_offset_sec", "last"))
                )
                _apply_last_updates(values, masks, last_time, grouped)
            elif agg == "sum":
                grouped = (
                    subset.groupby(["flat_bin", "feature_index"], as_index=False)
                    .agg(value=("_value", "sum"))
                )
                _apply_sum_updates(values, masks, grouped)
            elif agg == "max":
                grouped = (
                    subset.groupby(["flat_bin", "feature_index"], as_index=False)
                    .agg(value=("_value", "max"))
                )
                _apply_max_updates(values, masks, grouped)
            else:
                raise ValueError(f"Unsupported aggregation: {agg}")


def _process_lab_source(
    path: Path,
    cohort: pd.DataFrame,
    features: list[dict],
    itemid_to_feature: dict[int, int],
    values: np.memmap,
    masks: np.memmap,
    last_time: np.memmap,
    config: ProjectConfig,
) -> None:
    source_features = [feature for feature in features if feature["source"] == "lab"]
    if not source_features:
        return

    keep_itemids = {itemid for feature in source_features for itemid in feature["itemids"]}
    lookup = cohort[["hadm_id", "offset", "intime", "num_bins"]].copy()
    hadm_ids = set(lookup["hadm_id"].astype(int).tolist())

    reader = _read_csv(path, chunksize=config.chunk_size)
    for chunk_index, chunk in enumerate(reader):
        if config.max_chunks is not None and chunk_index >= config.max_chunks:
            break
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        chunk = chunk[chunk["itemid"].isin(keep_itemids)]
        if chunk.empty:
            continue

        chunk = chunk.merge(lookup, on="hadm_id", how="inner")
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])
        chunk["feature_index"] = chunk["itemid"].map(itemid_to_feature)
        chunk = chunk.dropna(subset=["feature_index"])
        chunk["feature_index"] = chunk["feature_index"].astype(int)

        chunk["bin_index"] = _bin_index(chunk["charttime"], chunk["intime"], config.bin_hours)
        chunk = chunk[(chunk["bin_index"] >= 0) & (chunk["bin_index"] < chunk["num_bins"])]
        if chunk.empty:
            continue

        chunk["_value"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["_value"])
        if chunk.empty:
            continue

        chunk["event_offset_sec"] = (chunk["charttime"] - chunk["intime"]).dt.total_seconds().astype(np.int32)
        chunk["flat_bin"] = chunk["offset"].astype(np.int64) + chunk["bin_index"].astype(np.int64)
        grouped = (
            chunk.sort_values("event_offset_sec")
            .groupby(["flat_bin", "feature_index"], as_index=False)
            .agg(value=("_value", "last"), event_offset_sec=("event_offset_sec", "last"))
        )
        _apply_last_updates(values, masks, last_time, grouped)


def _fit_feature_statistics(
    values: np.memmap,
    masks: np.memmap,
    cohort: pd.DataFrame,
    feature_names: list[str],
    config: ProjectConfig,
) -> list[dict[str, float | str]]:
    total_bins = values.shape[0]
    train_bin_mask = np.zeros(total_bins, dtype=bool)
    train_rows = cohort[cohort["split"] == "train"][["offset", "num_bins"]]
    for row in train_rows.itertuples(index=False):
        start = int(row.offset)
        stop = start + int(row.num_bins)
        train_bin_mask[start:stop] = True

    stats = []
    for feature_index, feature_name in enumerate(feature_names):
        feature_mask = np.asarray(masks[:, feature_index], dtype=bool) & train_bin_mask
        feature_values = np.asarray(values[:, feature_index], dtype=np.float32)
        observed_values = feature_values[feature_mask]
        if observed_values.size == 0:
            stats.append(
                {
                    "name": feature_name,
                    "lower": 0.0,
                    "upper": 0.0,
                    "mean": 0.0,
                    "std": 1.0,
                }
            )
            continue

        lower = float(np.quantile(observed_values, config.winsor_lower_quantile))
        upper = float(np.quantile(observed_values, config.winsor_upper_quantile))
        clipped = np.clip(observed_values, lower, upper)
        mean = float(clipped.mean())
        std = float(clipped.std())
        if std <= 1e-6:
            std = 1.0
        stats.append(
            {
                "name": feature_name,
                "lower": lower,
                "upper": upper,
                "mean": mean,
                "std": std,
            }
        )
    return stats


def _normalize_and_forward_fill(
    values: np.memmap,
    masks: np.memmap,
    cohort: pd.DataFrame,
    stats: list[dict[str, float | str]],
) -> None:
    for row in tqdm(cohort[["offset", "num_bins"]].itertuples(index=False), total=len(cohort), desc="Normalize/Ffill"):
        start = int(row.offset)
        stop = start + int(row.num_bins)
        stay_values = np.asarray(values[start:stop], dtype=np.float32)
        stay_masks = np.asarray(masks[start:stop], dtype=bool)
        time_index = np.arange(stop - start)

        for feature_index, feature_stats in enumerate(stats):
            observed = stay_masks[:, feature_index]
            column = stay_values[:, feature_index]
            if observed.any():
                lower = float(feature_stats["lower"])
                upper = float(feature_stats["upper"])
                mean = float(feature_stats["mean"])
                std = float(feature_stats["std"])
                column[observed] = (np.clip(column[observed], lower, upper) - mean) / std

            last_seen = np.where(observed, time_index, -1)
            np.maximum.accumulate(last_seen, out=last_seen)
            missing = last_seen < 0
            filled = np.zeros_like(column, dtype=np.float32)
            if (~missing).any():
                filled[~missing] = column[last_seen[~missing]]
            stay_values[:, feature_index] = filled

        values[start:stop] = stay_values


def _compute_deltas(masks: np.memmap, cohort: pd.DataFrame, config: ProjectConfig) -> np.memmap:
    deltas = open_memmap(
        config.observation_hourly_deltas_path,
        mode="w+",
        dtype=np.float32,
        shape=masks.shape,
    )

    for row in tqdm(cohort[["offset", "num_bins"]].itertuples(index=False), total=len(cohort), desc="Delta"):
        start = int(row.offset)
        stop = start + int(row.num_bins)
        stay_masks = np.asarray(masks[start:stop], dtype=bool)
        stay_deltas = np.zeros((stop - start, masks.shape[1]), dtype=np.float32)
        last_seen = np.full(masks.shape[1], -1, dtype=np.int32)
        for time_index in range(stop - start):
            observed = stay_masks[time_index]
            missing_delta = np.where(
                last_seen < 0,
                config.delta_cap_hours,
                np.minimum((time_index - last_seen) * config.bin_hours, config.delta_cap_hours),
            ).astype(np.float32)
            stay_deltas[time_index] = np.where(observed, 0.0, missing_delta)
            last_seen = np.where(observed, time_index, last_seen)
        deltas[start:stop] = stay_deltas

    return deltas


def build_hourly_observation_dataset(config: ProjectConfig) -> tuple[pd.DataFrame, dict]:
    config.state_from_observation_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_observation_cohort(config)
    catalog = load_catalog(config) if config.catalog_path.exists() else build_catalog(config)
    features, itemid_to_feature, feature_agg = _prepare_feature_maps(catalog)

    total_bins = int(cohort["num_bins"].sum())
    num_features = len(features)
    values, masks, last_time = _initialize_memmaps(config, total_bins, num_features)

    _process_direct_source(
        config.source_path("icu/inputevents.csv.gz"),
        "input",
        cohort,
        features,
        itemid_to_feature,
        feature_agg,
        values,
        masks,
        last_time,
        config,
    )
    _process_direct_source(
        config.source_path("icu/chartevents.csv.gz"),
        "chart",
        cohort,
        features,
        itemid_to_feature,
        feature_agg,
        values,
        masks,
        last_time,
        config,
    )
    _process_direct_source(
        config.source_path("icu/outputevents.csv.gz"),
        "output",
        cohort,
        features,
        itemid_to_feature,
        feature_agg,
        values,
        masks,
        last_time,
        config,
    )
    _process_lab_source(
        config.source_path("hosp/labevents.csv"),
        cohort,
        features,
        itemid_to_feature,
        values,
        masks,
        last_time,
        config,
    )

    stats = _fit_feature_statistics(
        values=values,
        masks=masks,
        cohort=cohort,
        feature_names=[feature["name"] for feature in features],
        config=config,
    )
    _normalize_and_forward_fill(values, masks, cohort, stats)
    deltas = _compute_deltas(masks, cohort, config)

    del last_time
    if config.observation_temp_last_time_path.exists():
        config.observation_temp_last_time_path.unlink()

    metadata = {
        "dynamic_feature_names": [feature["name"] for feature in features],
        "window_hours": config.window_hours,
        "window_stride_hours": config.window_stride_hours,
        "positive_window_gap_hours": config.positive_window_gap_hours,
        "bin_hours": config.bin_hours,
        "delta_cap_hours": config.delta_cap_hours,
        "total_stays": int(len(cohort)),
        "total_bins": total_bins,
    }

    with open(config.observation_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with open(config.observation_stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    del deltas
    return cohort, metadata


def build_window_metadata(config: ProjectConfig) -> pd.DataFrame:
    cohort = load_observation_cohort(config)
    records = []
    window_id = 0

    for row in tqdm(
        cohort[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "split",
                "offset",
                "num_bins",
                "in_hospital_mortality",
                "icu_los_days",
            ]
        ].itertuples(index=False),
        total=len(cohort),
        desc="Windows",
    ):
        max_start = int(row.num_bins) - config.window_bins
        if max_start < 0:
            continue
        starts = np.arange(0, max_start + 1, config.window_stride_bins, dtype=np.int32)
        positive_starts = starts + config.positive_window_gap_bins
        has_positive = positive_starts <= max_start
        positive_flat_start = np.where(has_positive, int(row.offset) + positive_starts, -1)

        stay_frame = pd.DataFrame(
            {
                "window_id": np.arange(window_id, window_id + len(starts), dtype=np.int64),
                "window_index": np.arange(len(starts), dtype=np.int32),
                "subject_id": int(row.subject_id),
                "hadm_id": int(row.hadm_id),
                "stay_id": int(row.stay_id),
                "split": row.split,
                "flat_start": int(row.offset) + starts,
                "start_bin": starts,
                "end_bin": starts + config.window_bins,
                "start_hour": starts * config.bin_hours,
                "end_hour": (starts + config.window_bins) * config.bin_hours,
                "positive_flat_start": positive_flat_start.astype(np.int64),
                "positive_start_bin": np.where(has_positive, positive_starts, -1).astype(np.int32),
                "positive_end_bin": np.where(has_positive, positive_starts + config.window_bins, -1).astype(np.int32),
                "in_hospital_mortality": int(row.in_hospital_mortality),
                "icu_los_days": float(row.icu_los_days),
            }
        )
        records.append(stay_frame)
        window_id += len(starts)

    if records:
        windows = pd.concat(records, ignore_index=True)
    else:
        windows = pd.DataFrame(
            columns=[
                "window_id",
                "window_index",
                "subject_id",
                "hadm_id",
                "stay_id",
                "split",
                "flat_start",
                "start_bin",
                "end_bin",
                "start_hour",
                "end_hour",
                "positive_flat_start",
                "positive_start_bin",
                "positive_end_bin",
                "in_hospital_mortality",
                "icu_los_days",
            ]
        )

    windows.to_csv(config.observation_window_metadata_path, index=False)
    return windows


def build_state_from_observation_dataset(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cohort, metadata = build_hourly_observation_dataset(config)
    windows = build_window_metadata(config)
    metadata = {
        **metadata,
        "total_windows": int(len(windows)),
        "total_positive_windows": int((windows["positive_flat_start"] >= 0).sum()) if len(windows) else 0,
    }
    with open(config.observation_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return cohort, windows, metadata


class ObservationWindowPairDataset(Dataset):
    def __init__(self, config: ProjectConfig, split: str) -> None:
        self.config = config
        self.window_bins = config.window_bins
        self.values = np.load(config.observation_hourly_values_path, mmap_mode="r")
        self.masks = np.load(config.observation_hourly_masks_path, mmap_mode="r")
        self.deltas = np.load(config.observation_hourly_deltas_path, mmap_mode="r")

        windows = load_window_metadata(config)
        windows = windows[(windows["split"] == split) & (windows["positive_flat_start"] >= 0)].copy()
        self.flat_starts = windows["flat_start"].to_numpy(dtype=np.int64)
        self.positive_flat_starts = windows["positive_flat_start"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.flat_starts)

    def _load_window(self, flat_start: int) -> torch.Tensor:
        flat_stop = flat_start + self.window_bins
        values = np.asarray(self.values[flat_start:flat_stop], dtype=np.float32)
        masks = np.asarray(self.masks[flat_start:flat_stop], dtype=np.float32)
        deltas = np.log1p(np.asarray(self.deltas[flat_start:flat_stop], dtype=np.float32))
        triplets = np.stack([values, masks, deltas], axis=-1)
        return torch.from_numpy(triplets).float()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = self._load_window(int(self.flat_starts[index]))
        positive = self._load_window(int(self.positive_flat_starts[index]))
        return anchor, positive


class ObservationWindowDataset(Dataset):
    def __init__(self, config: ProjectConfig, split: str | None = None, max_windows: int | None = None) -> None:
        self.config = config
        self.window_bins = config.window_bins
        self.values = np.load(config.observation_hourly_values_path, mmap_mode="r")
        self.masks = np.load(config.observation_hourly_masks_path, mmap_mode="r")
        self.deltas = np.load(config.observation_hourly_deltas_path, mmap_mode="r")

        windows = load_window_metadata(config)
        if split is not None:
            windows = windows[windows["split"] == split].copy()
        if max_windows is not None:
            windows = windows.head(max_windows).copy()
        self.window_metadata = windows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.window_metadata)

    def _load_window(self, flat_start: int) -> torch.Tensor:
        flat_stop = flat_start + self.window_bins
        values = np.asarray(self.values[flat_start:flat_stop], dtype=np.float32)
        masks = np.asarray(self.masks[flat_start:flat_stop], dtype=np.float32)
        deltas = np.log1p(np.asarray(self.deltas[flat_start:flat_stop], dtype=np.float32))
        return torch.from_numpy(np.stack([values, masks, deltas], axis=-1)).float()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, object]]:
        row = self.window_metadata.iloc[index]
        return self._load_window(int(row["flat_start"])), row.to_dict()
