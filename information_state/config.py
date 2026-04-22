"""Project configuration and curated feature definitions for Information State."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    regexes: tuple[str, ...]
    description: str
    value_column: str = "valuenum"
    time_column: str = "charttime"
    agg: str = "last"


@dataclass
class ProjectConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    raw_root: Path | None = None
    cohort_name: str = "all_adult_icu"
    build_workers: int = 1
    bin_hours: int = 1
    window_hours: int = 24
    window_stride_hours: int = 2
    positive_window_gap_hours: int = 2
    delta_cap_hours: int = 48
    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    min_age: int = 18
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    random_seed: int = 7
    chunk_size: int = 200_000
    max_stays: int | None = None
    max_chunks: int | None = None

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        if self.raw_root is None:
            env_root = os.environ.get("MIMIC_IV_ROOT")
            self.raw_root = Path(env_root).resolve() if env_root else self.project_root / "data"
        else:
            self.raw_root = Path(self.raw_root).resolve()
        self.cohort_name = self.cohort_name.strip().lower().replace("-", "_")
        if self.cohort_name == "aki":
            self.cohort_name = "aki_kdigo"
        if self.cohort_name not in {"all_adult_icu", "aki_kdigo"}:
            raise ValueError(f"Unsupported cohort_name={self.cohort_name!r}.")
        if self.build_workers < 1:
            raise ValueError("build_workers must be at least 1.")

        hour_fields = {
            "window_hours": self.window_hours,
            "window_stride_hours": self.window_stride_hours,
            "positive_window_gap_hours": self.positive_window_gap_hours,
        }
        for field_name, field_value in hour_fields.items():
            if field_value % self.bin_hours != 0:
                raise ValueError(f"{field_name} must be divisible by bin_hours={self.bin_hours}.")
        if not 0.0 <= self.winsor_lower_quantile < self.winsor_upper_quantile <= 1.0:
            raise ValueError("Winsor quantiles must satisfy 0 <= lower < upper <= 1.")

    @property
    def hosp_dir(self) -> Path:
        return self.raw_root / "hosp"

    @property
    def icu_dir(self) -> Path:
        return self.raw_root / "icu"

    def source_path(self, relative_path: str) -> Path:
        path = self.raw_root / relative_path
        if path.exists():
            return path
        if str(path).endswith(".gz"):
            plain = Path(str(path)[:-3])
            if plain.exists():
                return plain
        else:
            gz = Path(f"{path}.gz")
            if gz.exists():
                return gz
        raise FileNotFoundError(f"Could not resolve source file for {relative_path}.")

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def cohort_slug(self) -> str:
        return self.cohort_name

    @property
    def catalog_path(self) -> Path:
        return self.artifacts_dir / "resolved_catalog.json"

    @property
    def window_bins(self) -> int:
        return self.window_hours // self.bin_hours

    @property
    def window_stride_bins(self) -> int:
        return self.window_stride_hours // self.bin_hours

    @property
    def positive_window_gap_bins(self) -> int:
        return self.positive_window_gap_hours // self.bin_hours

    @property
    def state_from_observation_dir(self) -> Path:
        base_dir = self.artifacts_dir / "state_from_observation"
        if self.cohort_name == "all_adult_icu":
            return base_dir
        return base_dir / self.cohort_slug

    @property
    def observation_cohort_path(self) -> Path:
        return self.state_from_observation_dir / "cohort.csv"

    @property
    def observation_hourly_values_path(self) -> Path:
        return self.state_from_observation_dir / "hourly_values.npy"

    @property
    def observation_hourly_masks_path(self) -> Path:
        return self.state_from_observation_dir / "hourly_masks.npy"

    @property
    def observation_hourly_deltas_path(self) -> Path:
        return self.state_from_observation_dir / "hourly_deltas.npy"

    @property
    def observation_window_metadata_path(self) -> Path:
        return self.state_from_observation_dir / "window_metadata.csv"

    @property
    def observation_metadata_path(self) -> Path:
        return self.state_from_observation_dir / "hourly_metadata.json"

    @property
    def observation_stats_path(self) -> Path:
        return self.state_from_observation_dir / "feature_stats.json"

    @property
    def observation_temp_last_time_path(self) -> Path:
        return self.state_from_observation_dir / "_tmp_last_seen_seconds.npy"

    @property
    def observation_checkpoint_path(self) -> Path:
        return self.state_from_observation_dir / "state_from_observation_ssl.pt"

    @property
    def observation_history_path(self) -> Path:
        return self.state_from_observation_dir / "ssl_history.json"

    @property
    def embeddings_dir(self) -> Path:
        return self.state_from_observation_dir / "embeddings"

    @property
    def clusters_dir(self) -> Path:
        return self.state_from_observation_dir / "clusters"

    @property
    def evaluation_dir(self) -> Path:
        return self.state_from_observation_dir / "evaluation"

    @property
    def robustness_dir(self) -> Path:
        return self.state_from_observation_dir / "robustness"

    @property
    def manifests_dir(self) -> Path:
        return self.state_from_observation_dir / "manifests"

FEATURE_SPECS = [
    FeatureSpec(
        name="heart_rate",
        source="chart",
        regexes=(r"^Heart Rate$",),
        description="ICU heart rate",
    ),
    FeatureSpec(
        name="sbp",
        source="chart",
        regexes=(
            r"^Arterial Blood Pressure systolic$",
            r"^Non Invasive Blood Pressure systolic$",
        ),
        description="Systolic blood pressure",
    ),
    FeatureSpec(
        name="dbp",
        source="chart",
        regexes=(
            r"^Arterial Blood Pressure diastolic$",
            r"^Non Invasive Blood Pressure diastolic$",
        ),
        description="Diastolic blood pressure",
    ),
    FeatureSpec(
        name="map",
        source="chart",
        regexes=(
            r"^Arterial Blood Pressure mean$",
            r"^Non Invasive Blood Pressure mean$",
        ),
        description="Mean arterial pressure",
    ),
    FeatureSpec(
        name="resp_rate",
        source="chart",
        regexes=(r"^Respiratory Rate$",),
        description="Respiratory rate",
    ),
    FeatureSpec(
        name="temperature_c",
        source="chart",
        regexes=(r"^Temperature Celsius$",),
        description="Body temperature",
    ),
    FeatureSpec(
        name="spo2",
        source="chart",
        regexes=(r"^SpO2$",),
        description="Oxygen saturation",
    ),
    FeatureSpec(
        name="weight_kg",
        source="chart",
        regexes=(r"^Daily Weight$", r"^Admission Weight \(Kg\)$", r"^Weight Kg$"),
        description="Patient weight",
    ),
    FeatureSpec(
        name="creatinine",
        source="lab",
        regexes=(r"^Creatinine$",),
        description="Creatinine",
    ),
    FeatureSpec(
        name="bun",
        source="lab",
        regexes=(r"^Urea Nitrogen$",),
        description="Blood urea nitrogen",
    ),
    FeatureSpec(
        name="sodium",
        source="lab",
        regexes=(r"^Sodium$",),
        description="Sodium",
    ),
    FeatureSpec(
        name="potassium",
        source="lab",
        regexes=(r"^Potassium$",),
        description="Potassium",
    ),
    FeatureSpec(
        name="chloride",
        source="lab",
        regexes=(r"^Chloride$",),
        description="Chloride",
    ),
    FeatureSpec(
        name="bicarbonate",
        source="lab",
        regexes=(r"^Bicarbonate$",),
        description="Bicarbonate",
    ),
    FeatureSpec(
        name="glucose",
        source="lab",
        regexes=(r"^Glucose$",),
        description="Glucose",
    ),
    FeatureSpec(
        name="lactate",
        source="lab",
        regexes=(r"^Lactate$",),
        description="Lactate",
    ),
    FeatureSpec(
        name="wbc",
        source="lab",
        regexes=(r"^WBC$",),
        description="White blood cell count",
    ),
    FeatureSpec(
        name="hemoglobin",
        source="lab",
        regexes=(r"^Hemoglobin$",),
        description="Hemoglobin",
    ),
    FeatureSpec(
        name="platelets",
        source="lab",
        regexes=(r"^Platelet Count$",),
        description="Platelet count",
    ),
    FeatureSpec(
        name="bilirubin_total",
        source="lab",
        regexes=(r"^Bilirubin, Total$",),
        description="Total bilirubin",
    ),
    FeatureSpec(
        name="urine_output_ml",
        source="output",
        regexes=(r"Urine",),
        description="Urine output volume",
        value_column="value",
        agg="sum",
    ),
    FeatureSpec(
        name="vasopressor_event",
        source="input",
        regexes=(
            r"Norepinephrine",
            r"Epinephrine",
            r"Phenylephrine",
            r"Vasopressin",
            r"Dopamine",
        ),
        description="Any vasopressor administration",
        value_column="indicator",
        time_column="starttime",
        agg="max",
    ),
]
