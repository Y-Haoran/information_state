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
    history_hours: int = 24
    future_hours: int = 6
    bin_hours: int = 1
    min_age: int = 18
    long_los_days: float = 7.0
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

    @property
    def hosp_dir(self) -> Path:
        return self.raw_root / "hosp"

    @property
    def icu_dir(self) -> Path:
        return self.raw_root / "icu"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def cohort_path(self) -> Path:
        return self.artifacts_dir / "cohort.csv"

    @property
    def catalog_path(self) -> Path:
        return self.artifacts_dir / "resolved_catalog.json"

    @property
    def sequence_dataset_path(self) -> Path:
        return self.artifacts_dir / "sequence_dataset.npz"

    @property
    def sequence_metadata_path(self) -> Path:
        return self.artifacts_dir / "sequence_metadata.json"

    @property
    def tabular_features_path(self) -> Path:
        return self.artifacts_dir / "tabular_features.csv"

    @property
    def tabular_metadata_path(self) -> Path:
        return self.artifacts_dir / "tabular_metadata.json"

    @property
    def history_bins(self) -> int:
        return self.history_hours // self.bin_hours


STATIC_NUMERIC_COLUMNS = [
    "anchor_age",
    "diag_count",
]

STATIC_CATEGORICAL_COLUMNS = [
    "gender",
    "admission_type",
    "insurance",
    "race",
    "first_careunit",
]

TASK_COLUMNS = [
    "in_hospital_mortality",
    "long_icu_los",
    "vasopressor_next_6h",
]

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
