from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mimic_iv_project.metrics import binary_auprc, binary_auroc, binary_brier


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

PRIMARY_FEATURES = [
    "anchor_age",
    "in_icu_at_alert",
    "vasopressor_event_count_24h",
    "vasopressor_active_24h",
    "vasopressor_on_at_alert",
    "mechanical_ventilation_chart_events_24h",
    "mechanical_ventilation_24h",
    "prior_positive_specimens_24h",
    "prior_positive_specimens_7d",
    "prior_gp_positive_specimens_24h",
    "prior_gp_positive_specimens_7d",
    "lab_wbc_last_24h",
    "lab_wbc_min_24h",
    "lab_wbc_max_24h",
    "lab_wbc_count_24h",
    "lab_platelets_last_24h",
    "lab_platelets_min_24h",
    "lab_platelets_count_24h",
    "lab_creatinine_last_24h",
    "lab_creatinine_max_24h",
    "lab_creatinine_count_24h",
    "lab_lactate_last_24h",
    "lab_lactate_max_24h",
    "lab_lactate_count_24h",
    "vital_heart_rate_last_24h",
    "vital_heart_rate_min_24h",
    "vital_heart_rate_max_24h",
    "vital_heart_rate_count_24h",
    "vital_resp_rate_last_24h",
    "vital_resp_rate_max_24h",
    "vital_resp_rate_count_24h",
    "vital_temperature_c_last_24h",
    "vital_temperature_c_min_24h",
    "vital_temperature_c_max_24h",
    "vital_temperature_c_count_24h",
    "vital_map_last_24h",
    "vital_map_min_24h",
    "vital_map_count_24h",
    "vital_spo2_last_24h",
    "vital_spo2_min_24h",
    "vital_spo2_count_24h",
]

ENRICHED_FEATURES = PRIMARY_FEATURES + [
    "index_hours_draw_to_alert",
    "prealert_blood_culture_draws_6h",
    "prealert_blood_culture_draws_24h",
    "prealert_blood_culture_draws_7d",
    "prealert_additional_draws_24h",
    "prior_subject_s_aureus_positive_90d",
    "prior_subject_s_aureus_positive_365d",
    "prior_subject_cons_positive_90d",
    "prior_subject_cons_positive_365d",
    "prior_subject_any_staph_positive_365d",
    "prior_subject_s_aureus_positive_365d_flag",
    "prior_subject_cons_positive_365d_flag",
    "prior_subject_any_staph_positive_365d_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train enriched same-episode S. aureus models with process and prior-staph history features."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
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


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=DATETIME_FORMAT, errors="coerce", cache=True)


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


def _count_prior_events(sorted_ns: np.ndarray, current_ns: int, window_days: int | None) -> int:
    if sorted_ns.size == 0:
        return 0
    right = np.searchsorted(sorted_ns, current_ns, side="left")
    if window_days is None:
        return int(right)
    lower_ns = current_ns - window_days * 24 * 3600 * 1_000_000_000
    left = np.searchsorted(sorted_ns, lower_ns, side="left")
    return int(max(right - left, 0))


def _build_base_dataframe(features_path: Path) -> pd.DataFrame:
    data = pd.read_csv(features_path)
    organisms = data["organisms_json"].fillna("").astype(str).str.upper()
    data["target_s_aureus_same_episode"] = (
        organisms.str.contains("STAPH AUREUS") | organisms.str.contains("STAPHYLOCOCCUS AUREUS")
    ).astype(int)
    data["single_organism_episode"] = (data["org_polymicrobial_gp"] == 0).astype(int)
    data["urgent_emergency_admission"] = (
        data.get("admission_type_DIRECT EMER.", 0).astype(bool)
        | data.get("admission_type_EW EMER.", 0).astype(bool)
        | data.get("admission_type_URGENT", 0).astype(bool)
    ).astype(int)
    data["alert_time"] = _parse_datetime(data["alert_time"])
    return data


def _build_enriched_features(data: pd.DataFrame, specimen_path: Path) -> pd.DataFrame:
    specimen = pd.read_csv(
        specimen_path,
        usecols=[
            "micro_specimen_id",
            "subject_id",
            "hadm_id",
            "specimen_draw_time",
            "first_positive_result_time",
            "organisms_json",
        ],
    )
    specimen["specimen_draw_time"] = _parse_datetime(specimen["specimen_draw_time"])
    specimen["first_positive_result_time"] = _parse_datetime(specimen["first_positive_result_time"])
    specimen["micro_specimen_id"] = pd.to_numeric(specimen["micro_specimen_id"], errors="coerce").astype("Int64")
    specimen["subject_id"] = pd.to_numeric(specimen["subject_id"], errors="coerce").astype("Int64")
    specimen["hadm_id"] = pd.to_numeric(specimen["hadm_id"], errors="coerce").astype("Int64")

    out = data.merge(
        specimen[["micro_specimen_id", "specimen_draw_time"]].rename(columns={"specimen_draw_time": "index_specimen_draw_time"}),
        on="micro_specimen_id",
        how="left",
    )
    out["index_specimen_draw_time"] = _parse_datetime(out["index_specimen_draw_time"])
    out["index_hours_draw_to_alert"] = (
        (out["alert_time"] - out["index_specimen_draw_time"]).dt.total_seconds() / 3600.0
    )

    hadm_draws: dict[int, np.ndarray] = {}
    specimen_draws = specimen.dropna(subset=["hadm_id", "specimen_draw_time"]).copy()
    specimen_draws["hadm_id"] = specimen_draws["hadm_id"].astype(int)
    for hadm_id, group in specimen_draws.groupby("hadm_id", sort=False):
        hadm_draws[int(hadm_id)] = np.sort(group["specimen_draw_time"].astype("int64").to_numpy())

    org_upper = specimen["organisms_json"].fillna("").astype(str).str.upper()
    specimen["is_s_aureus_positive"] = (
        specimen["first_positive_result_time"].notna()
        & (org_upper.str.contains("STAPH AUREUS") | org_upper.str.contains("STAPHYLOCOCCUS AUREUS"))
    ).astype(int)
    specimen["is_cons_positive"] = (
        specimen["first_positive_result_time"].notna()
        & (
            org_upper.str.contains("COAGULASE NEGATIVE")
            | org_upper.str.contains("STAPHYLOCOCCUS EPIDERMIDIS")
        )
    ).astype(int)
    specimen["is_any_staph_positive"] = (
        specimen["first_positive_result_time"].notna() & org_upper.str.contains("STAPH")
    ).astype(int)

    subject_histories: dict[str, dict[int, np.ndarray]] = {
        "s_aureus": {},
        "cons": {},
        "any_staph": {},
    }
    for feature_name, flag_col in [
        ("s_aureus", "is_s_aureus_positive"),
        ("cons", "is_cons_positive"),
        ("any_staph", "is_any_staph_positive"),
    ]:
        subset = specimen[(specimen[flag_col] == 1) & specimen["subject_id"].notna()].copy()
        subset["subject_id"] = subset["subject_id"].astype(int)
        for subject_id, group in subset.groupby("subject_id", sort=False):
            subject_histories[feature_name][int(subject_id)] = np.sort(
                group["first_positive_result_time"].astype("int64").to_numpy()
            )

    out["prealert_blood_culture_draws_6h"] = 0
    out["prealert_blood_culture_draws_24h"] = 0
    out["prealert_blood_culture_draws_7d"] = 0
    out["prealert_additional_draws_24h"] = 0
    out["prior_subject_s_aureus_positive_90d"] = 0
    out["prior_subject_s_aureus_positive_365d"] = 0
    out["prior_subject_cons_positive_90d"] = 0
    out["prior_subject_cons_positive_365d"] = 0
    out["prior_subject_any_staph_positive_365d"] = 0

    for idx, row in out[["hadm_id", "subject_id", "alert_time"]].iterrows():
        hadm_id = int(row["hadm_id"])
        subject_id = int(row["subject_id"])
        alert_time = row["alert_time"]
        if pd.notna(alert_time):
            alert_ns = int(pd.Timestamp(alert_time).value)
            draw_ns = hadm_draws.get(hadm_id)
            if draw_ns is not None and draw_ns.size:
                right = np.searchsorted(draw_ns, alert_ns, side="right")
                left_6h = np.searchsorted(draw_ns, alert_ns - 6 * 3600 * 1_000_000_000, side="left")
                left_24h = np.searchsorted(draw_ns, alert_ns - 24 * 3600 * 1_000_000_000, side="left")
                left_7d = np.searchsorted(draw_ns, alert_ns - 7 * 24 * 3600 * 1_000_000_000, side="left")
                out.at[idx, "prealert_blood_culture_draws_6h"] = int(max(right - left_6h, 0))
                out.at[idx, "prealert_blood_culture_draws_24h"] = int(max(right - left_24h, 0))
                out.at[idx, "prealert_blood_culture_draws_7d"] = int(max(right - left_7d, 0))
                out.at[idx, "prealert_additional_draws_24h"] = int(max(right - left_24h - 1, 0))

            out.at[idx, "prior_subject_s_aureus_positive_90d"] = _count_prior_events(
                subject_histories["s_aureus"].get(subject_id, np.array([], dtype=np.int64)),
                alert_ns,
                90,
            )
            out.at[idx, "prior_subject_s_aureus_positive_365d"] = _count_prior_events(
                subject_histories["s_aureus"].get(subject_id, np.array([], dtype=np.int64)),
                alert_ns,
                365,
            )
            out.at[idx, "prior_subject_cons_positive_90d"] = _count_prior_events(
                subject_histories["cons"].get(subject_id, np.array([], dtype=np.int64)),
                alert_ns,
                90,
            )
            out.at[idx, "prior_subject_cons_positive_365d"] = _count_prior_events(
                subject_histories["cons"].get(subject_id, np.array([], dtype=np.int64)),
                alert_ns,
                365,
            )
            out.at[idx, "prior_subject_any_staph_positive_365d"] = _count_prior_events(
                subject_histories["any_staph"].get(subject_id, np.array([], dtype=np.int64)),
                alert_ns,
                365,
            )

    out["prior_subject_s_aureus_positive_365d_flag"] = (out["prior_subject_s_aureus_positive_365d"] > 0).astype(int)
    out["prior_subject_cons_positive_365d_flag"] = (out["prior_subject_cons_positive_365d"] > 0).astype(int)
    out["prior_subject_any_staph_positive_365d_flag"] = (out["prior_subject_any_staph_positive_365d"] > 0).astype(int)
    return out


def _train_one_cohort(
    data: pd.DataFrame,
    *,
    cohort_name: str,
    cohort_description: str,
    feature_columns: list[str],
    random_seed: int,
    n_estimators: int,
    max_depth: int,
):
    deps = _load_ml_deps()
    train_subjects, val_subjects, test_subjects = _subject_split(
        data["subject_id"].drop_duplicates().astype(int).to_numpy(),
        seed=random_seed,
    )
    data = data.copy()
    data["split"] = np.where(
        data["subject_id"].isin(train_subjects),
        "train",
        np.where(data["subject_id"].isin(val_subjects), "val", "test"),
    )

    x_train = data.loc[data["split"] == "train", feature_columns]
    x_val = data.loc[data["split"] == "val", feature_columns]
    x_test = data.loc[data["split"] == "test", feature_columns]
    y_train = data.loc[data["split"] == "train", "target_s_aureus_same_episode"].to_numpy(dtype=int)
    y_val = data.loc[data["split"] == "val", "target_s_aureus_same_episode"].to_numpy(dtype=int)
    y_test = data.loc[data["split"] == "test", "target_s_aureus_same_episode"].to_numpy(dtype=int)

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
        "cohort_name": cohort_name,
        "cohort_description": cohort_description,
        "feature_set_name": "PRIMARY_PLUS_PROCESS_AND_STAPH_HISTORY",
        "feature_count": len(feature_columns),
        "cohort": {
            "rows": int(len(data)),
            "unique_patients": int(data["subject_id"].nunique()),
            "unique_admissions": int(data["hadm_id"].nunique()),
            "positive_rows": int(data["target_s_aureus_same_episode"].sum()),
            "positive_prevalence": float(data["target_s_aureus_same_episode"].mean()),
            "icu_rate": float(data["in_icu_at_alert"].mean()),
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


def _write_report(path: Path, metrics: dict) -> None:
    primary = metrics["cohorts"]["primary_urgent_single_organism"]
    sensitivity = metrics["cohorts"]["sensitivity_all_single_organism"]
    lines = [
        "# Enriched `S. aureus` Same-Episode First-Alert Model",
        "",
        "## Added Features",
        "",
        "- process feature: draw-to-alert time",
        "- process features: number of blood-culture draws already present before alert in the prior 6h, 24h, and 7d",
        "- prior subject history: previous positive `S. aureus`, CoNS, and any-staphylococcal blood cultures",
        "",
        "## Primary Cohort",
        "",
        f"- rows: `{primary['cohort']['rows']:,}`",
        f"- `S. aureus` positives: `{primary['cohort']['positive_rows']:,}` ({primary['cohort']['positive_prevalence']:.2%})",
        "",
        "### Held-out Test Results",
        "",
        f"- Logistic Regression: AUROC `{primary['models']['logistic_regression']['test']['auroc']:.3f}`, AUPRC `{primary['models']['logistic_regression']['test']['auprc']:.3f}`, F1 `{primary['models']['logistic_regression']['test']['f1']:.3f}`",
        f"- XGBoost: AUROC `{primary['models']['xgboost']['test']['auroc']:.3f}`, AUPRC `{primary['models']['xgboost']['test']['auprc']:.3f}`, F1 `{primary['models']['xgboost']['test']['f1']:.3f}`",
        "",
        "## Sensitivity Cohort",
        "",
        f"- rows: `{sensitivity['cohort']['rows']:,}`",
        f"- `S. aureus` positives: `{sensitivity['cohort']['positive_rows']:,}` ({sensitivity['cohort']['positive_prevalence']:.2%})",
        "",
        f"- Logistic Regression: AUROC `{sensitivity['models']['logistic_regression']['test']['auroc']:.3f}`, AUPRC `{sensitivity['models']['logistic_regression']['test']['auprc']:.3f}`, F1 `{sensitivity['models']['logistic_regression']['test']['f1']:.3f}`",
        f"- XGBoost: AUROC `{sensitivity['models']['xgboost']['test']['auroc']:.3f}`, AUPRC `{sensitivity['models']['xgboost']['test']['auprc']:.3f}`, F1 `{sensitivity['models']['xgboost']['test']['f1']:.3f}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    artifacts_dir = project_root / "artifacts" / "blood_culture"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = _build_base_dataframe(artifacts_dir / "first_gp_alert_features.csv")
    enriched = _build_enriched_features(data, artifacts_dir / "blood_culture_specimen_subset.csv")

    single_org = enriched[enriched["single_organism_episode"] == 1].copy()
    primary = single_org[single_org["urgent_emergency_admission"] == 1].copy()

    primary_artifact = artifacts_dir / "s_aureus_same_episode_primary_urgent_enriched_features.csv"
    sensitivity_artifact = artifacts_dir / "s_aureus_same_episode_single_org_enriched_features.csv"
    primary.to_csv(primary_artifact, index=False)
    single_org.to_csv(sensitivity_artifact, index=False)

    metrics = {
        "task_name": "predict_s_aureus_from_same_first_gp_alert_episode_enriched",
        "clinical_question": (
            "At the first Gram-positive alert, can added process and prior staph-history features improve "
            "prediction of whether that same episode finalizes as S. aureus?"
        ),
        "added_feature_blocks": [
            "index draw-to-alert time",
            "pre-alert blood-culture draw counts",
            "prior subject S. aureus / CoNS / staphylococcal blood-culture history",
        ],
        "cohorts": {
            "primary_urgent_single_organism": _train_one_cohort(
                primary,
                cohort_name="primary_urgent_single_organism",
                cohort_description="single-organism first Gram-positive alerts, urgent or emergency admissions only",
                feature_columns=ENRICHED_FEATURES,
                random_seed=args.random_seed,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
            ),
            "sensitivity_all_single_organism": _train_one_cohort(
                single_org,
                cohort_name="sensitivity_all_single_organism",
                cohort_description="all single-organism first Gram-positive alerts",
                feature_columns=ENRICHED_FEATURES,
                random_seed=args.random_seed,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
            ),
        },
        "files": {
            "primary_features_csv": str(primary_artifact),
            "sensitivity_features_csv": str(sensitivity_artifact),
        },
    }

    metrics_path = reports_dir / "s_aureus_same_episode_enriched_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _write_report(reports_dir / "s_aureus_same_episode_enriched_report.md", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
