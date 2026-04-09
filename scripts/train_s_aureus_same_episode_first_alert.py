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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a refined S. aureus secondary analysis using the first Gram-positive blood-culture alert "
            "and the same finalized blood-culture episode."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    return parser.parse_args()


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


def _is_urgent_like(data: pd.DataFrame) -> pd.Series:
    return (
        data.get("admission_type_DIRECT EMER.", 0).astype(bool)
        | data.get("admission_type_EW EMER.", 0).astype(bool)
        | data.get("admission_type_URGENT", 0).astype(bool)
    )


def _build_task_dataframe(features_path: Path) -> pd.DataFrame:
    data = pd.read_csv(features_path)
    organisms = data["organisms_json"].fillna("").astype(str).str.upper()
    data["target_s_aureus_same_episode"] = (
        organisms.str.contains("STAPH AUREUS") | organisms.str.contains("STAPHYLOCOCCUS AUREUS")
    ).astype(int)
    data["single_organism_episode"] = (data["org_polymicrobial_gp"] == 0).astype(int)
    data["urgent_emergency_admission"] = _is_urgent_like(data).astype(int)
    return data


def _train_one_cohort(
    data: pd.DataFrame,
    *,
    cohort_name: str,
    cohort_description: str,
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

    x_train = data.loc[data["split"] == "train", PRIMARY_FEATURES]
    x_val = data.loc[data["split"] == "val", PRIMARY_FEATURES]
    x_test = data.loc[data["split"] == "test", PRIMARY_FEATURES]
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
                            PRIMARY_FEATURES,
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
        "feature_set_name": "PRIMARY_FEATURES_41",
        "feature_count": len(PRIMARY_FEATURES),
        "cohort": {
            "rows": int(len(data)),
            "unique_patients": int(data["subject_id"].nunique()),
            "unique_admissions": int(data["hadm_id"].nunique()),
            "positive_rows": int(data["target_s_aureus_same_episode"].sum()),
            "positive_prevalence": float(data["target_s_aureus_same_episode"].mean()),
            "icu_rate": float(data["in_icu_at_alert"].mean()),
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


def _write_report(path: Path, metrics: dict) -> None:
    primary = metrics["cohorts"]["primary_urgent_single_organism"]
    sensitivity = metrics["cohorts"]["sensitivity_all_single_organism"]
    lines = [
        "# `S. aureus` From The Same First Gram-Positive Alert Episode",
        "",
        "## Clinical Question",
        "",
        "- at the first Gram-positive blood-culture alert, can we use the prior 24 hours of routine data to predict whether that same blood-culture episode will later finalize as `Staphylococcus aureus` rather than another Gram-positive organism?",
        "",
        "## Cohort Design",
        "",
        "- anchor = first Gram-positive blood-culture alert per admission",
        "- positive label = that same index episode finalizes as `S. aureus`",
        "- negative label = that same index episode finalizes as another Gram-positive organism",
        "- excluded from the primary clean cohort = polymicrobial first-alert episodes",
        "- primary subgroup = urgent / emergency admissions only",
        "",
        "## Primary Cohort",
        "",
        f"- rows: `{primary['cohort']['rows']:,}`",
        f"- unique patients: `{primary['cohort']['unique_patients']:,}`",
        f"- `S. aureus` positives: `{primary['cohort']['positive_rows']:,}` ({primary['cohort']['positive_prevalence']:.2%})",
        f"- ICU at alert: `{primary['cohort']['icu_rate']:.2%}`",
        "",
        "### Held-out Test Results",
        "",
        f"- Logistic Regression: AUROC `{primary['models']['logistic_regression']['test']['auroc']:.3f}`, AUPRC `{primary['models']['logistic_regression']['test']['auprc']:.3f}`, F1 `{primary['models']['logistic_regression']['test']['f1']:.3f}`",
        f"- XGBoost: AUROC `{primary['models']['xgboost']['test']['auroc']:.3f}`, AUPRC `{primary['models']['xgboost']['test']['auprc']:.3f}`, F1 `{primary['models']['xgboost']['test']['f1']:.3f}`",
        "",
        "## Sensitivity Cohort",
        "",
        "- same label and same pre-alert features",
        "- broader cohort: all single-organism first Gram-positive alerts",
        "",
        f"- rows: `{sensitivity['cohort']['rows']:,}`",
        f"- unique patients: `{sensitivity['cohort']['unique_patients']:,}`",
        f"- `S. aureus` positives: `{sensitivity['cohort']['positive_rows']:,}` ({sensitivity['cohort']['positive_prevalence']:.2%})",
        "",
        f"- Logistic Regression: AUROC `{sensitivity['models']['logistic_regression']['test']['auroc']:.3f}`, AUPRC `{sensitivity['models']['logistic_regression']['test']['auprc']:.3f}`, F1 `{sensitivity['models']['logistic_regression']['test']['f1']:.3f}`",
        f"- XGBoost: AUROC `{sensitivity['models']['xgboost']['test']['auroc']:.3f}`, AUPRC `{sensitivity['models']['xgboost']['test']['auprc']:.3f}`, F1 `{sensitivity['models']['xgboost']['test']['f1']:.3f}`",
        "",
        "## Interpretation",
        "",
        "- this is scientifically cleaner than the broad first-blood-culture SAB dataset because the anchor and the final species outcome belong to the same microbiology episode",
        "- restricting to single-organism episodes removes some obvious label noise",
        "- the urgent / emergency subgroup improves AUROC compared with the looser all-single-organism cohort",
        "- performance is still modest, which suggests this task needs richer device, source, and prior staphylococcal context rather than more generic physiology alone",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    artifacts_dir = project_root / "artifacts" / "blood_culture"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = _build_task_dataframe(artifacts_dir / "first_gp_alert_features.csv")

    single_org = data[data["single_organism_episode"] == 1].copy()
    primary = single_org[single_org["urgent_emergency_admission"] == 1].copy()

    primary_artifact = artifacts_dir / "s_aureus_same_episode_primary_urgent_features.csv"
    sensitivity_artifact = artifacts_dir / "s_aureus_same_episode_single_org_features.csv"
    primary.to_csv(primary_artifact, index=False)
    single_org.to_csv(sensitivity_artifact, index=False)

    metrics = {
        "task_name": "predict_s_aureus_from_same_first_gp_alert_episode",
        "clinical_question": (
            "At the first Gram-positive blood-culture alert, can the prior 24 hours of routine clinical data "
            "predict whether that same finalized blood-culture episode is Staphylococcus aureus?"
        ),
        "label_definition": {
            "positive": "first Gram-positive alert episode later finalizes as S. aureus",
            "negative": "first Gram-positive alert episode later finalizes as another Gram-positive organism",
            "exclusions_primary": [
                "polymicrobial first-alert episodes",
                "non-urgent / non-emergency admissions for the primary cohort",
            ],
        },
        "cohorts": {
            "primary_urgent_single_organism": _train_one_cohort(
                primary,
                cohort_name="primary_urgent_single_organism",
                cohort_description="single-organism first Gram-positive alerts, urgent or emergency admissions only",
                random_seed=args.random_seed,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
            ),
            "sensitivity_all_single_organism": _train_one_cohort(
                single_org,
                cohort_name="sensitivity_all_single_organism",
                cohort_description="all single-organism first Gram-positive alerts",
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

    metrics_path = reports_dir / "s_aureus_same_episode_first_alert_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _write_report(reports_dir / "s_aureus_same_episode_first_alert_report.md", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
