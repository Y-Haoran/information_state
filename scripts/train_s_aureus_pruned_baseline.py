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
        description="Run importance-based feature selection for the secondary S. aureus first-alert task."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=15)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
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


def _select_nonredundant_features(
    importance_df: pd.DataFrame,
    corr: pd.DataFrame,
    max_features: int,
    corr_threshold: float,
) -> list[str]:
    selected: list[str] = []
    for feat in importance_df["feature"]:
        if feat not in corr.index:
            continue
        if all(abs(float(corr.loc[feat, kept])) <= corr_threshold for kept in selected):
            selected.append(feat)
        if len(selected) >= max_features:
            break
    return selected


def main() -> None:
    args = parse_args()
    deps = _load_ml_deps()
    project_root = args.project_root.resolve()
    features_path = project_root / "artifacts" / "blood_culture" / "first_gp_alert_features.csv"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(features_path)
    organisms = data["organisms_json"].fillna("").astype(str).str.upper()
    data["target_s_aureus"] = (
        organisms.str.contains("STAPH AUREUS") | organisms.str.contains("STAPHYLOCOCCUS AUREUS")
    ).astype(int)

    train_subjects, val_subjects, test_subjects = _subject_split(
        data["subject_id"].drop_duplicates().astype(int).to_numpy(),
        seed=args.random_seed,
    )
    data["split"] = np.where(
        data["subject_id"].isin(train_subjects),
        "train",
        np.where(data["subject_id"].isin(val_subjects), "val", "test"),
    )

    x_train = data.loc[data["split"] == "train", PRIMARY_FEATURES]
    x_val = data.loc[data["split"] == "val", PRIMARY_FEATURES]
    x_test = data.loc[data["split"] == "test", PRIMARY_FEATURES]
    y_train = data.loc[data["split"] == "train", "target_s_aureus"].to_numpy(dtype=int)
    y_val = data.loc[data["split"] == "val", "target_s_aureus"].to_numpy(dtype=int)
    y_test = data.loc[data["split"] == "test", "target_s_aureus"].to_numpy(dtype=int)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    selector_model = deps["xgb"].XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_seed,
        n_jobs=8,
    )
    selector_model.fit(x_train, y_train)

    dval = deps["xgb"].DMatrix(x_val[PRIMARY_FEATURES], feature_names=PRIMARY_FEATURES)
    shap_values = selector_model.get_booster().predict(dval, pred_contribs=True)
    importance_df = (
        pd.DataFrame(
            {
                "feature": PRIMARY_FEATURES,
                "mean_abs_shap": np.abs(shap_values[:, :-1]).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    feature_frame = data[PRIMARY_FEATURES].copy()
    nonconstant_features = [col for col in PRIMARY_FEATURES if feature_frame[col].nunique(dropna=True) > 1]
    corr = feature_frame[nonconstant_features].corr()
    selected = _select_nonredundant_features(importance_df, corr, args.max_features, args.corr_threshold)

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
                            selected,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", deps["LogisticRegression"](max_iter=2000, class_weight="balanced")),
        ]
    )
    logistic.fit(x_train[selected], y_train)
    val_prob_lr = logistic.predict_proba(x_val[selected])[:, 1]
    threshold_lr, best_val_f1_lr = _best_threshold_by_f1(y_val, val_prob_lr, deps)
    test_prob_lr = logistic.predict_proba(x_test[selected])[:, 1]

    pruned_xgb = deps["xgb"].XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_seed,
        n_jobs=8,
    )
    pruned_xgb.fit(x_train[selected], y_train)
    val_prob_xgb = pruned_xgb.predict_proba(x_val[selected])[:, 1]
    threshold_xgb, best_val_f1_xgb = _best_threshold_by_f1(y_val, val_prob_xgb, deps)
    test_prob_xgb = pruned_xgb.predict_proba(x_test[selected])[:, 1]

    importance_df.to_csv(reports_dir / "s_aureus_first_alert_shap_importance.csv", index=False)
    corr.to_csv(reports_dir / "s_aureus_first_alert_feature_correlation_matrix.csv")

    results = {
        "task_name": "predict_later_confirmed_s_aureus_from_first_gp_alert",
        "feature_selection_method": {
            "importance_source": "xgboost_mean_abs_shap_on_validation_split",
            "correlation_threshold": args.corr_threshold,
            "max_features": args.max_features,
        },
        "feature_set_name": "PRUNED_PRIMARY_FEATURES",
        "feature_count": len(selected),
        "feature_list": selected,
        "cohort": {
            "rows": int(len(data)),
            "unique_patients": int(data["subject_id"].nunique()),
            "unique_admissions": int(data["hadm_id"].nunique()),
            "positive_rows": int(data["target_s_aureus"].sum()),
            "positive_prevalence": float(data["target_s_aureus"].mean()),
            "train_rows": int((data["split"] == "train").sum()),
            "val_rows": int((data["split"] == "val").sum()),
            "test_rows": int((data["split"] == "test").sum()),
            "train_positive": int(y_train.sum()),
            "val_positive": int(y_val.sum()),
            "test_positive": int(y_test.sum()),
        },
        "top_importance_features": importance_df.head(15).to_dict(orient="records"),
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

    out_path = reports_dir / "s_aureus_first_alert_pruned_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
