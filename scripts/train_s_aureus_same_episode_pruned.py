from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_s_aureus_same_episode_enriched import (
    _best_threshold_by_f1,
    _classification_metrics,
    _load_ml_deps,
    _subject_split,
)


def _select_pruned_features(shap_df: pd.DataFrame, corr: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for feat in shap_df["feature"]:
        if feat not in corr.index:
            continue
        if all(abs(float(corr.loc[feat, kept])) <= 0.95 for kept in selected):
            selected.append(feat)
        if len(selected) >= 18:
            break

    required = [
        "anchor_age",
        "index_hours_draw_to_alert",
        "prior_subject_s_aureus_positive_365d",
        "prior_subject_cons_positive_365d",
        "prealert_blood_culture_draws_24h",
        "lab_creatinine_last_24h",
        "lab_platelets_last_24h",
        "vital_temperature_c_max_24h",
    ]
    for feat in required:
        if feat in corr.index and feat not in selected:
            if all(abs(float(corr.loc[feat, kept])) <= 0.98 for kept in selected):
                selected.append(feat)
    return [feat for feat in shap_df["feature"] if feat in selected]


def main() -> None:
    project_root = Path.cwd()
    features_path = (
        project_root / "artifacts" / "blood_culture" / "s_aureus_same_episode_primary_urgent_enriched_features.csv"
    )
    shap_path = project_root / "reports" / "s_aureus_same_episode_enriched_xgb_shap_importance.csv"
    corr_path = project_root / "reports" / "s_aureus_same_episode_enriched_feature_correlation_matrix.csv"
    out_path = project_root / "reports" / "s_aureus_same_episode_pruned_metrics.json"

    shap_df = pd.read_csv(shap_path)
    corr = pd.read_csv(corr_path, index_col=0)
    selected = _select_pruned_features(shap_df, corr)
    deps = _load_ml_deps()

    data = pd.read_csv(features_path)
    train_subjects, val_subjects, test_subjects = _subject_split(
        data["subject_id"].drop_duplicates().astype(int).to_numpy(),
        seed=7,
    )
    data["split"] = np.where(
        data["subject_id"].isin(train_subjects),
        "train",
        np.where(data["subject_id"].isin(val_subjects), "val", "test"),
    )

    x_train = data.loc[data["split"] == "train", selected]
    x_val = data.loc[data["split"] == "val", selected]
    x_test = data.loc[data["split"] == "test", selected]
    y_train = data.loc[data["split"] == "train", "target_s_aureus_same_episode"].to_numpy(dtype=int)
    y_val = data.loc[data["split"] == "val", "target_s_aureus_same_episode"].to_numpy(dtype=int)
    y_test = data.loc[data["split"] == "test", "target_s_aureus_same_episode"].to_numpy(dtype=int)

    logistic = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            selected,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    logistic.fit(x_train, y_train)
    val_prob_lr = logistic.predict_proba(x_val)[:, 1]
    threshold_lr, best_f1_lr = _best_threshold_by_f1(y_val, val_prob_lr, deps)
    test_prob_lr = logistic.predict_proba(x_test)[:, 1]

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=7,
        n_jobs=8,
    )
    xgb_model.fit(x_train, y_train)
    val_prob_xgb = xgb_model.predict_proba(x_val)[:, 1]
    threshold_xgb, best_f1_xgb = _best_threshold_by_f1(y_val, val_prob_xgb, deps)
    test_prob_xgb = xgb_model.predict_proba(x_test)[:, 1]

    results = {
        "task_name": "s_aureus_same_episode_pruned",
        "feature_set_name": "PRUNED_ENRICHED_FEATURES",
        "feature_count": len(selected),
        "feature_list": selected,
        "cohort": {
            "rows": int(len(data)),
            "unique_patients": int(data["subject_id"].nunique()),
            "unique_admissions": int(data["hadm_id"].nunique()),
            "positive_rows": int(data["target_s_aureus_same_episode"].sum()),
            "positive_prevalence": float(data["target_s_aureus_same_episode"].mean()),
        },
        "models": {
            "logistic_regression": {
                "validation": {**_classification_metrics(y_val, val_prob_lr, threshold_lr, deps), "best_f1_scan": best_f1_lr},
                "test": _classification_metrics(y_test, test_prob_lr, threshold_lr, deps),
            },
            "xgboost": {
                "validation": {**_classification_metrics(y_val, val_prob_xgb, threshold_xgb, deps), "best_f1_scan": best_f1_xgb},
                "test": _classification_metrics(y_test, test_prob_xgb, threshold_xgb, deps),
            },
        },
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
