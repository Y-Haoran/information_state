from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _subject_split(subjects: np.ndarray, seed: int = 7) -> tuple[set[int], set[int], set[int]]:
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(subjects), dtype=int)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train = set(subjects[:n_train].tolist())
    val = set(subjects[n_train:n_train + n_val].tolist())
    test = set(subjects[n_train + n_val:].tolist())
    return train, val, test


def _best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _binary_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def _binary_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapezoid(precision, recall))


def _binary_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true.astype(float)) ** 2))


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": _binary_auroc(y_true, y_prob),
        "auprc": _binary_auprc(y_true, y_prob),
        "brier": _binary_brier(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
    }


def _select_pruned_features(shap_df: pd.DataFrame, corr: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for feat in shap_df["feature"]:
        if feat not in corr.index:
            continue
        if all(abs(float(corr.loc[feat, kept])) <= 0.95 for kept in selected):
            selected.append(feat)
        if len(selected) >= 18:
            break

    for feat in ["anchor_age", "in_icu_at_alert", "vital_map_min_24h", "vital_temperature_c_max_24h", "lab_lactate_last_24h"]:
        if feat in corr.index and feat not in selected:
            if all(abs(float(corr.loc[feat, kept])) <= 0.98 for kept in selected):
                selected.append(feat)

    return [feat for feat in shap_df["feature"] if feat in selected]


def main() -> None:
    project_root = Path.cwd()
    features_path = project_root / "artifacts" / "blood_culture" / "first_gp_alert_features.csv"
    shap_path = project_root / "reports" / "blood_culture_primary_xgb_shap_importance.csv"
    corr_path = project_root / "reports" / "blood_culture_primary_feature_correlation_matrix.csv"
    out_path = project_root / "reports" / "blood_culture_important_pruned_metrics.json"

    shap_df = pd.read_csv(shap_path)
    corr = pd.read_csv(corr_path, index_col=0)
    selected = _select_pruned_features(shap_df, corr)

    df = pd.read_csv(features_path)
    data = df[df["is_high_confidence_binary"] == 1].copy()
    data["target_true_bsi"] = data["target_true_bsi"].astype(int)

    train_subjects, val_subjects, test_subjects = _subject_split(
        data["subject_id"].drop_duplicates().astype(int).to_numpy(),
        seed=7,
    )
    data["split"] = np.where(
        data["subject_id"].isin(train_subjects),
        "train",
        np.where(data["subject_id"].isin(val_subjects), "val", "test"),
    )

    X_train = data.loc[data["split"] == "train", selected]
    X_val = data.loc[data["split"] == "val", selected]
    X_test = data.loc[data["split"] == "test", selected]
    y_train = data.loc[data["split"] == "train", "target_true_bsi"].to_numpy(dtype=int)
    y_val = data.loc[data["split"] == "val", "target_true_bsi"].to_numpy(dtype=int)
    y_test = data.loc[data["split"] == "test", "target_true_bsi"].to_numpy(dtype=int)

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
    logistic.fit(X_train, y_train)
    val_prob_lr = logistic.predict_proba(X_val)[:, 1]
    threshold_lr, best_f1_lr = _best_threshold_by_f1(y_val, val_prob_lr)
    test_prob_lr = logistic.predict_proba(X_test)[:, 1]

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
    xgb_model.fit(X_train, y_train)
    val_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
    threshold_xgb, best_f1_xgb = _best_threshold_by_f1(y_val, val_prob_xgb)
    test_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    results = {
        "feature_set_name": "IMPORTANT_PRUNED_FEATURES",
        "feature_count": len(selected),
        "feature_list": selected,
        "cohort": {
            "rows": int(len(data)),
            "train_rows": int((data["split"] == "train").sum()),
            "val_rows": int((data["split"] == "val").sum()),
            "test_rows": int((data["split"] == "test").sum()),
            "train_positive": int(y_train.sum()),
            "val_positive": int(y_val.sum()),
            "test_positive": int(y_test.sum()),
        },
        "models": {
            "logistic_regression": {
                "validation": {**_classification_metrics(y_val, val_prob_lr, threshold_lr), "best_f1_scan": best_f1_lr},
                "test": _classification_metrics(y_test, test_prob_lr, threshold_lr),
            },
            "xgboost": {
                "validation": {**_classification_metrics(y_val, val_prob_xgb, threshold_xgb), "best_f1_scan": best_f1_xgb},
                "test": _classification_metrics(y_test, test_prob_xgb, threshold_xgb),
            },
        },
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
