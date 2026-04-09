from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_s_aureus_same_episode_enriched import (
    ENRICHED_FEATURES,
    _load_ml_deps,
    _best_threshold_by_f1,
    _classification_metrics,
    _subject_split,
)


def _save_barplot(
    data: pd.DataFrame,
    value_col: str,
    label_col: str,
    title: str,
    path: Path,
    color: str,
    n: int = 15,
) -> None:
    plot_df = data.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df[label_col], plot_df[value_col], color=color)
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_heatmap(corr: pd.DataFrame, path: Path) -> None:
    values = corr.to_numpy()
    mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    masked_values = np.ma.array(values, mask=mask)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(masked_values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Enriched Same-Episode Feature Correlation Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    project_root = Path.cwd()
    features_path = (
        project_root / "artifacts" / "blood_culture" / "s_aureus_same_episode_primary_urgent_enriched_features.csv"
    )
    reports_dir = project_root / "reports"
    figures_dir = project_root / "figures" / "s_aureus_same_episode"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(features_path)
    train_subjects, val_subjects, test_subjects = _subject_split(
        data["subject_id"].drop_duplicates().astype(int).to_numpy(),
        seed=7,
    )
    deps = _load_ml_deps()
    data["split"] = np.where(
        data["subject_id"].isin(train_subjects),
        "train",
        np.where(data["subject_id"].isin(val_subjects), "val", "test"),
    )

    x_train = data.loc[data["split"] == "train", ENRICHED_FEATURES]
    x_val = data.loc[data["split"] == "val", ENRICHED_FEATURES]
    x_test = data.loc[data["split"] == "test", ENRICHED_FEATURES]
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
                            ENRICHED_FEATURES,
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

    logistic_coefs = pd.DataFrame(
        {"feature": ENRICHED_FEATURES, "coefficient": logistic.named_steps["model"].coef_[0]}
    )
    logistic_coefs["abs_coefficient"] = logistic_coefs["coefficient"].abs()
    logistic_coefs = logistic_coefs.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    dtest = xgb.DMatrix(x_test[ENRICHED_FEATURES], feature_names=ENRICHED_FEATURES)
    shap_values = xgb_model.get_booster().predict(dtest, pred_contribs=True)
    shap_importance = pd.DataFrame(
        {"feature": ENRICHED_FEATURES, "mean_abs_shap": np.abs(shap_values[:, :-1]).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    feature_frame = data[ENRICHED_FEATURES].copy()
    feature_stats = pd.DataFrame(
        {
            "feature": ENRICHED_FEATURES,
            "non_missing_count": [int(feature_frame[col].notna().sum()) for col in ENRICHED_FEATURES],
            "observation_rate": [float(feature_frame[col].notna().mean()) for col in ENRICHED_FEATURES],
            "unique_non_missing": [int(feature_frame[col].nunique(dropna=True)) for col in ENRICHED_FEATURES],
            "variance": [float(feature_frame[col].var(skipna=True)) for col in ENRICHED_FEATURES],
        }
    )
    nonconstant_features = feature_stats.loc[feature_stats["unique_non_missing"] > 1, "feature"].tolist()
    corr = feature_frame[nonconstant_features].corr()

    corr_pairs = []
    for i, left in enumerate(nonconstant_features):
        for right in nonconstant_features[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            corr_pairs.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "correlation": float(value),
                    "abs_correlation": float(abs(value)),
                }
            )
    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    logistic_coefs.to_csv(reports_dir / "s_aureus_same_episode_enriched_logistic_coefficients.csv", index=False)
    shap_importance.to_csv(reports_dir / "s_aureus_same_episode_enriched_xgb_shap_importance.csv", index=False)
    feature_stats.sort_values(["observation_rate", "feature"]).to_csv(
        reports_dir / "s_aureus_same_episode_enriched_feature_observation_rates.csv",
        index=False,
    )
    corr.to_csv(reports_dir / "s_aureus_same_episode_enriched_feature_correlation_matrix.csv", index=True)
    corr_pairs_df.to_csv(reports_dir / "s_aureus_same_episode_enriched_feature_correlation_pairs.csv", index=False)

    _save_barplot(
        shap_importance,
        value_col="mean_abs_shap",
        label_col="feature",
        title="Enriched Same-Episode XGBoost Mean Absolute SHAP",
        path=reports_dir / "s_aureus_same_episode_enriched_xgb_shap_importance.png",
        color="#2d6a4f",
    )
    _save_barplot(
        logistic_coefs,
        value_col="abs_coefficient",
        label_col="feature",
        title="Enriched Same-Episode Logistic Coefficients",
        path=reports_dir / "s_aureus_same_episode_enriched_logistic_coefficients.png",
        color="#1d3557",
    )
    _save_heatmap(corr, reports_dir / "s_aureus_same_episode_enriched_feature_correlation.png")
    shutil.copy2(
        reports_dir / "s_aureus_same_episode_enriched_xgb_shap_importance.png",
        figures_dir / "xgb_shap_importance.png",
    )
    shutil.copy2(
        reports_dir / "s_aureus_same_episode_enriched_logistic_coefficients.png",
        figures_dir / "logistic_coefficients.png",
    )
    shutil.copy2(
        reports_dir / "s_aureus_same_episode_enriched_feature_correlation.png",
        figures_dir / "feature_correlation.png",
    )

    summary = {
        "feature_set_name": "ENRICHED_SAME_EPISODE_FEATURES",
        "feature_count": len(ENRICHED_FEATURES),
        "cohort": {
            "rows": int(len(data)),
            "unique_patients": int(data["subject_id"].nunique()),
            "unique_admissions": int(data["hadm_id"].nunique()),
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
        "top_xgb_shap_features": shap_importance.head(15).to_dict(orient="records"),
        "top_logistic_features": logistic_coefs.head(15).to_dict(orient="records"),
        "top_correlation_pairs": corr_pairs_df.head(20).to_dict(orient="records"),
        "files": {
            "logistic_coefficients_csv": str(reports_dir / "s_aureus_same_episode_enriched_logistic_coefficients.csv"),
            "xgb_shap_importance_csv": str(reports_dir / "s_aureus_same_episode_enriched_xgb_shap_importance.csv"),
            "feature_observation_rates_csv": str(reports_dir / "s_aureus_same_episode_enriched_feature_observation_rates.csv"),
            "correlation_matrix_csv": str(reports_dir / "s_aureus_same_episode_enriched_feature_correlation_matrix.csv"),
            "correlation_pairs_csv": str(reports_dir / "s_aureus_same_episode_enriched_feature_correlation_pairs.csv"),
            "xgb_shap_importance_png": str(reports_dir / "s_aureus_same_episode_enriched_xgb_shap_importance.png"),
            "logistic_coefficients_png": str(reports_dir / "s_aureus_same_episode_enriched_logistic_coefficients.png"),
            "correlation_png": str(reports_dir / "s_aureus_same_episode_enriched_feature_correlation.png"),
        },
    }
    (reports_dir / "s_aureus_same_episode_enriched_explainability_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
