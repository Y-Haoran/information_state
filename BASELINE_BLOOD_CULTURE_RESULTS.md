# Baseline Results For The First-Alert Model

## Main Baseline

The main baseline in this repo is the **41-feature first-alert model**.

It predicts:

- `0` = `probable_contaminant_or_low_significance_alert`
- `1` = `probable_clinically_significant_bsi_alert`

using only pre-alert clinical information.

## Cohort

- total first-alert rows: `5,546`
- high-confidence binary rows used for training: `2,506`

Split used for the 41-feature baseline:

- train rows: `1,764`
- validation rows: `374`
- test rows: `368`

Subject-level counts:

- train subjects: `1,658`
- validation subjects: `355`
- test subjects: `356`

## Feature Set

The clean baseline uses `41` features:

- age
- ICU status
- vasopressor and ventilation proxies
- prior microbiology counts
- WBC
- platelets
- creatinine
- lactate
- heart rate
- respiratory rate
- temperature
- MAP
- SpO2

Full list:

- [BLOOD_CULTURE_FEATURE_REFERENCE.md](BLOOD_CULTURE_FEATURE_REFERENCE.md)

Raw metrics JSON:

- [reports/blood_culture_primary_feature_metrics.json](reports/blood_culture_primary_feature_metrics.json)
- [PRIMARY_BASELINE_EXPLAINABILITY.md](PRIMARY_BASELINE_EXPLAINABILITY.md)

## Held-Out Test Results

| Model | F1 | Precision | Recall | Accuracy | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `0.767` | `0.638` | `0.961` | `0.717` | `0.798` | `0.728` | `0.182` |
| XGBoost | `0.761` | `0.653` | `0.910` | `0.723` | `0.809` | `0.765` | `0.178` |

Confusion counts:

- Logistic Regression: `TP=171`, `TN=93`, `FP=97`, `FN=7`
- XGBoost: `TP=162`, `TN=104`, `FP=86`, `FN=16`

## Interpretation

This is the clean result to focus on.

It is weaker than the larger exploratory model, but it is easier to defend because:

- it avoids organism-family shortcuts
- it avoids post-alert information
- it matches the first-alert clinical question directly

The tradeoff is clear:

- Logistic Regression gives slightly better F1 and recall
- XGBoost gives slightly better AUROC, AUPRC, accuracy, and Brier score

## Secondary Results

### Secondary sensitivity analysis: 18-feature pruned model

We also trained a smaller model using only the most important features after SHAP ranking and correlation pruning.

Saved metrics:

- [reports/blood_culture_important_pruned_metrics.json](reports/blood_culture_important_pruned_metrics.json)
- [scripts/train_pruned_feature_baseline.py](scripts/train_pruned_feature_baseline.py)

Held-out test results for the 18-feature pruned model:

| Model | F1 | Precision | Recall | Accuracy | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `0.771` | `0.646` | `0.955` | `0.726` | `0.791` | `0.723` | `0.188` |
| XGBoost | `0.756` | `0.641` | `0.921` | `0.712` | `0.808` | `0.757` | `0.178` |

Interpretation:

- pruning did **not** clearly outperform the 41-feature baseline
- logistic regression changed only slightly
- XGBoost stayed very similar
- this smaller model is still useful as a robustness and interpretability check

### Practical summary

The current hierarchy is:

- main baseline: 41-feature first-alert model
- secondary sensitivity model: 18-feature important-pruned model
- larger exploratory models: secondary only
