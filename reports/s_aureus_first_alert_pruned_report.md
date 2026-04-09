# `S. aureus` Feature-Selection Sensitivity Analysis

## Why We Tried Feature Selection

The full secondary `S. aureus` baseline used the same clean `41` pre-alert features as the main project. That feature set is clinically reasonable, but it includes several correlated summaries and some sparse variables. Because the baseline performance was modest, it was worth checking whether a smaller, less noisy feature set would help.

## How Features Were Selected

We used a simple importance-pruning approach:

1. fit an XGBoost model on the training split
2. rank the `41` features by mean absolute SHAP contribution on the validation split
3. remove highly correlated near-duplicates using absolute correlation `> 0.95`
4. keep at most `15` non-redundant features

Saved files:

- [reports/s_aureus_first_alert_pruned_metrics.json](reports/s_aureus_first_alert_pruned_metrics.json)
- [reports/s_aureus_first_alert_shap_importance.csv](reports/s_aureus_first_alert_shap_importance.csv)
- [reports/s_aureus_first_alert_feature_correlation_matrix.csv](reports/s_aureus_first_alert_feature_correlation_matrix.csv)
- [scripts/train_s_aureus_pruned_baseline.py](../scripts/train_s_aureus_pruned_baseline.py)

## Selected 15 Features

- `lab_platelets_min_24h`
- `anchor_age`
- `lab_creatinine_last_24h`
- `prior_positive_specimens_7d`
- `in_icu_at_alert`
- `lab_lactate_last_24h`
- `lab_creatinine_count_24h`
- `lab_lactate_max_24h`
- `vital_heart_rate_last_24h`
- `vital_resp_rate_last_24h`
- `vital_heart_rate_min_24h`
- `vital_heart_rate_max_24h`
- `vital_resp_rate_max_24h`
- `vital_temperature_c_count_24h`
- `vital_map_last_24h`

These selected features reinforce the general pattern from the full baseline:

- platelet features
- creatinine features
- age
- ICU acuity
- lactate
- heart rate / respiratory rate

## Comparison With The 41-Feature Baseline

Held-out test comparison:

| Model | Feature Set | F1 | AUROC | AUPRC | Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 41 features | `0.424` | `0.612` | `0.373` | `0.239` |
| Logistic Regression | 15 features | `0.433` | `0.638` | `0.397` | `0.236` |
| XGBoost | 41 features | `0.430` | `0.608` | `0.349` | `0.231` |
| XGBoost | 15 features | `0.427` | `0.604` | `0.342` | `0.234` |

## Main Interpretation

Feature selection helped the simpler linear model more than the tree model.

For Logistic Regression:

- AUROC improved from `0.612` to `0.638`
- AUPRC improved from `0.373` to `0.397`
- F1 improved slightly from `0.424` to `0.433`

For XGBoost:

- performance stayed about the same or became slightly worse

That suggests the current `S. aureus` task has some real signal, but also enough noise and redundancy that simpler models benefit from pruning.

## What This Means

This is encouraging, but not a full solution.

The pruned model is cleaner and a little better for Logistic Regression, but the task is still only moderately learnable with the current feature set. So the main conclusion does not change:

- the `S. aureus` task is clinically meaningful
- the current features are not rich enough to make it a strong primary project
- better device, source, and microbiology-context features are still the main priority

## Bottom Line

Feature selection was worth doing.

It shows that:

- some of the original `41` features were likely noisy or redundant for this task
- a smaller `15`-feature set is a cleaner secondary baseline
- but richer clinically targeted features are still needed if we want this `S. aureus` task to become a strong standalone model
