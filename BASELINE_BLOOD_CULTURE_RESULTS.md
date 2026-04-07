# Baseline Results For Gram-Positive Blood-Culture Labels

## Important Note

This file is now **out of date**.

Reason:

- these results were produced with the **older label version**
- the repo now uses a newer clinical-significance label:
  - `probable_clinically_significant_bsi_alert`
  - `probable_contaminant_or_low_significance_alert`

So please read this file as a historical baseline only.

The models need to be rerun on the new label set.

## What was trained

Two baseline models were trained on the high-confidence binary subset:

- Logistic Regression
- XGBoost

Target:

- `0` = `likely_contaminant`
- `1` = `likely_true_bsi`

The `indeterminate` group was excluded from training and evaluation.

## Cohort used for baseline training

- Total high-confidence binary rows: `3,251`
- Train rows: `2,269`
- Validation rows: `490`
- Test rows: `492`
- Train positives: `759`
- Validation positives: `147`
- Test positives: `160`

Subject-level split:

- Train subjects: `2,152`
- Validation subjects: `461`
- Test subjects: `462`

## Two feature settings were evaluated

### 1. Full features

This includes:

- alert context
- demographics
- prior culture history
- pre-alert labs
- pre-alert ICU vital summaries
- pre-alert antibiotics
- pre-alert ICU support proxies
- organism family features

### 2. No-organism features

This excludes organism-family indicators and is a more conservative test.

That matters because the current provisional label is partly defined using organism type. So the no-organism setting is a better estimate of how much signal comes from the rest of the EHR rather than from the label heuristic itself.

It still includes the newly added:

- pre-alert ICU vital-sign summaries
- vasopressor exposure features
- mechanical-ventilation proxy features

## Test-set results

### Full features

| Model | F1 | Precision | Recall | Accuracy | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `0.991` | `0.982` | `1.000` | `0.994` | `1.0000` | `0.9999` | `0.0020` |
| XGBoost | `1.000` | `1.000` | `1.000` | `1.000` | `1.0000` | `1.0000` | `0.0007` |

Confusion counts:

- Logistic Regression: `TP=160`, `TN=329`, `FP=3`, `FN=0`
- XGBoost: `TP=160`, `TN=332`, `FP=0`, `FN=0`

Interpretation:

These scores are almost certainly too optimistic for scientific interpretation, because organism-family features overlap strongly with how the current provisional labels were defined.

### No-organism features

| Model | F1 | Precision | Recall | Accuracy | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `0.561` | `0.522` | `0.606` | `0.691` | `0.729` | `0.565` | `0.210` |
| XGBoost | `0.672` | `0.601` | `0.762` | `0.758` | `0.834` | `0.729` | `0.161` |

Confusion counts:

- Logistic Regression: `TP=97`, `TN=243`, `FP=89`, `FN=63`
- XGBoost: `TP=122`, `TN=251`, `FP=81`, `FN=38`

Interpretation:

This is the more useful first result.

Even without organism-family features, there is still real predictive signal in the pre-alert EHR, and XGBoost still clearly outperforms logistic regression on this first baseline.

Adding ICU vitals and support proxies improved F1 for the no-organism models:

- Logistic Regression: `0.545 -> 0.561`
- XGBoost: `0.648 -> 0.672`

The gain is real but not dramatic, which makes sense because this is a hospital-wide cohort and only a minority of alerts have ICU charting available in the prior `24h`.

## Validation-threshold selection

The probability threshold was chosen on the validation split by maximizing validation F1, then applied unchanged to the test split.

Chosen thresholds:

- Full-feature logistic regression: `0.25`
- Full-feature XGBoost: `0.76`
- No-organism logistic regression: `0.53`
- No-organism XGBoost: `0.42`

## What these results mean

The main conclusion is not that the task is solved.

The safer conclusion is:

- the provisional label set is learnable
- some of the strongest signal comes from organism identity
- but even after removing organism-family features, the task still has meaningful signal
- ICU physiology adds incremental value, especially for recall in XGBoost
- XGBoost is the stronger first baseline

## Important caution

These are baseline results against a provisional label, not a clinician-validated gold standard.

So these numbers should be read as:

- a pipeline feasibility result
- an early benchmark
- not a final clinical-performance claim

## Files

- Metrics JSON: [reports/blood_culture_baseline_metrics.json](reports/blood_culture_baseline_metrics.json)
- Split counts: [reports/blood_culture_baseline_split_counts.json](reports/blood_culture_baseline_split_counts.json)
- Feature builder: [scripts/build_blood_culture_features.py](scripts/build_blood_culture_features.py)
- Trainer: [scripts/train_blood_culture_baselines.py](scripts/train_blood_culture_baselines.py)
