# `S. aureus` First-Alert Secondary Analysis

## Clinical Question

This secondary analysis asks:

- at the first Gram-positive blood-culture alert, which patients are more likely to have later-confirmed `Staphylococcus aureus` and should be prioritized for urgent review, repeat cultures, and source evaluation?

This is intentionally framed as a **prioritization** question, not an automatic antibiotic-escalation rule.

## Why This Is Secondary

The main project in the repo remains the contaminant-vs-clinically-significant first-alert model.

This `S. aureus` task is clinically meaningful, but it is harder because the current feature set is mostly general physiology and acuity. That may be enough to detect “sicker versus less sick” patients, but species-level prediction usually needs richer microbiology and source information.

## Cohort

- total first Gram-positive alert rows: `5,546`
- unique patients: `5,021`
- later-confirmed `S. aureus` rows: `1,454`
- prevalence: `26.2%`

## Current 41-Feature Baseline

Held-out test results:

| Model | F1 | Precision | Recall | Accuracy | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `0.424` | `0.344` | `0.553` | `0.611` | `0.611` | `0.373` |
| XGBoost | `0.430` | `0.318` | `0.660` | `0.546` | `0.608` | `0.349` |

Interpretation:

- there is some signal
- but the current baseline is modest
- this is not strong enough to replace microbiology workflow or justify aggressive treatment claims

## Practical Meaning

The current baseline supports this statement:

> Early routine clinical data alone contain limited but non-zero signal for later-confirmed `S. aureus` among first Gram-positive alerts.

It does **not** support this stronger statement:

> routine pre-alert vitals and labs are already sufficient to drive early species-targeted treatment decisions

## Most Important Next Features

To make this task stronger, the next feature set should prioritize:

- central-line and device features
- prior MRSA / prior staphylococcal history if available
- infection-source clues
- richer microbiology context
- leakage-safe note features later if needed

## Bottom Line

This is a good secondary task to keep developing because it is clinically meaningful, but with the current `41`-feature baseline it should remain a secondary analysis rather than the main project claim.
