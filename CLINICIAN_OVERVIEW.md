# Clinician Overview

## What This Project Does

This project asks:

When the **first Gram-positive blood-culture alert** appears in an admission, is it more likely:

- a real bloodstream infection signal
- or a contaminant / low-significance alert

The goal is early review support, not automatic treatment.

## Why We Think This Can Work

Our main hypothesis is:

- a clinically important Gram-positive alert often comes with a different patient-state pattern than a contaminant

In other words, true clinically important alerts may already show a measurable pre-alert footprint in:

- platelet and creatinine behavior
- temperature and other vital signs
- ICU-level support such as vasopressors or ventilation
- recent blood-culture history

## What One Case Means

One model row means:

- one hospital admission
- one first Gram-positive blood-culture alert

If a patient had several blood cultures drawn, that background can still influence the model through prior-history features, but the prediction target is tied to the **first alert**.

## What The Model Sees

The clean baseline model uses only information available **before or at the alert time**:

- age
- whether the patient is in ICU
- vasopressor and ventilation proxies
- prior positive blood-culture history
- recent labs
- recent vital signs

It does **not** use:

- future repeat cultures
- later antibiotic continuation
- organism-family identity

## What The Labels Mean

- `probable_clinically_significant_bsi_alert`
  - more likely a real and important infection alert
- `probable_contaminant_or_low_significance_alert`
  - more likely contamination or a low-significance alert
- `indeterminate`
  - not clear enough for the first binary model

## Dataset Size

- total first-alert rows: `5,546`
- clear binary subset used for the main model: `2,506`

## Main Baseline Performance

Held-out test performance for the clean 41-feature baseline:

- Logistic Regression: AUROC `0.80`, F1 `0.77`
- XGBoost: AUROC `0.81`, F1 `0.76`

Plain meaning:

- the model shows useful signal
- but this is still a research result, not a deployment-ready tool

## What Looks Most Important

In the current baseline, the strongest signals are mostly:

- platelet features
- creatinine features
- age
- ICU status
- temperature features

So the model appears to be using general illness severity and host-response pattern, not only one single measurement.

That is one of the main findings of the project:

- the model still works without using organism identity
- the strongest signal comes from patient physiology and acuity

## Best Files To Open

- main results: [BASELINE_BLOOD_CULTURE_RESULTS.md](BASELINE_BLOOD_CULTURE_RESULTS.md)
- feature list: [BLOOD_CULTURE_FEATURE_REFERENCE.md](BLOOD_CULTURE_FEATURE_REFERENCE.md)
- explainability summary: [PRIMARY_BASELINE_EXPLAINABILITY.md](PRIMARY_BASELINE_EXPLAINABILITY.md)
- alert-level dataset: `artifacts/blood_culture/first_gp_alert_dataset.csv`
