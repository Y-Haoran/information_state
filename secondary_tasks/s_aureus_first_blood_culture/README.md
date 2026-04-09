# Secondary Task: `S. aureus` From First Blood-Culture Event

## Why This Exists

This is a secondary analysis, not the main project in the repo.

The main project stays:

- first Gram-positive alert
- predict clinically significant alert vs contaminant / low-significance alert

This task moves earlier in the pathway.

It asks:

> Using the first 24 hours or 18 hours of clinical data after the first blood-culture event in an admission, can we predict which patients later have confirmed `Staphylococcus aureus` bacteremia and should be prioritized for urgent review, repeat cultures, and source evaluation?

## Anchor And Label

- one row = one admission
- anchor time = the first blood-culture event in that admission
- main feature window = `0-24h` after the anchor
- sensitivity window = `0-18h` after the anchor
- label = later-confirmed `Staphylococcus aureus` bacteremia in the same admission

This task does **not** anchor on the first positive alert.

## Cohort Size

- all admissions with a first blood-culture anchor: `92,229`
- eligible for the `24h` window: `87,436`
- eligible for the `18h` window: `89,177`

Later-confirmed `S. aureus` prevalence is low:

- `24h`: `1,448 / 87,436` = `1.66%`
- `18h`: `1,456 / 89,177` = `1.63%`

That low prevalence is the main reason this task is much harder than the first-alert contaminant-vs-significant project.

## Current Baseline Performance

Saved files:

- combined metrics: [reports/s_aureus_first_blood_culture_metrics.json](../../reports/s_aureus_first_blood_culture_metrics.json)
- short report: [reports/s_aureus_first_blood_culture_report.md](../../reports/s_aureus_first_blood_culture_report.md)
- builder + trainer: [scripts/run_s_aureus_first_blood_culture_experiment.py](../../scripts/run_s_aureus_first_blood_culture_experiment.py)

Held-out test results:

- `24h` Logistic Regression: AUROC `0.734`, AUPRC `0.049`, F1 `0.068`
- `24h` XGBoost: AUROC `0.706`, AUPRC `0.062`, F1 `0.095`
- `18h` Logistic Regression: AUROC `0.703`, AUPRC `0.038`, F1 `0.056`
- `18h` XGBoost: AUROC `0.689`, AUPRC `0.059`, F1 `0.117`

## Interpretation

This task is clinically meaningful, but the current early routine-feature baseline is weak.

The model has some ranking signal, but not enough yet for a strong early `S. aureus` prioritization model.

The main reasons are:

- the task is earlier than the first Gram-positive alert
- the positive class is rare
- the current feature set is still generic rather than source- or device-aware

## What We Need Next

- central-line and device features
- prior MRSA or prior staphylococcal history if it can be recovered safely
- source clues from diagnoses, procedures, and charted context
- richer microbiology context around the culture episode

## Bottom Line

This is a good clinical question, but it is not yet the main paper result.

Right now it should be read as:

- a realistic early-risk formulation
- a large-cohort feasibility analysis
- a baseline that shows why richer features are needed
