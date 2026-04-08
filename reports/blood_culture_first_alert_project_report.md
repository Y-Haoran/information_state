# First-Alert Gram-Positive Blood-Culture Project Report

## Motivation

Positive blood cultures with Gram-positive organisms are clinically difficult because the same alert can reflect two very different realities. In one patient, the alert may represent a true bloodstream infection that needs urgent attention. In another patient, the alert may be a contaminant or a low-significance signal that does not deserve the same level of escalation. This distinction matters because over-calling true infection can drive unnecessary antibiotics, while under-calling it can delay treatment for real bacteremia.

Our project focuses on one narrow and clinically recognizable decision point: the **first Gram-positive blood-culture alert** in a hospital admission. That is a clean moment in workflow. It is early enough to matter, but concrete enough to define in MIMIC-IV. Instead of asking a broad retrospective microbiology question, we ask whether the alert itself looks more like a clinically significant bloodstream infection signal or more like contamination / low significance.

To our knowledge, this exact framing remains relatively uncommon in MIMIC-IV work. We are not claiming a full systematic review, but we did not identify a published MIMIC-IV study with the same combination of design choices:

- one row per admission-level **first Gram-positive alert**
- no organism-identity shortcut in the main baseline
- pre-alert clinical features only
- labels tied to later microbiology and treatment behavior

That is the main motivation for keeping this project narrow and defensible.

## Core Hypothesis

Our working hypothesis is the **physiological footprint hypothesis**:

> When a Gram-positive blood-culture alert reflects a clinically meaningful bloodstream infection rather than contamination, the patient often already carries a measurable pre-alert physiological footprint.

In practical terms, that means a true clinically significant alert should be associated with a different pattern of host response and acuity than a likely contaminant. Even before organism speciation is known, we expect useful signal from:

- platelet behavior
- creatinine behavior
- lactate behavior
- temperature and other vital signs
- ICU support markers such as vasopressors and ventilation
- recent microbiology history

The important point is that the main baseline does **not** rely on organism-family identity. The model is asked to learn from the patient state around the alert, not from later microbiology detail.

## Dataset And Cohort

The starting source is MIMIC-IV microbiology data restricted to blood cultures. From there, we build a specimen-level subset and then an admission-level first-alert dataset.

Current cohort counts:

- total first Gram-positive alert rows: `5,546`
- unique patients in the full first-alert dataset: `5,021`
- unique admissions in the full first-alert dataset: `5,546`

Current label groups:

- `probable_clinically_significant_bsi_alert`: `1,246`
- `probable_contaminant_or_low_significance_alert`: `1,260`
- `indeterminate`: `3,040`

For the first binary baseline, we train only on the high-confidence subset:

- high-confidence rows: `2,506`
- unique patients in that subset: `2,369`
- unique admissions in that subset: `2,506`

This gives a reasonably balanced first binary cohort while still preserving an indeterminate group rather than forcing weak labels.

## Labeling Strategy

The label is not a direct MIMIC-IV column. It is an operational research label built from later microbiology pattern and later treatment behavior.

Positive label:

- `probable_clinically_significant_bsi_alert`

Negative label:

- `probable_contaminant_or_low_significance_alert`

Indeterminate cases are excluded from the first binary model.

This matters because the model inputs are strictly pre-alert, but the labels are allowed to use later information. That is the correct structure for a prediction problem. The model sees the patient state before or at alert time, while the label uses later evidence to define whether the alert was clinically important.

## Main Feature Set

The main clean baseline uses **41 features**. These are not prepackaged MIMIC-IV variables. They are engineered summaries extracted from raw MIMIC-IV tables.

The feature groups are:

- age
- ICU status at alert
- vasopressor and ventilation proxies
- prior positive blood-culture counts
- WBC summaries
- platelet summaries
- creatinine summaries
- lactate summaries
- heart-rate summaries
- respiratory-rate summaries
- temperature summaries
- MAP summaries
- SpO2 summaries

Each summary is defined from the pre-alert window only, mainly the 24 hours before the alert.

## Model Performance

The main reported baseline is the 41-feature first-alert model.

Held-out test results:

- Logistic Regression: AUROC `0.798`, F1 `0.767`
- XGBoost: AUROC `0.809`, F1 `0.761`

These are not perfect scores, but that is actually a strength here. The performance is useful without looking inflated. The model is learning real signal from a clinically limited pre-alert feature set rather than exploiting obvious future leakage.

We also trained an 18-feature pruned sensitivity model built from feature importance and redundancy reduction.

Held-out test results for the 18-feature model:

- Logistic Regression: AUROC `0.791`, F1 `0.771`
- XGBoost: AUROC `0.808`, F1 `0.756`

The smaller model stayed very close to the 41-feature baseline. That suggests the signal is not spread thinly across dozens of weak variables. A compact core feature set appears to carry much of the predictive value.

## Biggest Findings

### 1. Pre-alert physiology carries real signal

The most important result is that the model remains useful **without organism-family identity**. That supports the central hypothesis that a clinically significant alert often leaves a measurable host-response and acuity pattern before the full microbiology workup is complete.

### 2. Platelets and creatinine dominate the compact baseline

The strongest features in the explainability analysis were:

- `lab_platelets_last_24h`
- `lab_creatinine_last_24h`
- `lab_platelets_min_24h`
- `anchor_age`
- `lab_creatinine_max_24h`
- `in_icu_at_alert`
- temperature features

This is a clinically interesting pattern. The model seems to rely more on global illness severity, inflammatory physiology, and acuity than on any one narrow infection marker.

### 3. A smaller model performs almost the same

The 18-feature pruned model stayed close to the 41-feature baseline. That is a useful robustness finding because it suggests the task is not driven only by one large and fragile feature table.

### 4. The cohort is clinically meaningful but still conservative

We deliberately dropped `3,040` indeterminate cases and kept `2,506` high-confidence cases for the first binary model. This means the present model is best understood as a clean first baseline, not a final deployment cohort.

## Practical Interpretation

The current project does **not** prove that the model can replace clinical judgment. It shows something narrower and still important:

- at the moment of the first Gram-positive alert
- using only pre-alert patient data
- there is moderate, reproducible signal that helps distinguish a likely clinically significant alert from a likely contaminant / low-significance alert

That is a reasonable research contribution because it turns a familiar microbiology problem into a precise first-alert prediction task with interpretable features and a leakage-aware baseline.

## Next Steps

The strongest next steps are:

- calibration analysis
- subgroup analysis by ICU vs non-ICU
- prospective simplification of the feature set
- refinement of label validity in the indeterminate cases

For now, the main message is simple:

> A small, leakage-aware, first-alert model built from routine pre-alert clinical features can already recover useful signal in this problem, and the strongest signals come from platelet, creatinine, age, ICU acuity, and temperature-related features.
