# Clinician Overview

## What Is This Project Trying To Do?

This project looks at a simple question:

When the lab reports a first Gram-positive blood-culture alert, is this more likely:

- a real bloodstream infection that needs attention
- or a contaminant / low-significance result

## Why This Matters

In practice, some Gram-positive blood-culture results are very important.

Others are less important because they may reflect:

- skin contamination during sampling
- a low-significance alert
- an unclear result that needs review but not panic

The goal is to build a review tool, not a replacement for clinical judgement.

## What We Did

We built three dataset layers from MIMIC-IV:

### 1. Raw blood-culture rows

File:

- `artifacts/blood_culture/blood_culture_rows.csv`

This is the original microbiology table filtered to:

- `spec_type_desc = BLOOD CULTURE`

### 2. One row per specimen

File:

- `artifacts/blood_culture/blood_culture_specimen_subset.csv`

This is easier to read because one blood-culture specimen is summarized into one row.

### 3. First Gram-positive alert dataset

File:

- `artifacts/blood_culture/first_gp_alert_dataset.csv`

This is the first-alert dataset used for labeling and later modeling.

## What The Current Labels Mean

The current labels are:

- `probable_clinically_significant_bsi_alert`
- `probable_contaminant_or_low_significance_alert`
- `indeterminate`

Plain English:

- `probable clinically significant`
  - more likely a real and important infection signal
- `probable contaminant / low significance`
  - more likely not a major bloodstream infection signal
- `indeterminate`
  - not clear enough for strong labeling

## How We Built The Current Labels

The current label uses four simple ideas:

1. what organism pattern was seen
2. whether the organism repeated in another blood culture within `48h`
3. whether there were multiple blood-culture specimens in the episode
4. whether antibiotics were continued after the alert

This was done to make the label more clinically meaningful than a simple organism-only rule.

## Current Numbers

Current first Gram-positive alert dataset:

- total alerts: `5,546`
- probable clinically significant: `1,246`
- probable contaminant / low significance: `1,260`
- indeterminate: `3,040`

## Important Limitation

These labels are still research labels.

They are:

- clinically motivated
- better than the older simple rule
- but not the same as a full expert manual review of every chart

## Best File For A Quick Look

If you only want a quick look at the grouped specimen table, open:

- `artifacts/blood_culture/blood_culture_specimen_subset_preview.csv`

If you want the current first-alert labels, open:

- `artifacts/blood_culture/first_gp_alert_dataset.csv`
