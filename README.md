# Blood Culture Alert Project In MIMIC-IV

## Main Question

When a first Gram-positive blood-culture alert appears, can we tell whether it is:

- a **clinically important bloodstream infection alert**
- or a **likely contaminant / low-significance alert**

This project is meant to help with early review and prioritization. It is not an automatic treatment tool.

## Very Short Summary

This repo now has three main pieces for the blood-culture project:

1. a raw blood-culture subset from `microbiologyevents.csv`
2. a grouped one-row-per-specimen table
3. a first clinically meaningful label set for first Gram-positive alerts

## Best Files To Open First

If you want the simplest overview, start here:

- [CLINICIAN_OVERVIEW.md](CLINICIAN_OVERVIEW.md)

If you want to look at the actual local dataset files, these are the key ones:

- raw blood-culture rows:
  - `artifacts/blood_culture/blood_culture_rows.csv`
- grouped specimen table:
  - `artifacts/blood_culture/blood_culture_specimen_subset.csv`
- small preview of the grouped table:
  - `artifacts/blood_culture/blood_culture_specimen_subset_preview.csv`
- first Gram-positive alert dataset with the current label:
  - `artifacts/blood_culture/first_gp_alert_dataset.csv`
- current label summary:
  - `artifacts/blood_culture/blood_culture_label_metadata.json`

## What One Row Means

There are two different dataset styles in this project.

### 1. Raw row-level microbiology table

File:

- `artifacts/blood_culture/blood_culture_rows.csv`

Meaning:

- one row = one microbiology row from `microbiologyevents.csv`
- this is close to the original MIMIC-IV table
- one specimen can appear many times

### 2. Grouped specimen-level table

File:

- `artifacts/blood_culture/blood_culture_specimen_subset.csv`

Meaning:

- one row = one `micro_specimen_id`
- rows from the same specimen are grouped together
- this is easier to use for modeling and review

## Current Label

The old simple label based only on organism type and repeat positivity has now been replaced in the first-alert dataset.

The current label is:

- `probable_clinically_significant_bsi_alert`
- `probable_contaminant_or_low_significance_alert`
- `indeterminate`

Plain meaning:

- `probable_clinically_significant_bsi_alert`
  - more likely a real and clinically important infection signal
- `probable_contaminant_or_low_significance_alert`
  - more likely a contaminant or a low-significance alert
- `indeterminate`
  - not clear enough to trust for first-pass model training

## How The Current Label Is Built

The current label uses:

- organism pattern
- repeat blood-culture evidence within `48h`
- whether there were multiple blood-culture specimens in the episode
- post-alert antibiotic continuation in the `24-72h` window

It does **not** use pre-alert vitals or labs inside the label itself.

That is important, because the model should learn from those inputs, not have them baked into the label.

## Current Counts

Current first Gram-positive alert dataset:

- total rows: `5,546`
- `probable_clinically_significant_bsi_alert`: `1,246`
- `probable_contaminant_or_low_significance_alert`: `1,260`
- `indeterminate`: `3,040`
- high-confidence binary subset: `2,506`

## Important Caution

The blood-culture labels are still research labels.

That means:

- they are clinically motivated
- they are better than the older simple organism-only rule
- but they are still **not** a gold-standard manual review label

## Model Results

The current baseline results file is:

- [BASELINE_BLOOD_CULTURE_RESULTS.md](BASELINE_BLOOD_CULTURE_RESULTS.md)

Important:

- that file currently describes the **older label version**
- it should be treated as historical / outdated
- the baselines need to be rerun on the new label set

## Other Helpful Files

- [EDA_BLOOD_CULTURE_LABEL_VALIDITY.md](EDA_BLOOD_CULTURE_LABEL_VALIDITY.md)
  - early blood-culture EDA and cohort sizing
- [BLOOD_CULTURE_FEATURE_REFERENCE.md](BLOOD_CULTURE_FEATURE_REFERENCE.md)
  - feature list for the earlier baseline feature table
- [scripts/build_blood_culture_specimen_subset.py](scripts/build_blood_culture_specimen_subset.py)
  - builds the grouped specimen-level subset
- [scripts/build_blood_culture_labels.py](scripts/build_blood_culture_labels.py)
  - builds the current clinical-significance labels
- [scripts/build_blood_culture_features.py](scripts/build_blood_culture_features.py)
  - builds the tabular model features

## Current Next Step

The next correct step is:

1. rebuild the feature table using the new labels
2. rerun Logistic Regression and XGBoost
3. update the baseline results file with the new label set
