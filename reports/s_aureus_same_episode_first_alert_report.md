# `S. aureus` From The Same First Gram-Positive Alert Episode

## Clinical Question

- at the first Gram-positive blood-culture alert, can we use the prior 24 hours of routine data to predict whether that same blood-culture episode will later finalize as `Staphylococcus aureus` rather than another Gram-positive organism?

## Cohort Design

- anchor = first Gram-positive blood-culture alert per admission
- positive label = that same index episode finalizes as `S. aureus`
- negative label = that same index episode finalizes as another Gram-positive organism
- excluded from the primary clean cohort = polymicrobial first-alert episodes
- primary subgroup = urgent / emergency admissions only

## Primary Cohort

- rows: `3,877`
- unique patients: `3,588`
- `S. aureus` positives: `1,021` (26.33%)
- ICU at alert: `21.31%`

### Held-out Test Results

- Logistic Regression: AUROC `0.666`, AUPRC `0.382`, F1 `0.459`
- XGBoost: AUROC `0.640`, AUPRC `0.371`, F1 `0.465`

## Sensitivity Cohort

- same label and same pre-alert features
- broader cohort: all single-organism first Gram-positive alerts

- rows: `5,275`
- unique patients: `4,802`
- `S. aureus` positives: `1,397` (26.48%)

- Logistic Regression: AUROC `0.594`, AUPRC `0.363`, F1 `0.465`
- XGBoost: AUROC `0.631`, AUPRC `0.377`, F1 `0.490`

## Interpretation

- this is scientifically cleaner than the broad first-blood-culture SAB dataset because the anchor and the final species outcome belong to the same microbiology episode
- restricting to single-organism episodes removes some obvious label noise
- the urgent / emergency subgroup improves AUROC compared with the looser all-single-organism cohort
- performance is still modest, which suggests this task needs richer device, source, and prior staphylococcal context rather than more generic physiology alone
