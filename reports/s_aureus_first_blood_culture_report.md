# `S. aureus` From First Blood-Culture Event

## Clinical Question

- using the first 24 hours or 18 hours of clinical data after the first blood-culture event in an admission, can we predict which admissions later have confirmed `Staphylococcus aureus` bacteremia?

## Cohort

- total admissions with a first blood-culture anchor: `92,229`
- eligible for `24h` window: `87,436` admissions
  - later-confirmed `S. aureus`: `1,448` (1.66%)
- eligible for `18h` window: `89,177` admissions
  - later-confirmed `S. aureus`: `1,456` (1.63%)

## Baseline Results

### `24h`

- Logistic Regression: AUROC `0.734`, AUPRC `0.049`, F1 `0.068`
- XGBoost: AUROC `0.706`, AUPRC `0.062`, F1 `0.095`

### `18h`

- Logistic Regression: AUROC `0.703`, AUPRC `0.038`, F1 `0.056`
- XGBoost: AUROC `0.689`, AUPRC `0.059`, F1 `0.117`

## Interpretation

- this task is much earlier than the first-Gram-positive-alert analysis
- prevalence is much lower, so the class imbalance is harder
- the 0-24h window is the main analysis because it has more complete early data
- the 0-18h window is a sensitivity analysis for an earlier decision point

## Next Features To Add

- central-line and device features
- prior MRSA or prior staphylococcal history if available safely
- source clues from diagnoses, procedures, and targeted charting
- richer microbiology context
