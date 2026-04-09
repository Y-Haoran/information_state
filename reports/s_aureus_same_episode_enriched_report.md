# Enriched `S. aureus` Same-Episode First-Alert Model

## Added Features

- process feature: draw-to-alert time
- process features: number of blood-culture draws already present before alert in the prior 6h, 24h, and 7d
- prior subject history: previous positive `S. aureus`, CoNS, and any-staphylococcal blood cultures

## Primary Cohort

- rows: `3,877`
- `S. aureus` positives: `1,021` (26.33%)

### Held-out Test Results

- Logistic Regression: AUROC `0.807`, AUPRC `0.657`, F1 `0.589`
- XGBoost: AUROC `0.817`, AUPRC `0.704`, F1 `0.606`

## Sensitivity Cohort

- rows: `5,275`
- `S. aureus` positives: `1,397` (26.48%)

- Logistic Regression: AUROC `0.761`, AUPRC `0.644`, F1 `0.565`
- XGBoost: AUROC `0.811`, AUPRC `0.717`, F1 `0.632`
