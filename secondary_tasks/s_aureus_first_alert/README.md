# Secondary Task: Early `S. aureus` Prioritization

## Why This Exists

This folder describes a **secondary analysis**, not the main project.

The main project in this repo remains:

- first Gram-positive alert
- predict clinically significant alert vs contaminant / low-significance alert

This secondary task asks a different question:

> At the first Gram-positive alert, which patients are more likely to have later-confirmed `Staphylococcus aureus` and should be prioritized for urgent review, repeat cultures, and source evaluation?

## Refined Same-Episode Cohort

The cleaner version of this task is:

> At the first Gram-positive blood-culture alert, can the prior 24 hours of routine data predict whether that same index blood-culture episode will later finalize as `Staphylococcus aureus` rather than another Gram-positive organism?

Primary clean cohort:

- first Gram-positive alert per admission
- same-episode species label
- exclude polymicrobial first-alert episodes
- focus first on urgent / emergency admissions

Saved files:

- [reports/s_aureus_same_episode_first_alert_metrics.json](../../reports/s_aureus_same_episode_first_alert_metrics.json)
- [reports/s_aureus_same_episode_first_alert_report.md](../../reports/s_aureus_same_episode_first_alert_report.md)
- [scripts/train_s_aureus_same_episode_first_alert.py](../../scripts/train_s_aureus_same_episode_first_alert.py)
- enriched follow-up: [reports/s_aureus_same_episode_enriched_report.md](../../reports/s_aureus_same_episode_enriched_report.md)
- enriched metrics: [reports/s_aureus_same_episode_enriched_metrics.json](../../reports/s_aureus_same_episode_enriched_metrics.json)
- enriched trainer: [scripts/train_s_aureus_same_episode_enriched.py](../../scripts/train_s_aureus_same_episode_enriched.py)

Primary cohort counts:

- rows: `3,877`
- unique patients: `3,588`
- `S. aureus` positives: `1,021`
- prevalence: `26.3%`

Primary held-out test results:

- Logistic Regression: AUROC `0.666`, F1 `0.459`
- XGBoost: AUROC `0.640`, F1 `0.465`

### Enriched Follow-up

We then added:

- draw-to-alert time
- pre-alert blood-culture draw counts
- prior patient history of positive `S. aureus`, CoNS, and any-staphylococcal blood cultures

Primary cohort enriched results:

- Logistic Regression: AUROC `0.807`, AUPRC `0.657`, F1 `0.589`
- XGBoost: AUROC `0.817`, AUPRC `0.704`, F1 `0.606`

Sensitivity cohort:

- all single-organism first Gram-positive alerts
- rows: `5,275`
- `S. aureus` positives: `1,397`

Sensitivity held-out test results:

- Logistic Regression: AUROC `0.594`, F1 `0.465`
- XGBoost: AUROC `0.631`, F1 `0.490`

Sensitivity enriched results:

- Logistic Regression: AUROC `0.761`, AUPRC `0.644`, F1 `0.565`
- XGBoost: AUROC `0.811`, AUPRC `0.717`, F1 `0.632`

## Older Looser Cohort

Using the current first-alert dataset:

- total first Gram-positive alert rows: `5,546`
- later-confirmed `S. aureus` rows: `1,454`
- prevalence: `26.2%`

This means the task is feasible on sample size alone.

## Current Baseline

We tested the same clean `41` pre-alert features used in the main project:

- age
- ICU status
- vasopressor and ventilation proxies
- prior microbiology counts
- WBC
- platelets
- creatinine
- lactate
- heart rate
- respiratory rate
- temperature
- MAP
- SpO2

Saved metrics:

- [reports/s_aureus_first_alert_metrics.json](../../reports/s_aureus_first_alert_metrics.json)
- [reports/s_aureus_first_alert_report.md](../../reports/s_aureus_first_alert_report.md)
- trainer script: [scripts/train_s_aureus_first_alert_baseline.py](../../scripts/train_s_aureus_first_alert_baseline.py)
- pruned-feature sensitivity analysis: [reports/s_aureus_first_alert_pruned_report.md](../../reports/s_aureus_first_alert_pruned_report.md)
- pruned trainer script: [scripts/train_s_aureus_pruned_baseline.py](../../scripts/train_s_aureus_pruned_baseline.py)

Held-out test performance with the older looser `41`-feature baseline:

- Logistic Regression: AUROC `0.611`, F1 `0.424`
- XGBoost: AUROC `0.608`, F1 `0.430`

We also ran feature selection for this older looser version of the task.

Pruned `15`-feature results:

- Logistic Regression: AUROC `0.638`, F1 `0.433`
- XGBoost: AUROC `0.604`, F1 `0.427`

Interpretation:

- pruning helps Logistic Regression a bit
- pruning does not improve XGBoost
- so the current task still looks feature-limited rather than algorithm-limited

## Interpretation

The refined same-episode cohort is the better scientific version of this task because:

- the alert and the final `S. aureus` label belong to the same microbiology episode
- polymicrobial first-alert episodes are removed from the primary analysis
- urgent / emergency admissions are a cleaner first clinical subgroup

The added process and prior-staph-history features materially improved performance, which supports the idea that this task is more microbiology- and history-driven than physiology-driven.

It should still remain a secondary analysis until we add source- and device-aware features, but it is now much more convincing than the earlier physiology-only version.

## Why We Still Want To Work On It

Even though the current baseline is modest, the task is still clinically meaningful because later-confirmed `S. aureus` often triggers a different level of urgency than generic Gram-positive bacteremia.

Potential value:

- prioritize urgent microbiology review
- prioritize repeat cultures
- prioritize source investigation
- prioritize line / device assessment
- support faster recognition of patients who may need a more serious `S. aureus` workup

## What Features We Need Next

If we continue this task, the next feature set should be richer than the current `41` features.

The most important additions are:

- central-line and device information
- prior MRSA history, if recoverable safely
- infection-source clues
- more microbiology context
- targeted note information later, only if timing can be made leakage-safe

These additions fit the biology of the question much better than vitals and routine labs alone.

## Bottom Line

This `S. aureus` task is worth keeping because it is clinically meaningful, but the refined same-episode dataset should be the version readers look at first.

Current takeaway:

- keep the contaminant-vs-significant model as the primary project
- use the refined same-episode `S. aureus` cohort as the better secondary analysis
- improve it next with device, source, and prior-staphylococcal context
