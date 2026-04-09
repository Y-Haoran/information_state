# Secondary Task: Early `S. aureus` Prioritization

## Why This Exists

This folder describes a **secondary analysis**, not the main project.

The main project in this repo remains:

- first Gram-positive alert
- predict clinically significant alert vs contaminant / low-significance alert

This secondary task asks a different question:

> At the first Gram-positive alert, which patients are more likely to have later-confirmed `Staphylococcus aureus` and should be prioritized for urgent review, repeat cultures, and source evaluation?

## Current Cohort

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

Held-out test performance with the current `41`-feature baseline:

- Logistic Regression: AUROC `0.611`, F1 `0.424`
- XGBoost: AUROC `0.608`, F1 `0.430`

We also ran feature selection for this secondary task.

Pruned `15`-feature results:

- Logistic Regression: AUROC `0.638`, F1 `0.433`
- XGBoost: AUROC `0.604`, F1 `0.427`

Interpretation:

- pruning helps Logistic Regression a bit
- pruning does not improve XGBoost
- so the current task still looks feature-limited rather than algorithm-limited

## Interpretation

This is a clinically meaningful question, but the current feature set is **not strong enough** to make it the main project.

The signal is modest:

- better than random
- but much weaker than the main contaminant-vs-significant task

That means the current physiology-only baseline should be treated as:

- a feasibility analysis
- a secondary experiment
- a starting point for richer feature engineering

It should **not** yet be used to justify strong claims about species-level early treatment decisions.

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

This `S. aureus` task is worth keeping in the repo because it is clinically meaningful.

But the current evidence says:

- keep the existing first-alert project as the **primary story**
- keep this `S. aureus` task as a **secondary analysis**
- improve it with richer source- and device-aware features before treating it as a serious decision-support model
