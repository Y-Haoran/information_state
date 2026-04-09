# `S. aureus` Same-Episode Feature Reduction

## Current Feature Count

The current enriched same-episode model uses `54` features.

Those `54` features are:

- the original `41` pre-alert physiology and acuity features
- plus `13` added microbiology-process and prior-staphylococcal-history features

## Why We Reduced It

The enriched model performs well, but `54` features are harder to explain clinically.

So we ran:

- XGBoost SHAP-style feature importance
- logistic regression coefficient review
- feature-correlation screening
- retraining on a smaller pruned set

## Most Important Features

Top XGBoost SHAP-style features from the primary urgent / emergency same-episode cohort:

1. `prior_subject_s_aureus_positive_90d`
2. `index_hours_draw_to_alert`
3. `prealert_blood_culture_draws_7d`
4. `lab_platelets_last_24h`
5. `lab_platelets_min_24h`
6. `anchor_age`
7. `prior_subject_s_aureus_positive_365d`
8. `lab_creatinine_last_24h`
9. `prealert_blood_culture_draws_24h`
10. `lab_creatinine_max_24h`

## Clinical Meaning Of The Top Features

- `prior_subject_s_aureus_positive_90d`
  A recent prior positive `S. aureus` blood culture suggests recurrent infection, colonization, or a persistent unresolved source.

- `index_hours_draw_to_alert`
  Faster blood-culture positivity can reflect higher organism burden. In practice, a shorter draw-to-alert interval is often more concerning for true bacteremia.

- `prealert_blood_culture_draws_7d` and `prealert_blood_culture_draws_24h`
  Repeated blood-culture sampling before the alert means clinicians were already worried about infection persistence or bacteremia.

- `lab_platelets_last_24h` and `lab_platelets_min_24h`
  Lower platelets can track systemic inflammation, sepsis severity, or consumptive physiology.

- `anchor_age`
  Age is a broad host-risk marker and may capture differences in infection phenotype, comorbidity burden, and device exposure.

- `prior_subject_s_aureus_positive_365d`
  Longer-term prior `S. aureus` history still carries useful signal, though usually less strongly than recent history.

- `lab_creatinine_last_24h` and `lab_creatinine_max_24h`
  Renal dysfunction is not specific to `S. aureus`, but it marks physiologic stress and a sicker host phenotype.

These findings support the idea that this task is not only about generic illness severity. The model improved most when we added microbiology-process and prior-staphylococcal-history features.

## Feature Reduction Result

We pruned the model from `54` features down to `19` features.

Retained features:

- `prior_subject_s_aureus_positive_90d`
- `index_hours_draw_to_alert`
- `prealert_blood_culture_draws_7d`
- `lab_platelets_last_24h`
- `anchor_age`
- `prior_subject_s_aureus_positive_365d`
- `lab_creatinine_last_24h`
- `prealert_blood_culture_draws_24h`
- `in_icu_at_alert`
- `lab_lactate_max_24h`
- `prior_subject_cons_positive_365d`
- `prealert_blood_culture_draws_6h`
- `prior_positive_specimens_7d`
- `prior_subject_cons_positive_90d`
- `prior_subject_any_staph_positive_365d`
- `lab_creatinine_count_24h`
- `lab_wbc_last_24h`
- `vital_map_count_24h`
- `vital_temperature_c_max_24h`

## Performance Comparison

Primary urgent / emergency same-episode cohort:

- `41`-feature same-episode baseline
  - Logistic Regression: AUROC `0.666`, F1 `0.459`
  - XGBoost: AUROC `0.640`, F1 `0.465`

- `54`-feature enriched model
  - Logistic Regression: AUROC `0.807`, F1 `0.589`
  - XGBoost: AUROC `0.817`, F1 `0.606`

- `19`-feature pruned model
  - Logistic Regression: AUROC `0.812`, F1 `0.615`
  - XGBoost: AUROC `0.820`, F1 `0.604`

So the smaller `19`-feature model preserved the enriched model’s performance and slightly improved the logistic model.

## Recommended Current Model

The most practical current model is:

- cohort: primary urgent / emergency same-episode first Gram-positive alert cohort
- features: pruned `19`-feature set
- models to report:
  - Logistic Regression for interpretability
  - XGBoost for strongest ranking performance

## Files

- SHAP importance CSV: [s_aureus_same_episode_enriched_xgb_shap_importance.csv](/nfs/users/nfs_d/ds39/dosage_prediction/reports/s_aureus_same_episode_enriched_xgb_shap_importance.csv)
- logistic coefficients CSV: [s_aureus_same_episode_enriched_logistic_coefficients.csv](/nfs/users/nfs_d/ds39/dosage_prediction/reports/s_aureus_same_episode_enriched_logistic_coefficients.csv)
- pruned metrics JSON: [s_aureus_same_episode_pruned_metrics.json](/nfs/users/nfs_d/ds39/dosage_prediction/reports/s_aureus_same_episode_pruned_metrics.json)
- SHAP figure: [xgb_shap_importance.png](/nfs/users/nfs_d/ds39/dosage_prediction/figures/s_aureus_same_episode/xgb_shap_importance.png)
- coefficient figure: [logistic_coefficients.png](/nfs/users/nfs_d/ds39/dosage_prediction/figures/s_aureus_same_episode/logistic_coefficients.png)
- correlation figure: [feature_correlation.png](/nfs/users/nfs_d/ds39/dosage_prediction/figures/s_aureus_same_episode/feature_correlation.png)
