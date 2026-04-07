# Blood-Culture Feature Reference

## Why this file exists

People looking at the GitHub repo need to see:

- what the model features actually are
- how the final tabular dataset is structured
- an example table without exposing restricted MIMIC-derived patient rows

So this repo includes:

- this feature reference
- a synthetic example dataset with a few hundred rows

The synthetic file is only for schema illustration. It is **not** real patient data and it should not be used for scientific analysis.

## Files

- Feature builder: [scripts/build_blood_culture_features.py](scripts/build_blood_culture_features.py)
- Synthetic example dataset: [examples/blood_culture_feature_example_synthetic.csv](examples/blood_culture_feature_example_synthetic.csv)
- Real feature metadata: [artifacts/blood_culture/blood_culture_feature_metadata.json](artifacts/blood_culture/blood_culture_feature_metadata.json)

## Training target

For the first binary baseline:

- `target_true_bsi = 0` means `probable_contaminant_or_low_significance_alert`
- `target_true_bsi = 1` means `probable_clinically_significant_bsi_alert`
- `target_true_bsi = NaN` means `indeterminate`

Only the high-confidence binary subset is used for the first baseline model.

## Feature groups

The current tabular baseline uses `118` training features.

### 1. Alert context: 7 features

- `has_storetime`
- `has_charttime`
- `in_icu_at_alert`
- `anchor_age`
- `alert_hours_from_admit`
- `alert_weekend`
- `alert_night`

### 2. Organism-family indicators: 11 features

- `org_cons`
- `org_s_aureus`
- `org_enterococcus`
- `org_viridans_strep`
- `org_beta_or_anginosus_strep`
- `org_corynebacterium`
- `org_bacillus`
- `org_cutibacterium_propionibacterium`
- `org_lactobacillus`
- `org_polymicrobial_gp`
- `org_other_gp`

### 3. Prior microbiology history: 4 features

- `prior_positive_specimens_24h`
- `prior_positive_specimens_7d`
- `prior_gp_positive_specimens_24h`
- `prior_gp_positive_specimens_7d`

### 4. Lab summaries from the 24h lookback: 35 features

For each lab, the feature set includes:

- `last`
- `min`
- `max`
- `mean`
- `count`

Labs used:

- `wbc`
- `hemoglobin`
- `platelets`
- `creatinine`
- `lactate`
- `sodium`
- `potassium`

### 5. ICU vital-sign summaries from the 24h lookback: 25 features

For each vital, the feature set includes:

- `last`
- `min`
- `max`
- `mean`
- `count`

Vitals used:

- `heart_rate`
- `resp_rate`
- `temperature_c`
- `map`
- `spo2`

### 6. Pre-alert antibiotic exposure in the 24h lookback: 11 features

- `abx_total_admin_24h`
- `abx_total_admin_24h_flag`
- `abx_vancomycin_iv_like_24h`
- `abx_vancomycin_iv_like_24h_flag`
- `abx_linezolid_24h`
- `abx_linezolid_24h_flag`
- `abx_daptomycin_24h`
- `abx_daptomycin_24h_flag`
- `abx_broad_gram_negative_24h`
- `abx_broad_gram_negative_24h_flag`
- `abx_anti_mrsa_24h_flag`

### 7. ICU organ-support proxies from the 24h lookback: 5 features

- `vasopressor_event_count_24h`
- `vasopressor_active_24h`
- `vasopressor_on_at_alert`
- `mechanical_ventilation_chart_events_24h`
- `mechanical_ventilation_24h`

### 8. One-hot encoded demographics and admission context: 20 features

- sex indicators
- admission type indicators
- insurance group indicators
- race group indicators

## Coverage note

This cohort is hospital-wide, not ICU-only.

That means the ICU-derived feature groups above are informative but sparse:

- only about `13%` of high-confidence alerts have ICU vital-sign charting in the prior `24h`
- `6.64%` have vasopressor exposure in the prior `24h`
- `4.28%` are still on vasopressors at the alert time
- `6.58%` have the mechanical-ventilation proxy in the prior `24h`

So these features matter most for the ICU subset, while the broader hospital-wide model still relies heavily on labs, microbiology history, and antibiotic context.

## Important distinction

The full feature table also contains non-training columns such as:

- `hadm_id`
- `subject_id`
- `micro_specimen_id`
- `alert_time`
- `provisional_label`
- `label_source`

Those columns are useful for analysis and tracking, but they are not part of the training feature matrix.

## What is not used as a predictor

To avoid leakage, the current training feature list excludes:

- `repeat_any_positive_48h`
- `repeat_gp_positive_48h`
- `repeat_same_organism_48h`
- `repeat_positive_specimen_count_48h`
- `provisional_label`
- `label_source`

Those are used to create the provisional labels, so they cannot be fed back into the model.

## How to regenerate the real feature table

```bash
PYTHONPATH=. python3 scripts/build_blood_culture_features.py --project-root . --raw-root /path/to/mimic_root
```

## Why the synthetic example is useful

The synthetic CSV lets other people:

- inspect column names
- understand data types and rough ranges
- test parsing / loading code
- review the schema in GitHub

without publishing restricted MIMIC-derived patient rows.
