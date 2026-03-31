# Dosage Prediction And ICU Risk Forecasting With MIMIC-IV

## Main Question

This repo asks a simple question:

Can we use the first 24 hours of ICU data from MIMIC-IV to predict near-term deterioration and important clinical outcomes, and can a transformer that understands irregular medical time series do better than simpler baseline models?

## What This Project Does

This repo builds a full small research pipeline:

- it creates an adult ICU cohort from MIMIC-IV
- it turns raw hospital events into model-ready features
- it trains simple baseline models
- it trains a more interesting transformer model

The current prediction tasks are:

- `vasopressor_next_6h`
- `in_hospital_mortality`
- `long_icu_los`

## What We Actually Built

There are two model paths in the repo:

- Tabular baselines
  - logistic regression
  - random forest
  - XGBoost
- Sequence model
  - a patient-specific decay transformer

The transformer is the main novelty idea.

Why it is different:

- ICU data are irregular
- many values are missing
- older measurements should matter less
- the speed of that decay should depend on the patient

So instead of using one fixed decay for everyone, this model learns patient-specific decay rates from static context.

## What Data The Model Uses

Dynamic ICU features:

- heart rate
- blood pressure
- respiratory rate
- temperature
- SpO2
- weight
- creatinine
- BUN
- sodium
- potassium
- chloride
- bicarbonate
- glucose
- lactate
- WBC
- hemoglobin
- platelets
- bilirubin
- vasopressor indicator
- urine output

Static features:

- age
- sex
- admission type
- insurance
- race
- first ICU careunit
- diagnosis count

Each hour bin stores:

- feature value
- observation mask
- observation count
- time since last observation

## What We Achieved So Far

This repo already includes working code for:

- cohort building
- feature extraction
- hourly sequence construction
- tabular feature construction
- baseline training
- transformer training

The code has been smoke-tested end to end on a small sample.

That means:

- the pipeline runs
- the files connect correctly
- the training scripts work

## What This Repo Does Not Claim Yet

This is important.

This repo is not yet claiming final scientific performance.

Current limitations:

- the smoke-test results are only sanity checks
- full MIMIC-IV experiments still need to be run
- there is no external validation yet
- this is a research prototype, not a clinical tool

## Repo Structure

- `mimic_iv_project/config.py`
  - paths, feature definitions, task setup
- `mimic_iv_project/feature_catalog.py`
  - maps feature names to MIMIC item IDs
- `mimic_iv_project/data_pipeline.py`
  - builds cohort and model datasets
- `mimic_iv_project/train_baselines.py`
  - trains logistic regression, random forest, and XGBoost
- `mimic_iv_project/models.py`
  - defines the patient-specific decay transformer
- `mimic_iv_project/train_transformer.py`
  - trains the transformer

## Quick Start

Install dependencies:

```bash
python3 -m pip install --user -r requirements.txt
```

Point the code to your MIMIC-IV root directory.

That directory must contain:

- `hosp/`
- `icu/`

You can pass it directly:

```bash
PYTHONPATH=. python3 -m mimic_iv_project.data_pipeline --build-all --raw-root /path/to/mimic_root
```

Or use the environment variable:

```bash
export MIMIC_IV_ROOT=/path/to/mimic_root
PYTHONPATH=. python3 -m mimic_iv_project.data_pipeline --build-all
```

Build a small smoke-test sample first:

```bash
PYTHONPATH=. python3 -m mimic_iv_project.data_pipeline --build-all --raw-root /path/to/mimic_root --max-stays 256 --max-chunks 4 --project-root ./_smoke
```

Train baselines:

```bash
PYTHONPATH=. python3 -m mimic_iv_project.train_baselines
```

Train the transformer:

```bash
PYTHONPATH=. python3 -m mimic_iv_project.train_transformer --epochs 20 --batch-size 64
```

## Why This Might Be Effective

This setup is useful because it compares:

- strong simple baselines
- a sequence model that handles irregular timing better

That gives a cleaner research story:

- Are simple summary features already enough?
- Does a sequence model help?
- Does patient-specific decay help more than a plain transformer?

## If You Are New To The Repo

The best order is:

1. run the smoke build
2. inspect `artifacts/`
3. train the baselines
4. train the transformer
5. run the full-data build after that
