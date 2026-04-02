# MIMIC-IV EDA For Gram-Positive Blood-Culture Label Validity

## Project question

At the time a blood culture first becomes positive, can we predict whether a Gram-positive result is more likely to represent true bloodstream infection versus likely contamination, using only information available up to that alert time?

## Why this EDA matters

This is a promising project only if two things are true:

- MIMIC-IV contains a large enough blood-culture cohort with a usable alert timestamp.
- A reasonable proxy label can be built for `likely true BSI` versus `likely contamination`.

This report checks both.

## Data used in this pass

- `hosp/microbiologyevents.csv`
- `icu/icustays.csv.gz`

## Cohort definition used here

This is a first-pass label-validity EDA, not the final cohort logic.

- Unit: first Gram-positive candidate positive blood-culture alert per hospital admission
- Alert time: specimen-level `storetime`, with fallback to `charttime`, then `storedate/chartdate`
- Positive blood-culture row:
  - `spec_type_desc == BLOOD CULTURE`
  - non-empty `org_name`
  - `org_name != CANCELLED`

The specimen-level alert is then collapsed to one first Gram-positive alert per `hadm_id`.

## Overall positive blood-culture size

- Positive blood-culture rows: `70,055`
- Positive blood-culture specimens: `17,010`
- Admissions with at least one positive blood-culture specimen: `7,850`
- Positive blood-culture specimens with `storetime`: `99.99%`

This means MIMIC-IV does support an alert-time framing. The result timestamp is essentially always available for positive blood-culture specimens in this first pass.

## Gram-positive cohort size

- Gram-positive candidate positive specimens: `11,947`
- Admissions with at least one Gram-positive candidate specimen: `5,546`
- First Gram-positive alerts used in this EDA: `5,546`

So the cohort is large enough for:

- classical baselines
- a sequence model
- a three-way labeling strategy with an indeterminate group

## First-alert category split

The first alert on each admission was divided into provisional organism-based groups:

- `true_like`: `2,590` (`46.70%`)
- `contam_like`: `2,549` (`45.96%`)
- `ambiguous_gp`: `314` (`5.66%`)
- `mixed_gp`: `93` (`1.68%`)

These are not final labels. They are only first-pass microbiology categories.

## Organisms seen at first Gram-positive alert

Top Gram-positive organisms in the first-alert cohort:

- `STAPHYLOCOCCUS, COAGULASE NEGATIVE`: `1,842`
- `STAPH AUREUS COAG +`: `1,454`
- `ENTEROCOCCUS FAECIUM`: `603`
- `ENTEROCOCCUS FAECALIS`: `429`
- `STAPHYLOCOCCUS EPIDERMIDIS`: `427`
- `VIRIDANS STREPTOCOCCI`: `218`
- `CORYNEBACTERIUM SPECIES (DIPHTHEROIDS)`: `143`
- `LACTOBACILLUS SPECIES`: `78`
- `STREPTOCOCCUS ANGINOSUS (MILLERI) GROUP`: `71`
- `BETA STREPTOCOCCUS GROUP B`: `53`

This is exactly the kind of organism mix where contamination versus true infection is clinically relevant.

## Signal for label validity

The strongest first-pass validity check here was repeat blood-culture positivity within `48h` after the first alert.

Repeat positivity rates by first-alert category:

- `true_like`
  - any positive blood-culture specimen within `48h`: `41.16%`
  - repeat Gram-positive positive within `48h`: `40.27%`
  - repeat same-organism positive within `48h`: `39.77%`
- `contam_like`
  - any positive blood-culture specimen within `48h`: `14.28%`
  - repeat Gram-positive positive within `48h`: `13.34%`
  - repeat same-organism positive within `48h`: `12.48%`
- `ambiguous_gp`
  - any positive blood-culture specimen within `48h`: `21.02%`
  - repeat same-organism positive within `48h`: `18.47%`

That separation is important. In this dataset, `true_like` first alerts are about three times as likely as `contam_like` first alerts to have same-organism repeat positivity within `48h`.

That does not prove the label, but it does support the idea that the microbiology patterns are not random noise.

## ICU-only feasibility

- First Gram-positive alerts occurring during an ICU stay: `19.83%`

So an ICU-only version is possible, but it would be much smaller than the hospital-wide cohort. For this project, the cleaner starting point is likely hospital-wide rather than ICU-only.

## Provisional label design from this pass

The safest first modeling setup is a 3-group label design:

- `likely_contaminant`
- `likely_true_bsi`
- `indeterminate`

Using only microbiology-based high-confidence rules in this first pass:

- `likely_contaminant`: `2,185` (`39.40%`)
  - contaminant-like first alert
  - no repeat positive blood-culture specimen within `48h`
- `likely_true_bsi`: `1,066` (`19.22%`)
  - true-like first alert
  - repeat positive blood culture within `48h` or repeat same-organism positivity
- `indeterminate`: `2,295` (`41.38%`)

This is a useful result. It means we already have a reasonably sized high-confidence subset for a first baseline model, without pretending the ambiguous middle cases are clean labels.

## What this suggests

This project looks feasible.

The main reasons are:

- the cohort is large enough
- alert-time information is available
- the organism mix is clinically appropriate
- repeat-culture behavior separates contamination-like from true-like cases in a meaningful way

The main remaining problem is still label quality, not data volume.

## Important limitations

This is still only a first-pass proxy-label EDA.

Limitations:

- no clinician-reviewed gold-standard label set yet
- no bottle-level information on number of positive bottles in a set
- no line/device context yet
- no post-alert treatment logic yet
- some organisms, especially viridans streptococci and polymicrobial episodes, remain ambiguous

## Recommended next step

The next correct step is not a transformer.

It is:

1. refine the label rules with clinicians
2. add repeat-culture logic, line/device proxies, and post-alert antibiotic behavior
3. keep the first model on `likely_contaminant` versus `likely_true_bsi`
4. hold out `indeterminate` cases from the first training run

## Current step-by-step pipeline

Step 1 builds the first-alert cohort:

```bash
PYTHONPATH=. python3 scripts/build_blood_culture_cohort.py --project-root . --raw-root /path/to/mimic_root
```

Step 2 adds the provisional labels:

```bash
PYTHONPATH=. python3 scripts/build_blood_culture_labels.py --project-root .
```

Current generated artifacts:

- `artifacts/blood_culture/positive_blood_culture_specimens.csv`
- `artifacts/blood_culture/first_gp_alert_cohort.csv`
- `artifacts/blood_culture/first_gp_alert_labels.csv`
- `artifacts/blood_culture/first_gp_alert_dataset.csv`
- `artifacts/blood_culture/blood_culture_cohort_metadata.json`
- `artifacts/blood_culture/blood_culture_label_metadata.json`

## Files

- Script: [scripts/blood_culture_label_validity_eda.py](scripts/blood_culture_label_validity_eda.py)
- Cohort builder: [scripts/build_blood_culture_cohort.py](scripts/build_blood_culture_cohort.py)
- Label builder: [scripts/build_blood_culture_labels.py](scripts/build_blood_culture_labels.py)
- Summary JSON: [reports/blood_culture_label_validity_summary.json](reports/blood_culture_label_validity_summary.json)
