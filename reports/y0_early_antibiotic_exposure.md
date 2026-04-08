# Early Antibiotic Exposure In The `y=0` Alert Group

## Why This Matters

In the current project, `y=0` means:

- `probable_contaminant_or_low_significance_alert`

This is not a perfect gold-standard contaminant label. It is a conservative operational group: contaminant-prone organism pattern, no repeat positive confirmation, and no antibiotic continuation in the `24-72h` period after the alert.

That makes this group useful for one stewardship question:

- how often did these low-significance alerts still trigger **early antibiotic exposure** before treatment was stopped?

## Current Counts

- total `y=0` rows: `1,260`
- unique patients in `y=0`: `1,220`
- `y=0` rows with any systemic or anti-MRSA antibiotic in the first `24h` after alert: `60` (4.8%)
- `y=0` rows with anti-MRSA therapy in the first `24h` after alert: `20` (1.6%)
- `y=0` rows with any antibiotic continuation in `24-72h`: `0` by label construction

## Anti-MRSA Drugs Given In `y=0`

Among the `20` low-significance alerts that still received anti-MRSA therapy in the first `24h`, the medication breakdown was:

```text
medication | admin_rows | unique_admissions | unique_patients
-----------+------------+-------------------+----------------
Vancomycin | 23         | 15                | 15             
Daptomycin | 3          | 3                 | 3              
Linezolid  | 3          | 2                 | 2              
```

## Main Interpretation

The strongest concrete example is vancomycin:

- vancomycin appeared in `15` low-significance admissions
- that is `75.0%` of the low-significance alerts that received anti-MRSA therapy

So the stewardship story is real, but it should be stated carefully:

- the current dataset does **not** show that half of all alerts caused unnecessary prolonged antibiotic use
- it **does** show that a smaller subset of alerts later classified as low-significance still triggered short early exposure to drugs such as vancomycin, linezolid, and daptomycin

That is a defensible clinical motivation for the model:

> if the model can identify low-significance Gram-positive alerts earlier, it may help reduce some unnecessary early empiric antibiotic exposure, especially vancomycin-like treatment, before those alerts are recognized as low-value.
