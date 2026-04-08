from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ANTI_MRSA_PATTERN = r"VANCOMYCIN|LINEZOLID|DAPTOMYCIN|CEFTAROLINE"
MED_EXCLUDE_PATTERN = r"ORAL|ENEMA|LOCK|OPHTH|EYE|EAR|OTIC|TOPICAL|CREAM|OINTMENT|GEL|FLUSH|NEB|INHAL"


def _render_simple_table(df: pd.DataFrame) -> str:
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        lines.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if idx == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines)


def main() -> None:
    project_root = Path.cwd()
    data_path = project_root / "artifacts" / "blood_culture" / "first_gp_alert_dataset.csv"
    emar_path = Path(
        "/lustre/scratch127/mave/sge_analysis/team229/ds39/RD/LLaMA-Mesh/patient_AI/hosp/emar.csv"
    )
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        data_path,
        usecols=[
            "hadm_id",
            "subject_id",
            "alert_time",
            "provisional_label",
            "systemic_abx_admin_0_24h",
            "anti_mrsa_admin_0_24h",
            "systemic_abx_admin_24_72h",
            "anti_mrsa_admin_24_72h",
        ],
    )
    df["hadm_id"] = df["hadm_id"].astype("int64")
    df["subject_id"] = df["subject_id"].astype("int64")
    df["alert_time"] = pd.to_datetime(df["alert_time"], errors="coerce")

    y0 = df[df["provisional_label"] == "probable_contaminant_or_low_significance_alert"].copy()
    y0_any_early_abx = y0[(y0["systemic_abx_admin_0_24h"] > 0) | (y0["anti_mrsa_admin_0_24h"] > 0)].copy()
    y0_anti_mrsa = y0[y0["anti_mrsa_admin_0_24h"] > 0].copy()

    anti_lookup = y0_anti_mrsa[["hadm_id", "subject_id", "alert_time"]].copy()
    anti_lookup["lookup_key"] = anti_lookup["hadm_id"].astype(str) + "_" + anti_lookup["subject_id"].astype(str)
    anti_lookup_map = anti_lookup.set_index("lookup_key")["alert_time"].to_dict()
    anti_hadm_set = set(anti_lookup["hadm_id"].tolist())

    matched_rows: list[tuple[int, int, str, float]] = []
    reader = pd.read_csv(
        emar_path,
        usecols=["subject_id", "hadm_id", "charttime", "medication", "event_txt"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["subject_id", "hadm_id", "charttime", "medication"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id", "subject_id"])
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk["subject_id"] = chunk["subject_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(anti_hadm_set)]
        if chunk.empty:
            continue
        chunk["event_txt"] = chunk["event_txt"].fillna("").astype(str).str.lower()
        chunk = chunk[chunk["event_txt"].str.contains("admin", regex=False)]
        if chunk.empty:
            continue
        chunk["med_upper"] = chunk["medication"].astype(str).str.upper()
        chunk = chunk[chunk["med_upper"].str.contains(ANTI_MRSA_PATTERN, regex=True, na=False)]
        if chunk.empty:
            continue
        chunk = chunk[~chunk["med_upper"].str.contains(MED_EXCLUDE_PATTERN, regex=True, na=False)]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            lookup_key = f"{int(row.hadm_id)}_{int(row.subject_id)}"
            alert_time = anti_lookup_map.get(lookup_key)
            if alert_time is None:
                continue
            delta_hours = (row.charttime - alert_time).total_seconds() / 3600.0
            if 0 <= delta_hours <= 24:
                matched_rows.append(
                    (
                        int(row.subject_id),
                        int(row.hadm_id),
                        str(row.medication),
                        float(delta_hours),
                    )
                )

    anti_admin = pd.DataFrame(
        matched_rows,
        columns=["subject_id", "hadm_id", "medication", "delta_hours"],
    )
    anti_summary = (
        anti_admin.groupby("medication")
        .agg(admin_rows=("hadm_id", "size"), unique_admissions=("hadm_id", "nunique"), unique_patients=("subject_id", "nunique"))
        .sort_values(["unique_admissions", "admin_rows"], ascending=False)
        .reset_index()
    )
    anti_summary_table = _render_simple_table(anti_summary)

    summary = {
        "y0_rows": int(len(y0)),
        "y0_unique_patients": int(y0["subject_id"].nunique()),
        "y0_any_early_abx_rows": int(len(y0_any_early_abx)),
        "y0_any_early_abx_pct": float(len(y0_any_early_abx) / len(y0)),
        "y0_anti_mrsa_rows": int(len(y0_anti_mrsa)),
        "y0_anti_mrsa_pct": float(len(y0_anti_mrsa) / len(y0)),
        "y0_no_24_72h_continuation_rows": int(
            ((y0["systemic_abx_admin_24_72h"] == 0) & (y0["anti_mrsa_admin_24_72h"] == 0)).sum()
        ),
        "anti_mrsa_medication_breakdown": anti_summary.to_dict(orient="records"),
    }

    anti_summary.to_csv(reports_dir / "y0_anti_mrsa_medication_breakdown.csv", index=False)
    (reports_dir / "y0_early_antibiotic_exposure_summary.json").write_text(json.dumps(summary, indent=2))

    report = f"""# Early Antibiotic Exposure In The `y=0` Alert Group

## Why This Matters

In the current project, `y=0` means:

- `probable_contaminant_or_low_significance_alert`

This is not a perfect gold-standard contaminant label. It is a conservative operational group: contaminant-prone organism pattern, no repeat positive confirmation, and no antibiotic continuation in the `24-72h` period after the alert.

That makes this group useful for one stewardship question:

- how often did these low-significance alerts still trigger **early antibiotic exposure** before treatment was stopped?

## Current Counts

- total `y=0` rows: `{len(y0):,}`
- unique patients in `y=0`: `{y0['subject_id'].nunique():,}`
- `y=0` rows with any systemic or anti-MRSA antibiotic in the first `24h` after alert: `{len(y0_any_early_abx):,}` ({len(y0_any_early_abx) / len(y0):.1%})
- `y=0` rows with anti-MRSA therapy in the first `24h` after alert: `{len(y0_anti_mrsa):,}` ({len(y0_anti_mrsa) / len(y0):.1%})
- `y=0` rows with any antibiotic continuation in `24-72h`: `0` by label construction

## Anti-MRSA Drugs Given In `y=0`

Among the `20` low-significance alerts that still received anti-MRSA therapy in the first `24h`, the medication breakdown was:

```text
{anti_summary_table}
```

## Main Interpretation

The strongest concrete example is vancomycin:

- vancomycin appeared in `{int(anti_summary.loc[anti_summary['medication'].eq('Vancomycin'), 'unique_admissions'].sum()):,}` low-significance admissions
- that is `{(int(anti_summary.loc[anti_summary['medication'].eq('Vancomycin'), 'unique_admissions'].sum()) / len(y0_anti_mrsa)):.1%}` of the low-significance alerts that received anti-MRSA therapy

So the stewardship story is real, but it should be stated carefully:

- the current dataset does **not** show that half of all alerts caused unnecessary prolonged antibiotic use
- it **does** show that a smaller subset of alerts later classified as low-significance still triggered short early exposure to drugs such as vancomycin, linezolid, and daptomycin

That is a defensible clinical motivation for the model:

> if the model can identify low-significance Gram-positive alerts earlier, it may help reduce some unnecessary early empiric antibiotic exposure, especially vancomycin-like treatment, before those alerts are recognized as low-value.
"""
    (reports_dir / "y0_early_antibiotic_exposure.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
