# Changelog

## [Unreleased]

- Restructured the README around scientific motivation, current evidence, limitations, and usage.
- Added AKI-specific README figures for training dynamics, cluster separation, trajectory profiles,
  and observation robustness.
- Added a clearer front-page schematic and an explicit baseline strategy note for future AKI comparisons.
- Generalized `scripts/make_readme_figures.py` so README figures can be rebuilt from tracked run artifacts.

## [0.1.0] - 2026-04-21

- Refocused the repository on the `State-from-Observation` research pipeline.
- Added the end-to-end stages for SSL training, embedding extraction, clustering,
  phenotype evaluation, and observation robustness analysis.
- Added reproducibility manifests, run configuration snapshots, and git/dataset provenance capture.
- Added tests for tensor construction, model shapes, and a synthetic end-to-end smoke run.
- Added a synthetic demo notebook and top-level research software metadata.
