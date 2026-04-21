# Information State

`information_state` is a focused research software repository for `State-from-Observation`: a representation-learning framework that treats ICU EHR as an observation tensor rather than a flat time series.

```text
Input:
O ∈ R^(T × D × 3)
(value, mask, delta)

      ↓
Observation Encoder
encode each triplet into local representation

      ↓
State Formation Operator
jointly weigh information contribution
across variable, time, and observation context

      ↓
State Aggregation
pool all contextualized observations

      ↓
Latent Clinical State s

      ↓
SSL / clustering / downstream tasks
```

## Why This Repo Exists

The central claim is narrow:

- clinical state is not directly observed
- ICU data should be modeled as irregular observation triplets
- learned state should be sensitive to physiology but less brittle to observation density

This repo contains only the code needed to build, train, inspect, and stress-test that claim on MIMIC-IV.

## Package Layout

- `information_state/config.py`: project paths, feature definitions, artifact locations
- `information_state/feature_catalog.py`: curated variable resolution against MIMIC-IV dictionaries
- `information_state/observation_data.py`: hourly observation tensor construction and window datasets
- `information_state/state_from_observation.py`: observation encoder and state formation operator
- `information_state/contrastive.py`: symmetric InfoNCE objective
- `information_state/train_ssl.py`: contrastive pretraining entrypoint
- `information_state/extract_embeddings.py`: window-level latent state export
- `information_state/cluster_states.py`: clustering into candidate latent phenotypes
- `information_state/evaluate_phenotypes.py`: outcome, physiology, and transition summaries
- `information_state/evaluate_observation_robustness.py`: embedding drift under observation thinning
- `tests/`: tensor semantics, model-shape, and end-to-end synthetic smoke tests
- `notebooks/01_state_from_observation_demo.ipynb`: protected-data-free concept demo

## Install

For a lightweight setup:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

For a pinned baseline matching the current bounded run:

```bash
python3 -m pip install -r requirements-lock.txt
python3 -m pip install -e .
```

## Run

Train the SSL encoder:

```bash
python3 -m information_state.train_ssl \
  --raw-root /path/to/mimic-iv \
  --build-data \
  --window-hours 24 \
  --window-stride-hours 2 \
  --positive-window-gap-hours 2 \
  --epochs 50 \
  --batch-size 32 \
  --seed 7
```

Then run the downstream stages:

```bash
python3 -m information_state.extract_embeddings --split train val --seed 7
python3 -m information_state.cluster_states --split train --k 4 --seed 7
python3 -m information_state.evaluate_phenotypes
python3 -m information_state.evaluate_observation_robustness --split val --seed 7
```

The same flow is also exposed as console scripts after `pip install -e .`:

- `information-state-train`
- `information-state-extract`
- `information-state-cluster`
- `information-state-evaluate`
- `information-state-robustness`

## What You Should See After Running

Artifacts are written under `artifacts/state_from_observation/`. A complete bounded run now produces:

```text
artifacts/state_from_observation/
  cohort.csv
  feature_stats.json
  hourly_metadata.json
  hourly_values.npy
  hourly_masks.npy
  hourly_deltas.npy
  state_from_observation_ssl.pt
  ssl_history.json
  run_config.json
  window_metadata.csv
  manifests/
    train_ssl_latest.json
    extract_embeddings_latest.json
    cluster_states_latest.json
    evaluate_phenotypes_latest.json
    evaluate_observation_robustness_latest.json
  embeddings/
    train_embeddings.npy
    train_metadata.csv
    val_embeddings.npy
    val_metadata.csv
    embedding_manifest.json
    run_config.json
  clusters/
    cluster_assignments.csv
    cluster_model.npz
    cluster_summary.json
    run_config.json
  evaluation/
    cluster_outcomes.csv
    cluster_feature_profiles.csv
    cluster_trajectory_profiles.csv
    evaluation_report.md
    run_config.json
  robustness/
    robustness_metrics.csv
    robustness_summary.json
    embedding_drift_histogram.png
    run_config.json
```

### Concrete Bounded Run Example

The repo currently includes a bounded real-data smoke run on a small MIMIC-IV subset. It is useful as a proof-of-work artifact, not as a paper-ready result.

Bounded run snapshot:

- stays: `16`
- windows: `450`
- train embeddings: `(397, 32)`
- selected clustering: `k=4`
- train silhouette: `0.946`
- train Davies-Bouldin: `0.956`
- validation robustness cluster stability: `1.0`
- robustness plot: `artifacts/state_from_observation/robustness/embedding_drift_histogram.png`

Example cluster outcome summary from `cluster_outcomes.csv`:

| cluster | n_windows | n_stays | mortality_rate | icu_los_days_mean |
| --- | ---: | ---: | ---: | ---: |
| 0 | 375 | 11 | 0.189 | 9.292 |
| 1 | 9 | 1 | 0.000 | 15.613 |
| 2 | 7 | 1 | 0.000 | 15.613 |
| 3 | 6 | 1 | 0.000 | 15.613 |

These numbers come from the current local artifacts:

- `artifacts/state_from_observation/embeddings/embedding_manifest.json`
- `artifacts/state_from_observation/clusters/cluster_summary.json`
- `artifacts/state_from_observation/evaluation/cluster_outcomes.csv`
- `artifacts/state_from_observation/robustness/robustness_summary.json`

## Reproducibility Contract

Each pipeline stage now writes both:

- a stage-local `run_config.json`
- a timestamped manifest under `artifacts/state_from_observation/manifests/`

Those manifests include:

- CLI arguments
- serialized project config
- git commit and dirty-state status
- runtime context
- dataset artifact hashes
- output artifact paths

That means a checkpoint or clustering result can be traced back to:

- code version
- window length and stride
- positive-pair gap
- seed
- dataset metadata and feature statistics

## Tests and Demo

Scientific integrity checks live in `tests/`:

- tensor shape and binary mask semantics
- delta reset and capping semantics
- positive-window sampling gap rules
- model behavior on missing-heavy batches
- full synthetic `train → extract → cluster → evaluate → robustness` smoke run

Run them with:

```bash
python3 -m py_compile information_state/*.py
python3 -m unittest discover -s tests
```

For a data-free walkthrough of the idea itself, open:

- [notebooks/01_state_from_observation_demo.ipynb](notebooks/01_state_from_observation_demo.ipynb)

## Scope Boundaries

This repo does not include:

- earlier blood-culture classifiers
- unrelated treatment modeling tasks
- target trial emulation
- large downstream benchmark suites outside the `State-from-Observation` claim

That restriction is intentional. The goal is to keep the repo aligned with one scientific story.

## Citation and License

- citation metadata: [CITATION.cff](CITATION.cff)
- manuscript draft: [NATURE_STYLE_MANUSCRIPT_DRAFT.md](NATURE_STYLE_MANUSCRIPT_DRAFT.md)
- license: [LICENSE](LICENSE)
- contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- change history: [CHANGELOG.md](CHANGELOG.md)
