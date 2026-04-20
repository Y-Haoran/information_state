# Information State

`information_state` is a focused repository for `State-from-Observation`: a representation-learning framework that treats ICU EHR as an observation tensor rather than a flat time series.

Core architecture:

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

## Scope

This repo is intentionally narrow. It contains only:

- MIMIC-IV feature catalog resolution for a curated observation set
- hourly observation-tensor building with `value`, `mask`, and `delta`
- window sampling for adjacent positive pairs
- the `State-from-Observation` encoder and Clinical State Formation operator
- self-supervised contrastive training

It does not contain the earlier blood-culture classifiers or unrelated baseline tasks.

## Package Layout

- `information_state/config.py`: configuration, paths, and feature definitions
- `information_state/feature_catalog.py`: resolve curated variables against MIMIC-IV dictionaries
- `information_state/observation_data.py`: build hourly tensors and sliding windows
- `information_state/state_from_observation.py`: observation encoder and state formation operator
- `information_state/contrastive.py`: symmetric InfoNCE loss
- `information_state/train_ssl.py`: SSL training entrypoint

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 -m information_state.train_ssl \
  --raw-root /path/to/mimic-iv \
  --build-data \
  --window-hours 24 \
  --window-stride-hours 2 \
  --positive-window-gap-hours 2 \
  --epochs 50 \
  --batch-size 32
```

Artifacts are written under `artifacts/state_from_observation/`.
