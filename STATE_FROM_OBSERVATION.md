# State-from-Observation

This path adds a whole-MIMIC-IV self-supervised pipeline that treats ICU data as an observation field instead of a flat sequence.

Core design:

- input is an hourly observation tensor with triplets `[value, mask, delta]`
- representation learning happens over sliding `24h` windows by default
- positive pairs are adjacent windows from the same ICU stay
- the encoder forms latent state through a single state-formation operator that mixes content, variable relations, relative time, and observation-state interactions

Main files:

- data builder: `mimic_iv_project/observation_data.py`
- encoder: `mimic_iv_project/state_from_observation.py`
- contrastive loss: `mimic_iv_project/contrastive.py`
- SSL trainer: `mimic_iv_project/train_ssl.py`

Typical run:

```bash
cd dosage_prediction
python -m mimic_iv_project.train_ssl \
  --raw-root /path/to/mimic-iv \
  --build-data \
  --window-hours 24 \
  --window-stride-hours 2 \
  --positive-window-gap-hours 2 \
  --epochs 50 \
  --batch-size 32
```

Artifacts are written under `artifacts/state_from_observation/`.
