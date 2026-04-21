# Contributing

## Scope

This repository is intentionally narrow. Contributions should strengthen the
`State-from-Observation` pipeline itself:

- observation tensor construction
- state formation modeling
- self-supervised training
- embedding extraction, clustering, and phenotype evaluation
- robustness and reproducibility infrastructure

Please do not add unrelated modeling tasks or one-off project code.

## Development Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Before Opening a Pull Request

Run the local validation steps:

```bash
python3 -m py_compile information_state/*.py
python3 -m unittest discover -s tests
```

If you touch the real-data pipeline, include:

- the exact command you ran
- the output artifact directory
- the generated `run_config.json` or manifest path

## Coding Expectations

- preserve the repo's narrow research scope
- keep public entrypoints typed and documented
- prefer artifact-level reproducibility over ad hoc notes
- add or update tests when changing tensor construction, window sampling, or state formation logic
- avoid committing protected data or large generated artifacts

## Data and Privacy

Do not commit raw MIMIC-IV data, identifiers, or protected exports. The repo may
contain tiny synthetic examples and bounded metadata artifacts that are already
present locally for smoke testing, but raw source tables must remain outside the repo.
