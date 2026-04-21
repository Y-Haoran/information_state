# Nature-Style Manuscript Draft

This document is a submission-style manuscript draft for the `information_state` repository. It is written to publication standard in tone and structure, but it is intentionally conservative about the current evidence base. The present implementation has completed a bounded real-data smoke run on MIMIC-IV v3.1; full-cohort experiments, stronger baselines, and full ablations still need to be run before journal submission.

## Title

State-from-Observation learns latent clinical states from irregular intensive care measurements

## Abstract

Electronic health records from intensive care units are not direct measurements of patient state. They are sparse, irregular, and strongly shaped by clinical observation policy, because clinicians measure different variables at different times in response to evolving physiology. Here we introduce `State-from-Observation`, a representation-learning framework that treats intensive care data as an observation tensor rather than a flat time series. Each clinical position is encoded as a triplet comprising value, observation mask, and time since last observation. A state formation operator then jointly weights information contribution across variable identity, relative time, and observation context to form a latent clinical state. The framework is trained with self-supervised contrastive learning on adjacent windows from the same stay, enabling phenotype discovery without requiring dense labels. In a bounded real-data smoke run on MIMIC-IV v3.1, the full pipeline executed end-to-end, including tensor construction, self-supervised training, embedding extraction, clustering, phenotype summary, and robustness analysis. These pilot results establish technical feasibility and support the central hypothesis that clinically meaningful state representations can be learned directly from irregular observation processes. Full-scale experiments are now required to quantify phenotype quality, generalization, and robustness at publication strength.

## Main

Clinical machine learning has often treated intensive care unit data as if it were an ordinary multivariate time series. This simplification is convenient, but it misses a defining property of real-world clinical data: patients are not observed uniformly. Instead, clinicians selectively interrogate physiology. A lactate level that is newly ordered after a period of silence carries a different meaning from a value obtained during dense surveillance; a missing measurement is not merely absent data but part of the observation process itself. We therefore argue that latent clinical state should be learned from the joint structure of what was measured, when it was measured, and what value it returned.

This view motivates a reformulation of ICU trajectories as an observation tensor

\[
O \in \mathbb{R}^{T \times D \times 3},
\]

in which each position is represented by the triplet \((v_{t,d}, m_{t,d}, \Delta_{t,d})\), corresponding to value, observation mask, and time since last observation. Rather than building separate branches for variables, time, and missingness and later fusing them, `State-from-Observation` uses a single state formation operator that contextualizes each observation triplet against the full observation field. In this sense, the model does not attempt to reconstruct an imagined fully observed clinic. Instead, it accumulates information from a stream of irregular clinical queries and responses.

### Conceptual contribution

The central claim of this work is simple: clinical state is not directly observed; it is inferred from a stream of irregular observations, each carrying a context-dependent amount of information. This framing departs from imputation-centric modelling strategies in two ways. First, the observation process is promoted from nuisance to signal. Second, the unit of representation becomes a local observation event rather than a generic sequence token. The resulting architecture can therefore be interpreted as an observation-to-state learner rather than a direct analogue of natural language or vision transformers.

### Model overview

The model consists of three stages. First, an observation encoder maps each local triplet \((v, m, \Delta)\) to a hidden representation, augmented with variable and time embeddings. Second, a state formation operator computes context-sensitive interactions across all positions in the window by combining content similarity with learned variable-relation, relative-time, and observation-state terms. Third, the contextualized observations are aggregated into a latent state vector for the window. Self-supervised contrastive learning is then applied to adjacent windows from the same ICU stay, encouraging temporal consistency of nearby physiological states while separating unrelated windows across the batch.

This design is implemented in the repository as a complete pipeline: feature resolution from MIMIC-IV dictionaries, hourly tensor construction, sliding-window sampling, self-supervised training, single-window embedding extraction, clustering, phenotype profiling, and observation robustness analysis.

### Real-data pilot execution

To verify end-to-end functionality on real data, we ran the complete pipeline on MIMIC-IV v3.1 using a bounded smoke configuration. The source data were located under the PhysioNet MIMIC-IV v3.1 layout with gzipped `hosp` and `icu` tables. Because those tables are distributed in compressed form, the repository was extended to resolve either `.csv` or `.csv.gz` inputs automatically.

In this bounded real-data run, the pipeline processed 16 ICU stays and produced 450 windows in total, split into 397 training windows, 29 validation windows, and 24 test windows. A one-epoch self-supervised run completed successfully on CPU, after which embeddings were extracted for train and validation windows, clustered with \(k=4\), and passed through phenotype evaluation and robustness analysis. The train clustering run produced a reported silhouette score of approximately 0.94, but the cluster size distribution was highly imbalanced, with one dominant cluster containing 367 of 397 windows. The robustness evaluation on the validation split reported near-zero embedding drift under the current perturbation setting and complete cluster stability.

These pilot outputs should not yet be interpreted as biological or clinical evidence. The bounded run used only a very small subset of the available cohort and only one chunk per source table, which resulted in extremely sparse feature profiles. As a consequence, many cluster-level feature means were effectively zero after preprocessing, indicating that this smoke run primarily establishes technical correctness and artifact compatibility rather than phenotype validity. This is the expected role of the pilot experiment, and it is an appropriate foundation for the full-scale analysis.

### Why the current pilot is still informative

Even though the current run is not publication-grade as a result section, it already answers several critical feasibility questions. First, the observation tensor can be built directly from MIMIC-IV v3.1 without manual table conversion. Second, the state formation model trains end-to-end on real ICU windows. Third, the downstream scientific loop now exists in executable form: train, extract, cluster, evaluate, and test robustness. That loop is the essential infrastructure required for a serious phenotyping study.

### What the full study must show

A submission-ready version of this work will need stronger evidence in three domains. The first is representation quality: the method should be compared against GRU-style `value/mask/delta` encoders, vanilla flattened transformers, and ablated variants that remove observation-aware interaction terms. The second is phenotype quality: cluster stability, separation, and clinical coherence must be demonstrated across larger cohorts and multiple random initializations. The third is observation robustness: embeddings should remain stable under realistic perturbations of measurement density and ordering, while still remaining sensitive to genuine physiological change.

If these results are borne out at scale, the main contribution of the work will not be that it adds yet another attention variant to clinical deep learning. Rather, it will provide a principled alternative to the standard view of EHR time series by formalizing state as something learned from irregular observation processes.

## Methods

### Data representation

Each ICU stay is discretized into hourly bins. For each variable \(d\) and time step \(t\), the input is defined as

\[
o_{t,d} = (v_{t,d}, m_{t,d}, \Delta_{t,d}),
\]

where \(v_{t,d}\) is the normalized value, \(m_{t,d}\) indicates whether the variable was observed in that bin, and \(\Delta_{t,d}\) is the time since the last true observation of that variable. Values are winsorized using training-split quantiles, standardized, forward-filled within stay, and set to zero when a variable has not yet been observed. Delta values are capped and log-transformed before entering the model.

### Windowing

The current implementation uses fixed 24-hour windows with 1-hour resolution and a stride of 2 hours. Positive pairs for self-supervised training are adjacent windows from the same stay, offset by a small temporal gap. This design encourages temporal state consistency without requiring labels.

### Observation encoder

Each triplet is encoded by a small multilayer perceptron. Variable identity and time-step identity are injected as learned embeddings, producing a local hidden representation for each position.

### State formation operator

The state formation operator computes contextual interactions across all positions in the window. The current implementation combines: content-dependent similarity, a learned variable-relation bias, a learned relative-time bias, and an observation-state interaction term derived from mask and delta. This gives a unified operator over the full observation field rather than a sequence of independently learned branches.

### Self-supervised objective

The encoder is trained with a symmetric InfoNCE objective on adjacent windows. Projection-head outputs are used for contrastive training, whereas downstream phenotyping uses the encoder state itself rather than the projection head.

### Downstream analyses

The current repository supports four downstream analysis steps. First, `extract_embeddings.py` exports one latent state vector per window. Second, `cluster_states.py` fits KMeans clusterings over window-level embeddings and records internal clustering metrics. Third, `evaluate_phenotypes.py` summarizes outcome association, feature profiles, and within-stay cluster transitions. Fourth, `evaluate_observation_robustness.py` perturbs observation masks and re-encodes the same windows to quantify embedding drift and cluster-assignment stability.

## Current pilot results to replace before submission

The following values are real outputs from the bounded smoke run and are useful for internal tracking, but they should be replaced by full-scale estimates before submission:

- cohort size: 16 ICU stays
- total windows: 450
- training windows: 397
- validation windows: 29
- test windows: 24
- training epochs: 1
- train clustering: \(k=4\), silhouette \(\approx 0.94\)
- validation robustness: mean embedding drift \(\approx 2.97 \times 10^{-7}\), cluster stability \(= 1.0\)

These numbers mainly verify that the full pipeline executes and that outputs are written correctly. They do not yet constitute convincing scientific evidence.

## Discussion

The strongest aspect of the present work is conceptual coherence. By centring the observation process rather than treating it as a nuisance to be imputed away, the framework aligns more closely with how ICU data are actually generated. This gives the method an interpretable governing principle: latent state should be formed from a sequence of irregular clinical questions and answers.

The main weakness of the current state of the project is empirical immaturity. The repository now contains the full analysis path required for a phenotyping paper, but the executed evidence is still pilot-scale. A convincing paper will require larger runs, proper baseline comparisons, multiple ablations, and richer clinical interpretation. The current draft should therefore be understood as a high-quality manuscript scaffold paired with a technically validated codebase, not as a claim that the present results are already sufficient for a high-impact journal.

## Data availability

The method is designed for MIMIC-IV v3.1. Access to the raw data requires credentialed approval through PhysioNet and cannot be redistributed with this repository.

## Code availability

All code for data construction, self-supervised training, embedding extraction, clustering, phenotype evaluation, and observation robustness analysis is contained in this repository.

## Immediate next experiments

To upgrade this draft from pilot manuscript to submission candidate, the following experiments should be run next:

1. Full-cohort or syndrome-specific training with substantially more stays, more source chunks, and more epochs.
2. Baseline comparison against GRU-style `value/mask/delta` encoders and vanilla flattened transformers.
3. Ablations removing observation-state, variable-relation, and relative-time terms from the state formation operator.
4. Multi-seed clustering stability and broader `k` sweeps.
5. Clinical profiling on a sufficiently large cohort to yield non-degenerate feature distributions and interpretable trajectories.

## Suggested file use

This file is intended as the manuscript backbone for the project. The fastest path to a submission-grade document is:

1. keep the introduction, conceptual framing, and methods largely intact
2. replace the pilot-scale result paragraphs with full-scale estimates and figures
3. add proper baseline and ablation tables
4. tighten the discussion around whichever clinical cohort you choose for the main paper
