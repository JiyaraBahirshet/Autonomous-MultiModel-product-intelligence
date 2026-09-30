# Phase C — Advanced ABO Multimodal Product Understanding
## Completion, Evaluation, Reliability & Integrity Freeze Record

**Phase:** C — Advanced ABO Multimodal Product Understanding
**Dataset Scope:** ABO only
**Primary Target:** product_type
**Primary Class Space:** 549 common ABO classes
**Default Seed:** 20260827
**Status:** PHASE C COMPLETE / VERIFIED / FROZEN
**Final Integrity Status:** C6 VERIFIED

---

## 1. Executive Summary

Phase C extends the frozen Phase A and Phase B foundation by investigating whether a jointly learned multimodal model can improve ABO product understanding relative to the previously frozen B5 late-fusion baseline.

Phase C was deliberately restricted to the Amazon Berkeley Objects (ABO) dataset because ABO is the project's designated dataset for image-rich and multimodal product understanding.

The established dataset roles were preserved throughout Phase C:

| Dataset | Established Role | Phase C Treatment |
|---|---|---|
| Rakuten 2018 | Hierarchical categorization / taxonomy intelligence | FROZEN |
| ABO | Image-rich / multimodal product understanding | ACTIVE |
| MAVE | Attribute / text understanding | FROZEN |

Phase C did not reopen the completed B6 cross-dataset work, did not train on Rakuten or MAVE, and did not redefine the project's dataset roles.

The Phase C experiment used a locked experiment contract, deterministic seed, frozen split assignments, authoritative ABO B3 v002 representations, explicit missing-modality rules, validation-based model selection, isolated final test evaluation, and staged integrity audits.

### Final C-Base test result

On the frozen 7,346-record test benchmark:

| Metric | C-Base |
|---|---:|
| Accuracy | **0.786823** |
| Macro-F1 | **0.341346** |
| Weighted-F1 | **0.782435** |

Compared with the frozen B5 late-fusion baseline:

| Metric | B5 | C-Base | Delta |
|---|---:|---:|---:|
| Accuracy | 0.538660 | 0.786823 | +0.248163 |
| Macro-F1 | 0.122360 | 0.341346 | +0.218986 |
| Weighted-F1 | 0.507353 | 0.782435 | +0.275082 |

The comparison was conducted on the same frozen ABO common target space and frozen final test population.

---

# 2. Purpose of Phase C

Phase C answered the focused research question:

> Can a stronger jointly learned multimodal representation/model improve ABO product understanding compared with the frozen B5 late-fusion baseline under controlled and reproducible evaluation?

Phase C did not restart the project or replace earlier work.

It built on:

- the Phase A validated data foundation;
- frozen ABO split assignments;
- ABO B3 v002 representation;
- ABO B4.2 text baseline;
- ABO B5 multimodal late-fusion baseline;
- completed B6 cross-dataset intelligence.

Phase C therefore represents a controlled progression from independently trained modality models with late score fusion toward a jointly learned multimodal representation and classifier.

---

# 3. Scope and Boundaries

Phase C was strictly ABO-only.

The following boundaries were maintained:

1. Phase C training uses ABO.
2. Rakuten is not used for Phase C model training.
3. MAVE is not used for Phase C model training.
4. No three-dataset Phase C fusion is introduced.
5. B6 cross-dataset experiments remain frozen.
6. Frozen B3/B4/B5 artifacts are not overwritten.
7. Phase C uses the established ABO benchmark and common target space.
8. Final test data remains isolated from training and model selection.

Phase C is therefore an ABO multimodal modeling phase, not a replacement for the project's full multi-dataset architecture.

---

# 4. Frozen Starting Point

Phase C uses the authoritative ABO B3 v002 representation.

The authoritative text representation is:

ecord["b3_representation"]["text"]["combined_tokens"]

The authoritative image representation and metadata are derived from the B3 v002 image boundary.

Where an authoritative B3 v002 representation exists, Phase C uses it rather than reconstructing an equivalent representation independently from raw fields.

The frozen B4.2 and B5 results provide the baseline context.

---

# 5. Frozen B5 Comparator

The B5 multimodal baseline used late score fusion:

P_fusion = 0.9 P_text + 0.1 P_image

Frozen B5 benchmark:

- 549 common classes
- 7,346 paired test records

Frozen B5 test results:

- Accuracy: 0.538660
- Macro-F1: 0.122360
- Weighted-F1: 0.507353

B5 established that text and image information were complementary. It did not establish that a jointly learned multimodal representation would outperform late fusion.

That became the central Phase C experiment.

---

# 6. C1 — Experiment Definition and Contract

## Status

**C1 GATE PASSED / LOCKED**

C1 established the experimental contract before large-scale Phase C training.

The contract specified:

- research question;
- dataset boundary;
- task;
- target label space;
- primary population;
- authoritative representation;
- modality policy;
- missing-modality policy;
- baseline comparator;
- evaluation metrics;
- leakage controls;
- reproducibility controls;
- ablation requirements;
- test-isolation requirements.

### Primary benchmark

- Dataset: ABO
- Target: product_type
- Classes: 549
- Training population: 69,823
- Validation population: 69,867
- Test population: 7,346
- Seed: 20260827

### Missing-modality policy

The primary Phase C experiment requires both authoritative B3 v002 text and a physically available, documented ABO main image.

Records missing either modality are excluded from the primary jointly learned multimodal benchmark.

Missing modalities are never:

- fabricated;
- guessed;
- downloaded;
- synthetically generated;
- reconstructed outside the frozen B3 v002 boundary.

Missing-modality behavior may only be investigated through explicitly separated secondary robustness/fallback analyses.

---

# 7. Compute and Feasibility

Phase C feasibility was evaluated before large-scale training.

Compute environment:

- CPU: Intel Core i7-1355U
- Logical processors: 12
- GPU: Intel Iris Xe
- CUDA: unavailable
- RAM: 15.69 GB
- Python: 3.13
- PyTorch feasibility environment: 2.14.0+cpu
- Default seed: 20260827

### Text feasibility

The selected text encoder was ll-MiniLM-L6-v2.

Canonical maximum sequence length:

256

Token-preserving chunking was implemented at 254 content tokens per chunk, with explicit [CLS] and [SEP] handling.

For the exact 69,823-record training population:

- mean whitespace content tokens: 246.60
- median: 116
- P95: 852
- P99: 1,274
- maximum: 15,174
- 57.82% were within 256 tokens
- 42.18% exceeded 256 tokens

Token-preserving chunking:

- mean chunks: 2.448
- median chunks: 1
- maximum chunks: 158
- 99.68% used 16 or fewer chunks

### Image feasibility

ResNet-18 with ImageNet-1K pretrained weights was selected for the Phase C image representation.

Measured embedding:

- dimension: 512
- mean CPU runtime: 0.0668 seconds/image
- repeat maximum difference: 0
- approximate storage for 69,823 embeddings: 136.37 MiB

---

# 8. C2 — Implementation Smoke Test

C2 verified the implementation before large-scale training.

The final C-Base architecture was:

- text representation: 384 dimensions
- image representation: 512 dimensions
- text projection: 384 → 512 + ReLU
- image projection: 512 → 512 + ReLU
- fusion: 1024 → 512 + ReLU
- classifier: 512 → 549

Trainable parameters:

**1,266,213**

The C2 smoke test verified:

- forward execution;
- loss reduction across smoke-test steps;
- checkpoint save/reload;
- deterministic checkpoint behavior;
- target/class-space compatibility.

The smoke-test validation accuracy was explicitly treated as a smoke-test result and **not** as an evaluation result.

---

# 9. C3 — Embedding Preparation and Training

C3 prepared frozen text/image embeddings for the training and validation populations.

Final populations:

| Population | Eligible | Missing | Target Exclusions | Bad |
|---|---:|---:|---:|---:|
| Train | 69,823 | 461 | 0 | 0 |
| Validation | 69,867 | 70 | 59 | 0 |

Final embedding shapes:

### Train

- text: (69823, 384)
- image: (69823, 512)

### Validation

- text: (69867, 384)
- image: (69867, 512)

All arrays were finite and label/record-ID integrity checks passed.

### C-Base model selection

Best checkpoint:

- epoch: **17**
- selection metric: **validation Macro-F1**
- best validation Macro-F1: **0.4626669030182739**

The final C-Base test evaluation used this frozen epoch-17 checkpoint.

---

# 10. C4 — Final Evaluation

C4 generated the exact frozen test embeddings for the 7,346-record benchmark.

Test embedding shapes:

- text: (7346, 384)
- image: (7346, 512)

Test record IDs were unique and labels were valid.

No test data was used for model training or model selection.

## C4.2 Final C-Base result

Final test metrics:

- Accuracy: **0.7868227606860877**
- Fixed-549 Macro-F1: **0.3413462319143464**
- Weighted-F1: **0.7824345652636053**

These values were independently reproduced during later reliability and integrity audits.

## Comparison with B5

C-Base minus B5:

- Accuracy: **+0.248163**
- Macro-F1: **+0.218986**
- Weighted-F1: **+0.275082**

This demonstrates a measured improvement over the frozen B5 comparator under the locked Phase C benchmark.

---

# 11. C4.3 — Error Analysis

Final C-Base test results:

- test records: 7,346
- correct: 5,780
- incorrect: 1,566
- error rate: 0.213177
- unique true classes: 343
- unique predicted classes: 310
- union of observed classes: 364
- classes with zero test support: 206
- classes with zero predictions: 239
- nonzero directional confusion pairs: 776

Macro-F1 was calculated over the fixed 549-class space.

Representative high-frequency directional confusion pairs included:

- SHOES → BOOT: 62
- HOME_BED_AND_BATH → HOME: 38
- BOOT → SHOES: 37
- SHOES → SANDAL: 21
- SANDAL → SHOES: 19
- TABLE → HOME_FURNITURE_AND_DECOR: 18
- GROCERY → HEALTH_PERSONAL_CARE: 17
- HOME → HOME_BED_AND_BATH: 17
- HOME_FURNITURE_AND_DECOR → OTTOMAN: 13
- CHAIR → HOME_FURNITURE_AND_DECOR: 12

No causal interpretation was assigned to these confusions without representative inspection.

---

# 12. C5 — Reliability and Representation Analysis

C5 consisted of five controlled descriptive analyses.

These analyses used frozen outputs and did not perform:

- model retraining;
- checkpoint selection;
- hyperparameter tuning;
- calibration fitting;
- test-driven threshold optimization;
- selective prediction optimization;
- abstention optimization;
- human-review routing optimization;
- open-set training.

## C5.1 — Reliability Signal Inventory

Generated frozen signals included:

- logits;
- softmax probabilities;
- top-1 probability;
- top-2 probability;
- top-1/top-2 margin;
- predictive entropy;
- correctness;
- 512-dimensional joint representation.

All signals were finite.

Softmax row-sum maximum error:

4.768371582031e-07

C4.2 metrics were reproduced exactly.

### Scientific status

- raw probability behavior: **VERIFIED**
- margin behavior: **VERIFIED**
- entropy behavior: **VERIFIED**
- calibration: **UNKNOWN**
- selective prediction: **UNKNOWN**
- abstention: **UNKNOWN**
- human-review routing: **UNKNOWN**
- open-set detection: **UNKNOWN**

---

# 13. C5.2 — Confidence / Probability Analysis

Overall test population:

**7,346**

Mean top-1 probability:

**0.9108367919**

Median top-1 probability:

**0.99586761**

Mean top-1/top-2 margin:

**0.8470661642**

Mean predictive entropy:

**0.2534016835**

Correct versus incorrect predictions showed descriptive separation:

| Signal | Correct Mean | Incorrect Mean |
|---|---:|---:|
| Top-1 probability | 0.949644 | 0.767602 |
| Top-2 probability | 0.038956 | 0.155358 |
| Margin | 0.910688 | 0.612244 |
| Entropy | 0.148117 | 0.642001 |

At a diagnostic top-1 probability level of 0.9:

**597 incorrect predictions** still had top-1 probability ≥ 0.9.

Some incorrect predictions had probabilities extremely close to 1.

At a diagnostic level of 0.5:

**80 correct predictions** had top-1 probability < 0.5.

Therefore the signals contain information about correctness, but raw softmax probability cannot itself be treated as an already-calibrated operational confidence score.

Calibration remains **UNKNOWN**.

---

# 14. C5.3 — Margin and Ambiguity Analysis

Final test results:

- mean margin for correct predictions: 0.910688
- mean margin for incorrect predictions: 0.612244
- low-margin diagnostic < 0.10: 213 predictions
- high-margin diagnostic >= 0.90: 458 predictions
- unique top-2 competition pairs: 1,621
- unique directional error pairs: 776

Margin/correctness association was demonstrated descriptively.

The diagnostic thresholds were descriptive analysis thresholds only and were not selected as deployment thresholds.

Scientific status:

- margin behavior: **VERIFIED**
- margin/correctness association: **DEMONSTRATED**
- calibration: **UNKNOWN**
- selective prediction: **UNKNOWN**
- abstention: **UNKNOWN**
- human-review routing: **UNKNOWN**
- open-set detection: **UNKNOWN**

---

# 15. C5.4 — Modality Evidence Analysis

The analysis examined the contribution of the two learned modality branches.

Mean projected representation norms:

- text: 8.1076
- image: 21.5821

Mean modality strength shares:

- text: 0.2745
- image: 0.7255

The raw projected norms are not interpreted as direct modality importance because the two branches use different transformations and dimensions before fusion.

### Text ablation

Mean KL divergence between full and text-only predictions:

**1.5254**

Prediction change rate:

**40.6888%**

### Image ablation

Mean KL divergence between full and image-only predictions:

**2.1632**

Prediction change rate:

**50.5037%**

Incorrect samples showed greater sensitivity to modality ablation than correct samples.

This demonstrates that both modalities materially affect the learned multimodal predictions.

It does **not** establish a deployment-level modality importance ranking.

---

# 16. C5.5 — Representation Distance Analysis

C5.5 constructed class prototypes using the frozen **training population only** in the 512-dimensional C-Base joint representation space.

No test representations were used to construct prototypes.

Prototype support:

- minimum class support: 1
- maximum: 10,137
- mean: 127.1821
- median: 14
- zero-support classes: 0

Nearest-prototype diagnostic accuracy:

- Euclidean: 0.6365368908249387
- Cosine: 0.6455213721753336

Correct predictions tended to be closer to their true-class prototype and had more positive prototype margins.

Descriptive correlations with correctness:

- Euclidean margin: 0.449771
- cosine margin: 0.484176
- true Euclidean distance: -0.206252
- true cosine distance: -0.445024

This demonstrates meaningful class geometry in the learned joint representation.

It does not establish prototype distance as a calibrated confidence score or deployment-time routing mechanism.

---

# 17. C6 — Integrity and Reproducibility Freeze

## Status

**C6 VERIFIED**

C6 performed the final integrity and reproducibility audit.

Final C6 result:

status: VERIFIED

ailures: []

Seed:

20260827

Frozen benchmark:

- train: 69,823
- validation: 69,867
- test: 7,346
- classes: 549
- best epoch: 17
- selection metric: Macro-F1

C6 verified:

- C3 train labels;
- C3 class-space integrity;
- C3 test isolation;
- C3 population integrity;
- C3 seed;
- checkpoint existence;
- checkpoint best epoch;
- checkpoint selection metric;
- checkpoint seed;
- checkpoint test isolation;
- C4 test population;
- exact C4 metric reproduction;
- C4 test inference isolation;
- absence of test-based model selection;
- absence of frozen-artifact modifications;
- unique test IDs;
- prediction/label shape consistency;
- C5 seed consistency;
- C5 population consistency;
- C5.1 prediction agreement with C4;
- C5.5 train-only prototype construction;
- C5.2 absence of test-based training/selection/threshold/calibration fitting;
- C5.4 metric reproduction;
- cross-stage 549-class consistency;
- cross-stage 7,346-record population consistency.

### Authoritative C-Base checkpoint hash

SHA-256:

3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9

Checkpoint size:

15,222,122 bytes.

---

# 18. Final Scientific Findings

### Finding 1 — Joint multimodal learning produced a measured improvement over B5

C-Base achieved:

- Accuracy: 0.786823
- Macro-F1: 0.341346
- Weighted-F1: 0.782435

on the frozen 7,346-record benchmark.

### Finding 2 — The comparison was controlled

The comparison used:

- the same ABO common target space;
- frozen benchmark populations;
- isolated final test evaluation;
- deterministic seed;
- validation-based model selection;
- frozen B5 comparator.

### Finding 3 — Both modalities materially contribute

C5.4 showed that removing either modality changes predictions for substantial portions of the test set.

### Finding 4 — The learned representation contains class geometry

C5.5 demonstrated meaningful class-prototype structure in the learned 512-dimensional joint representation.

### Finding 5 — Model-derived signals contain information about correctness

Top-1 probability, margin, entropy, modality sensitivity, and prototype geometry showed descriptive associations with prediction correctness.

### Finding 6 — Raw confidence signals are not sufficient evidence for autonomous routing

High-confidence errors remain present.

Therefore, probability and margin signals cannot be treated as already-calibrated operational confidence.

---

# 19. What Phase C Does NOT Claim

Phase C does **not** establish:

- that the model is calibrated;
- that the model reliably knows when it is uncertain;
- that the model can reliably abstain;
- that the model can automatically route difficult products to human reviewers;
- that the model performs open-set detection;
- that the model detects unknown products;
- that the model is autonomous;
- that softmax probability is a calibrated confidence score;
- that a particular test-set threshold is a valid deployment threshold.

These capabilities require dedicated experiments and evidence.

---

# 20. Relationship to the Overall Project

The overall project remains a multi-dataset product-intelligence system.

The dataset roles remain:

- **Rakuten 2018:** hierarchical categorization / taxonomy intelligence
- **ABO:** image-rich / multimodal product understanding
- **MAVE:** attribute / text understanding

Phase C advances only the ABO multimodal component.

It does not replace:

- Rakuten hierarchical intelligence;
- MAVE attribute/text intelligence;
- B6 cross-dataset intelligence.

It also does not complete the project's future uncertainty, selective prediction, human-review, or open-set capabilities.

---

# 21. Phase C Artifact and Reproducibility Policy

Phase C generated artifacts remain under the repository's existing models/ ignore policy.

The local Phase C artifact tree contains generated embeddings, checkpoints, predictions, probabilities, and analysis arrays.

These generated artifacts are preserved locally and were validated by C6.

The Git repository records the Phase C implementation and completion/freeze documentation without overriding the repository's existing large-model/data artifact policy.

The authoritative C-Base checkpoint SHA-256 is:

3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9

---

# 22. Final Phase C Status

**PHASE C — COMPLETE**

**C1 — VERIFIED / LOCKED**

**C2 — VERIFIED / COMPLETE**

**C3 — VERIFIED / COMPLETE**

**C4 — VERIFIED / COMPLETE**

**C5.1 — VERIFIED / COMPLETE**

**C5.2 — VERIFIED / COMPLETE**

**C5.3 — VERIFIED / COMPLETE**

**C5.4 — VERIFIED / COMPLETE**

**C5.5 — VERIFIED / COMPLETE**

**C6 — VERIFIED / COMPLETE**

### Final statement

Phase C is complete, scientifically verified, reproducible under the frozen experiment contract, and frozen.

No further Phase C rerun, retraining, test tuning, calibration fitting, threshold optimization, or artifact modification is required for the completed Phase C result.
