# Autonomous MultiModel Product Intelligence — Master Roadmap
*Continuing from the audited three-dataset foundation. This document does not restart dataset selection, does not replace the locked Rakuten / ABO / MAVE decisions, and does not claim any ML implementation has already occurred unless the project record says so.*

---

## 1. Project Understanding

### 1.1 Vision
The project is not a single-dataset classifier. It is a **product-intelligence system** that combines image and text into a unified product representation, reasons over a multi-level category hierarchy, extracts structured attributes, quantifies its own uncertainty, and — critically — is allowed to **not answer**: it can abstain on weak evidence, route ambiguous cases to a human reviewer, and flag products that fall outside the known category space entirely rather than force-fitting them into a familiar class.

### 1.2 The Five Core Capabilities
1. **Multimodal Product Understanding** — fuse product image and text into one useful representation.
2. **Hierarchical Categorization** — predict at multiple taxonomy levels (e.g., Apparel → Footwear → Running Shoes), not a single flat label.
3. **Confidence / Uncertainty Awareness** — know when the model's prediction is trustworthy vs. shaky.
4. **Selective Prediction / Human Review** — abstain and route low-confidence cases instead of forcing an answer.
5. **Open-Set / Unknown Detection** — recognize products that don't belong to any known category rather than mis-assigning them to the nearest familiar class.

These last three are **system-level, cross-cutting capabilities** — they are not "owned" by any single dataset the way capabilities 1 and 2 are.

### 1.3 The Three-Dataset Architecture — and exactly why each exists

| Dataset | Primary role | Why this dataset specifically |
|---|---|---|
| **Rakuten 2018** | Hierarchical Categorization | Only dataset with an explicit, multi-level (depth 1–8), large-scale (1M instance) product taxonomy. This is the taxonomy backbone. |
| **Amazon Berkeley Objects (ABO)** | Image-rich / Multimodal Product Understanding | The only dataset with audited, high-integrity image↔metadata alignment (100% resolution, 0 mismatches in the audit sample) at meaningful scale (147,702 listings). This supplies the visual grounding. |
| **MAVE** | Attribute / Text Understanding | The only dataset built specifically around structured attribute extraction from product text, now materialized as a 2,907,358-record canonical dataset with full provenance tracking. This supplies fine-grained semantic/attribute signal that neither Rakuten (anonymized categories, no attributes) nor ABO (2.80% description coverage) can provide. |

**Why not force one dataset to do everything:** each dataset was audited and found to be strong in exactly one dimension and weak in the others — Rakuten has hierarchy but anonymized labels and no images or attributes; ABO has images but thin attribute/description coverage (materials 18.98%, descriptions 2.80%); MAVE has rich attribute supervision but is text/metadata-only. Rather than dilute any one dataset's strength by asking it to also cover a role it audited poorly on, the project deliberately keeps them specialized and plans to integrate their *representations and capabilities*, not their raw records, at the system level.

### 1.4 What has been audited and validated per dataset

**Rakuten 2018**
- Full sizes: 800,000 train / 200,000 test / 1,000,000 total instances.
- Hierarchy audit: max depth 8, average depth ~4.0, 3,008 unique complete paths (train), 2,184 (test), 14 root categories — matches the original challenge documentation independently.
- Structural checks: 0 test-only category paths, 0 parent-child conflicts, 0 missing titles, 0 missing category paths, all titles unique in both splits.
- Known limitation: category IDs are intentionally anonymized (semantic names not recoverable) — a property of the released data, not an audit defect.

**ABO**
- Full size: 147,702 product listings (147,127 with main images) — this, not the 398,212 image-record count, is the correct product-instance figure.
- Audit sample (not the full dataset): 495 sampled image paths → 100% (495/495) official image ID resolution → 1,786 matched products → 100% (1,786/1,786) valid image files → 0 alignment mismatches.
- Audited field coverage on the matched sample: Title 100%, Category 100%, Color 69.48%, Style 42.67%, Material 18.98%, Description 2.80%.
- Diversity check: sample spanned varied categories (phone cases, shoes, office products, home/bed/bath, sandals); noted that one image can map to multiple listings (495 images → 1,786 products), flagged as a future leakage concern for splitting, not a defect.

**MAVE**
- Required universe: 2,907,358 ASINs.
- Composition: 2,900,210 direct Amazon-metadata records + 7,148 MAVE-fallback records = 2,907,358 (100.00% coverage).
- Label recovery inputs: 2,226,509 positive MAVE label records (6,697 missing-ASIN matches) and 1,248,009 negative MAVE label records (2,135 missing-ASIN matches); overlap exists between positive/negative matches so these don't sum directly — final unique fallback count is exactly 7,148.
- 28 Amazon metadata files processed.
- Fallback source breakdown: mave_positive 5,013, mave_positive_and_negative 1,684, mave_negative 451 (= 7,148).
- Direct Amazon-metadata field coverage: Title 100%, Amazon category path 100%, Feature 91.10%, Description 89.20%, Brand 86.39%, imageURL/imageURLHighRes 63.66%, Amazon main category 62.36%, Price 44.90%.
- Final quality audit: PASS — 0 duplicate ASINs, 0 missing required, 0 unexpected ASINs, 0 malformed JSON/objects, 0 records without an ASIN.
- Scan-vs-final distinction explicitly documented: the raw scan saw 36,851 duplicate ASIN occurrences and 11,627,406 out-of-required-set records — these are **source-scan statistics**, superseded by the final filtered/deduplicated dataset, and must not be quoted as if they were final-dataset defects.

### 1.5 Full sizes vs. audit samples vs. record types — kept distinct
- **Full dataset sizes:** Rakuten 1,000,000; ABO 147,702 listings; MAVE 2,907,358 records.
- **Audit sample sizes (not full-dataset claims):** ABO's 495 image paths / 1,786 matched products.
- **Image-record counts ≠ product-instance counts:** ABO has 398,212 image *records* but only 147,702 product *listings* — these are not interchangeable.
- **Source-scan statistics ≠ final validation statistics:** MAVE's 36,851 duplicate occurrences and 11,627,406 outside-required-set figures are scan-level; the final dataset has 0 on both counts.

### 1.6 MAVE canonical dataset & provenance design
Each record carries explicit provenance rather than presenting a flattened, source-agnostic view:
- `metadata_source`: `amazon_metadata` (direct) or `mave_fallback`.
- `mave_label_source`: distinguishes `mave_positive`, `mave_negative`, `mave_positive_and_negative` fallback evidence.
- `mave_information_available`: whether MAVE information exists for the record.
- `metadata_complete`: distinguishes direct/complete metadata from fallback records.
- Missing Amazon fields on fallback records are left **empty**, never fabricated.
- `mave_category` / `mave_attributes` preserve available MAVE information specifically for fallback records.

Artifacts already produced: `mave_final_dataset.jsonl`, `mave_final_dataset_statistics.txt`, `mave_final_dataset_validation.txt`, `MAVE_FINAL_DATASET_QUALITY_AUDIT.txt`, `build_final_mave_dataset.py`, `final_dataset_quality_audit.py`.

### 1.7 Completed vs. not yet implemented
**Completed:** capability definition; multi-dataset strategy decision; Rakuten selection + audit + hierarchy validation; ABO selection + audit + alignment validation; MAVE selection, ASIN-universe definition, fallback recovery, canonical dataset construction, and final quality audit (PASS).

**Explicitly NOT yet implemented** (the record is deliberate about this): any ML model training, cross-dataset integration, the multimodal fusion layer, the confidence/uncertainty layer, the selective-prediction/human-review layer, and the open-set/unknown-detection layer. The record states plainly it does not claim implementation work that hasn't happened.

### 1.8 Key engineering constraints already established
- **Rakuten:** preserve hierarchical paths — do not flatten prematurely.
- **ABO:** image reuse across listings (495 images → 1,786 products) means train/val/test splits must be **grouped by image**, not by listing, to prevent leakage.
- **MAVE:** preserve the direct-metadata vs. fallback distinction throughout downstream processing; never conflate the two provenance types.
- **Missing values:** handled explicitly everywhere; never silently fabricated.
- **Closed-set vs. open-set:** treated as fundamentally distinct evaluation regimes, not the same task with a threshold bolted on.
- **Uncertainty / selective prediction:** system-level capabilities layered on top of the three datasets, not owned by any one of them.

### MY UNDERSTANDING OF THE CURRENT PROJECT STATE

**Completed**
- Five-capability definition and multi-dataset strategy locked.
- Rakuten: selected, audited, hierarchy-validated. LOCKED.
- ABO: selected, audited, image-metadata alignment validated. LOCKED.
- MAVE: selected, canonical dataset built (2,907,358 records), final quality audit PASSED. LOCKED.

**Locked decisions**
- Three-dataset architecture (Rakuten / ABO / MAVE) will not be replaced without new audit evidence.
- Rakuten category IDs stay anonymized — no invented semantic mapping.
- ABO product-instance count is 147,702, not 398,212.
- MAVE fallback records stay explicitly flagged, never merged silently into "complete" metadata.

**Not yet implemented**
- Any preprocessing pipeline for ML consumption.
- Any baseline or research model, for any capability.
- Multimodal fusion.
- Confidence/uncertainty estimation.
- Selective prediction / human review logic.
- Open-set / unknown detection.
- Any cross-dataset integration.

**Immediate next stage**
Dataset-specific preprocessing and data-engineering pipelines (Phase A), followed by dataset-specific baselines (Phase B) — establishing per-dataset ML groundwork **before** any fusion or system-level capability work begins.

---

## 2. Current State
See §1.7–1.8 and the table below.

| Area | State |
|---|---|
| Core capability definition | Established |
| Rakuten | Locked and audited |
| ABO | Locked and audited |
| MAVE | Locked; canonical dataset built and audited (PASS) |
| Three-dataset foundation | Established |
| Cross-dataset integration | Future work |
| ML model development | Not yet started |
| Confidence/uncertainty layer | Future work |
| Selective prediction / human review | Future work |
| Open-set / unknown detection | Future work |

---

## 3. Locked Decisions (do not revisit without new evidence)
- Dataset 1 = Rakuten 2018 → Hierarchical Categorization.
- Dataset 2 = ABO → Image-rich / Multimodal Product Understanding.
- Dataset 3 = MAVE → Attribute / Text Understanding.
- ABO is *not* Dataset 3 (2.80% description coverage disqualifies it as the primary text/attribute source).
- MAVE fallback records are not a full Amazon-metadata replacement.
- Rakuten category IDs remain anonymized.
- The five capabilities are system-level; the three datasets are specialized foundations, not five dataset-to-capability 1:1 mappings.

---

## 4. Gaps Remaining
These are genuine gaps in the current record — not the same thing as "insufficient data," which would require new audit evidence to claim:
1. **No documented cross-dataset compatibility analysis** — Rakuten, ABO, and MAVE were audited independently; nothing in the record establishes how their category vocabularies, if at all, might be aligned or bridged.
2. **No documented preprocessing/tokenization/image-pipeline decisions** — audits establish *what's in the data*, not yet *how it will be fed to models*.
3. **No baseline model of any kind exists yet**, per any capability.
4. **No leakage-safe split has been constructed yet** — the record identifies the ABO image-reuse leakage *risk* but hasn't built the grouped split.
5. **No open-set evaluation protocol defined** — which categories/products would represent "unknown" is UNKNOWN in the current record.
6. **No compute/infrastructure plan documented.**

None of these gaps justify replacing or re-auditing Rakuten, ABO, or MAVE — they are next-phase engineering and modeling gaps, not dataset-selection gaps.

---

## 5. Proposed End-to-End Roadmap

Each phase below states Objective, Why, Dataset(s), Inputs, Outputs, Methodology, Recommended algorithms, Evaluation metrics, Validation checks, Leakage risks, Failure modes, Deliverables, and Exit criteria (what must be true before moving on).

### Phase A — Data Engineering & Preprocessing
- **Objective:** Turn the three audited-but-raw datasets into ML-ready artifacts without losing provenance.
- **Why:** Nothing downstream can start safely until each dataset has a defined, reproducible, leakage-aware preprocessing pipeline.
- **Dataset(s):** All three, independently.
- **Inputs:** Rakuten train/test files; ABO listings + image metadata + images; MAVE canonical `mave_final_dataset.jsonl`.
- **Outputs:** Three cleaned, versioned, ML-ready datasets with fixed schemas and documented splits.
- **Methodology:** Rakuten — preserve full hierarchical path as a structured label (not a flattened string); build a path→node lookup. ABO — build an **image-grouped** listing index so no image ID appears in more than one split; resolve `main_image_id` chains once and cache. MAVE — keep `metadata_source`/`mave_label_source` fields intact; do not backfill missing Amazon fields.
- **Recommended tooling:** pandas/pyarrow for tabular; pillow/webdataset for ABO images; jsonlines streaming for MAVE (2.9M records won't comfortably load naively in memory in all environments).
- **Evaluation metrics:** N/A (engineering phase) — validated via structural checks (row counts, null-rate deltas pre/post pipeline).
- **Validation checks:** Re-run the same audit checks from the project record after preprocessing (unique titles, 0 orphaned category paths, 0 duplicate ASINs) to confirm preprocessing didn't silently corrupt anything.
- **Leakage risks:** ABO image reuse (primary risk in this phase); Rakuten test-set category paths must remain a subset of train (already confirmed 0 test-only paths — preprocessing must not break this).
- **Failure modes:** Silent flattening of Rakuten hierarchy; naive random ABO split (would leak reused images across splits); accidentally fabricating "empty string vs. null" semantics differently for MAVE fallback fields than for direct metadata.
- **Deliverables:** Three versioned processed datasets + a data-engineering README documenting every transformation.
- **Exit criteria:** All three datasets pass a re-run of their original structural audits, and ABO has a documented image-grouped split.

### Phase B — Dataset-Specific Baselines
- **Objective:** Establish a simple, well-understood baseline model per dataset before any fusion or advanced technique.
- **Why:** Provides a floor to measure every later, more complex model against; also surfaces unexpected data issues early and cheaply.
- **Dataset(s):** Each independently (Rakuten, ABO, MAVE).
- **Inputs:** Phase A outputs.
- **Outputs:** Three baseline models + baseline metric reports.
- **Methodology:** Rakuten — flat or shallow-hierarchy text classifier on titles (TF-IDF + linear classifier, or a small transformer) predicting leaf category, evaluated with hierarchy-aware metrics even at baseline. ABO — simple CNN/ViT image classifier on the 576 category labels using the image-grouped split. MAVE — attribute-extraction baseline (span-tagging or sequence classification) on text fields where `metadata_complete` is true.
- **Recommended algorithms:** Logistic regression / linear SVM on TF-IDF (Rakuten baseline); ResNet-50 or a small pretrained ViT fine-tune (ABO baseline); BiLSTM-CRF or a small pretrained transformer for token classification (MAVE baseline).
- **Evaluation metrics:** Rakuten — top-1 leaf accuracy, macro-F1, plus a hierarchy-aware metric (see §11). ABO — top-1/top-5 accuracy, macro-F1. MAVE — attribute-level precision/recall/F1.
- **Validation checks:** Confirm baseline results are stable across seeds; sanity-check confusion patterns align with known category overlaps.
- **Leakage risks:** Same as Phase A, now realized in actual model performance if unaddressed (watch for suspiciously high ABO accuracy as a leakage smell test).
- **Failure modes:** Baselines that are too weak to be informative (undertrained) or accidentally strong due to un-caught leakage.
- **Deliverables:** Baseline model artifacts, metric reports, error analysis notebook per dataset.
- **Exit criteria:** All three baselines trained, evaluated, and their failure patterns documented.

### Phase C — Hierarchical Categorization (Rakuten)
- **Objective:** Move from flat baseline to a genuinely hierarchy-aware classifier.
- **Why:** Capability 2 requires multi-level prediction, not just leaf accuracy.
- **Dataset(s):** Rakuten.
- **Inputs:** Phase B Rakuten baseline + full path labels.
- **Outputs:** A hierarchy-aware classifier producing predictions at every level of the path.
- **Methodology:** Compare flat classification (predict leaf, derive ancestors) vs. local-per-level classifiers vs. a single model with hierarchical loss.
- **Recommended algorithms:** Hierarchical softmax or per-level classification heads sharing a text encoder; hierarchy-aware losses (e.g., top-down conditional loss, or a Hierarchical Cross-Entropy that penalizes distant-in-tree errors more).
- **Evaluation metrics:** Per-level accuracy, hierarchical F1, tree-distance error (how far, in the taxonomy, is a wrong prediction from the truth).
- **Validation checks:** Predicted paths must be *valid* (no orphaned nodes / impossible parent-child pairs) — re-use the same parent-child consistency check from the original audit.
- **Leakage risks:** Low here (text-only, unique titles) — main risk is overfitting to memorized short titles rather than generalizable taxonomy features.
- **Failure modes:** Model exploits surface title patterns instead of semantic category understanding; deep-level (near-leaf) accuracy collapses due to class sparsity (thousands of leaf paths with imbalanced support).
- **Deliverables:** Hierarchical classifier, per-level metric report, tree-distance error analysis.
- **Exit criteria:** Hierarchical model beats flat baseline on tree-distance error and per-level accuracy at all levels, not just top-level.

### Phase D — Product Image Understanding (ABO)
- **Objective:** Build a strong visual product representation and category classifier.
- **Why:** Capability 1 (multimodal) needs a visual encoder; ABO is the audited source for this.
- **Dataset(s):** ABO.
- **Inputs:** Phase B ABO baseline, image-grouped splits.
- **Outputs:** A visual encoder + image-based category classifier, reusable downstream as a fusion component.
- **Methodology:** Fine-tune a pretrained visual backbone; evaluate transferability of learned embeddings (not just classification accuracy) since the encoder will later feed the fusion phase.
- **Recommended algorithms:** ViT or ConvNeXt fine-tuning; consider CLIP-style pretrained visual towers to ease later multimodal fusion (image encoder already lives in a joint embedding space with text).
- **Evaluation metrics:** Top-1/top-5 accuracy, macro-F1, embedding-space quality checks (e.g., nearest-neighbor retrieval precision).
- **Validation checks:** Confirm no image ID crosses the train/val/test boundary (re-verify the grouped split holds after any resampling/augmentation).
- **Leakage risks:** Highest-risk phase given the documented image-reuse pattern (495 images → 1,786 products) — must re-verify grouping before every training run, not just once.
- **Failure modes:** Overfitting to background/packaging cues rather than product identity; class imbalance across the 576 categories.
- **Deliverables:** Visual encoder checkpoint, classification report, embedding-quality report.
- **Exit criteria:** Visual encoder achieves stable performance and passes an explicit "zero leaked images across splits" check.

### Phase E — Text / Attribute Understanding (MAVE)
- **Objective:** Build a strong attribute-extraction model that also respects provenance distinctions.
- **Why:** Capability 1's text side and general attribute understanding depend on this.
- **Dataset(s):** MAVE.
- **Inputs:** Phase B MAVE baseline, canonical dataset with provenance fields intact.
- **Outputs:** Attribute-extraction model (and/or attribute-aware text encoder) usable downstream in fusion.
- **Methodology:** Train primarily on `metadata_source = amazon_metadata` records (higher completeness); evaluate separately on `mave_fallback` records to check whether the model still performs reasonably where Amazon metadata was absent — this is an important, dataset-record-supported evaluation split the project record explicitly enables via its provenance design.
- **Recommended algorithms:** Transformer-based sequence/span-tagging models for attribute-value extraction (e.g., BIO-tagging transformer); consider a joint category+attribute multi-task head since Amazon category path has 100% coverage.
- **Evaluation metrics:** Attribute-level precision/recall/F1, exact-match span accuracy, separate reporting for direct-metadata vs. fallback subsets.
- **Validation checks:** Confirm missing-field handling in training matches the "empty, not fabricated" rule from the canonical dataset design.
- **Leakage risks:** Low (text-only); watch for near-duplicate titles/descriptions across ASINs from the same brand inflating apparent performance.
- **Failure modes:** Model learns to lean entirely on high-coverage fields (Title, Feature) and fails badly on sparse ones (Price at 44.90% coverage); fallback-record performance silently ignored if not evaluated separately.
- **Deliverables:** Attribute extractor, split-wise (direct vs. fallback) evaluation report.
- **Exit criteria:** Reasonable attribute-F1 on both direct-metadata and fallback subsets, reported separately, not blended.

### Phase F — Multimodal Representation / Fusion
- **Objective:** Combine ABO's visual encoder and MAVE/Rakuten's text understanding into one product representation.
- **Why:** This is Capability 1 directly, and is the technical bridge between the three previously-independent per-dataset efforts.
- **Dataset(s):** Primarily ABO (has both image and text) as the fusion-training source, since it is the only dataset with genuinely paired image+text at the record level; Rakuten and MAVE contribute pretrained/transferred text representations.
- **Inputs:** Phase D visual encoder, Phase E text encoder (and/or Phase C hierarchy-aware text encoder), ABO's paired image+text records.
- **Outputs:** A fused multimodal product embedding.
- **Methodology:** Compare early fusion (concatenate raw features early) vs. late fusion (combine independent predictions) vs. cross-attention fusion.
- **Recommended algorithms:** CLIP-style contrastive image-text alignment as a pretraining step; cross-attention transformer fusion for the final representation; late-fusion ensembling as a cheap, interpretable baseline.
- **Evaluation metrics:** Downstream task performance (category accuracy, attribute F1) using the fused representation vs. single-modality baselines; embedding retrieval metrics (image→text and text→image retrieval).
- **Validation checks:** Ablation-verify the fused model actually beats the stronger single-modality baseline — fusion that underperforms the best single modality is a red flag, not a success.
- **Leakage risks:** Same ABO image-grouping risk, now compounded because both towers train jointly.
- **Failure modes:** One modality "dominates" and the other is effectively ignored (a known multimodal failure mode); fusion architecture overfits to ABO's specific category distribution and doesn't generalize.
- **Deliverables:** Fused encoder, multimodal ablation report (§12 gives ablation detail).
- **Exit criteria:** Fusion beats best single-modality baseline on at least one held-out task; modality-dominance check passed (both modalities shown to contribute via ablation).

### Phase G — Confidence & Uncertainty Estimation
- **Objective:** Attach calibrated confidence to predictions from the fused system.
- **Why:** Capability 3; required before selective prediction (Phase H) can be meaningful.
- **Dataset(s):** All three, wherever a trained classifier exists.
- **Inputs:** Phase C/D/E/F trained models.
- **Outputs:** Calibrated confidence scores accompanying every prediction.
- **Methodology:** Start with post-hoc calibration (cheap, well-understood) before investing in ensembles.
- **Recommended algorithms/trade-offs:**
  - *Temperature scaling* — baseline choice; cheap, requires only a held-out calibration set; limited expressiveness (single scalar).
  - *Deep ensembles* — stronger uncertainty estimates, captures epistemic uncertainty; expensive (Nx training cost); best as an intermediate/research-tier model, not the first thing built.
  - *MC Dropout* — cheaper approximation to ensembling if the architecture already uses dropout; noisier estimates than true ensembles.
  - *Conformal prediction* — distribution-free coverage guarantees, well suited to Rakuten's hierarchical setting (can produce prediction sets rather than single points); recommend as the primary research-tier method given the project's emphasis on rigor.
- **Evaluation metrics:** Expected Calibration Error (ECE), Brier score, reliability diagrams.
- **Validation checks:** Calibration set must be leakage-free from both training and any earlier calibration steps.
- **Leakage risks:** Reusing training data for calibration inflates confidence artificially.
- **Failure modes:** Confidence scores that are well-calibrated in aggregate but badly miscalibrated on rare/tail categories (especially likely given Rakuten's long-tail of 3,008 leaf paths and MAVE's sparse Price/fallback fields).
- **Deliverables:** Calibrated confidence module, ECE/Brier reports, per-category calibration breakdown (not just aggregate).
- **Exit criteria:** ECE below an agreed threshold on the held-out calibration set, with no severe per-category miscalibration outliers.

### Phase H — Selective Prediction / Abstention / Human Review
- **Objective:** Use calibrated confidence to decide when to predict vs. abstain and route to a human.
- **Why:** Capability 4.
- **Dataset(s):** All three, via the calibrated system from Phase G.
- **Inputs:** Phase G calibrated confidence outputs.
- **Outputs:** A selective-prediction policy with a defined coverage/risk trade-off curve.
- **Methodology:** Set a confidence threshold via risk-coverage analysis on a validation set; evaluate at multiple operating points rather than one fixed threshold.
- **Recommended algorithms:** Selective classification with a reject option (Chow's rule generalization); risk-coverage curve optimization; conformal-prediction-based abstention (natural extension of Phase G if conformal prediction was chosen there).
- **Evaluation metrics:** Selective risk at fixed coverage, risk-coverage curve/AUC, human-review workload reduction (% of cases automatically resolved vs. routed).
- **Validation checks:** Confirm abstention rate is not concentrated entirely in specific categories in a way that would systematically overload human reviewers with certain product types.
- **Leakage risks:** Threshold tuned on the same data used for final reporting would overstate selective performance.
- **Failure modes:** Threshold that abstains too rarely (unsafe) or too often (defeats automation purpose); abstention correlated with the exact long-tail/sparse-field categories the system was weakest on to begin with (Rakuten's deep leaf nodes, MAVE's fallback records) — worth checking explicitly.
- **Deliverables:** Selective prediction policy, risk-coverage report, recommended operating point(s) with justification.
- **Exit criteria:** A documented risk-coverage trade-off with at least one operating point that meaningfully reduces human workload without unacceptable risk increase.

### Phase I — Open-Set / Unknown Detection
- **Objective:** Detect products that don't belong to any known category.
- **Why:** Capability 5.
- **Dataset(s):** Primarily Rakuten and ABO (have closed category sets); MAVE contributes attribute-level anomaly signal.
- **Inputs:** Phase F fused representation, Phase G confidence outputs.
- **Outputs:** An open-set detector layered on top of the closed-set classifier.
- **Methodology:** This is a genuinely new experimental setup, not a threshold on existing confidence — requires deliberately held-out categories to simulate "unknown" (see §11 for how to construct this without contaminating training).
- **Recommended algorithms/trade-offs:**
  - *Distance-based detection* (e.g., Mahalanobis distance or nearest-class-mean in embedding space) — simple, interpretable, works well when the fused embedding space (Phase F) is reasonably structured; limited if embeddings aren't well-separated.
  - *Energy-based OOD scoring* — often stronger than softmax-confidence-based OOD detection with minimal added complexity; recommend as the primary baseline for this phase.
  - *Open-set recognition methods* (e.g., OpenMax) — purpose-built for this exact problem, extends softmax with an explicit "unknown" class; good intermediate-tier method.
  - *Full open-set-trained models with explicit reject class* — highest complexity, best asymptotic performance; research-tier only, after simpler methods are benchmarked.
- **Evaluation metrics:** AUROC/AUPRC for known-vs-unknown discrimination, false acceptance rate, false rejection rate, open-set F1.
- **Validation checks:** Verify "unknown" categories used for evaluation were never seen during training/calibration in any capacity (including as negative examples).
- **Leakage risks:** The single largest risk in this phase — if held-out "unknown" categories leak into training/calibration/threshold-setting, open-set metrics become meaningless. Must be a hard separation.
- **Failure modes:** Detector that just re-implements low-confidence detection from Phase G without adding real open-set discrimination; detector overfits to the specific synthetic "unknown" categories chosen for evaluation and doesn't generalize to genuinely novel products.
- **Deliverables:** Open-set detector, AUROC/AUPRC report, explicit documentation of which categories were held out as "unknown" and why.
- **Exit criteria:** Open-set detector meaningfully outperforms a naive low-confidence-threshold baseline on AUROC.

### Phase J — Cross-Dataset Integration
- **Objective:** Reconcile the three independently-trained-and-evaluated dataset efforts into a coherent system.
- **Why:** Up to this point, Rakuten/ABO/MAVE work has been mostly parallel; the system needs them to interoperate.
- **Dataset(s):** All three.
- **Inputs:** All prior phase outputs.
- **Outputs:** A unified interface where a single product (regardless of source dataset) can be run through hierarchy, image, and attribute pathways.
- **Methodology:** Since the three datasets don't share a common product-ID space (this is UNKNOWN/not established in the record — no evidence of ASIN/Rakuten-ID overlap was audited), integration is primarily **capability-level**, not **record-level**: a single trained system accepts (image, text) input regardless of origin and applies hierarchy reasoning, attribute extraction, and multimodal fusion together.
- **Evaluation metrics:** End-to-end task performance on held-out real products (not just per-dataset test splits).
- **Validation checks:** Confirm the unified system's Rakuten-derived hierarchy predictions, ABO-derived visual features, and MAVE-derived attributes remain mutually consistent for the same input product.
- **Leakage risks:** Low if per-phase leakage was already controlled; new risk is subtle metric conflation across differently-sourced test sets.
- **Failure modes:** System performs well per-dataset but incoherently on genuinely new products that don't resemble any single dataset's distribution.
- **Deliverables:** Unified inference interface, cross-dataset consistency report.
- **Exit criteria:** A single input product produces coherent, non-contradictory outputs across all three capability pathways.

### Phase K — End-to-End Product-Intelligence Architecture
- **Objective:** Assemble Phases C–I into the full pipeline described in §13.
- **Why:** This is the actual deliverable system.
- **Dataset(s):** All three, via their trained components.
- **Inputs:** All prior artifacts.
- **Outputs:** A single callable system: product in → (category path, attributes, confidence, accept/review/unknown decision) out.
- **Methodology:** Integration engineering — wire together encoders, fusion, calibration, selective prediction, and open-set detection into one inference path.
- **Evaluation metrics:** End-to-end latency, end-to-end task accuracy, decision-layer distribution (% accept / % review / % unknown) on a realistic mixed test set.
- **Validation checks:** No component silently bypassed; decision layer logic auditable end-to-end.
- **Leakage risks:** N/A at this stage if prior phases were clean; primary risk is now integration bugs.
- **Failure modes:** Components trained/evaluated well in isolation but interact poorly (e.g., calibration thresholds tuned before fusion no longer valid after fusion changes representation).
- **Deliverables:** End-to-end system, integration test suite.
- **Exit criteria:** System runs end-to-end on held-out data with all five capabilities demonstrably active.

### Phase L — Rigorous Evaluation & Ablation Studies
- **Objective:** Prove the system's design choices matter (see §12 for specific ablations).
- **Why:** Distinguishes an engineering project from a research contribution.
- **Dataset(s):** All three.
- **Deliverables:** Full ablation report, final benchmark tables.
- **Exit criteria:** Every major architectural choice (fusion strategy, calibration method, open-set method) has at least one supporting ablation.

### Phase M — Deployment / System Engineering (if appropriate)
- **Objective:** Package the system for real use, if the project's goals extend beyond research.
- **Why:** Only relevant if productionization is an actual goal — **this is not established either way in the current record**, flagged here as conditional.
- **Methodology:** Model serving, batching for the 2.9M-scale MAVE data, monitoring for calibration drift over time.
- **Exit criteria:** Only pursue this phase once K and L are complete and a deployment goal is explicitly confirmed.

---

## 6. Dataset-Specific ML Plan (summary table)

| Dataset | Primary phase | Core model type | Key metric |
|---|---|---|---|
| Rakuten | C | Hierarchy-aware text classifier | Tree-distance error, per-level accuracy |
| ABO | D | Fine-tuned visual encoder | Top-1/5 accuracy, embedding retrieval quality |
| MAVE | E | Attribute-extraction transformer | Attribute-level F1 (direct vs. fallback, reported separately) |

---

## 7. Multimodal Strategy
Fusion (Phase F) should be trained primarily on ABO, since it is the only dataset with genuinely paired image+text at record level. Rakuten and MAVE contribute transferred/pretrained text representations rather than joint training data, because neither has both modalities paired the way ABO does. Recommend starting with **late fusion** as a fast, interpretable baseline, then moving to **cross-attention fusion** as the research-tier approach, using CLIP-style contrastive pretraining as a bridge between the two. Every fusion variant must be ablation-tested against the stronger single-modality baseline (§12).

---

## 8. Uncertainty Strategy
Start cheap (temperature scaling) to establish a calibration baseline quickly, then invest in conformal prediction as the primary research-tier method — it fits naturally with Rakuten's hierarchical output (prediction sets over tree nodes) and gives distribution-free coverage guarantees, which is a stronger property than ensembling alone provides. Deep ensembles and MC dropout are reasonable intermediate comparisons but shouldn't be the final answer given their cost (ensembles) or noisiness (MC dropout). Calibration must be evaluated **per-category**, not just in aggregate, given Rakuten's long tail and MAVE's uneven field coverage.

---

## 9. Selective Prediction Strategy
Build directly on the calibrated confidence from Phase G. Report full risk-coverage curves, not a single threshold, and explicitly check whether abstention concentrates on already-weak subpopulations (deep Rakuten leaves, MAVE fallback records, low-Material-coverage ABO products) — if so, the "human review" queue will be systematically biased toward certain product types, which is worth surfacing rather than hiding.

---

## 10. Open-Set Strategy
Construct held-out "unknown" categories *before* any training begins, from Rakuten's or ABO's category sets, and keep them completely excluded from all training, calibration, and threshold-tuning — this hard separation is the single most important methodological safeguard in the whole roadmap, since leakage here invalidates the open-set metrics entirely. Start with energy-based OOD scoring as the primary baseline (better than raw softmax confidence, similar complexity), compare against distance-based detection and OpenMax as intermediate methods, and reserve full open-set-trained architectures for the research tier.

---

## 11. Evaluation Strategy

**Splits:**
- Rakuten: already has an audited train/test split with 0 test-only paths — reuse it; carve a validation subset from train, preserving path distribution (stratify by root category at minimum, given only 14 roots).
- ABO: build a fresh, **image-grouped** split (no image ID crosses split boundaries) — this doesn't yet exist in the record and must be constructed in Phase A.
- MAVE: split by ASIN, stratified to preserve the direct-metadata vs. fallback ratio (2,900,210 vs. 7,148) in each split so fallback performance can always be evaluated with adequate sample size.

**Cross-dataset compatibility:** no shared product-ID space is established across the three datasets in the current record (UNKNOWN) — treat integration as capability-level (§Phase J), not record-level joining.

**Recommended metrics by capability:**
- Hierarchy (Rakuten): macro/micro-F1 per level, top-k accuracy for leaf prediction, tree-distance error, hierarchical F1.
- Image (ABO): top-1/top-5 accuracy, macro-F1, retrieval precision on embeddings.
- Attributes (MAVE): precision/recall/F1 per attribute type, exact-match span accuracy — reported separately for direct-metadata vs. fallback subsets.
- Calibration: ECE, Brier score, reliability diagrams, per-category calibration breakdown.
- Selective prediction: selective risk at fixed coverage, risk-coverage AUC, human-review workload reduction.
- Open-set: AUROC/AUPRC, false acceptance/rejection rate, open-set F1.

Accuracy alone is explicitly insufficient anywhere in this system — every capability above has a dedicated, non-accuracy metric because the whole point of the project (uncertainty, abstention, unknown detection) is invisible to plain accuracy.

---

## 12. Ablation / Research Strategy

| Ablation | What it tests | Relevant phase |
|---|---|---|
| Text-only vs. image-only vs. multimodal | Whether fusion actually helps over either single modality | F |
| Early fusion vs. late fusion vs. cross-attention | Which fusion architecture is worth its added complexity | F |
| With vs. without hierarchy-aware loss | Whether flat classification is "good enough" for Rakuten | C |
| With vs. without attribute information in the fused representation | Whether MAVE's contribution is load-bearing or marginal | F, E |
| Calibrated vs. uncalibrated confidence | Whether calibration work in Phase G is justified | G |
| Forced prediction vs. selective prediction | Quantifies the actual risk reduction from abstention | H |
| Closed-set vs. open-set evaluation | Shows the system isn't just relabeling everything as "known" | I |
| Energy-based vs. distance-based vs. OpenMax unknown detection | Which open-set method earns its complexity | I |

Every ablation should report the same metric suite as the corresponding phase, on the same held-out data, to keep comparisons fair.

---

## 13. Final Architecture

**Recommended (not yet decided in the report — this is new guidance):**

```
Product (image, text)
   → ABO-trained visual encoder (Phase D)
   → Rakuten/MAVE-informed text encoder (Phase C/E)
   → Multimodal fusion (Phase F)
   → Hierarchical categorization head + attribute extraction head
   → Calibrated confidence (Phase G)
   → Decision layer:
        - high confidence, known category → Accept
        - low confidence, known category  → Human Review (Phase H)
        - open-set signal fires           → Unknown (Phase I)
```

**What's already documented vs. what's proposed here:** the project record establishes *that* the five capabilities exist and *that* the three datasets are complementary — it does not document a specific architecture diagram. Everything in the diagram above is a recommendation for review, not a locked decision.

---

## 14. Risks and Critical Review

Acting as a critical reviewer, not a cheerleader:

1. **No cross-dataset product-ID overlap established.** *Severity: Medium.* If Rakuten, ABO, and MAVE never actually describe overlapping products, "integration" is necessarily capability-level, not data-level — the system may never be validated on a single product seen through all three lenses simultaneously. *Mitigation:* explicitly test the unified system only on genuinely new products at inference time, not on a fictional "shared" test set. *Blocks progress?* No — but should be stated as a known limitation of the final system, not glossed over.

2. **ABO image leakage risk is identified but not yet resolved.** *Severity: High.* 495 images mapping to 1,786 products is a real, quantified reuse pattern; an ungrouped split would silently inflate every ABO metric downstream, including the fusion and open-set phases that depend on ABO. *Mitigation:* image-grouped splitting, done once in Phase A and re-verified at every subsequent training run. *Blocks progress?* Yes — should block Phase D and everything downstream of it until resolved.

3. **MAVE fallback records (7,148) are a small, possibly non-representative slice.** *Severity: Low-Medium.* The fallback category distribution is heavily skewed (Collectible Trading Cards alone = 4,530 of 7,148, i.e. ~63%). A model evaluated only in aggregate could look fine on fallback records while actually just being good at trading cards. *Mitigation:* report fallback performance broken out by category, not just in aggregate. *Blocks progress?* No, but should shape how fallback results are interpreted.

4. **Rakuten's anonymized categories limit interpretability and error analysis.** *Severity: Medium.* Without semantic names, qualitative error analysis (e.g., "is the model confusing Shoes with Sandals?") is impossible — only tree-distance and path-structure analysis is available. *Mitigation:* rely on quantitative tree-distance metrics rather than qualitative category-name inspection; accept this as a permanent constraint of the data, not something to work around. *Blocks progress?* No.

5. **Domain shift between datasets is unaddressed.** *Severity: Medium.* Rakuten is a different marketplace/locale (a Japan-origin e-commerce challenge) than ABO/MAVE (Amazon-sourced). Text style, category granularity, and even units/currency conventions likely differ. A text encoder trained on one may not transfer cleanly to the other. *Mitigation:* validate any cross-dataset text-encoder transfer empirically before relying on it in Phase C or F; don't assume transfer works. *Blocks progress?* No, but should temper expectations for Phase J.

6. **Open-set evaluation construction is entirely undefined so far.** *Severity: High.* Nothing in the record specifies which categories would represent "unknown," and getting this wrong (e.g., leaking held-out categories into calibration) invalidates Phase I's results completely. *Mitigation:* the hard train/calibration/eval separation described in §10, decided and documented *before* any Phase I code is written. *Blocks progress?* Yes, specifically for Phase I — should not be improvised mid-phase.

7. **MAVE's Price field coverage is low (44.90%).** *Severity: Low.* Not a blocker for the core capabilities (none of the five capabilities are price-dependent), but worth flagging if price is ever assumed available downstream.

8. **Reproducibility depends on artifacts not yet inspected here.** *Severity: Low-Medium.* The record lists MAVE build/audit scripts (`build_final_mave_dataset.py`, `final_dataset_quality_audit.py`) but this roadmap hasn't verified their contents — reproducibility of the 2,907,358-record dataset from raw sources should be spot-checked before heavy downstream investment. *Blocks progress?* No, but recommend as an early Phase A task.

9. **Computational scale is a real constraint, not yet planned.** *Severity: Medium.* MAVE alone is 2.9M records; Rakuten is 1M; ABO has ~147K images. Training multiple encoders plus fusion plus ensembles (if chosen for uncertainty) at this scale needs an explicit compute budget, which is UNKNOWN in the current record. *Mitigation:* scope Phase A/B experiments on subsamples first, scale up only after methodology is validated. *Blocks progress?* No, but should inform every phase's timeline.

---

## 15. First 5 Tasks to Execute ("DO THIS NEXT")
1. **Build the ABO image-grouped train/val/test split** and re-verify it against the leakage pattern already identified (495→1,786) — this unblocks Phase D and is the single highest-severity open risk.
2. **Stand up the Rakuten preprocessing pipeline** preserving full hierarchical paths, and re-run the original structural audit checks (0 test-only paths, 0 parent-child conflicts) against the processed output to confirm nothing broke.
3. **Stand up the MAVE preprocessing pipeline**, preserving `metadata_source`/`mave_label_source` provenance fields, and produce stratified splits that keep the fallback-record ratio consistent across train/val/test.
4. **Spot-check reproducibility** of `build_final_mave_dataset.py` and `final_dataset_quality_audit.py` against a sample, before investing further downstream.
5. **Train the three dataset-specific baselines** (Rakuten flat classifier, ABO image classifier, MAVE attribute extractor) to establish floors for every later comparison.

## 16. Recommended Implementation Order
A → B → {C, D, E in parallel, since they're dataset-independent} → F → G → H and I (can partially overlap, since both build on G, but I has a hard prerequisite of its own held-out unknown-category design) → J → K → L → (M, conditionally).

**Dependencies:** F depends on completed C/D/E encoders. G depends on F (or at minimum on any trained classifier). H depends on G. I depends on F and G, plus its own independent held-out-category design work that can start as early as Phase A in parallel. J depends on C–I all being complete. K depends on J. L depends on K but its ablation *design* (§12) can be drafted early, in parallel with earlier phases, then executed once components exist.

**Can be done in parallel:** Phase C (Rakuten), Phase D (ABO), Phase E (MAVE) are fully independent and should be parallelized. Ablation design (§12) and open-set held-out-category selection (§10) can both be drafted during Phase A/B.

**Should NOT be started yet:** Phase M (deployment) until K and L are complete and a deployment goal is confirmed; Phase I implementation before its held-out "unknown" categories are formally defined and isolated; any fusion work (Phase F) before Phase A's ABO leakage-safe split exists.

---

## 17. Definition of "Project Complete"

**Minimum viable research system:** Phases A–I completed with at least one working method per capability (a hierarchical classifier, a visual encoder, an attribute extractor, a fusion method, a calibration method, a selective-prediction policy, and an open-set detector), each with honest, non-inflated metrics and documented leakage safeguards — even if each individual method is the "baseline-tier" option rather than the most advanced one.

**Advanced / industry-level version:** All of the above plus Phase J (real cross-dataset integration testing), Phase K (a single callable end-to-end system), Phase L (full ablation suite substantiating every architectural choice), and Phase M if a deployment goal is confirmed — with research-tier methods (conformal prediction, cross-attention fusion, energy-based or OpenMax open-set detection) preferred over baseline-tier ones throughout.

**Strongest possible research contribution:** A demonstrated, ablation-supported case that (a) multimodal fusion of independently-audited, complementary datasets meaningfully improves hierarchical categorization and attribute extraction over single-source baselines, (b) calibrated uncertainty enables a favorable risk-coverage trade-off for human-review routing, and (c) open-set detection reliably separates known from genuinely unknown products without contaminating the closed-set evaluation — all built on top of a dataset foundation whose provenance, sizing, and leakage risks are audited and explicitly documented rather than assumed, which is not yet common practice in product-intelligence literature and is the most defensible, citable claim this project's engineering rigor supports.

---

*This roadmap is a planning document for review before any implementation begins, per the project's own Part 10 instruction. Items marked UNKNOWN reflect genuine gaps in the current project record, not omissions in this roadmap.*
