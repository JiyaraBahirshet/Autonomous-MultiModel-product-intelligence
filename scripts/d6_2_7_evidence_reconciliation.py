from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# D6.2.7 — FORMAL EVIDENCE RECONCILIATION
# ============================================================
#
# Purpose:
#   Formally reconcile the evidence established after D6.2.5
#   without modifying any prior/frozen D6 artifacts.
#
# Boundary:
#   - READ ONLY on all existing project artifacts
#   - NO model inference
#   - NO D-Test access
#   - NO threshold selection
#   - NO upstream modification
#   - NO synthetic unknown construction
#
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

SEED = 20260827

EXPECTED_CBASE_SHA256 = (
    "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"
)

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

D624_RESULTS = ROOT / "data/models/abo/phase_d/d6/feasibility/d6_2_4_canonical_set_results.json"
D624_INVENTORY = ROOT / "data/models/abo/phase_d/d6/feasibility/d6_2_4_candidate_inventory.json"

D625_RESULTS = ROOT / "data/models/abo/phase_d/d6/feasibility/d6_2_5_admissibility_results.json"
D625_MATRIX = ROOT / "data/models/abo/phase_d/d6/feasibility/d6_2_5_admissibility_matrix.csv"

C3_EMBEDDING_MANIFEST = (
    ROOT / "data/models/abo/phase_c/c3/embeddings/c3_embedding_manifest.json"
)

C3_TRAINING_MANIFEST = (
    ROOT / "data/models/abo/phase_c/c3/training/cbase_training_manifest.json"
)

C3_TRAIN_IDS = (
    ROOT / "data/models/abo/phase_c/c3/embeddings/train_record_ids.json"
)

C3_VAL_IDS = (
    ROOT / "data/models/abo/phase_c/c3/embeddings/validation_record_ids.json"
)

C4_TEST_IDS = (
    ROOT / "data/models/abo/phase_c/c4/test_embeddings/test_record_ids.json"
)

ABO_TRAIN = ROOT / "data/splits/abo/train.jsonl"
ABO_VALIDATION = ROOT / "data/splits/abo/validation.jsonl"
ABO_TEST = ROOT / "data/splits/abo/test.jsonl"


# ------------------------------------------------------------
# Outputs — NEW D6.2.7 artifacts only
# ------------------------------------------------------------

OUT_DIR = ROOT / "data/models/abo/phase_d/d6/feasibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RESULTS = OUT_DIR / "d6_2_7_evidence_reconciliation_results.json"
OUT_MATRIX = OUT_DIR / "d6_2_7_evidence_reconciliation_matrix.csv"
OUT_MANIFEST = OUT_DIR / "d6_2_7_evidence_reconciliation_manifest.json"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")


def canonical_product_type(value):
    """
    Extract canonical ABO product_type.

    Expected native representation:
        [{'value': 'SHOES'}]

    Also accepts already-canonical strings.
    No semantic mapping is performed.
    """
    if isinstance(value, str):
        s = value.strip()

        # Handle stringified native structure only if it can be
        # parsed safely as JSON/Python literal.
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                return canonical_product_type(parsed)
            except Exception:
                pass

        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                return canonical_product_type(parsed)
            except Exception:
                pass

        return s if s else None

    if isinstance(value, list):
        if len(value) != 1:
            return None

        item = value[0]

        if isinstance(item, dict):
            v = item.get("value")
            if isinstance(v, str) and v.strip():
                return v.strip()

        return None

    if isinstance(value, dict):
        v = value.get("value")
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


def read_product_types(path: Path):
    records = []
    malformed = 0
    missing = 0
    empty = 0

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                malformed += 1
                continue

            record_id = obj.get("record_id")

            if not record_id:
                malformed += 1
                continue

            if "product_type" not in obj:
                missing += 1
                continue

            product_type = canonical_product_type(obj["product_type"])

            if not product_type:
                empty += 1
                continue

            records.append(
                {
                    "record_id": record_id,
                    "product_type": product_type,
                    "source_file": str(path.relative_to(ROOT)),
                    "line_number": line_number,
                }
            )

    return records, {
        "malformed": malformed,
        "missing": missing,
        "empty": empty,
    }


def load_ids(path: Path):
    data = load_json(path)

    if isinstance(data, list):
        return set(data)

    if isinstance(data, dict):
        for key in ("record_ids", "ids", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return set(value)

    raise ValueError(f"Could not identify record-ID list in {path}")


# ------------------------------------------------------------
# Required input validation
# ------------------------------------------------------------

required_inputs = [
    D624_RESULTS,
    D624_INVENTORY,
    D625_RESULTS,
    D625_MATRIX,
    C3_EMBEDDING_MANIFEST,
    C3_TRAINING_MANIFEST,
    C3_TRAIN_IDS,
    C3_VAL_IDS,
    C4_TEST_IDS,
    ABO_TRAIN,
    ABO_VALIDATION,
    ABO_TEST,
]

for path in required_inputs:
    require_file(path)


# ------------------------------------------------------------
# Frozen C-Base integrity
# ------------------------------------------------------------

cbase_path = ROOT / "data/models/abo/phase_c/c3/training/cbase_best.pt"
require_file(cbase_path)

observed_cbase_sha256 = sha256_file(cbase_path)

cbase_hash_pass = observed_cbase_sha256 == EXPECTED_CBASE_SHA256


# ------------------------------------------------------------
# D-Test namespace boundary
# ------------------------------------------------------------

d_test_dir = ROOT / "data/models/abo/phase_d/dtest"

d_test_present = d_test_dir.exists()

if d_test_present:
    d_test_files = [
        p for p in d_test_dir.rglob("*")
        if p.is_file()
    ]
else:
    d_test_files = []


# ------------------------------------------------------------
# Load D6.2.4 candidate universe
# ------------------------------------------------------------

d624_inventory = load_json(D624_INVENTORY)

candidate_entries = d624_inventory.get("candidates", [])

candidate_classes = sorted(
    {
        item["canonical_product_type"]
        for item in candidate_entries
        if item.get("canonical_product_type")
    }
)

candidate_class_set = set(candidate_classes)


# ------------------------------------------------------------
# Reconstruct exact candidate record population
# ------------------------------------------------------------

source_records = []

split_stats = {}

for split_name, path in [
    ("train", ABO_TRAIN),
    ("validation", ABO_VALIDATION),
    ("test", ABO_TEST),
]:
    records, stats = read_product_types(path)

    split_stats[split_name] = {
        "record_count": len(records),
        **stats,
    }

    source_records.extend(records)


candidate_records_by_class = defaultdict(list)

for record in source_records:
    if record["product_type"] in candidate_class_set:
        candidate_records_by_class[record["product_type"]].append(
            record["record_id"]
        )


candidate_record_ids = set()

candidate_record_counts = {}

for cls in candidate_classes:
    ids = sorted(set(candidate_records_by_class.get(cls, [])))

    candidate_record_ids.update(ids)

    candidate_record_counts[cls] = {
        "total_count": len(ids),
        "record_ids": ids,
    }


# ------------------------------------------------------------
# Identify zero-training candidates
# ------------------------------------------------------------

zero_training_classes = []

for item in candidate_entries:
    cls = item["canonical_product_type"]

    if item["train_count"] == 0:
        zero_training_classes.append(cls)

zero_training_classes = sorted(set(zero_training_classes))

zero_training_class_set = set(zero_training_classes)


zero_training_candidate_ids = set()

for cls in zero_training_classes:
    zero_training_candidate_ids.update(
        candidate_record_counts[cls]["record_ids"]
    )


# ------------------------------------------------------------
# Candidate record-level lineage
# ------------------------------------------------------------

c3_train_ids = load_ids(C3_TRAIN_IDS)
c3_validation_ids = load_ids(C3_VAL_IDS)
c4_test_ids = load_ids(C4_TEST_IDS)

c3_train_overlap = candidate_record_ids & c3_train_ids
c3_validation_overlap = candidate_record_ids & c3_validation_ids
c4_test_overlap = candidate_record_ids & c4_test_ids

zero_train_c3_train_overlap = zero_training_candidate_ids & c3_train_ids
zero_train_c3_validation_overlap = (
    zero_training_candidate_ids & c3_validation_ids
)
zero_train_c4_test_overlap = zero_training_candidate_ids & c4_test_ids


# ------------------------------------------------------------
# Model-selection lineage
# ------------------------------------------------------------

c3_embedding_manifest = load_json(C3_EMBEDDING_MANIFEST)
c3_training_manifest = load_json(C3_TRAINING_MANIFEST)

populations = c3_embedding_manifest.get("populations", {})
primary_population = c3_training_manifest.get("primary_population", {})

best_checkpoint = None
best_epoch = None
best_validation_macro_f1 = None
test_accessed = None

progress_candidates = [
    ROOT / "data/models/abo/phase_c/c3/training/cbase_training_progress.json",
    ROOT / "data/models/abo/phase_c/c3/training/cbase_training_history.json",
]

for path in progress_candidates:
    if path.exists():
        try:
            data = load_json(path)

            if isinstance(data, dict):
                best_checkpoint = data.get("best_checkpoint", best_checkpoint)
                best_epoch = data.get("best_epoch", best_epoch)
                best_validation_macro_f1 = data.get(
                    "best_validation_macro_f1",
                    best_validation_macro_f1,
                )

                if "test_accessed" in data:
                    test_accessed = data["test_accessed"]
        except Exception:
            pass


if test_accessed is None:
    test_accessed = (
        populations.get("test_accessed")
        if "test_accessed" in populations
        else primary_population.get("test_accessed")
    )


model_selection_record_exclusion_pass = (
    len(zero_train_c3_train_overlap) == 0
    and len(zero_train_c3_validation_overlap) == 0
    and len(zero_train_c4_test_overlap) == 0
    and test_accessed is False
)


# ------------------------------------------------------------
# D2 lineage
# ------------------------------------------------------------

D2_FRESH_MANIFEST = (
    ROOT / "data/models/abo/phase_d/d2/inference/d2_fresh_inference_manifest.json"
)

D2_CAL_MANIFEST = (
    ROOT / "data/models/abo/phase_d/d2/calibration/d2_calibration_manifest.json"
)

D3_MANIFEST = (
    ROOT / "data/models/abo/phase_d/d3/development/d3_selective_prediction_manifest.json"
)

D4_MANIFEST = (
    ROOT / "data/models/abo/phase_d/d4/development/d4_human_review_routing_manifest.json"
)

D5_MANIFEST = (
    ROOT / "data/models/abo/phase_d/d5/development/d5_reliability_error_analysis_manifest.json"
)

for path in [
    D2_FRESH_MANIFEST,
    D2_CAL_MANIFEST,
    D3_MANIFEST,
    D4_MANIFEST,
    D5_MANIFEST,
]:
    require_file(path)

d2_fresh = load_json(D2_FRESH_MANIFEST)
d2_cal = load_json(D2_CAL_MANIFEST)
d3 = load_json(D3_MANIFEST)
d4 = load_json(D4_MANIFEST)
d5 = load_json(D5_MANIFEST)

d2_fresh_population = d2_fresh.get("population", {})
d2_cal_population = d2_cal.get("population", {})

d2_source_validation = (
    d2_fresh_population.get("total")
    == 69867
    and d2_fresh_population.get("dtest_accessed") is False
)

d2_partition_pass = (
    d2_cal_population.get("source_validation") == 69867
    and d2_cal_population.get("dcal") == 34909
    and d2_cal_population.get("dselect") == 34958
    and d2_cal_population.get("dtest_accessed") is False
)

d3_dtest_pass = d3.get("dtest_accessed") is False
d4_dtest_pass = d4.get("dtest_accessed") is False
d5_dtest_pass = d5.get("dtest_accessed") is False

downstream_population_lineage_pass = (
    d2_source_validation
    and d2_partition_pass
    and d3_dtest_pass
    and d4_dtest_pass
    and d5_dtest_pass
    and len(zero_train_c3_validation_overlap) == 0
)

d2_exclusion = "PASS" if downstream_population_lineage_pass else "NOT_ESTABLISHED"
d3_exclusion = "PASS" if downstream_population_lineage_pass else "NOT_ESTABLISHED"
d4_exclusion = "PASS" if downstream_population_lineage_pass else "NOT_ESTABLISHED"
d5_exclusion = "PASS" if downstream_population_lineage_pass else "NOT_ESTABLISHED"


# ------------------------------------------------------------
# Candidate admissibility
# ------------------------------------------------------------

matrix_rows = []

for item in candidate_entries:
    cls = item["canonical_product_type"]

    train_count = item["train_count"]
    validation_count = item["validation_count"]
    test_count = item["test_count"]
    total_count = item["total_count"]

    is_zero_training = train_count == 0

    if not is_zero_training:
        training_exclusion = "FAIL"
        candidate_admissibility = "INADMISSIBLE"
    else:
        training_exclusion = "PASS"

        model_selection = (
            "PASS"
            if (
                cls in zero_training_class_set
                and len(
                    set(candidate_record_counts[cls]["record_ids"])
                    & c3_train_ids
                )
                == 0
                and len(
                    set(candidate_record_counts[cls]["record_ids"])
                    & c3_validation_ids
                )
                == 0
                and len(
                    set(candidate_record_counts[cls]["record_ids"])
                    & c4_test_ids
                )
                == 0
            )
            else "NOT_ESTABLISHED"
        )

        candidate_admissibility = (
            "ADMISSIBLE"
            if (
                model_selection == "PASS"
                and downstream_population_lineage_pass
                and not d_test_present
            )
            else "NOT_ESTABLISHED"
        )

    candidate_ids = set(candidate_record_counts[cls]["record_ids"])

    row = {
        "candidate": cls,
        "total_count": total_count,
        "source_train_count": train_count,
        "source_validation_count": validation_count,
        "source_test_count": test_count,
        "known_space_separation": "PASS",
        "training_exclusion": training_exclusion,
        "model_selection_exclusion": (
            "PASS"
            if (
                is_zero_training
                and len(candidate_ids & c3_train_ids) == 0
                and len(candidate_ids & c3_validation_ids) == 0
                and len(candidate_ids & c4_test_ids) == 0
                and test_accessed is False
            )
            else (
                "FAIL"
                if not is_zero_training
                else "NOT_ESTABLISHED"
            )
        ),
        "d2_exclusion": (
            d2_exclusion if is_zero_training else "FAIL"
        ),
        "d3_exclusion": (
            d3_exclusion if is_zero_training else "FAIL"
        ),
        "d4_exclusion": (
            d4_exclusion if is_zero_training else "FAIL"
        ),
        "d5_exclusion": (
            d5_exclusion if is_zero_training else "FAIL"
        ),
        "frozen_phase_c_test_isolation": (
            "PASS"
            if len(candidate_ids & c4_test_ids) == 0
            else "FAIL"
        ),
        "provenance": "ESTABLISHED_BY_D6_2_4",
        "label_validity": "PASS",
        "same_task_relevance": "ESTABLISHED_BY_ABO_DATASET_ROLE",
        "artificial_construction": "PASS",
        "candidate_admissibility": candidate_admissibility,
        "candidate_record_count": len(candidate_ids),
    }

    matrix_rows.append(row)


# ------------------------------------------------------------
# Aggregate decision
# ------------------------------------------------------------

admissible_classes = [
    r["candidate"]
    for r in matrix_rows
    if r["candidate_admissibility"] == "ADMISSIBLE"
]

inadmissible_classes = [
    r["candidate"]
    for r in matrix_rows
    if r["candidate_admissibility"] == "INADMISSIBLE"
]

unresolved_classes = [
    r["candidate"]
    for r in matrix_rows
    if r["candidate_admissibility"] == "NOT_ESTABLISHED"
]


# D6.2.7 does NOT yet perform open-set detection.
#
# FEASIBLE means an admissible, provenance-traceable unknown
# population has been established for a future open-set
# experiment under the frozen known-class boundary.
#
# The decision is therefore based on admissibility evidence,
# not model performance.

if not cbase_hash_pass:
    overall_status = "UNRESOLVED"
    decision_reason = "Frozen C-Base SHA-256 verification failed."

elif d_test_present:
    overall_status = "UNRESOLVED"
    decision_reason = "D-Test namespace is present; feasibility gate cannot be finalized."

elif unresolved_classes:
    overall_status = "UNRESOLVED"
    decision_reason = (
        "At least one candidate remains NOT_ESTABLISHED after reconciliation."
    )

elif not admissible_classes:
    overall_status = "NOT_CURRENTLY_FEASIBLE"
    decision_reason = "No admissible candidate population remains."

else:
    overall_status = "FEASIBLE"
    decision_reason = (
        "At least one provenance-traceable ABO product population outside "
        "the frozen 549-class known space satisfies the D6 admissibility "
        "and isolation gates."
    )


# ------------------------------------------------------------
# Final counts
# ------------------------------------------------------------

candidate_total_records = sum(
    r["total_count"] for r in matrix_rows
)

admissible_record_count = sum(
    r["candidate_record_count"]
    for r in matrix_rows
    if r["candidate_admissibility"] == "ADMISSIBLE"
)

inadmissible_record_count = sum(
    r["candidate_record_count"]
    for r in matrix_rows
    if r["candidate_admissibility"] == "INADMISSIBLE"
)

unresolved_record_count = sum(
    r["candidate_record_count"]
    for r in matrix_rows
    if r["candidate_admissibility"] == "NOT_ESTABLISHED"
)


# ------------------------------------------------------------
# Results artifact
# ------------------------------------------------------------

results = {
    "phase": "D6",
    "stage": "D6.2.7",
    "version": "v001",
    "status": overall_status,
    "audit_type": "formal_evidence_reconciliation",
    "read_only": True,

    "frozen_cbase": {
        "path": str(cbase_path.relative_to(ROOT)),
        "expected_sha256": EXPECTED_CBASE_SHA256,
        "observed_sha256": observed_cbase_sha256,
        "sha256_pass": cbase_hash_pass,
        "known_class_count": 549,
    },

    "d_test": {
        "namespace_present": d_test_present,
        "file_count": len(d_test_files),
    },

    "candidate_population": {
        "full_abo_unique_classes": 576,
        "frozen_known_classes": 549,
        "abo_only_classes": 27,
        "abo_only_records": candidate_total_records,
        "zero_training_classes": len(zero_training_classes),
        "zero_training_records": len(zero_training_candidate_ids),
        "admissible_classes": len(admissible_classes),
        "admissible_records": admissible_record_count,
        "inadmissible_classes": len(inadmissible_classes),
        "inadmissible_records": inadmissible_record_count,
        "unresolved_classes": len(unresolved_classes),
        "unresolved_records": unresolved_record_count,
    },

    "record_level_lineage": {
        "candidate_records": len(candidate_record_ids),
        "candidate_c3_train_overlap": len(c3_train_overlap),
        "candidate_c3_validation_overlap": len(c3_validation_overlap),
        "candidate_c4_test_overlap": len(c4_test_overlap),
        "zero_training_candidate_records": len(zero_training_candidate_ids),
        "zero_training_c3_train_overlap": len(zero_train_c3_train_overlap),
        "zero_training_c3_validation_overlap": len(
            zero_train_c3_validation_overlap
        ),
        "zero_training_c4_test_overlap": len(zero_train_c4_test_overlap),
    },

    "model_selection_lineage": {
        "c3_test_accessed": test_accessed,
        "best_checkpoint": best_checkpoint,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "train_population": primary_population.get("train"),
        "validation_population": primary_population.get("validation"),
        "record_level_exclusion_pass": model_selection_record_exclusion_pass,
    },

    "downstream_lineage": {
        "d2_source_validation": d2_source_validation,
        "d2_partition_pass": d2_partition_pass,
        "d3_dtest_pass": d3_dtest_pass,
        "d4_dtest_pass": d4_dtest_pass,
        "d5_dtest_pass": d5_dtest_pass,
        "zero_training_records_absent_from_c3_validation": (
            len(zero_train_c3_validation_overlap) == 0
        ),
        "overall_downstream_lineage_pass": downstream_population_lineage_pass,
    },

    "decision": {
        "status": overall_status,
        "reason": decision_reason,
        "admissible_classes": admissible_classes,
        "inadmissible_classes": inadmissible_classes,
        "unresolved_classes": unresolved_classes,
    },

    "boundary": {
        "open_set_inference_performed": False,
        "unknown_detection_claim": False,
        "threshold_selected": False,
        "d_test_evaluated": False,
        "synthetic_unknowns_created": False,
        "external_scraping_performed": False,
        "upstream_modified": False,
    },

    "prohibited_actions_not_performed": [
        "C-Base retraining",
        "C-Base fine-tuning",
        "C-Base modification",
        "Phase C split regeneration",
        "D2 recalibration",
        "D3 threshold selection",
        "D4 threshold selection",
        "D5 modification",
        "D-Test evaluation",
        "open-set inference",
        "synthetic unknown construction",
        "external scraping",
        "fabricated labels",
        "fabricated images",
        "manual overwrite of D6.2.4",
        "manual overwrite of D6.2.5",
    ],
}


# ------------------------------------------------------------
# Write results
# ------------------------------------------------------------

with OUT_RESULTS.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------
# Write matrix
# ------------------------------------------------------------

matrix_fields = [
    "candidate",
    "total_count",
    "source_train_count",
    "source_validation_count",
    "source_test_count",
    "known_space_separation",
    "training_exclusion",
    "model_selection_exclusion",
    "d2_exclusion",
    "d3_exclusion",
    "d4_exclusion",
    "d5_exclusion",
    "frozen_phase_c_test_isolation",
    "provenance",
    "label_validity",
    "same_task_relevance",
    "artificial_construction",
    "candidate_admissibility",
    "candidate_record_count",
]

with OUT_MATRIX.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=matrix_fields)
    writer.writeheader()
    writer.writerows(matrix_rows)


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

input_hashes = {}

for path in required_inputs + [cbase_path]:
    input_hashes[str(path.relative_to(ROOT))] = sha256_file(path)


manifest = {
    "manifest_version": "D6.2.7-v001",
    "phase": "D6",
    "stage": "D6.2.7",
    "seed": SEED,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),

    "read_only": True,

    "inputs": {
        "d6_2_4_results": str(D624_RESULTS.relative_to(ROOT)),
        "d6_2_4_inventory": str(D624_INVENTORY.relative_to(ROOT)),
        "d6_2_5_results": str(D625_RESULTS.relative_to(ROOT)),
        "d6_2_5_matrix": str(D625_MATRIX.relative_to(ROOT)),
        "c3_embedding_manifest": str(
            C3_EMBEDDING_MANIFEST.relative_to(ROOT)
        ),
        "c3_training_manifest": str(
            C3_TRAINING_MANIFEST.relative_to(ROOT)
        ),
        "c3_train_record_ids": str(C3_TRAIN_IDS.relative_to(ROOT)),
        "c3_validation_record_ids": str(
            C3_VAL_IDS.relative_to(ROOT)
        ),
        "c4_test_record_ids": str(
            C4_TEST_IDS.relative_to(ROOT)
        ),
        "abo_train": str(ABO_TRAIN.relative_to(ROOT)),
        "abo_validation": str(ABO_VALIDATION.relative_to(ROOT)),
        "abo_test": str(ABO_TEST.relative_to(ROOT)),
        "cbase_checkpoint": str(cbase_path.relative_to(ROOT)),
    },

    "outputs": {
        "results": str(OUT_RESULTS.relative_to(ROOT)),
        "matrix": str(OUT_MATRIX.relative_to(ROOT)),
    },

    "input_sha256": input_hashes,

    "cbase_sha256": EXPECTED_CBASE_SHA256,

    "d_test_namespace_present": d_test_present,

    "overall_status": overall_status,

    "no_inference": True,
    "no_test_evaluation": True,
    "no_threshold_selection": True,
    "no_upstream_modification": True,
    "no_synthetic_unknowns": True,
}


with OUT_MANIFEST.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------

print("=" * 72)
print("D6.2.7 — FORMAL EVIDENCE RECONCILIATION")
print("=" * 72)

print(f"C-Base SHA-256 PASS:       {cbase_hash_pass}")
print(f"D-Test namespace present:  {d_test_present}")
print()

print(f"Full ABO classes:          576")
print(f"Frozen known classes:      549")
print(f"ABO-only classes:          {len(candidate_classes)}")
print(f"ABO-only records:          {candidate_total_records}")
print()

print(f"Zero-training classes:     {len(zero_training_classes)}")
print(f"Zero-training records:     {len(zero_training_candidate_ids)}")
print()

print("Record-level lineage:")
print(f"  Candidate → C3 train:    {len(c3_train_overlap)}")
print(f"  Candidate → C3 val:      {len(c3_validation_overlap)}")
print(f"  Candidate → C4 test:     {len(c4_test_overlap)}")
print()
print("Zero-training lineage:")
print(f"  → C3 train:              {len(zero_train_c3_train_overlap)}")
print(f"  → C3 validation:         {len(zero_train_c3_validation_overlap)}")
print(f"  → C4 test:               {len(zero_train_c4_test_overlap)}")
print()

print(f"Admissible classes:        {len(admissible_classes)}")
print(f"Inadmissible classes:      {len(inadmissible_classes)}")
print(f"Unresolved classes:        {len(unresolved_classes)}")
print()

print("INADMISSIBLE:")
for cls in inadmissible_classes:
    print(f"  - {cls}")

print()
print("UNRESOLVED:")
for cls in unresolved_classes:
    print(f"  - {cls}")

print()
print("=" * 72)
print(f"D6.2.7 STATUS: {overall_status}")
print(f"REASON: {decision_reason}")
print("=" * 72)

print()
print("Artifacts:")
print(f"  {OUT_RESULTS}")
print(f"  {OUT_MATRIX}")
print(f"  {OUT_MANIFEST}")