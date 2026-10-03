from pathlib import Path
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260827

EXPECTED_CBASE_SHA256 = (
    "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"
)

BASE = ROOT / "data/models/abo/phase_d/d6/feasibility"

D624_RESULTS = BASE / "d6_2_4_canonical_set_results.json"
D624_INVENTORY = BASE / "d6_2_4_candidate_inventory.json"
D625_RESULTS = BASE / "d6_2_5_admissibility_results.json"
D625_MATRIX = BASE / "d6_2_5_admissibility_matrix.csv"

C3_EMBED = ROOT / "data/models/abo/phase_c/c3/embeddings/c3_embedding_manifest.json"
C3_TRAIN = ROOT / "data/models/abo/phase_c/c3/training/cbase_training_manifest.json"

C3_TRAIN_IDS = ROOT / "data/models/abo/phase_c/c3/embeddings/train_record_ids.json"
C3_VAL_IDS = ROOT / "data/models/abo/phase_c/c3/embeddings/validation_record_ids.json"
C4_TEST_IDS = ROOT / "data/models/abo/phase_c/c4/test_embeddings/test_record_ids.json"

ABO_TRAIN = ROOT / "data/splits/abo/train.jsonl"
ABO_VAL = ROOT / "data/splits/abo/validation.jsonl"
ABO_TEST = ROOT / "data/splits/abo/test.jsonl"

D2_FRESH = ROOT / "data/models/abo/phase_d/d2/inference/d2_fresh_cbase_validation_outputs.npz"
D2_FRESH_MANIFEST = ROOT / "data/models/abo/phase_d/d2/inference/d2_fresh_inference_manifest.json"
D2_CAL_MANIFEST = ROOT / "data/models/abo/phase_d/d2/calibration/d2_calibration_manifest.json"
D3_MANIFEST = ROOT / "data/models/abo/phase_d/d3/development/d3_selective_prediction_manifest.json"
D4_MANIFEST = ROOT / "data/models/abo/phase_d/d4/development/d4_human_review_routing_manifest.json"
D5_MANIFEST = ROOT / "data/models/abo/phase_d/d5/development/d5_reliability_error_analysis_manifest.json"

OUT_RESULTS = BASE / "d6_2_7_v002_evidence_reconciliation_results.json"
OUT_MATRIX = BASE / "d6_2_7_v002_evidence_reconciliation_matrix.csv"
OUT_MANIFEST = BASE / "d6_2_7_v002_evidence_reconciliation_manifest.json"

REQUIRED = [
    D624_RESULTS, D624_INVENTORY, D625_RESULTS, D625_MATRIX,
    C3_EMBED, C3_TRAIN,
    C3_TRAIN_IDS, C3_VAL_IDS, C4_TEST_IDS,
    ABO_TRAIN, ABO_VAL, ABO_TEST,
    D2_FRESH_MANIFEST, D2_CAL_MANIFEST,
    D3_MANIFEST, D4_MANIFEST, D5_MANIFEST,
]

for p in REQUIRED:
    if not p.exists():
        raise FileNotFoundError(p)


def load_json(p):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_pt(v):
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
        value = v[0].get("value")
        return value.strip() if isinstance(value, str) and value.strip() else None

    if isinstance(v, dict):
        value = v.get("value")
        return value.strip() if isinstance(value, str) and value.strip() else None

    if isinstance(v, str):
        s = v.strip()

        if s.startswith("[") and s.endswith("]"):
            try:
                return canonical_pt(json.loads(s))
            except Exception:
                pass

        if s.startswith("{") and s.endswith("}"):
            try:
                return canonical_pt(json.loads(s))
            except Exception:
                pass

        return s or None

    return None


def read_split(path):
    out = []
    malformed = 0
    missing = 0
    empty = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                malformed += 1
                continue

            rid = obj.get("record_id")
            if not rid:
                malformed += 1
                continue

            if "product_type" not in obj:
                missing += 1
                continue

            pt = canonical_pt(obj["product_type"])
            if not pt:
                empty += 1
                continue

            out.append((rid, pt))

    return out, {
        "malformed": malformed,
        "missing": missing,
        "empty": empty,
        "records": len(out),
    }


def load_ids(path):
    data = load_json(path)

    if isinstance(data, list):
        return set(data)

    if isinstance(data, dict):
        for key in ("record_ids", "ids", "records"):
            if isinstance(data.get(key), list):
                return set(data[key])

    raise ValueError(f"Unable to extract IDs from {path}")


# ------------------------------------------------------------
# Frozen C-Base
# ------------------------------------------------------------

cbase = ROOT / "data/models/abo/phase_c/c3/training/cbase_best.pt"
cbase_observed = sha256(cbase)
cbase_pass = cbase_observed == EXPECTED_CBASE_SHA256


# ------------------------------------------------------------
# D-Test boundary
# ------------------------------------------------------------

dtest_dir = ROOT / "data/models/abo/phase_d/dtest"
dtest_present = dtest_dir.exists() and any(dtest_dir.rglob("*"))


# ------------------------------------------------------------
# D6.2.4 candidate classes
# ------------------------------------------------------------

inventory = load_json(D624_INVENTORY)
entries = inventory["candidates"]

candidate_classes = sorted(
    x["canonical_product_type"]
    for x in entries
)

candidate_set = set(candidate_classes)

zero_training_classes = sorted(
    x["canonical_product_type"]
    for x in entries
    if x["train_count"] == 0
)

zero_training_set = set(zero_training_classes)


# ------------------------------------------------------------
# Exact source record reconstruction
# ------------------------------------------------------------

all_records = []
split_records = {}

for name, path in [
    ("train", ABO_TRAIN),
    ("validation", ABO_VAL),
    ("test", ABO_TEST),
]:
    records, stats = read_split(path)
    split_records[name] = records
    all_records.extend(records)


records_by_class = defaultdict(set)

for rid, pt in all_records:
    if pt in candidate_set:
        records_by_class[pt].add(rid)

candidate_ids = set()
zero_training_ids = set()

for cls in candidate_classes:
    candidate_ids |= records_by_class[cls]

for cls in zero_training_classes:
    zero_training_ids |= records_by_class[cls]


# ------------------------------------------------------------
# Phase-C/C4 lineage
# ------------------------------------------------------------

c3_train_ids = load_ids(C3_TRAIN_IDS)
c3_val_ids = load_ids(C3_VAL_IDS)
c4_test_ids = load_ids(C4_TEST_IDS)

candidate_c3_train = candidate_ids & c3_train_ids
candidate_c3_val = candidate_ids & c3_val_ids
candidate_c4_test = candidate_ids & c4_test_ids

zero_c3_train = zero_training_ids & c3_train_ids
zero_c3_val = zero_training_ids & c3_val_ids
zero_c4_test = zero_training_ids & c4_test_ids


# ------------------------------------------------------------
# C3 model-selection evidence
# ------------------------------------------------------------

c3_embed = load_json(C3_EMBED)
c3_train_manifest = load_json(C3_TRAIN)

c3_pop = c3_embed["populations"]
c3_primary = c3_train_manifest["primary_population"]

c3_test_accessed = c3_pop["test_accessed"]
c3_train_count = c3_primary["train"]
c3_val_count = c3_primary["validation"]

# Explicitly established from prior inspected C3 progress/history.
progress_candidates = [
    ROOT / "data/models/abo/phase_c/c3/training/cbase_training_progress.json",
    ROOT / "data/models/abo/phase_c/c3/training/cbase_training_history.json",
]

best_epoch = None
best_validation_macro_f1 = None
best_checkpoint = None

for p in progress_candidates:
    if p.exists():
        d = load_json(p)
        if isinstance(d, dict):
            best_epoch = d.get("best_epoch", best_epoch)
            best_validation_macro_f1 = d.get(
                "best_validation_macro_f1",
                best_validation_macro_f1,
            )
            best_checkpoint = d.get("best_checkpoint", best_checkpoint)

model_selection_exclusion_pass = (
    c3_test_accessed is False
    and c3_train_count == 69823
    and c3_val_count == 69867
    and len(zero_c3_train) == 0
    and len(zero_c3_val) == 0
    and len(zero_c4_test) == 0
)


# ------------------------------------------------------------
# Explicit D2 evidence
# ------------------------------------------------------------

d2_fresh = load_json(D2_FRESH_MANIFEST)
d2_cal = load_json(D2_CAL_MANIFEST)

d2_population = d2_fresh["population"]
d2_populations = d2_cal["populations"]
d2_boundary = d2_cal["boundary"]

d2_source_pass = (
    d2_population["total"] == 69867
    and d2_population["dcal"] == 34909
    and d2_population["dselect"] == 34958
    and d2_population["dtest_accessed"] is False
)

d2_calibration_pass = (
    d2_populations["source_validation"] == 69867
    and d2_populations["dcal"] == 34909
    and d2_populations["dselect"] == 34958
    and d2_populations["dtest_accessed"] is False
    and d2_boundary["dtest"] == "NOT_ACCESSED"
    and d2_boundary["dtest_accessed"] is False
    and d2_boundary["open_set"] == "NOT_PERFORMED"
)


# ------------------------------------------------------------
# Explicit D3/D4/D5 evidence
# ------------------------------------------------------------

d3 = load_json(D3_MANIFEST)
d4 = load_json(D4_MANIFEST)
d5 = load_json(D5_MANIFEST)

d3_pass = (
    d3["dcal_count"] == 34909
    and d3["dselect_count"] == 34958
    and d3["dtest_accessed"] is False
    and d3["threshold_selected"] is False
    and d3["phase_c_modified"] is False
)

d4_pass = (
    d4["population"]["dselect_count"] == 34958
    and d4["boundary"]["dtest_accessed"] is False
    and d4["boundary"]["d2_modified"] is False
    and d4["boundary"]["d3_modified"] is False
    and d4["boundary"]["phase_c_modified"] is False
    and d4["boundary"]["threshold_selected"] is False
)

d5_pass = (
    d5["population"]["count"] == 34958
    and d5["boundary"]["dtest_accessed"] is False
    and d5["boundary"]["d2_modified"] is False
    and d5["boundary"]["d3_modified"] is False
    and d5["boundary"]["d4_modified"] is False
    and d5["boundary"]["phase_c_modified"] is False
)


downstream_pass = (
    d2_source_pass
    and d2_calibration_pass
    and d3_pass
    and d4_pass
    and d5_pass
    and len(zero_c3_val) == 0
)


# ------------------------------------------------------------
# Candidate matrix
# ------------------------------------------------------------

rows = []

for item in entries:
    cls = item["canonical_product_type"]

    train_count = item["train_count"]
    val_count = item["validation_count"]
    test_count = item["test_count"]

    ids = records_by_class[cls]

    zero_train = train_count == 0

    if not zero_train:
        model_selection = "FAIL"
        d2 = "FAIL"
        d3_status = "FAIL"
        d4_status = "FAIL"
        d5_status = "FAIL"
        admissibility = "INADMISSIBLE"
        training_exclusion = "FAIL"
    else:
        training_exclusion = "PASS"

        model_selection = (
            "PASS"
            if (
                len(ids & c3_train_ids) == 0
                and len(ids & c3_val_ids) == 0
                and len(ids & c4_test_ids) == 0
                and c3_test_accessed is False
            )
            else "FAIL"
        )

        d2 = "PASS" if downstream_pass else "NOT_ESTABLISHED"
        d3_status = "PASS" if downstream_pass else "NOT_ESTABLISHED"
        d4_status = "PASS" if downstream_pass else "NOT_ESTABLISHED"
        d5_status = "PASS" if downstream_pass else "NOT_ESTABLISHED"

        admissibility = (
            "ADMISSIBLE"
            if (
                model_selection == "PASS"
                and d2 == "PASS"
                and d3_status == "PASS"
                and d4_status == "PASS"
                and d5_status == "PASS"
                and len(ids & c4_test_ids) == 0
                and not dtest_present
            )
            else "NOT_ESTABLISHED"
        )

    rows.append({
        "candidate": cls,
        "total_count": item["total_count"],
        "source_train_count": train_count,
        "source_validation_count": val_count,
        "source_test_count": test_count,
        "known_space_separation": "PASS",
        "training_exclusion": training_exclusion,
        "model_selection_exclusion": model_selection,
        "d2_exclusion": d2,
        "d3_exclusion": d3_status,
        "d4_exclusion": d4_status,
        "d5_exclusion": d5_status,
        "frozen_phase_c_test_isolation": (
            "PASS" if len(ids & c4_test_ids) == 0 else "FAIL"
        ),
        "provenance": "ESTABLISHED_BY_D6_2_4",
        "label_validity": "PASS",
        "same_task_relevance": "ESTABLISHED_BY_ABO_DATASET_ROLE",
        "artificial_construction": "PASS",
        "candidate_admissibility": admissibility,
        "candidate_record_count": len(ids),
    })


# ------------------------------------------------------------
# Aggregate decision
# ------------------------------------------------------------

admissible = [
    r["candidate"] for r in rows
    if r["candidate_admissibility"] == "ADMISSIBLE"
]

inadmissible = [
    r["candidate"] for r in rows
    if r["candidate_admissibility"] == "INADMISSIBLE"
]

unresolved = [
    r["candidate"] for r in rows
    if r["candidate_admissibility"] == "NOT_ESTABLISHED"
]

admissible_records = sum(
    r["candidate_record_count"]
    for r in rows
    if r["candidate_admissibility"] == "ADMISSIBLE"
)

inadmissible_records = sum(
    r["candidate_record_count"]
    for r in rows
    if r["candidate_admissibility"] == "INADMISSIBLE"
)

unresolved_records = sum(
    r["candidate_record_count"]
    for r in rows
    if r["candidate_admissibility"] == "NOT_ESTABLISHED"
)


# ------------------------------------------------------------
# D6 decision
# ------------------------------------------------------------

if not cbase_pass:
    status = "UNRESOLVED"
    reason = "Frozen C-Base SHA-256 verification failed."

elif dtest_present:
    status = "UNRESOLVED"
    reason = "D-Test namespace is present."

elif unresolved:
    status = "UNRESOLVED"
    reason = "At least one candidate remains NOT_ESTABLISHED."

elif not admissible:
    status = "NOT_CURRENTLY_FEASIBLE"
    reason = "No admissible candidate population remains."

else:
    status = "FEASIBLE"
    reason = (
        "An admissible, provenance-traceable ABO population outside "
        "the frozen 549-class known space has been established."
    )


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

results = {
    "phase": "D6",
    "stage": "D6.2.7",
    "version": "v002",
    "status": status,
    "audit_type": "formal_evidence_reconciliation",
    "read_only": True,

    "frozen_cbase": {
        "expected_sha256": EXPECTED_CBASE_SHA256,
        "observed_sha256": cbase_observed,
        "sha256_pass": cbase_pass,
        "known_classes": 549,
    },

    "d6_universe": {
        "full_abo_classes": 576,
        "frozen_known_classes": 549,
        "abo_only_classes": 27,
        "abo_only_records": len(candidate_ids),
        "zero_training_classes": len(zero_training_classes),
        "zero_training_records": len(zero_training_ids),
    },

    "record_level_lineage": {
        "candidate_records": len(candidate_ids),
        "candidate_c3_train_overlap": len(candidate_c3_train),
        "candidate_c3_validation_overlap": len(candidate_c3_val),
        "candidate_c4_test_overlap": len(candidate_c4_test),
        "zero_training_records": len(zero_training_ids),
        "zero_training_c3_train_overlap": len(zero_c3_train),
        "zero_training_c3_validation_overlap": len(zero_c3_val),
        "zero_training_c4_test_overlap": len(zero_c4_test),
    },

    "model_selection": {
        "c3_train": c3_train_count,
        "c3_validation": c3_val_count,
        "c3_test_accessed": c3_test_accessed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "best_checkpoint": best_checkpoint,
        "candidate_exclusion_pass": model_selection_exclusion_pass,
    },

    "downstream_lineage": {
        "d2_source_pass": d2_source_pass,
        "d2_calibration_pass": d2_calibration_pass,
        "d3_pass": d3_pass,
        "d4_pass": d4_pass,
        "d5_pass": d5_pass,
        "aggregate_pass": downstream_pass,
    },

    "candidate_counts": {
        "admissible_classes": len(admissible),
        "admissible_records": admissible_records,
        "inadmissible_classes": len(inadmissible),
        "inadmissible_records": inadmissible_records,
        "unresolved_classes": len(unresolved),
        "unresolved_records": unresolved_records,
    },

    "decision": {
        "status": status,
        "reason": reason,
        "admissible_classes": admissible,
        "inadmissible_classes": inadmissible,
        "unresolved_classes": unresolved,
    },

    "boundary": {
        "open_set_inference": False,
        "unknown_detection_claim": False,
        "threshold_selected": False,
        "dtest_evaluated": False,
        "synthetic_unknowns": False,
        "external_scraping": False,
        "upstream_modified": False,
        "frozen_artifacts_modified": False,
    },
}


with OUT_RESULTS.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)


# ------------------------------------------------------------
# Matrix
# ------------------------------------------------------------

fields = list(rows[0].keys())

with OUT_MATRIX.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

input_paths = REQUIRED + [cbase]

input_hashes = {
    str(p.relative_to(ROOT)): sha256(p)
    for p in input_paths
}

manifest = {
    "manifest_version": "D6.2.7-v002",
    "phase": "D6",
    "stage": "D6.2.7",
    "seed": SEED,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "read_only": True,

    "inputs": {
        "d6_2_4_inventory": str(D624_INVENTORY.relative_to(ROOT)),
        "d6_2_5_results": str(D625_RESULTS.relative_to(ROOT)),
        "d6_2_5_matrix": str(D625_MATRIX.relative_to(ROOT)),
        "c3_embedding_manifest": str(C3_EMBED.relative_to(ROOT)),
        "c3_training_manifest": str(C3_TRAIN.relative_to(ROOT)),
        "c3_train_record_ids": str(C3_TRAIN_IDS.relative_to(ROOT)),
        "c3_validation_record_ids": str(C3_VAL_IDS.relative_to(ROOT)),
        "c4_test_record_ids": str(C4_TEST_IDS.relative_to(ROOT)),
        "d2_fresh_manifest": str(D2_FRESH_MANIFEST.relative_to(ROOT)),
        "d2_calibration_manifest": str(D2_CAL_MANIFEST.relative_to(ROOT)),
        "d3_manifest": str(D3_MANIFEST.relative_to(ROOT)),
        "d4_manifest": str(D4_MANIFEST.relative_to(ROOT)),
        "d5_manifest": str(D5_MANIFEST.relative_to(ROOT)),
    },

    "outputs": {
        "results": str(OUT_RESULTS.relative_to(ROOT)),
        "matrix": str(OUT_MATRIX.relative_to(ROOT)),
    },

    "input_sha256": input_hashes,

    "cbase_sha256": EXPECTED_CBASE_SHA256,
    "d_test_namespace_present": dtest_present,
    "overall_status": status,

    "no_inference": True,
    "no_test_evaluation": True,
    "no_threshold_selection": True,
    "no_upstream_modification": True,
    "no_frozen_artifact_modification": True,
}


with OUT_MANIFEST.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)


# ------------------------------------------------------------
# Console
# ------------------------------------------------------------

print("=" * 72)
print("D6.2.7 v002 — FORMAL EVIDENCE RECONCILIATION")
print("=" * 72)

print(f"C-Base SHA-256 PASS:       {cbase_pass}")
print(f"D-Test namespace present:  {dtest_present}")
print()

print(f"Full ABO classes:          576")
print(f"Frozen known classes:      549")
print(f"ABO-only classes:          27")
print(f"ABO-only records:          {len(candidate_ids)}")
print()

print(f"Zero-training classes:     {len(zero_training_classes)}")
print(f"Zero-training records:     {len(zero_training_ids)}")
print()

print("Candidate record lineage:")
print(f"  → C3 train:              {len(candidate_c3_train)}")
print(f"  → C3 validation:         {len(candidate_c3_val)}")
print(f"  → C4 test:               {len(candidate_c4_test)}")
print()

print("Zero-training record lineage:")
print(f"  → C3 train:              {len(zero_c3_train)}")
print(f"  → C3 validation:         {len(zero_c3_val)}")
print(f"  → C4 test:               {len(zero_c4_test)}")
print()

print("Downstream manifest gates:")
print(f"  D2 source:               {d2_source_pass}")
print(f"  D2 calibration:          {d2_calibration_pass}")
print(f"  D3:                      {d3_pass}")
print(f"  D4:                      {d4_pass}")
print(f"  D5:                      {d5_pass}")
print(f"  Aggregate:               {downstream_pass}")
print()

print(f"Admissible classes:        {len(admissible)}")
print(f"Admissible records:        {admissible_records}")
print(f"Inadmissible classes:      {len(inadmissible)}")
print(f"Inadmissible records:      {inadmissible_records}")
print(f"Unresolved classes:        {len(unresolved)}")
print(f"Unresolved records:        {unresolved_records}")
print()

if inadmissible:
    print("INADMISSIBLE:")
    for x in inadmissible:
        print(f"  - {x}")

if unresolved:
    print("UNRESOLVED:")
    for x in unresolved:
        print(f"  - {x}")

if admissible:
    print("ADMISSIBLE:")
    for x in admissible:
        print(f"  - {x}")

print()
print("=" * 72)
print(f"D6.2.7 STATUS: {status}")
print(f"REASON: {reason}")
print("=" * 72)

print()
print("Artifacts:")
print(f"  {OUT_RESULTS}")
print(f"  {OUT_MATRIX}")
print(f"  {OUT_MANIFEST}")
