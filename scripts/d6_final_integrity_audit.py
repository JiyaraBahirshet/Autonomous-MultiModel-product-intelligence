from pathlib import Path
import hashlib
import json
import csv
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CBASE = (
    "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"
)

BASE = ROOT / "data/models/abo/phase_d/d6/feasibility"

FILES = {
    "d6_2_4_results":
        BASE / "d6_2_4_canonical_set_results.json",
    "d6_2_4_inventory":
        BASE / "d6_2_4_candidate_inventory.json",
    "d6_2_5_results":
        BASE / "d6_2_5_admissibility_results.json",
    "d6_2_7_results":
        BASE / "d6_2_7_v002_evidence_reconciliation_results.json",
    "d6_2_7_matrix":
        BASE / "d6_2_7_v002_evidence_reconciliation_matrix.csv",
    "d6_2_7_manifest":
        BASE / "d6_2_7_v002_evidence_reconciliation_manifest.json",

    "cbase":
        ROOT / "data/models/abo/phase_c/c3/training/cbase_best.pt",

    "c3_embedding_manifest":
        ROOT / "data/models/abo/phase_c/c3/embeddings/c3_embedding_manifest.json",
    "c3_training_manifest":
        ROOT / "data/models/abo/phase_c/c3/training/cbase_training_manifest.json",

    "c3_train_ids":
        ROOT / "data/models/abo/phase_c/c3/embeddings/train_record_ids.json",
    "c3_validation_ids":
        ROOT / "data/models/abo/phase_c/c3/embeddings/validation_record_ids.json",
    "c4_test_ids":
        ROOT / "data/models/abo/phase_c/c4/test_embeddings/test_record_ids.json",

    "d2_fresh_manifest":
        ROOT / "data/models/abo/phase_d/d2/inference/d2_fresh_inference_manifest.json",
    "d2_calibration_manifest":
        ROOT / "data/models/abo/phase_d/d2/calibration/d2_calibration_manifest.json",
    "d3_manifest":
        ROOT / "data/models/abo/phase_d/d3/development/d3_selective_prediction_manifest.json",
    "d4_manifest":
        ROOT / "data/models/abo/phase_d/d4/development/d4_human_review_routing_manifest.json",
    "d5_manifest":
        ROOT / "data/models/abo/phase_d/d5/development/d5_reliability_error_analysis_manifest.json",
}

OUT = BASE / "d6_final_integrity_audit_results.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


passes = []
warnings = []
failures = []


def check(name, condition, detail=""):
    if condition:
        passes.append((name, detail))
        print(f"[PASS] {name}")
    else:
        failures.append((name, detail))
        print(f"[FAIL] {name}: {detail}")


print("=" * 80)
print("D6 FINAL INTEGRITY AUDIT")
print("=" * 80)

# ------------------------------------------------------------
# 1. Required artifacts
# ------------------------------------------------------------

for name, path in FILES.items():
    check(
        f"required artifact: {name}",
        path.exists(),
        str(path),
    )

# Stop only if fundamental files are missing.
if failures:
    print()
    print("Fundamental artifact check failed; audit cannot continue safely.")
    raise SystemExit(1)


# ------------------------------------------------------------
# 2. Frozen C-Base integrity
# ------------------------------------------------------------

observed_cbase = sha256(FILES["cbase"])

check(
    "frozen C-Base SHA-256",
    observed_cbase == EXPECTED_CBASE,
    f"observed={observed_cbase}",
)


# ------------------------------------------------------------
# 3. D-Test boundary
# ------------------------------------------------------------

dtest_dir = ROOT / "data/models/abo/phase_d/dtest"
dtest_present = dtest_dir.exists() and any(dtest_dir.rglob("*"))

check(
    "D-Test namespace absent",
    not dtest_present,
    str(dtest_dir),
)


# ------------------------------------------------------------
# 4. D6.2.4 universe
# ------------------------------------------------------------

d624_results = load(FILES["d6_2_4_results"])
d624_inventory = load(FILES["d6_2_4_inventory"])

known_classes = set(
    d624_results["frozen_known_classes"]
    if isinstance(d624_results.get("frozen_known_classes"), list)
    else []
)

candidates = d624_inventory["candidates"]

candidate_names = {
    x["canonical_product_type"]
    for x in candidates
}

check(
    "D6.2.4 candidate count = 27",
    len(candidate_names) == 27,
    f"observed={len(candidate_names)}",
)

# Use the authoritative candidate inventory counts.
candidate_records = sum(x["total_count"] for x in candidates)
zero_train = [
    x for x in candidates
    if x["train_count"] == 0
]
training_exposed = [
    x for x in candidates
    if x["train_count"] > 0
]

zero_train_records = sum(x["total_count"] for x in zero_train)
training_exposed_records = sum(x["total_count"] for x in training_exposed)

check(
    "ABO-only candidate records = 122",
    candidate_records == 122,
    f"observed={candidate_records}",
)

check(
    "zero-training candidate classes = 23",
    len(zero_train) == 23,
    f"observed={len(zero_train)}",
)

check(
    "zero-training candidate records = 89",
    zero_train_records == 89,
    f"observed={zero_train_records}",
)

check(
    "training-exposed candidate classes = 4",
    len(training_exposed) == 4,
    f"observed={len(training_exposed)}",
)

check(
    "training-exposed candidate records = 33",
    training_exposed_records == 33,
    f"observed={training_exposed_records}",
)


# ------------------------------------------------------------
# 5. D6.2.7 formal reconciliation
# ------------------------------------------------------------

d627 = load(FILES["d6_2_7_results"])

check(
    "D6.2.7 status = FEASIBLE",
    d627.get("status") == "FEASIBLE",
    f"observed={d627.get('status')}",
)

u = d627["d6_universe"]

check(
    "D6.2.7 full ABO universe = 576",
    u["full_abo_classes"] == 576,
    f"observed={u['full_abo_classes']}",
)

check(
    "D6.2.7 frozen known space = 549",
    u["frozen_known_classes"] == 549,
    f"observed={u['frozen_known_classes']}",
)

check(
    "D6.2.7 ABO-only classes = 27",
    u["abo_only_classes"] == 27,
    f"observed={u['abo_only_classes']}",
)

check(
    "D6.2.7 ABO-only records = 122",
    u["abo_only_records"] == 122,
    f"observed={u['abo_only_records']}",
)


# ------------------------------------------------------------
# 6. Record-level isolation
# ------------------------------------------------------------

lineage = d627["record_level_lineage"]

check(
    "candidate → C3 train overlap = 0",
    lineage["candidate_c3_train_overlap"] == 0,
    str(lineage["candidate_c3_train_overlap"]),
)

check(
    "candidate → C3 validation overlap = 0",
    lineage["candidate_c3_validation_overlap"] == 0,
    str(lineage["candidate_c3_validation_overlap"]),
)

check(
    "candidate → C4 frozen test overlap = 0",
    lineage["candidate_c4_test_overlap"] == 0,
    str(lineage["candidate_c4_test_overlap"]),
)

check(
    "zero-training → C3 train overlap = 0",
    lineage["zero_training_c3_train_overlap"] == 0,
    str(lineage["zero_training_c3_train_overlap"]),
)

check(
    "zero-training → C3 validation overlap = 0",
    lineage["zero_training_c3_validation_overlap"] == 0,
    str(lineage["zero_training_c3_validation_overlap"]),
)

check(
    "zero-training → C4 test overlap = 0",
    lineage["zero_training_c4_test_overlap"] == 0,
    str(lineage["zero_training_c4_test_overlap"]),
)


# ------------------------------------------------------------
# 7. Model-selection boundary
# ------------------------------------------------------------

ms = d627["model_selection"]

check(
    "C3 training population = 69,823",
    ms["c3_train"] == 69823,
    str(ms["c3_train"]),
)

check(
    "C3 validation population = 69,867",
    ms["c3_validation"] == 69867,
    str(ms["c3_validation"]),
)

check(
    "C3 test access = false",
    ms["c3_test_accessed"] is False,
    str(ms["c3_test_accessed"]),
)

check(
    "candidate model-selection exclusion = PASS",
    ms["candidate_exclusion_pass"] is True,
    str(ms["candidate_exclusion_pass"]),
)


# ------------------------------------------------------------
# 8. D2–D5 downstream lineage
# ------------------------------------------------------------

down = d627["downstream_lineage"]

for stage in ["d2_source_pass", "d2_calibration_pass",
              "d3_pass", "d4_pass", "d5_pass"]:
    check(
        f"D6 downstream evidence: {stage}",
        down[stage] is True,
        str(down[stage]),
    )

check(
    "D2–D5 aggregate lineage = PASS",
    down["aggregate_pass"] is True,
    str(down["aggregate_pass"]),
)


# ------------------------------------------------------------
# 9. Candidate decision counts
# ------------------------------------------------------------

counts = d627["candidate_counts"]

check(
    "admissible classes = 23",
    counts["admissible_classes"] == 23,
    str(counts["admissible_classes"]),
)

check(
    "admissible records = 89",
    counts["admissible_records"] == 89,
    str(counts["admissible_records"]),
)

check(
    "inadmissible classes = 4",
    counts["inadmissible_classes"] == 4,
    str(counts["inadmissible_classes"]),
)

check(
    "inadmissible records = 33",
    counts["inadmissible_records"] == 33,
    str(counts["inadmissible_records"]),
)

check(
    "unresolved classes = 0",
    counts["unresolved_classes"] == 0,
    str(counts["unresolved_classes"]),
)

check(
    "unresolved records = 0",
    counts["unresolved_records"] == 0,
    str(counts["unresolved_records"]),
)


# ------------------------------------------------------------
# 10. Explicit inadmissible classes
# ------------------------------------------------------------

expected_inadmissible = {
    "HAIRBAND",
    "PUNCHING_BAG",
    "SALWAR_SUIT_SET",
    "TREADMILL",
}

actual_inadmissible = set(
    d627["decision"]["inadmissible_classes"]
)

check(
    "training-exposed classes exactly identified",
    actual_inadmissible == expected_inadmissible,
    f"observed={sorted(actual_inadmissible)}",
)


# ------------------------------------------------------------
# 11. Explicit admissible class count
# ------------------------------------------------------------

actual_admissible = set(
    d627["decision"]["admissible_classes"]
)

check(
    "admissible class count = 23",
    len(actual_admissible) == 23,
    f"observed={len(actual_admissible)}",
)

check(
    "admissible + inadmissible = all ABO-only candidates",
    actual_admissible | actual_inadmissible == candidate_names,
    (
        f"admissible={len(actual_admissible)}, "
        f"inadmissible={len(actual_inadmissible)}, "
        f"candidates={len(candidate_names)}"
    ),
)

check(
    "admissible ∩ inadmissible = empty",
    actual_admissible.isdisjoint(actual_inadmissible),
    "",
)


# ------------------------------------------------------------
# 12. Boundary controls
# ------------------------------------------------------------

boundary = d627["boundary"]

for field, expected in {
    "open_set_inference": False,
    "unknown_detection_claim": False,
    "threshold_selected": False,
    "dtest_evaluated": False,
    "synthetic_unknowns": False,
    "external_scraping": False,
    "upstream_modified": False,
    "frozen_artifacts_modified": False,
}.items():
    check(
        f"D6 boundary: {field} = {expected}",
        boundary[field] is expected,
        str(boundary[field]),
    )


# ------------------------------------------------------------
# 13. Manifest integrity
# ------------------------------------------------------------

manifest = load(FILES["d6_2_7_manifest"])

check(
    "D6.2.7 manifest version",
    manifest.get("manifest_version") == "D6.2.7-v002",
    str(manifest.get("manifest_version")),
)

check(
    "D6.2.7 manifest read_only = true",
    manifest.get("read_only") is True,
    str(manifest.get("read_only")),
)

check(
    "D6.2.7 manifest status = FEASIBLE",
    manifest.get("overall_status") == "FEASIBLE",
    str(manifest.get("overall_status")),
)

check(
    "manifest C-Base SHA matches frozen SHA",
    manifest.get("cbase_sha256") == EXPECTED_CBASE,
    str(manifest.get("cbase_sha256")),
)

check(
    "manifest D-Test absent",
    manifest.get("d_test_namespace_present") is False,
    str(manifest.get("d_test_namespace_present")),
)

check(
    "manifest no inference",
    manifest.get("no_inference") is True,
    str(manifest.get("no_inference")),
)

check(
    "manifest no test evaluation",
    manifest.get("no_test_evaluation") is True,
    str(manifest.get("no_test_evaluation")),
)

check(
    "manifest no threshold selection",
    manifest.get("no_threshold_selection") is True,
    str(manifest.get("no_threshold_selection")),
)

check(
    "manifest no upstream modification",
    manifest.get("no_upstream_modification") is True,
    str(manifest.get("no_upstream_modification")),
)

check(
    "manifest no frozen-artifact modification",
    manifest.get("no_frozen_artifact_modification") is True,
    str(manifest.get("no_frozen_artifact_modification")),
)


# ------------------------------------------------------------
# 14. Cross-check D2–D5 directly
# ------------------------------------------------------------

d2fresh = load(FILES["d2_fresh_manifest"])
d2cal = load(FILES["d2_calibration_manifest"])
d3 = load(FILES["d3_manifest"])
d4 = load(FILES["d4_manifest"])
d5 = load(FILES["d5_manifest"])

d2_record_ids_source = d2fresh["source"]["record_ids"].replace("\\", "/")

check(
    "D2 fresh source is C3 validation",
    d2_record_ids_source.endswith(
        "c3/embeddings/validation_record_ids.json"
    ),
    d2fresh["source"]["record_ids"],
)
check(
    "D2 fresh dtest_accessed = false",
    d2fresh["population"]["dtest_accessed"] is False,
    str(d2fresh["population"]["dtest_accessed"]),
)

check(
    "D2 calibration dtest_accessed = false",
    d2cal["populations"]["dtest_accessed"] is False,
    str(d2cal["populations"]["dtest_accessed"]),
)

check(
    "D3 dtest_accessed = false",
    d3["dtest_accessed"] is False,
    str(d3["dtest_accessed"]),
)

check(
    "D3 threshold not selected",
    d3["threshold_selected"] is False,
    str(d3["threshold_selected"]),
)

check(
    "D4 dtest_accessed = false",
    d4["boundary"]["dtest_accessed"] is False,
    str(d4["boundary"]["dtest_accessed"]),
)

check(
    "D4 human performance not evaluated",
    d4["boundary"]["human_performance_evaluated"] is False,
    str(d4["boundary"]["human_performance_evaluated"]),
)

check(
    "D5 dtest_accessed = false",
    d5["boundary"]["dtest_accessed"] is False,
    str(d5["boundary"]["dtest_accessed"]),
)


# ------------------------------------------------------------
# 15. Final decision
# ------------------------------------------------------------

audit_status = "PASS" if not failures else "FAIL"

if not failures:
    final_status = "VERIFIED"
else:
    final_status = "NOT_VERIFIED"

result = {
    "phase": "D6",
    "stage": "final_integrity_audit",
    "version": "v001",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": final_status,
    "audit_status": audit_status,

    "summary": {
        "full_abo_classes": 576,
        "frozen_known_classes": 549,
        "abo_only_classes": 27,
        "abo_only_records": 122,
        "admissible_classes": 23,
        "admissible_records": 89,
        "inadmissible_classes": 4,
        "inadmissible_records": 33,
        "unresolved_classes": 0,
        "unresolved_records": 0,
    },

    "frozen_cbase": {
        "expected_sha256": EXPECTED_CBASE,
        "observed_sha256": observed_cbase,
    },

    "dtest_namespace_present": dtest_present,

    "passes": [
        {"name": n, "detail": d}
        for n, d in passes
    ],

    "warnings": [
        {"name": n, "detail": d}
        for n, d in warnings
    ],

    "failures": [
        {"name": n, "detail": d}
        for n, d in failures
    ],

    "boundary": {
        "open_set_detection_performed": False,
        "unknown_detection_claim": False,
        "threshold_selected": False,
        "dtest_evaluated": False,
        "upstream_modified": False,
        "frozen_artifacts_modified": False,
    },

    "freeze_recommendation": (
        "D6 VERIFIED; Phase D may proceed to final freeze."
        if not failures
        else
        "Do not freeze Phase D; resolve failed integrity checks."
    ),
}

with OUT.open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)


print()
print("=" * 80)
print("D6 FINAL INTEGRITY AUDIT SUMMARY")
print("=" * 80)
print(f"PASS CHECKS:      {len(passes)}")
print(f"WARNINGS:         {len(warnings)}")
print(f"FAILURES:         {len(failures)}")
print()
print(f"D6 AUDIT STATUS:  {audit_status}")
print(f"D6 STATUS:        {final_status}")
print()

if failures:
    print("FAILURES:")
    for name, detail in failures:
        print(f"  - {name}: {detail}")

print()
print("D6 SUMMARY:")
print("  Full ABO classes:          576")
print("  Frozen known classes:      549")
print("  ABO-only classes:           27")
print("  ABO-only records:          122")
print("  Admissible classes:         23")
print("  Admissible records:         89")
print("  Inadmissible classes:        4")
print("  Inadmissible records:       33")
print("  Unresolved classes:          0")
print("  D-Test namespace:       ABSENT")
print()
print(f"Audit artifact:")
print(f"  {OUT}")
print("=" * 80)
