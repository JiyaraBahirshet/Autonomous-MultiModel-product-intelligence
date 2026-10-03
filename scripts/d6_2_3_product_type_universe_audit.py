import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")
EXPECTED_CBASE_SHA = "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"

KNOWN_CLASSES_PATH = ROOT / "data/models/abo/phase_c/c3/embeddings/c3_embedding_manifest.json"
CBASE_PATH = ROOT / "data/models/abo/phase_c/c3/training/cbase_best.pt"

SPLITS = {
    "train": ROOT / "data/splits/abo/train.jsonl",
    "validation": ROOT / "data/splits/abo/validation.jsonl",
    "test": ROOT / "data/splits/abo/test.jsonl",
}

OUT_DIR = ROOT / "data/models/abo/phase_d/d6/feasibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INVENTORY_OUT = OUT_DIR / "d6_2_3_product_type_universe_inventory.json"
RESULTS_OUT = OUT_DIR / "d6_2_3_product_type_universe_results.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_known_classes():
    with open(KNOWN_CLASSES_PATH, "r", encoding="utf-8") as f:
        obj = json.load(f)

    classes = obj.get("target", {}).get("classes")
    if not isinstance(classes, list):
        raise RuntimeError("Could not find target.classes in c3_embedding_manifest.json")

    classes = [str(x) for x in classes]
    if len(classes) != 549 or len(set(classes)) != 549:
        raise RuntimeError(
            f"Frozen class universe invalid: count={len(classes)}, unique={len(set(classes))}"
        )
    return set(classes)


def scan_split(path):
    counter = Counter()
    malformed = 0
    rows = 0
    missing_product_type = 0
    empty_product_type = 0
    examples = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            rows += 1
            try:
                rec = json.loads(line)
            except Exception:
                malformed += 1
                continue

            value = rec.get("product_type")

            if value is None:
                missing_product_type += 1
                continue

            if isinstance(value, list):
                values = [str(x).strip() for x in value if str(x).strip()]
            else:
                values = [str(value).strip()] if str(value).strip() else []

            if not values:
                empty_product_type += 1
                continue

            # product_type is expected to be a single categorical value.
            # If a list is ever encountered, record every explicit value but flag it.
            for v in values:
                counter[v] += 1
                examples.setdefault(v, {"line": line_no})

    return {
        "path": str(path),
        "rows": rows,
        "malformed_json_rows": malformed,
        "missing_product_type": missing_product_type,
        "empty_product_type": empty_product_type,
        "unique_product_types": len(counter),
        "counts": dict(counter),
        "examples": examples,
    }


print("=" * 72)
print("D6.2.3 EXACT ABO PRODUCT_TYPE UNIVERSE AUDIT")
print("READ-ONLY AUDIT")
print("  - no model inference")
print("  - no D-Test evaluation")
print("  - no threshold selection")
print("  - no artifact modification")
print("  - no synthetic/external data")
print("=" * 72)

# 1. Frozen C-Base boundary
print("\n1. FROZEN C-BASE CHECK")
observed_sha = sha256(CBASE_PATH)
print(f"  observed: {observed_sha}")
print(f"  expected: {EXPECTED_CBASE_SHA}")
if observed_sha != EXPECTED_CBASE_SHA:
    raise RuntimeError("C-Base SHA mismatch. STOP.")
print("  HASH: PASS")

# 2. D-Test namespace boundary
print("\n2. D-TEST BOUNDARY")
dtest_dir = ROOT / "data/models/abo/phase_d/dtest"
dtest_present = dtest_dir.exists()
print(f"  D-Test namespace present: {dtest_present}")
if dtest_present:
    raise RuntimeError("D-Test namespace exists. STOP.")

# 3. Frozen known classes
print("\n3. FROZEN 549 CLASS UNIVERSE")
known = load_known_classes()
print(f"  source: {KNOWN_CLASSES_PATH}")
print(f"  count: {len(known)}")
print(f"  unique: {len(known)}")

# 4. Scan all ABO splits
print("\n4. EXACT PRODUCT_TYPE EXTRACTION")
split_results = {}
union_counts = Counter()
category_split_presence = defaultdict(set)

for split_name, path in SPLITS.items():
    print(f"\n  SOURCE: {path}")
    if not path.exists():
        raise RuntimeError(f"Required ABO split missing: {path}")

    result = scan_split(path)
    split_results[split_name] = result

    print(f"    rows scanned: {result['rows']}")
    print(f"    malformed JSON: {result['malformed_json_rows']}")
    print(f"    missing product_type: {result['missing_product_type']}")
    print(f"    empty product_type: {result['empty_product_type']}")
    print(f"    unique product_type: {result['unique_product_types']}")

    for cls, n in result["counts"].items():
        union_counts[cls] += n
        category_split_presence[cls].add(split_name)

full_universe = set(union_counts)

print("\n5. FULL ABO PRODUCT_TYPE UNIVERSE")
print(f"  total unique product_type values: {len(full_universe)}")
print(f"  expected project-record count: 576")
print(f"  576 COUNT REPRODUCED: {len(full_universe) == 576}")

# 6. Compare against frozen 549
outside_known = sorted(full_universe - known)
known_missing_from_full = sorted(known - full_universe)

print("\n6. SET COMPARISON")
print(f"  frozen known classes: {len(known)}")
print(f"  full ABO classes: {len(full_universe)}")
print(f"  full ABO minus frozen 549: {len(outside_known)}")
print(f"  frozen 549 absent from full ABO universe: {len(known_missing_from_full)}")

if outside_known:
    print("\n  CANDIDATE CLASSES OUTSIDE FROZEN 549:")
    for cls in outside_known:
        counts = {
            split: split_results[split]["counts"].get(cls, 0)
            for split in SPLITS
        }
        print(
            f"    {cls!r}: total={union_counts[cls]}, "
            f"train={counts['train']}, "
            f"validation={counts['validation']}, "
            f"test={counts['test']}"
        )

# 7. Per-class split membership
candidate_inventory = []
for cls in outside_known:
    candidate_inventory.append({
        "product_type": cls,
        "total_records": union_counts[cls],
        "split_counts": {
            split: split_results[split]["counts"].get(cls, 0)
            for split in SPLITS
        },
        "split_membership": sorted(category_split_presence[cls]),
        "outside_frozen_549": True,
    })

# 8. Candidate exposure flags
# Presence in train is direct evidence that the category exists in the C-Base
# training population. This audit does not infer model-selection exposure from
# mere presence; it records the observable split evidence only.
for item in candidate_inventory:
    item["present_in_cbase_training_split"] = item["split_counts"]["train"] > 0
    item["present_in_cbase_validation_split"] = item["split_counts"]["validation"] > 0
    item["present_in_cbase_test_split"] = item["split_counts"]["test"] > 0

# 9. Explicit feasibility gate
print("\n7. FEASIBILITY GATE")
if len(full_universe) != 576:
    status = "UNRESOLVED"
    reason = "The exact 576-category ABO universe was not reproduced from the split artifacts."
elif not outside_known:
    status = "NOT_CURRENTLY_FEASIBLE"
    reason = "No product_type values exist outside the frozen 549-class space."
elif any(x["present_in_cbase_training_split"] for x in candidate_inventory):
    status = "UNRESOLVED"
    reason = (
        "Candidate categories exist outside the frozen 549 names but some are present "
        "in the C-Base training split; they cannot be treated as admissible unknowns "
        "without further protocol-level evidence."
    )
else:
    status = "EVIDENCE_CANDIDATES_FOUND"
    reason = (
        "Candidate product_type values outside the frozen 549 class names were found "
        "and none occur in the C-Base training split. This is not yet a declaration "
        "of FEASIBLE; model-selection and D2-D5 exposure still require explicit evidence."
    )

print(f"  status: {status}")
print(f"  reason: {reason}")

# 10. Save audit artifacts
inventory = {
    "phase": "D6",
    "subphase": "D6.2.3",
    "analysis": "exact_abo_product_type_universe",
    "read_only": True,
    "cbase_sha256": observed_sha,
    "frozen_known_class_count": len(known),
    "frozen_known_classes": sorted(known),
    "split_results": split_results,
    "full_product_type_count": len(full_universe),
    "full_product_type_values": sorted(full_universe),
    "candidate_classes_outside_frozen_549": candidate_inventory,
    "known_classes_missing_from_full_universe": known_missing_from_full,
}

results = {
    "phase": "D6",
    "subphase": "D6.2.3",
    "status": status,
    "reason": reason,
    "cbase_sha256": observed_sha,
    "dtest_accessed": False,
    "model_inference_performed": False,
    "threshold_selected": False,
    "frozen_known_class_count": len(known),
    "full_abo_product_type_count": len(full_universe),
    "expected_project_record_category_count": 576,
    "category_count_reproduced": len(full_universe) == 576,
    "outside_frozen_549_count": len(outside_known),
    "outside_frozen_549_classes": outside_known,
    "candidate_inventory": candidate_inventory,
    "known_classes_missing_from_full_universe": known_missing_from_full,
}

with open(INVENTORY_OUT, "w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False)

with open(RESULTS_OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n8. ARTIFACTS")
print(f"  {INVENTORY_OUT}")
print(f"  {RESULTS_OUT}")

print("\nD6.2.3 AUDIT COMPLETE")
print("STOP CONDITION:")
print("  No open-set inference performed.")
print("  No threshold selected.")
print("  No D-Test accessed.")
print("  No candidate declared FEASIBLE automatically.")
