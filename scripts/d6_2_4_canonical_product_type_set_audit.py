from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# D6.2.4 v002
# CANONICAL ABO PRODUCT_TYPE SET AUDIT
#
# READ-ONLY
# ============================================================

ROOT = Path.cwd()

EXPECTED_CBASE_SHA = (
    "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"
)

CBASE_CHECKPOINT = (
    ROOT / "data" / "models" / "abo" / "phase_c" /
    "c3" / "training" / "cbase_best.pt"
)

CBASE_CLASS_MANIFEST = (
    ROOT / "data" / "models" / "abo" / "phase_c" /
    "c3" / "embeddings" / "c3_embedding_manifest.json"
)

SPLITS = {
    "train": ROOT / "data" / "splits" / "abo" / "train.jsonl",
    "validation": ROOT / "data" / "splits" / "abo" / "validation.jsonl",
    "test": ROOT / "data" / "splits" / "abo" / "test.jsonl",
}

OUT_DIR = (
    ROOT / "data" / "models" / "abo" /
    "phase_d" / "d6" / "feasibility"
)

RESULTS_PATH = OUT_DIR / "d6_2_4_canonical_set_results.json"
INVENTORY_PATH = OUT_DIR / "d6_2_4_candidate_inventory.json"
MANIFEST_PATH = OUT_DIR / "d6_2_4_canonical_set_manifest.json"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message: str):
    print(f"FAIL: {message}")
    sys.exit(1)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonicalize_product_type(value):
    """
    Deterministic normalization of the OBSERVED ABO structure.

    Expected observed structure:
        [
            {"value": "ABIS_BEAUTY"}
        ]

    Also permits the equivalent one-element tuple/list nesting
    only when the resulting structure is unambiguous.

    No semantic remapping is performed.
    """

    if not isinstance(value, list):
        return None, "not_list"

    if len(value) != 1:
        return None, "list_length_not_one"

    item = value[0]

    if not isinstance(item, dict):
        return None, "list_item_not_dict"

    if set(item.keys()) != {"value"}:
        return None, "unexpected_dict_keys"

    inner = item["value"]

    if not isinstance(inner, str):
        return None, "value_not_string"

    canonical = inner.strip()

    if not canonical:
        return None, "empty_value"

    return canonical, "list_dict_value"


def load_cbase_classes():
    if not CBASE_CLASS_MANIFEST.exists():
        fail(f"Missing class manifest: {CBASE_CLASS_MANIFEST}")

    manifest = load_json(CBASE_CLASS_MANIFEST)

    try:
        classes = manifest["target"]["classes"]
    except Exception:
        fail("Could not find target.classes in C-Base embedding manifest.")

    if not isinstance(classes, list):
        fail("target.classes is not a list.")

    classes = [str(x).strip() for x in classes]

    if len(classes) != 549:
        fail(f"Expected 549 C-Base classes, found {len(classes)}.")

    if len(set(classes)) != 549:
        fail("C-Base class names are not unique.")

    return classes


def inspect_examples(path: Path, n=5):
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except Exception:
                continue

            if isinstance(record, dict) and "product_type" in record:
                value = record["product_type"]

                examples.append({
                    "line": line_number,
                    "python_type": type(value).__name__,
                    "repr": repr(value),
                })

                if len(examples) >= n:
                    break

    return examples


def scan_split(path: Path, split_name: str):
    if not path.exists():
        fail(f"Missing split: {path}")

    rows = 0
    malformed = 0
    missing = 0
    empty = 0

    raw_counter = Counter()
    canonical_counter = Counter()
    methods = Counter()
    failures = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            rows += 1

            try:
                record = json.loads(line)
            except Exception:
                malformed += 1
                continue

            if not isinstance(record, dict):
                malformed += 1
                continue

            if "product_type" not in record:
                missing += 1
                continue

            value = record["product_type"]

            if value is None:
                empty += 1
                continue

            raw_counter[repr(value)] += 1

            canonical, method = canonicalize_product_type(value)

            if canonical is None:
                failures[method] += 1
            else:
                canonical_counter[canonical] += 1
                methods[method] += 1

    return {
        "split": split_name,
        "path": str(path.relative_to(ROOT)),
        "rows": rows,
        "malformed": malformed,
        "missing": missing,
        "empty": empty,
        "raw_unique": len(raw_counter),
        "canonical_unique": len(canonical_counter),
        "canonical_counts": canonical_counter,
        "normalization_methods": dict(methods),
        "normalization_failures": dict(failures),
        "examples": inspect_examples(path),
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 72)
print("D6.2.4 v002 CANONICAL ABO PRODUCT_TYPE SET AUDIT")
print("READ-ONLY AUDIT")
print("  - no model inference")
print("  - no D-Test evaluation")
print("  - no threshold selection")
print("  - no upstream artifact modification")
print("  - no synthetic/external data")
print("  - structure inspection + deterministic normalization only")
print("=" * 72)


# ------------------------------------------------------------
# 1. Frozen C-Base
# ------------------------------------------------------------

print("\n1. FROZEN C-BASE CHECKPOINT HASH")

if not CBASE_CHECKPOINT.exists():
    fail(f"Missing checkpoint: {CBASE_CHECKPOINT}")

observed_sha = sha256_file(CBASE_CHECKPOINT)

print(f"  observed: {observed_sha}")
print(f"  expected: {EXPECTED_CBASE_SHA}")

if observed_sha != EXPECTED_CBASE_SHA:
    fail("C-Base SHA mismatch.")

print("  HASH: PASS")


# ------------------------------------------------------------
# 2. D-Test boundary
# ------------------------------------------------------------

print("\n2. D-TEST BOUNDARY")

dtest_dir = ROOT / "data" / "models" / "abo" / "phase_d" / "dtest"

print(f"  D-Test namespace present: {dtest_dir.exists()}")

if dtest_dir.exists():
    fail("D-Test namespace exists.")


# ------------------------------------------------------------
# 3. Frozen 549 universe
# ------------------------------------------------------------

print("\n3. FROZEN 549 CLASS UNIVERSE")

frozen_classes = load_cbase_classes()
frozen_set = set(frozen_classes)

print(f"  source: {CBASE_CLASS_MANIFEST.relative_to(ROOT)}")
print(f"  count: {len(frozen_classes)}")
print(f"  unique: {len(frozen_set)}")

print("  CLASS UNIVERSE: PASS")


# ------------------------------------------------------------
# 4. Structure inspection
# ------------------------------------------------------------

print("\n4. ACTUAL PRODUCT_TYPE STRUCTURE INSPECTION")

for split_name, path in SPLITS.items():
    examples = inspect_examples(path, n=3)

    print(f"\n  {split_name}: {path.relative_to(ROOT)}")

    for ex in examples:
        print(f"    line: {ex['line']}")
        print(f"    python type: {ex['python_type']}")
        print(f"    repr: {ex['repr']}")

print("\n  STRUCTURE INSPECTION COMPLETE")


# ------------------------------------------------------------
# 5. Extraction + canonicalization
# ------------------------------------------------------------

print("\n5. ABO PRODUCT_TYPE EXTRACTION + CANONICALIZATION")

split_results = {}

combined_counts = Counter()
all_failures = Counter()

for split_name, path in SPLITS.items():

    result = scan_split(path, split_name)
    split_results[split_name] = result

    combined_counts.update(result["canonical_counts"])
    all_failures.update(result["normalization_failures"])

    print(f"\n  SOURCE: {result['path']}")
    print(f"    rows scanned: {result['rows']}")
    print(f"    malformed JSON: {result['malformed']}")
    print(f"    missing product_type: {result['missing']}")
    print(f"    empty product_type: {result['empty']}")
    print(f"    raw unique product_type: {result['raw_unique']}")
    print(f"    canonical unique product_type: {result['canonical_unique']}")
    print(f"    normalization failures: "
          f"{sum(result['normalization_failures'].values())}")

    if result["normalization_failures"]:
        print("    failure types:")
        for k, v in result["normalization_failures"].items():
            print(f"      {k}: {v}")


# ------------------------------------------------------------
# 6. Normalization gate
# ------------------------------------------------------------

print("\n6. NORMALIZATION GATE")

failure_total = sum(all_failures.values())

print(f"  total normalization failures: {failure_total}")

if failure_total != 0:
    fail(
        "Canonicalization is not complete. "
        "No set comparison will be performed."
    )

print("  NORMALIZATION: PASS")


# ------------------------------------------------------------
# 7. Full ABO universe
# ------------------------------------------------------------

print("\n7. FULL ABO PRODUCT_TYPE UNIVERSE")

full_abo_set = set(combined_counts.keys())

print(f"  canonical unique categories: {len(full_abo_set)}")
print("  expected categories: 576")

if len(full_abo_set) != 576:
    fail(
        f"Expected 576 canonical categories but found "
        f"{len(full_abo_set)}."
    )

print("  576 COUNT: PASS")


# ------------------------------------------------------------
# 8. Split union
# ------------------------------------------------------------

print("\n8. SPLIT UNION INTEGRITY")

train_set = set(split_results["train"]["canonical_counts"])
val_set = set(split_results["validation"]["canonical_counts"])
test_set = set(split_results["test"]["canonical_counts"])

union_set = train_set | val_set | test_set

print(f"  train unique: {len(train_set)}")
print(f"  validation unique: {len(val_set)}")
print(f"  test unique: {len(test_set)}")
print(f"  union unique: {len(union_set)}")

if union_set != full_abo_set:
    missing = sorted(full_abo_set - union_set)
    extra = sorted(union_set - full_abo_set)

    print(f"  missing from union: {missing[:20]}")
    print(f"  unexpected in union: {extra[:20]}")

    fail("Split union does not equal full ABO universe.")

print("  SPLIT UNION: PASS")


# ------------------------------------------------------------
# 9. Exact set difference
# ------------------------------------------------------------

print("\n9. EXACT 549-vs-576 SET COMPARISON")

intersection = frozen_set & full_abo_set
abo_only = full_abo_set - frozen_set
cbase_only = frozen_set - full_abo_set

print(f"  frozen known classes: {len(frozen_set)}")
print(f"  full ABO classes: {len(full_abo_set)}")
print(f"  intersection: {len(intersection)}")
print(f"  ABO-only: {len(abo_only)}")
print(f"  C-Base-only: {len(cbase_only)}")


# ------------------------------------------------------------
# 10. Set identities
# ------------------------------------------------------------

print("\n10. SET IDENTITY CHECKS")

lhs_abo = len(intersection) + len(abo_only)
lhs_cbase = len(intersection) + len(cbase_only)

print(f"  intersection + ABO-only = {lhs_abo}")
print(f"  expected ABO universe = 576")

print(f"  intersection + C-Base-only = {lhs_cbase}")
print(f"  expected C-Base universe = 549")

if lhs_abo != 576:
    fail("ABO set identity failed.")

if lhs_cbase != 549:
    fail("C-Base set identity failed.")

print("  SET IDENTITIES: PASS")


# ------------------------------------------------------------
# 11. Candidate inventory
# ------------------------------------------------------------

print("\n11. TRUE ABO-ONLY CANDIDATE INVENTORY")

candidate_inventory = []

for category in sorted(abo_only):

    train_count = split_results["train"]["canonical_counts"].get(
        category, 0
    )
    val_count = split_results["validation"]["canonical_counts"].get(
        category, 0
    )
    test_count = split_results["test"]["canonical_counts"].get(
        category, 0
    )

    row = {
        "canonical_product_type": category,
        "total_count": train_count + val_count + test_count,
        "train_count": train_count,
        "validation_count": val_count,
        "test_count": test_count,
        "present_in_train": train_count > 0,
        "present_in_validation": val_count > 0,
        "present_in_test": test_count > 0,
    }

    candidate_inventory.append(row)

    print(
        f"  {category}: "
        f"total={row['total_count']}, "
        f"train={train_count}, "
        f"validation={val_count}, "
        f"test={test_count}"
    )


# ------------------------------------------------------------
# 12. Exposure summary
# ------------------------------------------------------------

print("\n12. CANDIDATE EXPOSURE SUMMARY")

train_exposed = [
    x for x in candidate_inventory
    if x["present_in_train"]
]

validation_exposed = [
    x for x in candidate_inventory
    if x["present_in_validation"]
]

test_exposed = [
    x for x in candidate_inventory
    if x["present_in_test"]
]

print(f"  ABO-only candidate classes: {len(candidate_inventory)}")
print(f"  present in train: {len(train_exposed)}")
print(f"  present in validation: {len(validation_exposed)}")
print(f"  present in test: {len(test_exposed)}")


# ------------------------------------------------------------
# 13. Feasibility boundary
# ------------------------------------------------------------

print("\n13. FEASIBILITY BOUNDARY")

print(
    "  ABO-only membership is NOT sufficient for open-set feasibility."
)

print(
    "  Remaining admissibility checks require:"
)

print("    - known-space separation")
print("    - provenance")
print("    - label validity")
print("    - training exclusion")
print("    - model-selection exclusion")
print("    - D2-D5 exclusion")
print("    - test isolation")
print("    - same-task relevance")
print("    - no artificial construction")

print("\n  D6 FEASIBILITY STATUS: UNRESOLVED")


# ------------------------------------------------------------
# 14. Save artifacts
# ------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "audit": "D6.2.4",
    "version": "v002",
    "status": "EXECUTED",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),

    "read_only": True,
    "model_inference_performed": False,
    "d_test_evaluation_performed": False,
    "threshold_selected": False,
    "upstream_artifacts_modified": False,
    "synthetic_data_created": False,
    "external_data_used": False,

    "cbase": {
        "checkpoint": str(CBASE_CHECKPOINT.relative_to(ROOT)),
        "sha256_observed": observed_sha,
        "sha256_expected": EXPECTED_CBASE_SHA,
        "hash_pass": observed_sha == EXPECTED_CBASE_SHA,
        "class_count": len(frozen_set),
        "class_source": str(
            CBASE_CLASS_MANIFEST.relative_to(ROOT)
        ),
    },

    "abo_universe": {
        "canonical_count": len(full_abo_set),
        "expected_count": 576,
        "count_pass": len(full_abo_set) == 576,
    },

    "set_comparison": {
        "frozen_known_count": len(frozen_set),
        "abo_full_count": len(full_abo_set),
        "intersection_count": len(intersection),
        "abo_only_count": len(abo_only),
        "cbase_only_count": len(cbase_only),
        "intersection": sorted(intersection),
        "abo_only": sorted(abo_only),
        "cbase_only": sorted(cbase_only),
        "identity_abo_pass": lhs_abo == 576,
        "identity_cbase_pass": lhs_cbase == 549,
    },

    "candidate_exposure": {
        "candidate_count": len(candidate_inventory),
        "train_exposed_count": len(train_exposed),
        "validation_exposed_count": len(validation_exposed),
        "test_exposed_count": len(test_exposed),
    },

    "feasibility": {
        "status": "UNRESOLVED",
        "reason": (
            "Exact canonical category difference has been established, "
            "but open-set admissibility requires a separate provenance "
            "and exposure audit."
        ),
    },
}

with RESULTS_PATH.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


inventory = {
    "audit": "D6.2.4",
    "version": "v002",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "candidates": candidate_inventory,
}

with INVENTORY_PATH.open("w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False)


manifest = {
    "manifest_version": "d6.2.4-v002",
    "audit": "D6.2.4",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),

    "inputs": {
        "cbase_checkpoint": str(
            CBASE_CHECKPOINT.relative_to(ROOT)
        ),
        "cbase_class_manifest": str(
            CBASE_CLASS_MANIFEST.relative_to(ROOT)
        ),
        "splits": {
            k: str(v.relative_to(ROOT))
            for k, v in SPLITS.items()
        },
    },

    "frozen_constraints": {
        "cbase_sha256": EXPECTED_CBASE_SHA,
        "known_classes": 549,
        "expected_abo_classes": 576,
        "model_inference": False,
        "d_test_evaluation": False,
        "threshold_selection": False,
        "synthetic_data": False,
        "external_data": False,
    },

    "results": {
        "abo_count": len(full_abo_set),
        "intersection": len(intersection),
        "abo_only": len(abo_only),
        "cbase_only": len(cbase_only),
    },

    "feasibility_status": "UNRESOLVED",
}

with MANIFEST_PATH.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n14. ARTIFACTS")

print(f"  {RESULTS_PATH.relative_to(ROOT)}")
print(f"  {INVENTORY_PATH.relative_to(ROOT)}")
print(f"  {MANIFEST_PATH.relative_to(ROOT)}")

print("\n" + "=" * 72)
print("D6.2.4 v002 AUDIT COMPLETE")
print("=" * 72)

print(f"  Frozen C-Base classes: {len(frozen_set)}")
print(f"  Full ABO classes: {len(full_abo_set)}")
print(f"  Exact intersection: {len(intersection)}")
print(f"  True ABO-only candidates: {len(abo_only)}")
print(f"  C-Base-only classes: {len(cbase_only)}")

print("\nSTOP CONDITIONS:")
print("  No model inference.")
print("  No D-Test evaluation.")
print("  No threshold selection.")
print("  No upstream modification.")
print("  No synthetic/external data.")
print("  Open-set feasibility remains UNRESOLVED.")