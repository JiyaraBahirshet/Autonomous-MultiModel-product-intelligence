from pathlib import Path
import json
import csv
import hashlib
import numpy as np

ROOT = Path.cwd()

ABO_ROOT = ROOT / "data" / "models" / "abo"
PHASE_C = ABO_ROOT / "phase_c"
D6_ROOT = ABO_ROOT / "phase_d" / "d6" / "feasibility"

CBASE = PHASE_C / "c3" / "training" / "cbase_best.pt"
C4_RESULTS = PHASE_C / "c4" / "evaluation" / "cbase_test_results.json"
C4_MANIFEST = PHASE_C / "c4" / "evaluation" / "c4_2_evaluation_manifest.json"
C4_PRED = PHASE_C / "c4" / "evaluation" / "cbase_test_predictions.npz"

EXPECTED_CBASE_SHA = (
    "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"
)

D6_ROOT.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def walk_json_files():
    roots = [
        ABO_ROOT / "b3.2",
        ABO_ROOT / "b4.2",
        PHASE_C,
    ]

    files = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.json"))

    return sorted(set(files))


def recursive_find(obj, wanted_keys, path=""):
    found = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)

            if k in wanted_keys:
                found.append((p, v))

            found.extend(recursive_find(v, wanted_keys, p))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(recursive_find(v, wanted_keys, f"{path}[{i}]"))

    return found


print_header("D6.2 CANDIDATE POPULATION RESOLUTION")

print("READ-ONLY AUDIT")
print("  - no model inference")
print("  - no D-Test access")
print("  - no threshold selection")
print("  - no artifact modification")
print("  - no synthetic data")
print()

# ----------------------------------------------------------------------
# 1. Frozen C-Base check
# ----------------------------------------------------------------------

print_header("1. FROZEN C-BASE CHECK")

if not CBASE.exists():
    raise RuntimeError(f"Missing C-Base checkpoint: {CBASE}")

observed_sha = sha256(CBASE)

print(f"C-Base: {CBASE}")
print(f"SHA-256: {observed_sha}")
print("EXPECTED repr:", repr(EXPECTED_CBASE_SHA))
print("OBSERVED repr:", repr(observed_sha))
print("EXPECTED len:", len(EXPECTED_CBASE_SHA))
print("OBSERVED len:", len(observed_sha))
print("DIRECT EQUALITY:", observed_sha == EXPECTED_CBASE_SHA)
print(
    "HASH:",
    "PASS" if observed_sha == EXPECTED_CBASE_SHA else "FAIL"
)

if observed_sha != EXPECTED_CBASE_SHA:
    raise RuntimeError("C-Base hash mismatch. STOP.")

# ----------------------------------------------------------------------
# 2. D-Test boundary
# ----------------------------------------------------------------------

print_header("2. D-TEST BOUNDARY")

DTEST = ROOT / "data" / "models" / "abo" / "phase_d" / "dtest"

print(
    "D-Test namespace:",
    "ABSENT" if not DTEST.exists() else "PRESENT"
)

if DTEST.exists():
    raise RuntimeError(
        "D-Test namespace exists. This resolver will not inspect it."
    )

# ----------------------------------------------------------------------
# 3. Inspect authoritative JSON artifacts for class information
# ----------------------------------------------------------------------

print_header("3. CLASS-UNIVERSE EVIDENCE")

wanted_keys = {
    "classes",
    "class_names",
    "class_ids",
    "class_to_idx",
    "idx_to_class",
    "label_classes",
    "target_classes",
    "num_classes",
    "n_classes",
    "product_type",
    "categories",
    "category",
    "unique_categories",
}

evidence = []

for path in walk_json_files():
    try:
        obj = load_json(path)
    except Exception:
        continue

    matches = recursive_find(obj, wanted_keys)

    if matches:
        evidence.append(
            {
                "path": str(path.relative_to(ROOT)),
                "matches": [
                    {
                        "key": key,
                        "type": type(value).__name__,
                        "preview": (
                            value
                            if isinstance(value, (str, int, float, bool))
                            else (
                                list(value.keys())[:20]
                                if isinstance(value, dict)
                                else (
                                    value[:20]
                                    if isinstance(value, list)
                                    else str(value)[:500]
                                )
                            )
                        ),
                    }
                    for key, value in matches
                ],
            }
        )

print(f"JSON artifacts containing class evidence: {len(evidence)}")

for item in evidence:
    print()
    print("SOURCE:", item["path"])
    for match in item["matches"]:
        print(" ", match["key"], "=>", match["preview"])

evidence_path = D6_ROOT / "d6_class_universe_evidence.json"
evidence_path.write_text(
    json.dumps(evidence, indent=2, default=str) + "\n",
    encoding="utf-8",
)

# ----------------------------------------------------------------------
# 4. Inspect NPZ prediction artifacts WITHOUT running inference
# ----------------------------------------------------------------------

print_header("4. EXISTING NPZ ARTIFACTS")

npz_candidates = sorted(
    p for p in PHASE_C.rglob("*.npz")
    if "phase_d" not in str(p).lower()
)

print(f"NPZ artifacts discovered: {len(npz_candidates)}")

for p in npz_candidates:
    try:
        with np.load(p, allow_pickle=False) as data:
            print()
            print("SOURCE:", p.relative_to(ROOT))
            print(" KEYS:", list(data.files))

            for key in data.files:
                arr = data[key]
                print(
                    f"  {key}: shape={getattr(arr, 'shape', None)}, "
                    f"dtype={getattr(arr, 'dtype', None)}"
                )
    except Exception as e:
        print("  READ ERROR:", repr(e))

# ----------------------------------------------------------------------
# 5. C4 test artifact inspection
# ----------------------------------------------------------------------

print_header("5. C4 TEST ARTIFACT METADATA")

for path in [C4_RESULTS, C4_MANIFEST]:
    print()
    print("SOURCE:", path.relative_to(ROOT))

    if not path.exists():
        print("  MISSING")
        continue

    obj = load_json(path)

    # Print only potentially relevant fields.
    matches = recursive_find(
        obj,
        {
            "classes",
            "class_names",
            "class_ids",
            "num_classes",
            "n_classes",
            "test_count",
            "population_count",
            "split",
            "split_name",
            "label",
            "target",
            "prediction",
        },
    )

    for key, value in matches:
        if isinstance(value, dict):
            preview = list(value.keys())[:20]
        elif isinstance(value, list):
            preview = value[:20]
        else:
            preview = value

        print(" ", key, "=>", preview)

# ----------------------------------------------------------------------
# 6. Inspect C4 prediction labels if available
# ----------------------------------------------------------------------

print_header("6. C4 PREDICTION LABEL UNIVERSE")

if C4_PRED.exists():

    with np.load(C4_PRED, allow_pickle=False) as data:

        print("C4 prediction keys:", list(data.files))

        for key in data.files:
            arr = data[key]

            if arr.ndim == 1:
                try:
                    unique = np.unique(arr)
                    print(
                        f"{key}: shape={arr.shape}, "
                        f"unique_count={len(unique)}, "
                        f"min={unique.min()}, "
                        f"max={unique.max()}"
                    )
                except Exception:
                    pass

            elif arr.ndim == 2:
                print(
                    f"{key}: shape={arr.shape}; "
                    f"2-D artifact not interpreted as class universe."
                )

else:
    print("C4 prediction artifact missing.")

# ----------------------------------------------------------------------
# 7. Search for explicit 576/549 category evidence
# ----------------------------------------------------------------------

print_header("7. 576 vs 549 EVIDENCE SEARCH")

matches_576 = []
matches_549 = []

for path in walk_json_files():

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    if "576" in text:
        matches_576.append(str(path.relative_to(ROOT)))

    if "549" in text:
        matches_549.append(str(path.relative_to(ROOT)))

print("Artifacts containing literal 576:", len(matches_576))

for p in matches_576:
    print(" ", p)

print()
print("Artifacts containing literal 549:", len(matches_549))

for p in matches_549:
    print(" ", p)

# ----------------------------------------------------------------------
# 8. Write resolution manifest
# ----------------------------------------------------------------------

print_header("8. D6.2 RESOLUTION STATUS")

result = {
    "phase": "D6",
    "substage": "D6.2",
    "version": "v001",
    "status": "EVIDENCE_COLLECTED",
    "cbase_sha256": observed_sha,
    "cbase_hash_matches": observed_sha == EXPECTED_CBASE_SHA,
    "dtest_accessed": False,
    "model_inference": False,
    "threshold_selected": False,
    "synthetic_data": False,
    "candidate_feasibility": "NOT_YET_ESTABLISHED",
    "class_evidence_artifact": str(
        evidence_path.relative_to(ROOT)
    ),
}

result_path = (
    D6_ROOT / "d6_candidate_resolution_results.json"
)

result_path.write_text(
    json.dumps(result, indent=2) + "\n",
    encoding="utf-8",
)

print("Status: EVIDENCE_COLLECTED")
print("Candidate feasibility: NOT_YET_ESTABLISHED")
print()
print("Generated:")
print(
    f"  {evidence_path.relative_to(ROOT)}"
)
print(
    f"  {result_path.relative_to(ROOT)}"
)
print()
print("STOP CONDITION:")
print(
    "No candidate has been declared FEASIBLE. "
    "No open-set inference has been performed."
)