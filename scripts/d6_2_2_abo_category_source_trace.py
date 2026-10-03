#!/usr/bin/env python3
"""
D6.2.2 — ABO authoritative category-source trace
READ-ONLY AUDIT

Purpose:
  Trace the local ABO artifacts that can reproduce the documented
  "576 unique product categories" count, without inference.

Safety:
  - no model inference
  - no D-Test evaluation/access
  - no threshold selection
  - no artifact modification
  - no synthetic/external data

The audit prioritizes non-test ABO B2/split artifacts. It reports schemas,
category-like fields, row counts, and unique-category counts. It does not
declare any category an unknown class.
"""

from pathlib import Path
import json
import hashlib
from collections import Counter

ROOT = Path.cwd()
EXPECTED_CBASE_SHA = "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"

CBASE = ROOT / "data/models/abo/phase_c/c3/training/cbase_best.pt"
EMBED_MANIFEST = ROOT / "data/models/abo/phase_c/c3/embeddings/c3_embedding_manifest.json"
TRAIN_MANIFEST = ROOT / "data/models/abo/phase_c/c3/training/cbase_training_manifest.json"

TARGETS = [
    ROOT / "data/splits/abo/train.jsonl",
    ROOT / "data/splits/abo/validation.jsonl",
    ROOT / "data/processed/abo/b2/train.jsonl",
    ROOT / "data/processed/abo/b2/validation.jsonl",
    ROOT / "data/representations/abo/b3/train.jsonl",
    ROOT / "data/representations/abo/b3/validation.jsonl",
    ROOT / "reports/validation/abo_raw_validation.json",
    ROOT / "reports/preprocessing/abo_b2_preprocessing.json",
    ROOT / "data/processed/abo/b2/statistics.json",
]

CATEGORY_KEYS = {
    "category", "product_type", "product_category", "category_name",
    "category_path", "category_id", "category_label", "productType",
    "product_type_name", "product_category_name"
}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def inspect_json(path):
    try:
        x = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  JSON READ ERROR: {e}")
        return
    print(f"  JSON type: {type(x).__name__}")
    if isinstance(x, dict):
        print(f"  top-level keys: {list(x.keys())}")
        hits = []
        for k, v in x.items():
            kl = str(k).lower()
            if any(t in kl for t in ("categor", "product_type", "producttype")):
                hits.append((k, type(v).__name__, v if not isinstance(v, (dict,list)) else f"<{len(v)} items>"))
        print(f"  category-like top-level fields: {hits}")
    elif isinstance(x, list):
        print(f"  list length: {len(x)}")
        if x and isinstance(x[0], dict):
            print(f"  first-record keys: {list(x[0].keys())}")

def scan_jsonl(path, max_records=None):
    rows = 0
    malformed = 0
    category_values = {}
    first = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_records is not None and rows >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                malformed += 1
                continue
            rows += 1
            if first is None:
                first = r
            if isinstance(r, dict):
                for k, v in r.items():
                    kl = str(k).lower()
                    if (
                        k in CATEGORY_KEYS
                        or "categor" in kl
                        or "product_type" in kl
                        or "producttype" in kl
                    ):
                        if isinstance(v, (str, int, float)) and v not in (None, ""):
                            category_values.setdefault(k, Counter())[str(v)] += 1
    print(f"  rows scanned: {rows}")
    print(f"  malformed JSON rows: {malformed}")
    if isinstance(first, dict):
        print(f"  first-record keys: {list(first.keys())}")
    for k, c in sorted(category_values.items()):
        print(f"  FIELD {k!r}: unique={len(c)}, top5={c.most_common(5)}")
        if len(c) == 576:
            print(f"  *** EXACT 576 CATEGORY FIELD FOUND: {k} ***")
    return category_values

print("D6.2.2 ABO AUTHORITATIVE CATEGORY-SOURCE TRACE")
print("READ-ONLY AUDIT")
print("  - no model inference")
print("  - no D-Test evaluation")
print("  - no threshold selection")
print("  - no artifact modification")
print("  - no synthetic/external data")
print()

print("1. FROZEN C-BASE CHECK")
if not CBASE.exists():
    raise SystemExit(f"C-Base missing: {CBASE}")
observed = sha256(CBASE)
print(f"  observed: {observed}")
print(f"  expected: {EXPECTED_CBASE_SHA}")
print(f"  HASH: {'PASS' if observed == EXPECTED_CBASE_SHA else 'FAIL'}")
if observed != EXPECTED_CBASE_SHA:
    raise SystemExit("STOP: C-Base hash mismatch.")

print("\n2. FROZEN CLASS COUNT/CANONICAL NAME CHECK")
m = json.loads(EMBED_MANIFEST.read_text(encoding="utf-8"))
classes = m.get("target", {}).get("classes")
if classes is None:
    classes = m.get("target_classes")
print(f"  class count: {len(classes) if isinstance(classes,list) else 'NOT FOUND'}")
print(f"  unique count: {len(set(classes)) if isinstance(classes,list) else 'NOT FOUND'}")

tm = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
print(f"  training integrity.target_classes: {tm.get('integrity',{}).get('target_classes')}")
print(f"  training model.num_classes: {tm.get('model',{}).get('num_classes')}")
print(f"  training_config.num_classes: {tm.get('training_config',{}).get('num_classes')}")

print("\n3. TARGETED ABO SOURCE TRACE")
found = 0
for p in TARGETS:
    print(f"\nSOURCE: {p}")
    if not p.exists():
        print("  STATUS: ABSENT")
        continue
    found += 1
    print(f"  size_bytes: {p.stat().st_size}")
    if p.suffix == ".jsonl":
        # Scan complete train/validation sources; these are non-Phase-C-test
        # populations and are needed to reproduce the category universe.
        scan_jsonl(p)
    else:
        inspect_json(p)

print("\n4. RECURSIVE ABO B2/B3 SOURCE CANDIDATES")
roots = [
    ROOT / "data/processed/abo/b2",
    ROOT / "data/splits/abo",
    ROOT / "data/representations/abo/b3",
]
for root in roots:
    print(f"\nROOT: {root}")
    if not root.exists():
        print("  ABSENT")
        continue
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".txt", ".csv"}:
            print(f"  {p.relative_to(ROOT)}")

print("\n5. INTERPRETATION GATE")
print("  This audit does NOT infer that 576 - 549 = 27 unknown classes.")
print("  It only identifies whether an existing local ABO artifact can reproduce")
print("  the documented 576-category universe.")
print("  Any candidate population still requires separate provenance/exposure checks.")

print("\nD6.2.2 TRACE COMPLETE")
print(f"Targeted sources present: {found}/{len(TARGETS)}")
