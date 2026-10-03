from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
D6_ROOT = ROOT / "data" / "models" / "abo" / "phase_d" / "d6" / "feasibility"
CFG_PATH = ROOT / "configs" / "phase_d" / "d6_v001_open_set_feasibility_config.json"

CBASE = ROOT / "data" / "models" / "abo" / "phase_c" / "c3" / "training" / "cbase_best.pt"
EXPECTED_CBASE_SHA = "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"
DTEST = ROOT / "data" / "models" / "abo" / "phase_d" / "dtest"

D2_ROOT = ROOT / "data" / "models" / "abo" / "phase_d" / "d2"
D3_ROOT = ROOT / "data" / "models" / "abo" / "phase_d" / "d3"
D4_ROOT = ROOT / "data" / "models" / "abo" / "phase_d" / "d4"
D5_ROOT = ROOT / "data" / "models" / "abo" / "phase_d" / "d5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def inventory(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for p in root.rglob("*"):
        if p.is_file():
            rows.append({
                "path": str(p.relative_to(ROOT)),
                "size_bytes": p.stat().st_size,
                "suffix": p.suffix.lower(),
            })
    return rows


def approved_roots() -> list[Path]:
    return [
        ROOT / "data" / "models" / "abo",
        ROOT / "reports" / "phase_c",
        ROOT / "configs",
    ]


def discover_structured_evidence() -> list[dict[str, Any]]:
    terms = {
        "549", "576", "7346", "69823", "69867", "698284",
        "train", "training", "validation", "val", "test", "split",
        "class", "classes", "target", "product_type", "category",
        "cbase", "phase_c", "record", "listing", "image",
    }
    hits = []
    seen = set()
    for root in approved_roots():
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            if "phase_d" in str(p).lower():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                obj = load_json(p)
            except Exception:
                continue
            found = []
            for path, value in flatten(obj):
                text = f"{path}={value}".lower()
                if any(t in text for t in terms):
                    found.append({"field": path, "value": value})
            if found:
                hits.append({
                    "path": str(p.relative_to(ROOT)),
                    "evidence": found[:200],
                })
    return hits


def check_frozen_artifacts() -> dict[str, Any]:
    out: dict[str, Any] = {
        "cbase": {},
        "dtest": {},
        "d2_d5_presence": {},
    }
    if CBASE.exists():
        observed = sha256(CBASE)
        out["cbase"] = {
            "path": str(CBASE.relative_to(ROOT)),
            "exists": True,
            "sha256": observed,
            "expected_sha256": EXPECTED_CBASE_SHA,
            "hash_match": observed == EXPECTED_CBASE_SHA,
        }
    else:
        out["cbase"] = {"exists": False, "hash_match": False}

    out["dtest"] = {
        "path": str(DTEST.relative_to(ROOT)),
        "exists": DTEST.exists(),
        "accessed_by_audit": False,
    }
    for name, root in [("D2", D2_ROOT), ("D3", D3_ROOT), ("D4", D4_ROOT), ("D5", D5_ROOT)]:
        out["d2_d5_presence"][name] = root.exists()
    return out


def main() -> None:
    print("=" * 72)
    print("D6 v001 OPEN-SET FEASIBILITY AUDIT")
    print("=" * 72)
    print()
    print("Audit only:")
    print("  - no model inference")
    print("  - no open-set threshold selection")
    print("  - no D-Test access")
    print("  - no synthetic/external unknown data")
    print("  - no upstream modification")
    print()

    D6_ROOT.mkdir(parents=True, exist_ok=True)
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "phase": "D6",
        "version": "v001",
        "purpose": "open_set_feasibility_gate",
        "seed": 20260827,
        "known_class_count": 549,
        "frozen_phase_c_test_count": 7346,
        "cbase_checkpoint": str(CBASE.relative_to(ROOT)),
        "cbase_expected_sha256": EXPECTED_CBASE_SHA,
        "candidate_policy": "existing_project_artifacts_only",
        "dtest_accessed": False,
        "model_inference": False,
        "threshold_selected": False,
        "synthetic_unknowns": False,
        "external_scraping": False,
        "cross_dataset_unknown_mapping_assumed": False,
        "result_adaptive": False,
    }
    CFG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print("1. FROZEN BOUNDARY CHECK")
    frozen = check_frozen_artifacts()
    cbase = frozen["cbase"]
    print("  C-Base:", "PASS" if cbase.get("hash_match") else "FAIL")
    if cbase.get("exists"):
        print(f"    SHA-256 observed={cbase['sha256']}")
    print("  D-Test namespace:", "ABSENT" if not DTEST.exists() else "PRESENT")
    for name, present in frozen["d2_d5_presence"].items():
        print(f"  {name} namespace present={present}")

    print()
    print("2. APPROVED-SCOPE FILE INVENTORY")
    file_inventory = []
    for root in approved_roots():
        file_inventory.extend(inventory(root))
    print(f"  Files inventoried: {len(file_inventory)}")

    inventory_obj = {
        "phase": "D6",
        "version": "v001",
        "inventory_scope": [str(r.relative_to(ROOT)) for r in approved_roots()],
        "file_inventory": file_inventory,
        "candidate_population_status": "NOT_ESTABLISHED",
        "policy": "No candidate is admissible merely because an artifact or class count exists.",
    }
    inventory_path = D6_ROOT / "d6_candidate_population_inventory.json"
    inventory_path.write_text(json.dumps(inventory_obj, indent=2) + "\n", encoding="utf-8")

    print()
    print("3. STRUCTURED EVIDENCE DISCOVERY")
    evidence = discover_structured_evidence()
    print(f"  JSON artifacts containing potentially relevant evidence: {len(evidence)}")
    evidence_path = D6_ROOT / "d6_json_evidence_discovery.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    for item in evidence[:30]:
        print(f"  - {item['path']} ({len(item['evidence'])} evidence fields)")

    print()
    print("4. ADMISSIBILITY MATRIX")
    fields = [
        "candidate_id", "source_dataset", "source_artifact", "population_description",
        "record_count", "known_space_separation", "provenance", "label_validity",
        "training_exclusion", "model_selection_exclusion", "d2_d5_exclusion",
        "test_isolation", "same_task_relevance", "artificial_construction",
        "overall_status", "evidence",
    ]
    matrix_path = D6_ROOT / "d6_admissibility_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print("  No candidate promoted to FEASIBLE by inference.")
    print("  Matrix status: NOT ESTABLISHED")

    print()
    print("5. FEASIBILITY RESULT")
    result = {
        "phase": "D6",
        "version": "v001",
        "status": "UNRESOLVED",
        "reason": (
            "The feasibility audit inventories approved project artifacts and structured evidence, "
            "but does not infer an admissible unknown population from filenames, counts, predictions, "
            "or cross-dataset similarity. Explicit provenance, known-space separation, and exclusion "
            "evidence are required before a candidate can be marked FEASIBLE."
        ),
        "dtest_accessed": False,
        "model_inference": False,
        "threshold_selected": False,
        "synthetic_unknowns": False,
        "external_scraping": False,
        "cross_dataset_unknown_mapping_assumed": False,
        "cbase_hash_match": cbase.get("hash_match", False),
    }
    result_path = D6_ROOT / "d6_open_set_feasibility_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "phase": "D6",
        "version": "v001",
        "status": "UNRESOLVED",
        "config": str(CFG_PATH.relative_to(ROOT)),
        "inventory": str(inventory_path.relative_to(ROOT)),
        "evidence_discovery": str(evidence_path.relative_to(ROOT)),
        "admissibility_matrix": str(matrix_path.relative_to(ROOT)),
        "results": str(result_path.relative_to(ROOT)),
        "dtest_accessed": False,
        "model_inference": False,
        "cbase_sha256": cbase.get("sha256"),
    }
    manifest_path = D6_ROOT / "d6_open_set_feasibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print("=" * 72)
    print("D6 v001 AUDIT COMPLETE")
    print("=" * 72)
    print("Status: UNRESOLVED")
    print("No candidate was declared admissible without explicit evidence.")
    print()
    print("Generated:")
    for p in [CFG_PATH, inventory_path, matrix_path, result_path, manifest_path]:
        print(f"  {p.relative_to(ROOT)}")
    print()
    print("Important: review the evidence before any candidate is promoted.")


if __name__ == "__main__":
    main()
