"""
D6.2.5 v001 — Candidate Exposure & Provenance Admissibility Audit

Purpose
-------
Audit the 27 ABO-only product_type candidates established by D6.2.4
against the D6 open-set feasibility admissibility gates.

READ-ONLY:
- Does not modify Phase A/B/C artifacts.
- Does not modify D2-D5 artifacts.
- Does not run C-Base inference.
- Does not access/evaluate D-Test predictions.
- Does not select an open-set threshold.
- Does not create an unknown population.
- Does not promote feasibility automatically.

Core principle
--------------
FAIL = direct evidence of inadmissibility.
PASS = direct evidence satisfying a gate.
NOT_ESTABLISHED = evidence is insufficient.

A candidate is never declared FEASIBLE merely because it is
outside the frozen 549-class C-Base output space.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 20260827

FROZEN_CBASE_SHA256 = (
    "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"
)

EXPECTED_CBASE_CLASSES = 549
EXPECTED_ABO_UNIVERSE = 576

PROJECT_ROOT = Path(__file__).resolve().parents[1]

D6_DIR = PROJECT_ROOT / "data" / "models" / "abo" / "phase_d" / "d6" / "feasibility"

D6_2_4_RESULTS = D6_DIR / "d6_2_4_canonical_set_results.json"
D6_2_4_INVENTORY = D6_DIR / "d6_2_4_candidate_inventory.json"

CBASE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c3"
    / "training"
    / "cbase_training_manifest.json"
)

C3_EMBEDDING_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c3"
    / "embeddings"
    / "c3_embedding_manifest.json"
)

# Frozen ABO source populations.
ABO_SPLITS = {
    "train": PROJECT_ROOT / "data" / "splits" / "abo" / "train.jsonl",
    "validation": PROJECT_ROOT / "data" / "splits" / "abo" / "validation.jsonl",
    "test": PROJECT_ROOT / "data" / "splits" / "abo" / "test.jsonl",
}

# Phase C benchmark manifests / outputs are discovered rather than
# assumed to have one exact filename.
PHASE_C_ROOT = PROJECT_ROOT / "data" / "models" / "abo" / "phase_c"

D2_ROOT = PROJECT_ROOT / "data" / "models" / "abo" / "phase_d" / "d2"
D3_ROOT = PROJECT_ROOT / "data" / "models" / "abo" / "phase_d" / "d3"
D4_ROOT = PROJECT_ROOT / "data" / "models" / "abo" / "phase_d" / "d4"
D5_ROOT = PROJECT_ROOT / "data" / "models" / "abo" / "phase_d" / "d5"

OUTPUT_RESULTS = D6_DIR / "d6_2_5_admissibility_results.json"
OUTPUT_MATRIX = D6_DIR / "d6_2_5_admissibility_matrix.csv"
OUTPUT_MANIFEST = D6_DIR / "d6_2_5_admissibility_manifest.json"


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonical_product_type(value: Any) -> str | None:
    """
    Canonicalize the observed ABO product_type structure.

    Expected observed structure:
        [{"value": "SHOES"}]

    Also tolerates a plain string because earlier audit artifacts
    may contain normalized values.
    """
    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    if isinstance(value, list):
        values = []

        for item in value:
            if isinstance(item, dict):
                v = item.get("value")
                if isinstance(v, str) and v.strip():
                    values.append(v.strip())
            elif isinstance(item, str) and item.strip():
                values.append(item.strip())

        if len(values) == 1:
            return values[0]

    return None


def read_jsonl_product_types(path: Path) -> tuple[Counter, Counter]:
    """
    Returns:
        counts by canonical product_type
        record IDs by canonical product_type
    """
    counts = Counter()
    ids = defaultdict(Counter)

    if not path.exists():
        return counts, ids

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                continue

            pt = canonical_product_type(record.get("product_type"))

            if pt is None:
                continue

            counts[pt] += 1

            record_id = (
                record.get("record_id")
                or record.get("id")
                or record.get("asin")
                or record.get("item_id")
            )

            if record_id is not None:
                ids[pt][str(record_id)] += 1

    return counts, ids


def recursive_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    return [
        p
        for p in root.rglob("*")
        if p.is_file()
    ]


def namespace_present(root: Path) -> bool:
    return root.exists() and any(root.rglob("*"))


def find_text_hits(root: Path, candidates: set[str]) -> dict[str, list[str]]:
    """
    Evidence discovery only.

    Searches text-like project artifacts for exact candidate strings.
    This is NOT treated as semantic evidence by itself.
    """
    hits = defaultdict(list)

    if not root.exists():
        return hits

    extensions = {
        ".json",
        ".jsonl",
        ".csv",
        ".txt",
        ".md",
        ".log",
    }

    for path in recursive_files(root):
        if path.suffix.lower() not in extensions:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for candidate in candidates:
            if candidate in text:
                hits[candidate].append(
                    str(path.relative_to(PROJECT_ROOT))
                )

    return hits


def get_cbase_classes() -> list[str]:
    manifest = load_json(C3_EMBEDDING_MANIFEST)

    classes = (
        manifest
        .get("target", {})
        .get("classes")
    )

    if not isinstance(classes, list):
        raise RuntimeError(
            "C3 embedding manifest does not contain target.classes."
        )

    classes = [
        str(x).strip()
        for x in classes
        if str(x).strip()
    ]

    return classes


def get_d6_candidates() -> list[str]:
    """
    Read the authoritative D6.2.4 candidate inventory.

    D6.2.4 v002 schema:
        {
            "audit": "D6.2.4",
            "version": "v002",
            "candidates": [
                {
                    "canonical_product_type": "...",
                    ...
                }
            ]
        }

    This function performs no inference or reconstruction.
    """
    inventory = load_json(D6_2_4_INVENTORY)

    if not isinstance(inventory, dict):
        raise RuntimeError(
            "D6.2.4 candidate inventory is not a JSON object."
        )

    candidates_raw = inventory.get("candidates")

    if not isinstance(candidates_raw, list):
        raise RuntimeError(
            "D6.2.4 candidate inventory does not contain "
            "the expected 'candidates' list."
        )

    candidates = []

    for i, item in enumerate(candidates_raw):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"D6.2.4 candidate entry {i} is not an object."
            )

        name = item.get("canonical_product_type")

        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(
                f"D6.2.4 candidate entry {i} has no valid "
                "'canonical_product_type'."
            )

        candidates.append(name.strip())

    candidates = sorted(set(candidates))

    if len(candidates) != 27:
        raise RuntimeError(
            "D6.2.4 candidate inventory did not resolve exactly "
            f"27 candidates; resolved {len(candidates)}."
        )

    return candidates

def inspect_json_structure(path: Path) -> dict[str, Any]:
    """
    Structural inspection only.
    """
    out = {
        "exists": path.exists(),
        "path": str(path.relative_to(PROJECT_ROOT))
        if path.exists()
        else str(path),
        "type": None,
        "keys": [],
    }

    if not path.exists():
        return out

    try:
        obj = load_json(path)
    except Exception as e:
        out["error"] = repr(e)
        return out

    out["type"] = type(obj).__name__

    if isinstance(obj, dict):
        out["keys"] = sorted(obj.keys())

    return out


# ============================================================
# MAIN AUDIT
# ============================================================

def main() -> int:
    D6_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("D6.2.5 v001 CANDIDATE EXPOSURE & PROVENANCE ADMISSIBILITY AUDIT")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Frozen C-Base integrity
    # --------------------------------------------------------

    cbase_checkpoint_candidates = [
        PROJECT_ROOT
        / "data"
        / "models"
        / "abo"
        / "phase_c"
        / "c3"
        / "training"
        / "cbase_best.pt"
    ]

    cbase_checkpoint = next(
        (p for p in cbase_checkpoint_candidates if p.exists()),
        None,
    )

    if cbase_checkpoint is None:
        raise RuntimeError(
            "Frozen C-Base checkpoint not found."
        )

    observed_sha = sha256_file(cbase_checkpoint)

    cbase_hash_pass = observed_sha == FROZEN_CBASE_SHA256

    print("\n[1] FROZEN C-BASE")
    print("    checkpoint:", cbase_checkpoint)
    print("    observed SHA:", observed_sha)
    print("    expected SHA:", FROZEN_CBASE_SHA256)
    print("    status:", "PASS" if cbase_hash_pass else "FAIL")

    if not cbase_hash_pass:
        raise RuntimeError(
            "Frozen C-Base SHA mismatch. Stop audit."
        )

    # --------------------------------------------------------
    # 2. D-Test boundary
    # --------------------------------------------------------

    dtest_root = (
        PROJECT_ROOT
        / "data"
        / "models"
        / "abo"
        / "phase_d"
        / "d_test"
    )

    dtest_present = namespace_present(dtest_root)

    print("\n[2] D-TEST BOUNDARY")
    print("    D-Test namespace present:", dtest_present)

    if dtest_present:
        print(
            "    WARNING: D-Test namespace exists. "
            "This audit will not read/evaluate it."
        )

    # --------------------------------------------------------
    # 3. Frozen 549 class universe
    # --------------------------------------------------------

    cbase_classes = get_cbase_classes()

    if len(cbase_classes) != EXPECTED_CBASE_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CBASE_CLASSES} C-Base classes, "
            f"found {len(cbase_classes)}."
        )

    cbase_set = set(cbase_classes)

    print("\n[3] FROZEN 549 CLASS UNIVERSE")
    print("    source:", C3_EMBEDDING_MANIFEST.relative_to(PROJECT_ROOT))
    print("    count:", len(cbase_classes))
    print("    unique:", len(cbase_set))

    # --------------------------------------------------------
    # 4. D6.2.4 candidate universe
    # --------------------------------------------------------

    candidates = get_d6_candidates()
    candidate_set = set(candidates)

    print("\n[4] D6.2.4 CANDIDATES")
    print("    candidate count:", len(candidates))

    overlap = candidate_set & cbase_set

    print("    candidate/C-Base overlap:", len(overlap))

    if overlap:
        print("    FAIL: candidates overlap frozen known space:")
        for x in sorted(overlap):
            print("      ", x)

    # --------------------------------------------------------
    # 5. ABO split exposure
    # --------------------------------------------------------

    split_counts = {}
    split_ids = {}

    print("\n[5] ABO SOURCE SPLIT EXPOSURE")

    for split_name, path in ABO_SPLITS.items():
        counts, ids = read_jsonl_product_types(path)

        split_counts[split_name] = counts
        split_ids[split_name] = ids

        print(
            f"    {split_name}: "
            f"records={sum(counts.values())}, "
            f"unique_product_types={len(counts)}"
        )

    # --------------------------------------------------------
    # 6. Candidate exposure matrix
    # --------------------------------------------------------

    candidate_records = {}

    for candidate in candidates:
        train_n = split_counts["train"].get(candidate, 0)
        val_n = split_counts["validation"].get(candidate, 0)
        test_n = split_counts["test"].get(candidate, 0)

        candidate_records[candidate] = {
            "train_count": train_n,
            "validation_count": val_n,
            "test_count": test_n,
            "total_count": train_n + val_n + test_n,
        }

    # --------------------------------------------------------
    # 7. Training exclusion gate
    # --------------------------------------------------------

    print("\n[6] TRAINING EXCLUSION")

    for candidate in candidates:
        n = candidate_records[candidate]["train_count"]

        status = "PASS" if n == 0 else "FAIL"

        print(
            f"    {candidate}: {status} "
            f"(train_count={n})"
        )

    # --------------------------------------------------------
    # 8. Model-selection / Phase-C population discovery
    # --------------------------------------------------------

    print("\n[7] PHASE-C / MODEL-SELECTION EXPOSURE DISCOVERY")

    phase_c_files = recursive_files(PHASE_C_ROOT)

    candidate_text_hits_phase_c = find_text_hits(
        PHASE_C_ROOT,
        candidate_set,
    )

    phase_c_candidate_hits = {
        candidate: sorted(
            candidate_text_hits_phase_c.get(candidate, [])
        )
        for candidate in candidates
    }

    for candidate in candidates:
        hits = phase_c_candidate_hits[candidate]

        if hits:
            status = "EVIDENCE_PRESENT"
        else:
            status = "NO_DIRECT_TEXT_HIT"

        print(
            f"    {candidate}: {status} "
            f"(files={len(hits)})"
        )

    # --------------------------------------------------------
    # 9. D2-D5 exposure discovery
    # --------------------------------------------------------

    print("\n[8] D2-D5 EXPOSURE DISCOVERY")

    downstream_roots = {
        "D2": D2_ROOT,
        "D3": D3_ROOT,
        "D4": D4_ROOT,
        "D5": D5_ROOT,
    }

    downstream_hits = {}

    for stage, root in downstream_roots.items():
        hits = find_text_hits(root, candidate_set)

        downstream_hits[stage] = {
            candidate: sorted(hits.get(candidate, []))
            for candidate in candidates
        }

        present = sum(
            bool(downstream_hits[stage][candidate])
            for candidate in candidates
        )

        print(
            f"    {stage}: candidate text evidence in "
            f"{present}/27 candidates"
        )

    # --------------------------------------------------------
    # 10. Frozen Phase-C test isolation
    # --------------------------------------------------------

    print("\n[9] FROZEN PHASE-C TEST ISOLATION")

    # Search only metadata/manifests and prediction-support artifacts.
    # Do not read C-Base prediction values and do not run inference.
    phase_c_test_hits = find_text_hits(
        PHASE_C_ROOT,
        candidate_set,
    )

    # This is deliberately classified as evidence discovery rather
    # than automatic exposure determination.
    for candidate in candidates:
        hits = phase_c_test_hits.get(candidate, [])

        print(
            f"    {candidate}: "
            f"{'TEXT_EVIDENCE_PRESENT' if hits else 'NO_DIRECT_TEXT_HIT'}"
        )

    # --------------------------------------------------------
    # 11. Provenance / source validity
    # --------------------------------------------------------

    print("\n[10] PROVENANCE & LABEL VALIDITY")

    # D6.2.4 already established that the 27 names are canonical
    # ABO product_type values. This audit records that fact but
    # does not reinterpret semantic meaning.
    provenance_status = {
        candidate: {
            "source": "ABO product_type",
            "canonicalized": True,
            "label_validity": "ESTABLISHED_BY_D6_2_4",
            "same_task_relevance": "ESTABLISHED_BY_ABO_DATASET_ROLE",
        }
        for candidate in candidates
    }

    # --------------------------------------------------------
    # 12. Artificial construction check
    # --------------------------------------------------------

    print("\n[11] ARTIFICIAL CONSTRUCTION")

    artificial_construction_status = {
        candidate: {
            "status": "PASS",
            "basis": (
                "Candidate is an observed canonical ABO product_type "
                "from the frozen ABO source universe; no synthetic "
                "candidate construction performed by this audit."
            ),
        }
        for candidate in candidates
    }

    # --------------------------------------------------------
    # 13. Build admissibility matrix
    # --------------------------------------------------------

    matrix_rows = []

    for candidate in candidates:
        train_n = candidate_records[candidate]["train_count"]
        val_n = candidate_records[candidate]["validation_count"]
        test_n = candidate_records[candidate]["test_count"]

        training_status = (
            "PASS" if train_n == 0 else "FAIL"
        )

        known_space_status = (
            "FAIL" if candidate in cbase_set else "PASS"
        )

        phase_c_hits = phase_c_candidate_hits[candidate]

        # Presence of text in Phase-C artifacts is not sufficient
        # to prove actual population exposure, because manifests and
        # inventories may contain class names. Therefore:
        #
        # - do NOT convert a text hit into FAIL.
        # - mark exposure as NOT_ESTABLISHED until actual population
        #   membership can be traced.
        #
        model_selection_status = (
            "NOT_ESTABLISHED"
            if not phase_c_hits
            else "NOT_ESTABLISHED"
        )

        d2_status = (
            "NOT_ESTABLISHED"
            if not downstream_hits["D2"][candidate]
            else "NOT_ESTABLISHED"
        )

        d3_status = (
            "NOT_ESTABLISHED"
            if not downstream_hits["D3"][candidate]
            else "NOT_ESTABLISHED"
        )

        d4_status = (
            "NOT_ESTABLISHED"
            if not downstream_hits["D4"][candidate]
            else "NOT_ESTABLISHED"
        )

        d5_status = (
            "NOT_ESTABLISHED"
            if not downstream_hits["D5"][candidate]
            else "NOT_ESTABLISHED"
        )

        # IMPORTANT:
        # Original ABO split test presence is recorded as factual
        # exposure information, but is NOT equated with the frozen
        # Phase-C C-Base test population.
        #
        # Therefore this remains a separate evidence field.
        frozen_phase_c_test_status = "NOT_ESTABLISHED"

        provenance = provenance_status[candidate]
        artificial = artificial_construction_status[candidate]

        row = {
            "candidate": candidate,

            "known_space_separation": known_space_status,

            "source_train_count": train_n,
            "source_validation_count": val_n,
            "source_test_count": test_n,

            "training_exclusion": training_status,
            "model_selection_exclusion": model_selection_status,

            "d2_exclusion": d2_status,
            "d3_exclusion": d3_status,
            "d4_exclusion": d4_status,
            "d5_exclusion": d5_status,

            "frozen_phase_c_test_isolation": frozen_phase_c_test_status,

            "provenance": provenance["label_validity"],
            "same_task_relevance": provenance["same_task_relevance"],

            "artificial_construction": artificial["status"],

            "phase_c_text_evidence_files": phase_c_hits,
            "d2_text_evidence_files": downstream_hits["D2"][candidate],
            "d3_text_evidence_files": downstream_hits["D3"][candidate],
            "d4_text_evidence_files": downstream_hits["D4"][candidate],
            "d5_text_evidence_files": downstream_hits["D5"][candidate],
        }

        matrix_rows.append(row)

    # --------------------------------------------------------
    # 14. Candidate-level feasibility determination
    # --------------------------------------------------------

    for row in matrix_rows:
        statuses = [
            row["known_space_separation"],
            row["training_exclusion"],
            row["model_selection_exclusion"],
            row["d2_exclusion"],
            row["d3_exclusion"],
            row["d4_exclusion"],
            row["d5_exclusion"],
            row["frozen_phase_c_test_isolation"],
            row["provenance"],
            row["same_task_relevance"],
            row["artificial_construction"],
        ]

        if "FAIL" in statuses:
            row["candidate_admissibility"] = "INADMISSIBLE"
        elif all(s == "PASS" for s in statuses):
            row["candidate_admissibility"] = "ADMISSIBLE"
        else:
            row["candidate_admissibility"] = "NOT_ESTABLISHED"

    # --------------------------------------------------------
    # 15. Overall D6 feasibility status
    # --------------------------------------------------------

    admissible = [
        r for r in matrix_rows
        if r["candidate_admissibility"] == "ADMISSIBLE"
    ]

    inadmissible = [
        r for r in matrix_rows
        if r["candidate_admissibility"] == "INADMISSIBLE"
    ]

    unresolved = [
        r for r in matrix_rows
        if r["candidate_admissibility"] == "NOT_ESTABLISHED"
    ]

    if admissible:
        overall_status = "FEASIBLE_CANDIDATES_IDENTIFIED"
    elif unresolved:
        overall_status = "UNRESOLVED"
    else:
        overall_status = "NOT_CURRENTLY_FEASIBLE"

    # --------------------------------------------------------
    # 16. Results
    # --------------------------------------------------------

    results = {
        "phase": "D6",
        "stage": "D6.2.5",
        "version": "v001",
        "status": overall_status,
        "audit_type": "candidate_exposure_and_provenance_admissibility",
        "read_only": True,

        "frozen_cbase": {
            "checkpoint": str(cbase_checkpoint.relative_to(PROJECT_ROOT)),
            "sha256": observed_sha,
            "expected_sha256": FROZEN_CBASE_SHA256,
            "hash_status": "PASS",
            "num_classes": len(cbase_classes),
        },

        "d_test_namespace_present": dtest_present,

        "universe": {
            "frozen_known_classes": EXPECTED_CBASE_CLASSES,
            "full_abo_classes": EXPECTED_ABO_UNIVERSE,
            "d6_2_4_candidates": len(candidates),
        },

        "candidate_counts": {
            "admissible": len(admissible),
            "inadmissible": len(inadmissible),
            "unresolved": len(unresolved),
        },

        "candidate_records": candidate_records,
        "candidate_matrix": matrix_rows,

        "important_boundary": (
            "Original ABO split counts do not by themselves establish "
            "membership in the frozen Phase-C C-Base training, validation, "
            "D2/D3/D4/D5, or final test populations. Population-level "
            "membership must be traced from authoritative Phase-C/D "
            "artifacts before an exposure gate can be marked PASS or FAIL."
        ),

        "prohibited_actions_not_performed": [
            "no_cbase_inference",
            "no_d_test_evaluation",
            "no_threshold_selection",
            "no_open_set_threshold",
            "no_model_retraining",
            "no_finetuning",
            "no_class_space_modification",
            "no_split_regeneration",
            "no_synthetic_unknowns",
            "no_external_scraping",
            "no_label_fabrication",
            "no_upstream_artifact_modification",
        ],
    }

    # --------------------------------------------------------
    # 17. Write JSON results
    # --------------------------------------------------------

    with OUTPUT_RESULTS.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # 18. Write CSV matrix
    # --------------------------------------------------------

    import csv

    fieldnames = [
        "candidate",
        "known_space_separation",
        "source_train_count",
        "source_validation_count",
        "source_test_count",
        "training_exclusion",
        "model_selection_exclusion",
        "d2_exclusion",
        "d3_exclusion",
        "d4_exclusion",
        "d5_exclusion",
        "frozen_phase_c_test_isolation",
        "provenance",
        "same_task_relevance",
        "artificial_construction",
        "candidate_admissibility",
    ]

    with OUTPUT_MATRIX.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in matrix_rows:
            writer.writerow({
                k: row[k]
                for k in fieldnames
            })

    # --------------------------------------------------------
    # 19. Manifest
    # --------------------------------------------------------

    manifest = {
        "manifest_version": "D6.2.5-v001",
        "phase": "D6",
        "stage": "D6.2.5",
        "seed": SEED,
        "read_only": True,

        "inputs": {
            "d6_2_4_results": str(
                D6_2_4_RESULTS.relative_to(PROJECT_ROOT)
            ),
            "d6_2_4_inventory": str(
                D6_2_4_INVENTORY.relative_to(PROJECT_ROOT)
            ),
            "cbase_training_manifest": str(
                CBASE_MANIFEST.relative_to(PROJECT_ROOT)
            ),
            "c3_embedding_manifest": str(
                C3_EMBEDDING_MANIFEST.relative_to(PROJECT_ROOT)
            ),
            "abo_train": str(
                ABO_SPLITS["train"].relative_to(PROJECT_ROOT)
            ),
            "abo_validation": str(
                ABO_SPLITS["validation"].relative_to(PROJECT_ROOT)
            ),
            "abo_test": str(
                ABO_SPLITS["test"].relative_to(PROJECT_ROOT)
            ),
        },

        "outputs": {
            "results": str(
                OUTPUT_RESULTS.relative_to(PROJECT_ROOT)
            ),
            "matrix": str(
                OUTPUT_MATRIX.relative_to(PROJECT_ROOT)
            ),
        },

        "cbase_sha256": observed_sha,

        "d_test_namespace_present": dtest_present,

        "overall_status": overall_status,

        "no_inference": True,
        "no_test_evaluation": True,
        "no_threshold_selection": True,
        "no_upstream_modification": True,
    }

    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # 20. Final terminal summary
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("D6.2.5 AUDIT SUMMARY")
    print("=" * 72)

    print("C-Base hash:", "PASS")
    print("C-Base classes:", len(cbase_classes))
    print("D6.2.4 candidates:", len(candidates))
    print("Candidate/C-Base overlap:", len(overlap))

    print("\nCandidate status:")
    print("  ADMISSIBLE:", len(admissible))
    print("  INADMISSIBLE:", len(inadmissible))
    print("  NOT_ESTABLISHED:", len(unresolved))

    if inadmissible:
        print("\nInadmissible candidates:")
        for row in inadmissible:
            print("  ", row["candidate"])

    if unresolved:
        print("\nUnresolved candidates:")
        for row in unresolved:
            print("  ", row["candidate"])

    print("\nOverall D6 feasibility status:", overall_status)

    print("\nArtifacts:")
    print(" ", OUTPUT_RESULTS.relative_to(PROJECT_ROOT))
    print(" ", OUTPUT_MATRIX.relative_to(PROJECT_ROOT))
    print(" ", OUTPUT_MANIFEST.relative_to(PROJECT_ROOT))

    print("\nIMPORTANT:")
    print(
        "Do not interpret original ABO train/validation/test counts as "
        "proof of Phase-C/D exposure. Actual population membership must "
        "be traced from authoritative frozen artifacts."
    )

    print("\nD6.2.5 EXECUTION COMPLETE")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())