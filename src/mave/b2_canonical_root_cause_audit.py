#!/usr/bin/env python3
"""
MAVE B2 -> canonical-data READ-ONLY ROOT-CAUSE AUDIT

Investigates why the 7,148 MAVE-supervised records traced from frozen B3
have no approved usable text at canonical/B2 level.

STRICTLY READ-ONLY:
- never modifies canonical, B2, or B3 datasets
- never changes labels
- never creates processed data
- writes only the audit JSON report
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL = REPO_ROOT / "MAVE_Audit" / "mave_final_dataset.jsonl"
B2_ROOT = REPO_ROOT / "data" / "processed" / "mave" / "b2"
B3_ROOT = REPO_ROOT / "data" / "representations" / "mave" / "b3"
REPORT_DIR = REPO_ROOT / "reports" / "baseline" / "mave"
REPORT_PATH = REPORT_DIR / "mave_b2_canonical_root_cause_audit.json"

SPLITS = ("train", "validation", "test")
TEXT_FIELDS = (
    "title", "description", "brand", "feature",
    "amazon_category_path", "amazon_main_category",
)
TRACE_FIELDS = TEXT_FIELDS + ("mave_category", "mave_attributes")


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return bool(value)


def approved_text_available(record: dict[str, Any]) -> bool:
    return any(nonempty(record.get(k)) for k in TEXT_FIELDS)


def supervision_available(record: dict[str, Any]) -> bool:
    if nonempty(record.get("mave_category")) or nonempty(record.get("mave_attributes")):
        return True
    s = record.get("supervision")
    if isinstance(s, dict):
        return (
            s.get("label_available") is True
            or s.get("mave_information_available") is True
        )
    return record.get("mave_information_available") is True


def load_json(line: str, path: Path, line_no: int) -> dict[str, Any]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path}:{line_no}: expected JSON object")
    return obj


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_b3_supervised():
    targets: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {}

    for split in SPLITS:
        path = B3_ROOT / f"{split}.jsonl"
        records = supervised = duplicates = 0
        seen: set[str] = set()

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = load_json(line, path, line_no)
                records += 1
                asin = r.get("asin")
                if not isinstance(asin, str) or not asin:
                    raise ValueError(f"{path}:{line_no}: missing/invalid ASIN")
                if asin in seen:
                    duplicates += 1
                seen.add(asin)

                if supervision_available(r):
                    supervised += 1
                    bp = r.get("b2_provenance") or {}
                    targets[asin] = {
                        "split": split,
                        "record_id": r.get("record_id"),
                        "source_asin": bp.get("source_asin"),
                    }

        stats[split] = {
            "records": records,
            "supervised_records": supervised,
            "duplicate_asins": duplicates,
        }

    return targets, stats


def scan_canonical(target_asins: set[str]):
    matches: dict[str, dict[str, Any]] = {}
    total = 0

    with CANONICAL.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = load_json(line, CANONICAL, line_no)
            total += 1
            asin = r.get("asin")
            if asin not in target_asins:
                continue
            if asin in matches:
                raise ValueError(f"Duplicate canonical target ASIN: {asin}")

            matches[asin] = {
                "record_id": r.get("record_id"),
                "metadata_source": r.get("metadata_source"),
                "metadata_complete": r.get("metadata_complete"),
                "text_available": approved_text_available(r),
                "image_reference_count": sum(
                    len(r.get(k)) if isinstance(r.get(k), list) else 0
                    for k in ("imageURL", "imageURLHighRes")
                ),
                "field_presence": {
                    k: nonempty(r.get(k)) for k in TRACE_FIELDS
                },
            }

    return matches, {
        "total_records_scanned": total,
        "target_matches": len(matches),
        "target_records_with_approved_text": sum(
            x["text_available"] for x in matches.values()
        ),
        "target_records_with_image_references": sum(
            x["image_reference_count"] > 0 for x in matches.values()
        ),
    }


def scan_b2(target_asins: set[str]):
    matches: dict[str, dict[str, Any]] = {}
    split_counts = Counter()

    for split in SPLITS:
        path = B2_ROOT / f"{split}.jsonl"
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = load_json(line, path, line_no)
                asin = r.get("asin")
                if asin not in target_asins:
                    continue
                if asin in matches:
                    raise ValueError(f"Duplicate B2 target ASIN: {asin}")

                bp = r.get("b2_provenance") or {}
                matches[asin] = {
                    "split": split,
                    "record_id": r.get("record_id"),
                    "source_asin": bp.get("source_asin"),
                    "text_available": approved_text_available(r),
                    "supervision_available": supervision_available(r),
                    "metadata_source": r.get("metadata_source"),
                    "metadata_complete": r.get("metadata_complete"),
                    "field_presence": {
                        k: nonempty(r.get(k)) for k in TRACE_FIELDS
                    },
                }
                split_counts[split] += 1

    record_ids = [
        x["record_id"] for x in matches.values()
        if isinstance(x["record_id"], str)
    ]
    id_counts = Counter(record_ids)

    return matches, {
        "target_matches": len(matches),
        "target_records_with_approved_text": sum(
            x["text_available"] for x in matches.values()
        ),
        "target_records_with_supervision": sum(
            x["supervision_available"] for x in matches.values()
        ),
        "target_source_asin_mismatches": sum(
            x["source_asin"] != asin for asin, x in matches.items()
        ),
        "target_duplicate_record_ids": sum(
            n - 1 for n in id_counts.values() if n > 1
        ),
        "target_matches_by_split": dict(split_counts),
    }


def classify(targets, canonical, b2):
    c = Counter()
    for asin in targets:
        if asin not in canonical:
            c["MISSING_CANONICAL_RECORD"] += 1
        elif asin not in b2:
            c["MISSING_B2_RECORD"] += 1
        elif not canonical[asin]["text_available"] and not b2[asin]["text_available"]:
            c["SOURCE_AND_B2_NO_APPROVED_TEXT"] += 1
        elif canonical[asin]["text_available"] and not b2[asin]["text_available"]:
            c["TEXT_LOST_DURING_B2_PROCESSING"] += 1
        elif not canonical[asin]["text_available"] and b2[asin]["text_available"]:
            c["B2_TEXT_NOT_PRESENT_CANONICALLY"] += 1
        else:
            c["TEXT_AVAILABLE_FOR_FURTHER_INTERFACE_REVIEW"] += 1
    return c


def field_summary(source: dict[str, dict[str, Any]]):
    return {
        field: sum(x["field_presence"].get(field, False) for x in source.values())
        for field in TRACE_FIELDS
    }


def main():
    print("=" * 70)
    print("MAVE B2 -> CANONICAL DATA ROOT-CAUSE AUDIT")
    print("=" * 70)
    print("\nREAD-ONLY AUDIT")
    print("No canonical, B2, or B3 dataset modification will be performed.\n")

    required = (
        [CANONICAL]
        + [B2_ROOT / f"{s}.jsonl" for s in SPLITS]
        + [B3_ROOT / f"{s}.jsonl" for s in SPLITS]
    )
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    print("Scanning frozen B3 supervised records...")
    targets, b3_stats = scan_b3_supervised()
    target_asins = set(targets)
    print(f"  Supervised B3 ASINs: {len(target_asins):,}")

    print("\nScanning canonical MAVE dataset...")
    canonical, canonical_stats = scan_canonical(target_asins)
    print(f"  Canonical matches: {len(canonical):,}")

    print("\nScanning frozen B2 splits...")
    b2, b2_stats = scan_b2(target_asins)
    print(f"  B2 matches: {len(b2):,}")

    classifications = classify(targets, canonical, b2)

    if (
        len(canonical) == len(target_asins)
        and len(b2) == len(target_asins)
        and classifications["SOURCE_AND_B2_NO_APPROVED_TEXT"] == len(target_asins)
    ):
        root_cause = "SOURCE_LEVEL_TEXT_ABSENCE_FOR_MAVE_SUPERVISED_RECORDS"
        status = "CONFIRMED_SOURCE_LEVEL_DISJOINTNESS"
    elif classifications["TEXT_LOST_DURING_B2_PROCESSING"] > 0:
        root_cause = "TEXT_LOST_DURING_B2_PROCESSING"
        status = "B2_PROCESSING_REQUIRES_REVIEW"
    elif classifications["TEXT_AVAILABLE_FOR_FURTHER_INTERFACE_REVIEW"] > 0:
        root_cause = "APPROVED_TEXT_EXISTS_FOR_SUPERVISED_RECORDS"
        status = "FURTHER_B4_INTERFACE_REVIEW_REQUIRED"
    else:
        root_cause = "UNRESOLVED_PROVENANCE_CONDITION"
        status = "REVIEW_REQUIRED"

    report = {
        "audit": {
            "name": "MAVE B2 -> canonical data root-cause audit",
            "version": "v001",
            "read_only": True,
            "dataset_modification": False,
        },
        "paths": {
            "canonical": str(CANONICAL.relative_to(REPO_ROOT)),
            "b2_root": str(B2_ROOT.relative_to(REPO_ROOT)),
            "b3_root": str(B3_ROOT.relative_to(REPO_ROOT)),
        },
        "b3_supervision_population": {
            "total_supervised_asins": len(target_asins),
            "split_stats": b3_stats,
        },
        "canonical_trace": canonical_stats,
        "b2_trace": b2_stats,
        "classification": dict(classifications),
        "field_presence_among_supervised_targets": {
            "canonical": field_summary(canonical),
            "b2": field_summary(b2),
        },
        "root_cause": root_cause,
        "status": status,
        "integrity_checks": {
            "canonical_target_coverage": len(canonical) == len(target_asins),
            "b2_target_coverage": len(b2) == len(target_asins),
            "b2_source_asin_mismatches": b2_stats["target_source_asin_mismatches"],
            "b2_duplicate_target_record_ids": b2_stats["target_duplicate_record_ids"],
        },
        "source_sha256": {
            "canonical": sha256(CANONICAL),
            "b2_train": sha256(B2_ROOT / "train.jsonl"),
            "b2_validation": sha256(B2_ROOT / "validation.jsonl"),
            "b2_test": sha256(B2_ROOT / "test.jsonl"),
            "b3_train": sha256(B3_ROOT / "train.jsonl"),
            "b3_validation": sha256(B3_ROOT / "validation.jsonl"),
            "b3_test": sha256(B3_ROOT / "test.jsonl"),
        },
        "notes": [
            "Strictly read-only.",
            "Image references are not treated as text inputs.",
            "MAVE labels are not fabricated or inferred.",
            "Approved text means non-empty product metadata in the defined text fields.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\n" + "-" * 70)
    print("ROOT-CAUSE SUMMARY")
    print("-" * 70)
    print(f"Supervised targets             : {len(target_asins):,}")
    print(f"Canonical matches               : {len(canonical):,}")
    print(f"B2 matches                      : {len(b2):,}")
    print(f"Canonical targets with text     : {canonical_stats['target_records_with_approved_text']:,}")
    print(f"B2 targets with text            : {b2_stats['target_records_with_approved_text']:,}")
    print(f"B2 source-ASIN mismatches       : {b2_stats['target_source_asin_mismatches']:,}")
    print(f"\nRoot cause classification       : {root_cause}")
    print(f"STATUS                          : {status}")
    print("\nReport:")
    print(f"  {REPORT_PATH}")


if __name__ == "__main__":
    main()
