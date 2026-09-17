from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

B3_ROOT = PROJECT_ROOT / "data" / "representations" / "mave" / "b3"
REPORT_DIR = PROJECT_ROOT / "reports" / "baseline" / "mave"
REPORT_PATH = REPORT_DIR / "mave_b3_b4.3_interface_audit.json"

SPLITS = ("train", "validation", "test")


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Stream JSONL records without loading the full file into memory."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected JSON object"
                )

            yield line_number, record


def is_nonempty_text(record: Dict[str, Any]) -> bool:
    """
    Determine whether B3 exposes usable text through combined_tokens.

    This audit intentionally checks the B3 representation actually consumed
    by the downstream interface rather than reconstructing text from raw
    source fields.
    """
    b3 = record.get("b3_representation")

    if not isinstance(b3, dict):
        return False

    text = b3.get("text")
    if not isinstance(text, dict):
        return False

    tokens = text.get("combined_tokens")

    if not isinstance(tokens, list):
        return False

    return any(
        isinstance(token, str) and token.strip()
        for token in tokens
    )


def has_mave_supervision(record: Dict[str, Any]) -> bool:
    """
    B4.3 supervision candidate:

    The B3 structured representation reports MAVE information as available
    and contains a non-empty MAVE category or attribute supervision source.
    """
    structured = record.get("b3_representation", {}).get("structured", {})

    if not isinstance(structured, dict):
        return False

    field_presence = structured.get("field_presence", {})

    if not isinstance(field_presence, dict):
        return False

    information_available = bool(
        field_presence.get("mave_information_available", False)
    )

    mave_category = record.get("mave_category")
    mave_attributes = record.get("mave_attributes")

    category_available = (
        isinstance(mave_category, str)
        and bool(mave_category.strip())
    )

    attributes_available = (
        isinstance(mave_attributes, list)
        and len(mave_attributes) > 0
    )

    return information_available and (
        category_available or attributes_available
    )


def get_label_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the exact B3/B2 MAVE supervision fields without modifying them."""
    return {
        "mave_information_available": record.get(
            "mave_information_available"
        ),
        "mave_category": record.get("mave_category"),
        "mave_attributes": record.get("mave_attributes"),
        "mave_label_source": record.get("mave_label_source"),
        "metadata_source": record.get("metadata_source"),
        "supervision": record.get("supervision"),
    }


def inspect_split(split: str) -> Dict[str, Any]:
    path = B3_ROOT / f"{split}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Missing B3 split: {path}")

    total = 0
    unique_asins = set()
    unique_record_ids = set()

    duplicate_asins = 0
    duplicate_record_ids = 0

    malformed_identity = 0
    split_mismatches = 0

    usable_text = 0
    supervision_candidates = 0
    supervision_candidates_with_text = 0
    supervision_candidates_without_text = 0

    category_records = 0
    attribute_records = 0
    information_records = 0

    missing_b3_representation = 0
    missing_b3_text = 0
    missing_structured = 0

    supervision_examples: List[Dict[str, Any]] = []

    for line_number, record in iter_jsonl(path):
        total += 1

        asin = record.get("asin")
        record_id = record.get("record_id")

        if not isinstance(asin, str) or not asin.strip():
            malformed_identity += 1
        else:
            if asin in unique_asins:
                duplicate_asins += 1
            unique_asins.add(asin)

        if not isinstance(record_id, str) or not record_id.strip():
            malformed_identity += 1
        else:
            if record_id in unique_record_ids:
                duplicate_record_ids += 1
            unique_record_ids.add(record_id)

        actual_split = record.get("split")
        frozen_split = record.get("frozen_split")

        if actual_split != split or frozen_split != split:
            split_mismatches += 1

        b3 = record.get("b3_representation")

        if not isinstance(b3, dict):
            missing_b3_representation += 1
        else:
            if not isinstance(b3.get("text"), dict):
                missing_b3_text += 1

            if not isinstance(b3.get("structured"), dict):
                missing_structured += 1

        text_available = is_nonempty_text(record)

        if text_available:
            usable_text += 1

        if has_mave_supervision(record):
            supervision_candidates += 1

            if text_available:
                supervision_candidates_with_text += 1

                if len(supervision_examples) < 5:
                    supervision_examples.append(
                        {
                            "line": line_number,
                            "asin": asin,
                            "record_id": record_id,
                            "label_payload": get_label_payload(record),
                        }
                    )
            else:
                supervision_candidates_without_text += 1

        if isinstance(record.get("mave_category"), str):
            if record["mave_category"].strip():
                category_records += 1

        if (
            isinstance(record.get("mave_attributes"), list)
            and len(record["mave_attributes"]) > 0
        ):
            attribute_records += 1

        if record.get("mave_information_available") is True:
            information_records += 1

    return {
        "split": split,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "records": total,
        "unique_asins": len(unique_asins),
        "unique_record_ids": len(unique_record_ids),
        "duplicate_asins": duplicate_asins,
        "duplicate_record_ids": duplicate_record_ids,
        "malformed_identity": malformed_identity,
        "split_mismatches": split_mismatches,
        "usable_b3_text": usable_text,
        "mave_information_records": information_records,
        "mave_category_records": category_records,
        "mave_attribute_records": attribute_records,
        "supervision_candidates": supervision_candidates,
        "supervision_candidates_with_text": supervision_candidates_with_text,
        "supervision_candidates_without_text": (
            supervision_candidates_without_text
        ),
        "missing_b3_representation": missing_b3_representation,
        "missing_b3_text": missing_b3_text,
        "missing_b3_structured": missing_structured,
        "supervision_examples": supervision_examples,
    }


def compute_cross_split_leakage(
    split_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform a second streaming pass over B3 to establish cross-split
    ASIN/record-ID intersections.
    """
    asin_sets: Dict[str, set] = {}
    record_id_sets: Dict[str, set] = {}

    for split in SPLITS:
        path = B3_ROOT / f"{split}.jsonl"

        asins = set()
        record_ids = set()

        for _, record in iter_jsonl(path):
            asin = record.get("asin")
            record_id = record.get("record_id")

            if isinstance(asin, str) and asin.strip():
                asins.add(asin)

            if isinstance(record_id, str) and record_id.strip():
                record_ids.add(record_id)

        asin_sets[split] = asins
        record_id_sets[split] = record_ids

    asin_leaks = {}
    record_id_leaks = {}

    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            asin_intersection = asin_sets[left] & asin_sets[right]
            record_intersection = (
                record_id_sets[left] & record_id_sets[right]
            )

            key = f"{left}_vs_{right}"

            asin_leaks[key] = {
                "count": len(asin_intersection),
                "examples": sorted(asin_intersection)[:10],
            }

            record_id_leaks[key] = {
                "count": len(record_intersection),
                "examples": sorted(record_intersection)[:10],
            }

    total_asin_leaks = sum(
        value["count"] for value in asin_leaks.values()
    )

    total_record_id_leaks = sum(
        value["count"] for value in record_id_leaks.values()
    )

    return {
        "asin": {
            "pairwise": asin_leaks,
            "total_pairwise_intersections": total_asin_leaks,
        },
        "record_id": {
            "pairwise": record_id_leaks,
            "total_pairwise_intersections": total_record_id_leaks,
        },
    }


def classify_root_cause(
    split_results: Dict[str, Dict[str, Any]],
    leakage: Dict[str, Any],
) -> str:
    total_candidates = sum(
        r["supervision_candidates"] for r in split_results.values()
    )

    candidates_with_text = sum(
        r["supervision_candidates_with_text"]
        for r in split_results.values()
    )

    usable_text = sum(
        r["usable_b3_text"] for r in split_results.values()
    )

    missing_b3 = sum(
        r["missing_b3_representation"]
        for r in split_results.values()
    )

    missing_text = sum(
        r["missing_b3_text"]
        for r in split_results.values()
    )

    if missing_b3 > 0:
        return "B3_SCHEMA_OR_ARTIFACT_MISSING"

    if total_candidates == 0:
        return "NO_MAVE_SUPERVISION_CANDIDATES"

    if candidates_with_text == 0 and usable_text > 0:
        return "LABEL_INPUT_DISJOINTNESS_MAVE_LABELS_HAVE_NO_USABLE_B3_TEXT"

    if candidates_with_text > 0:
        return "B4.3_INPUTS_AVAILABLE"

    if missing_text > 0:
        return "B3_TEXT_INTERFACE_INCOMPLETE"

    if (
        leakage["asin"]["total_pairwise_intersections"] > 0
        or leakage["record_id"]["total_pairwise_intersections"] > 0
    ):
        return "LEAKAGE_REQUIRES_REVIEW"

    return "UNCLASSIFIED_INTERFACE_STATE"


def main() -> int:
    print("=" * 70)
    print("MAVE B3 → B4.3 INTERFACE AUDIT")
    print("=" * 70)

    print()
    print(f"B3 root: {B3_ROOT}")

    if not B3_ROOT.exists():
        print()
        print("ERROR: B3 representation directory does not exist.")
        return 1

    split_results: Dict[str, Dict[str, Any]] = {}

    for split in SPLITS:
        print()
        print(f"Inspecting frozen B3 {split} split...")

        result = inspect_split(split)
        split_results[split] = result

        print(f"  Records                         : {result['records']:,}")
        print(f"  Unique ASINs                    : {result['unique_asins']:,}")
        print(f"  Duplicate ASINs                 : {result['duplicate_asins']:,}")
        print(
            f"  Duplicate record IDs            : "
            f"{result['duplicate_record_ids']:,}"
        )
        print(
            f"  B3 representation missing      : "
            f"{result['missing_b3_representation']:,}"
        )
        print(
            f"  Usable B3 text                 : "
            f"{result['usable_b3_text']:,}"
        )
        print(
            f"  MAVE information records       : "
            f"{result['mave_information_records']:,}"
        )
        print(
            f"  MAVE category records          : "
            f"{result['mave_category_records']:,}"
        )
        print(
            f"  MAVE attribute records         : "
            f"{result['mave_attribute_records']:,}"
        )
        print(
            f"  Supervised candidates           : "
            f"{result['supervision_candidates']:,}"
        )
        print(
            f"  Candidates + usable B3 text    : "
            f"{result['supervision_candidates_with_text']:,}"
        )
        print(
            f"  Candidates without B3 text     : "
            f"{result['supervision_candidates_without_text']:,}"
        )

    print()
    print("Checking cross-split identity leakage...")

    leakage = compute_cross_split_leakage(split_results)

    print(
        "  Cross-split ASIN intersections  : "
        f"{leakage['asin']['total_pairwise_intersections']:,}"
    )

    print(
        "  Cross-split record-ID intersections: "
        f"{leakage['record_id']['total_pairwise_intersections']:,}"
    )

    root_cause = classify_root_cause(
        split_results,
        leakage,
    )

    total_records = sum(
        result["records"] for result in split_results.values()
    )

    total_information = sum(
        result["mave_information_records"]
        for result in split_results.values()
    )

    total_category = sum(
        result["mave_category_records"]
        for result in split_results.values()
    )

    total_attributes = sum(
        result["mave_attribute_records"]
        for result in split_results.values()
    )

    total_candidates = sum(
        result["supervision_candidates"]
        for result in split_results.values()
    )

    total_candidates_with_text = sum(
        result["supervision_candidates_with_text"]
        for result in split_results.values()
    )

    total_candidates_without_text = sum(
        result["supervision_candidates_without_text"]
        for result in split_results.values()
    )

    total_usable_text = sum(
        result["usable_b3_text"] for result in split_results.values()
    )

    if (
        total_candidates > 0
        and total_candidates_with_text > 0
        and leakage["asin"]["total_pairwise_intersections"] == 0
        and leakage["record_id"]["total_pairwise_intersections"] == 0
    ):
        status = "PASS_INTERFACE_AVAILABLE"
    else:
        status = "INTERFACE_REQUIRES_REVIEW"

    report = {
        "audit": {
            "name": "MAVE B3 to B4.3 interface audit",
            "version": "v001",
            "read_only": True,
            "model_training_performed": False,
        },
        "source": {
            "b3_root": str(B3_ROOT.relative_to(PROJECT_ROOT)),
            "splits": list(SPLITS),
        },
        "splits": split_results,
        "aggregate": {
            "total_records": total_records,
            "usable_b3_text": total_usable_text,
            "mave_information_records": total_information,
            "mave_category_records": total_category,
            "mave_attribute_records": total_attributes,
            "supervision_candidates": total_candidates,
            "supervision_candidates_with_text": total_candidates_with_text,
            "supervision_candidates_without_text": (
                total_candidates_without_text
            ),
        },
        "leakage": leakage,
        "root_cause_classification": root_cause,
        "status": status,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("-" * 70)
    print("B3 → B4.3 INTERFACE AUDIT SUMMARY")
    print("-" * 70)
    print(f"Total records                 : {total_records:,}")
    print(f"Usable B3 text                : {total_usable_text:,}")
    print(f"MAVE supervision candidates   : {total_candidates:,}")
    print(f"Candidates + B3 text         : {total_candidates_with_text:,}")
    print(
        f"Candidates without B3 text    : "
        f"{total_candidates_without_text:,}"
    )
    print(
        f"Cross-split ASIN leakage      : "
        f"{leakage['asin']['total_pairwise_intersections']:,}"
    )
    print(
        f"Cross-split record-ID leakage : "
        f"{leakage['record_id']['total_pairwise_intersections']:,}"
    )
    print()
    print(f"ROOT-CAUSE CLASSIFICATION: {root_cause}")
    print(f"STATUS: {status}")
    print()
    print("Report:")
    print(f"  {REPORT_PATH}")

    return 0 if status == "PASS_INTERFACE_AVAILABLE" else 2


if __name__ == "__main__":
    sys.exit(main())