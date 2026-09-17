"""
MAVE B3 -> B4.3 Provenance / Label-to-Input Trace Audit

Purpose
-------
Read-only forensic audit of the 7,148 MAVE-supervised records.

Trace:
    Canonical MAVE dataset
        ->
    frozen B2 split
        ->
    frozen B3 representation
        ->
    B4.3 supervision/input interface

This script DOES NOT:
- modify any dataset
- rewrite B2/B3
- repair labels
- fabricate text
- backfill Amazon metadata
- download images
- train a model

It only determines where the MAVE supervision exists and whether
usable approved input exists at each stage.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_PATH = (
    PROJECT_ROOT
    / "mave_audit"
    / "mave_final_dataset.jsonl"
)

B2_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mave"
    / "b2"
)

B3_DIR = (
    PROJECT_ROOT
    / "data"
    / "representations"
    / "mave"
    / "b3"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "baseline"
    / "mave"
)

REPORT_PATH = (
    REPORT_DIR
    / "mave_b4_provenance_trace_audit.json"
)

SPLITS = ("train", "validation", "test")

TEXT_FIELDS = (
    "title",
    "description",
    "brand",
    "amazon_category_path",
    "amazon_main_category",
    "feature",
)


def load_json_line(
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for current_line, line in enumerate(handle, start=1):
            if current_line == line_number:
                return json.loads(line)

    raise RuntimeError(
        f"Unable to read line {line_number} from {path}"
    )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, None, exc
                continue

            yield line_number, record, None


def has_source_text(record: dict[str, Any]) -> bool:
    """
    Determine whether at least one approved B2/B3 text field
    contains usable textual content.
    """

    for field in TEXT_FIELDS:
        value = record.get(field)

        if isinstance(value, str):
            if value.strip():
                return True

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return True

                if isinstance(item, dict):
                    for nested_value in item.values():
                        if (
                            isinstance(nested_value, str)
                            and nested_value.strip()
                        ):
                            return True

    return False


def has_b3_text(record: dict[str, Any]) -> bool:
    b3 = record.get("b3_representation")

    if not isinstance(b3, dict):
        return False

    text = b3.get("text")

    if not isinstance(text, dict):
        return False

    combined_tokens = text.get("combined_tokens")

    return (
        isinstance(combined_tokens, list)
        and len(combined_tokens) > 0
    )


def has_mave_supervision(record: dict[str, Any]) -> bool:
    return (
        record.get("mave_information_available") is True
        or bool(record.get("mave_category"))
        or bool(record.get("mave_attributes"))
        or bool(record.get("mave_label_source"))
        or (
            isinstance(record.get("supervision"), dict)
            and record["supervision"].get("label_available") is True
        )
    )


def supervision_type(record: dict[str, Any]) -> str:
    category = bool(record.get("mave_category"))
    attributes = bool(record.get("mave_attributes"))

    if category and attributes:
        return "category_and_attributes"

    if category:
        return "category_only"

    if attributes:
        return "attributes_only"

    if record.get("mave_information_available") is True:
        return "information_only"

    return "none"


def record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def collect_supervised_b3_records():
    """
    First pass:
    collect only the ASINs that actually carry MAVE supervision
    in the frozen B3 artifacts.

    This avoids loading the full 2.9M-record dataset into memory.
    """

    targets: dict[str, dict[str, Any]] = {}

    for split in SPLITS:
        path = B3_DIR / f"{split}.jsonl"

        print(f"Scanning B3 {split} for supervised records...")

        count = 0

        for line_number, record, error in iter_jsonl(path):

            if error is not None:
                raise RuntimeError(
                    f"B3 malformed JSON: {path}:{line_number}: {error}"
                )

            if not isinstance(record, dict):
                continue

            if not has_mave_supervision(record):
                continue

            asin = record.get("asin")

            if not asin:
                raise RuntimeError(
                    f"Supervised B3 record missing ASIN: "
                    f"{path}:{line_number}"
                )

            if asin in targets:
                raise RuntimeError(
                    f"Supervised ASIN appears more than once: {asin}"
                )

            targets[asin] = {
                "asin": asin,
                "split": split,
                "b3_line": line_number,
                "b3_record_id": record.get("record_id"),
                "b3_record_hash": record_hash(record),
                "b3_metadata_source": record.get("metadata_source"),
                "b3_mave_label_source": record.get(
                    "mave_label_source"
                ),
                "b3_mave_information_available": (
                    record.get("mave_information_available")
                ),
                "b3_mave_category_present": bool(
                    record.get("mave_category")
                ),
                "b3_mave_attributes_present": bool(
                    record.get("mave_attributes")
                ),
                "b3_supervision_type": supervision_type(record),
                "b3_has_source_text": has_source_text(record),
                "b3_has_usable_text": has_b3_text(record),
            }

            count += 1

        print(f"  Supervised records found: {count:,}")

    return targets


def trace_canonical(
    targets: dict[str, dict[str, Any]],
):
    """
    Trace supervised ASINs back to the canonical final dataset.
    """

    found = set()
    malformed = 0
    duplicate_targets = 0

    canonical_details: dict[str, dict[str, Any]] = {}

    print()
    print("Scanning canonical MAVE dataset...")

    for line_number, record, error in iter_jsonl(CANONICAL_PATH):

        if error is not None:
            malformed += 1
            continue

        if not isinstance(record, dict):
            continue

        asin = record.get("asin")

        if asin not in targets:
            continue

        if asin in found:
            duplicate_targets += 1
            continue

        found.add(asin)

        canonical_details[asin] = {
            "canonical_line": line_number,
            "canonical_record_hash": record_hash(record),
            "canonical_metadata_source": record.get(
                "metadata_source"
            ),
            "canonical_mave_label_source": record.get(
                "mave_label_source"
            ),
            "canonical_mave_information_available": (
                record.get("mave_information_available")
            ),
            "canonical_mave_category_present": bool(
                record.get("mave_category")
            ),
            "canonical_mave_attributes_present": bool(
                record.get("mave_attributes")
            ),
            "canonical_supervision_type": supervision_type(record),
            "canonical_has_source_text": has_source_text(record),
            "canonical_title_present": bool(
                record.get("title")
            ),
            "canonical_description_present": bool(
                record.get("description")
            ),
            "canonical_brand_present": bool(
                record.get("brand")
            ),
            "canonical_feature_present": bool(
                record.get("feature")
            ),
            "canonical_category_path_present": bool(
                record.get("amazon_category_path")
            ),
            "canonical_main_category_present": bool(
                record.get("amazon_main_category")
            ),
        }

    return (
        canonical_details,
        malformed,
        duplicate_targets,
    )


def trace_b2(
    targets: dict[str, dict[str, Any]],
):
    """
    Trace supervised ASINs through frozen B2.
    """

    b2_details: dict[str, dict[str, Any]] = {}

    print()
    print("Scanning frozen B2 splits...")

    for split in SPLITS:

        path = B2_DIR / f"{split}.jsonl"

        for line_number, record, error in iter_jsonl(path):

            if error is not None:
                raise RuntimeError(
                    f"B2 malformed JSON: {path}:{line_number}: {error}"
                )

            if not isinstance(record, dict):
                continue

            asin = record.get("asin")

            if asin not in targets:
                continue

            if asin in b2_details:
                raise RuntimeError(
                    f"Supervised ASIN duplicated across B2: {asin}"
                )

            provenance = record.get("b2_provenance")

            b2_details[asin] = {
                "b2_split": split,
                "b2_line": line_number,
                "b2_record_id": record.get("record_id"),
                "b2_record_hash": record_hash(record),
                "b2_source_artifact": (
                    provenance.get("source_artifact")
                    if isinstance(provenance, dict)
                    else None
                ),
                "b2_source_split": (
                    provenance.get("source_split")
                    if isinstance(provenance, dict)
                    else None
                ),
                "b2_source_asin": (
                    provenance.get("source_asin")
                    if isinstance(provenance, dict)
                    else None
                ),
                "b2_metadata_source": record.get(
                    "metadata_source"
                ),
                "b2_mave_label_source": record.get(
                    "mave_label_source"
                ),
                "b2_mave_information_available": (
                    record.get("mave_information_available")
                ),
                "b2_mave_category_present": bool(
                    record.get("mave_category")
                ),
                "b2_mave_attributes_present": bool(
                    record.get("mave_attributes")
                ),
                "b2_supervision_type": supervision_type(record),
                "b2_has_source_text": has_source_text(record),
            }

    return b2_details


def classify_trace(
    target: dict[str, Any],
    canonical: dict[str, Any] | None,
    b2: dict[str, Any] | None,
):
    if canonical is None:
        return "MISSING_FROM_CANONICAL"

    if b2 is None:
        return "MISSING_FROM_B2"

    if target["split"] != b2["b2_split"]:
        return "SPLIT_MISMATCH"

    if canonical["canonical_metadata_source"] != "mave_fallback":
        return "LABEL_NOT_FROM_EXPECTED_FALLBACK_SOURCE"

    if not canonical["canonical_has_source_text"]:
        return "MAVE_LABEL_WITHOUT_CANONICAL_APPROVED_TEXT"

    if not b2["b2_has_source_text"]:
        return "INPUT_LOST_OR_ABSENT_AT_B2"

    if not target["b3_has_usable_text"]:
        return "INPUT_LOST_AT_B3"

    return "LABEL_AND_INPUT_ALIGNED"


def main():

    print("=" * 70)
    print("MAVE B4.3 — PROVENANCE / LABEL-TO-INPUT TRACE AUDIT")
    print("=" * 70)
    print()
    print("READ-ONLY AUDIT")
    print("No dataset modification will be performed.")
    print()

    for path in (
        CANONICAL_PATH,
        B2_DIR,
        B3_DIR,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = collect_supervised_b3_records()

    print()
    print(
        f"Total supervised B3 ASINs: {len(targets):,}"
    )

    canonical_details, canonical_malformed, canonical_duplicates = (
        trace_canonical(targets)
    )

    b2_details = trace_b2(targets)

    traces = []

    classifications = Counter()

    missing_canonical = []
    missing_b2 = []

    for asin, target in targets.items():

        canonical = canonical_details.get(asin)
        b2 = b2_details.get(asin)

        classification = classify_trace(
            target,
            canonical,
            b2,
        )

        classifications[classification] += 1

        if canonical is None:
            missing_canonical.append(asin)

        if b2 is None:
            missing_b2.append(asin)

        trace = {
            **target,
            "canonical": canonical,
            "b2": b2,
            "classification": classification,
        }

        traces.append(trace)

    canonical_source_counts = Counter(
        item["canonical"]["canonical_metadata_source"]
        for item in traces
        if item.get("canonical")
    )

    b2_source_counts = Counter(
        item["b2"]["b2_metadata_source"]
        for item in traces
        if item.get("b2")
    )

    canonical_text_count = sum(
        1
        for item in traces
        if (
            item.get("canonical")
            and item["canonical"]["canonical_has_source_text"]
        )
    )

    b2_text_count = sum(
        1
        for item in traces
        if (
            item.get("b2")
            and item["b2"]["b2_has_source_text"]
        )
    )

    b3_text_count = sum(
        1
        for item in traces
        if item["b3_has_usable_text"]
    )

    b2_source_asin_mismatch = sum(
        1
        for item in traces
        if (
            item.get("b2")
            and item["b2"]["b2_source_asin"]
            != item["asin"]
        )
    )

    b2_record_id_mismatch = sum(
        1
        for item in traces
        if (
            item.get("b2")
            and item["b2"]["b2_record_id"]
            != item["b3_record_id"]
        )
    )

    b2_split_mismatch = sum(
        1
        for item in traces
        if (
            item.get("b2")
            and item["b2"]["b2_split"]
            != item["split"]
        )
    )

    report = {
        "report": "mave_b4_provenance_trace_audit",
        "report_version": "v001",
        "dataset": "MAVE",
        "stage": "B4.3",
        "audit_type": "read_only_provenance_trace",
        "modifies_source_data": False,
        "model_training_performed": False,
        "downloads_performed": False,
        "imputation_performed": False,
        "label_fabrication_performed": False,

        "artifacts": {
            "canonical": str(
                CANONICAL_PATH.relative_to(PROJECT_ROOT)
            ),
            "b2_directory": str(
                B2_DIR.relative_to(PROJECT_ROOT)
            ),
            "b3_directory": str(
                B3_DIR.relative_to(PROJECT_ROOT)
            ),
        },

        "population": {
            "supervised_b3_asins": len(targets),
            "canonical_matches": len(canonical_details),
            "b2_matches": len(b2_details),
            "b3_supervised_records": len(targets),
        },

        "source_distribution": {
            "canonical_metadata_source": dict(
                sorted(canonical_source_counts.items())
            ),
            "b2_metadata_source": dict(
                sorted(b2_source_counts.items())
            ),
        },

        "input_availability": {
            "canonical_approved_text": canonical_text_count,
            "b2_approved_text": b2_text_count,
            "b3_usable_text": b3_text_count,
        },

        "integrity_checks": {
            "canonical_malformed_records": canonical_malformed,
            "canonical_duplicate_target_matches": (
                canonical_duplicates
            ),
            "missing_canonical_targets": len(
                missing_canonical
            ),
            "missing_b2_targets": len(missing_b2),
            "b2_source_asin_mismatches": (
                b2_source_asin_mismatch
            ),
            "b2_record_id_mismatches": (
                b2_record_id_mismatch
            ),
            "b2_split_mismatches": b2_split_mismatch,
        },

        "classification_counts": dict(
            sorted(classifications.items())
        ),

        "root_cause_indicators": {
            "all_supervised_records_have_no_b3_text": (
                b3_text_count == 0
            ),
            "all_supervised_records_have_no_b2_text": (
                b2_text_count == 0
            ),
            "all_supervised_records_have_no_canonical_text": (
                canonical_text_count == 0
            ),
            "all_supervised_records_are_mave_fallback": (
                canonical_source_counts.get(
                    "mave_fallback",
                    0,
                )
                == len(targets)
            ),
        },

        "traces": traces,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("-" * 70)
    print("PROVENANCE TRACE SUMMARY")
    print("-" * 70)

    print(
        f"Supervised ASINs traced       : {len(targets):,}"
    )

    print(
        f"Canonical matches             : "
        f"{len(canonical_details):,}"
    )

    print(
        f"B2 matches                    : "
        f"{len(b2_details):,}"
    )

    print(
        f"Canonical records with text   : "
        f"{canonical_text_count:,}"
    )

    print(
        f"B2 records with text          : "
        f"{b2_text_count:,}"
    )

    print(
        f"B3 records with usable text   : "
        f"{b3_text_count:,}"
    )

    print()
    print("Classification:")
    for key, value in sorted(classifications.items()):
        print(f"  {key:<45}: {value:,}")

    print()
    print("Integrity:")
    print(
        f"  Missing canonical targets   : "
        f"{len(missing_canonical):,}"
    )
    print(
        f"  Missing B2 targets          : "
        f"{len(missing_b2):,}"
    )
    print(
        f"  B2 source ASIN mismatches   : "
        f"{b2_source_asin_mismatch:,}"
    )
    print(
        f"  B2 record-ID mismatches     : "
        f"{b2_record_id_mismatch:,}"
    )
    print(
        f"  B2 split mismatches         : "
        f"{b2_split_mismatch:,}"
    )

    print()
    print("-" * 70)

    if (
        len(targets) == 0
        or len(canonical_details) != len(targets)
        or len(b2_details) != len(targets)
    ):
        status = "FAIL"

    elif (
        canonical_text_count == 0
        and b2_text_count == 0
        and b3_text_count == 0
    ):
        status = "CONFIRMED_LABEL_INPUT_DISJOINTNESS"

    else:
        status = "REVIEW_REQUIRED"

    print(f"STATUS: {status}")

    print()
    print("Report:")
    print(f"  {REPORT_PATH}")


if __name__ == "__main__":
    main()