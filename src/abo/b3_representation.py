"""
ABO B3 — Feature / Representation Construction

Purpose
-------
Construct deterministic B3 representations from the frozen ABO B2 artifacts.

IMPORTANT IMAGE UPDATE
----------------------
The ABO physical image archive was expanded after the original B3 artifact
was created. Therefore B3 v002 recalculates physical image availability
against the ACTUAL extracted image filesystem.

B3 does NOT:
- modify B2 artifacts
- modify frozen train/validation/test splits
- download images
- guess image paths
- fabricate images
- train learned models
- perform fusion

Physical image availability is determined only by:
    ABO_Audit/images/small/<relative_path>

The documented B2 relative_path is used exactly as supplied.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "abo"
    / "b2"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "representations"
    / "abo"
    / "b3"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "representation"
)

# Actual extracted ABO image root.
IMAGE_ROOT = (
    PROJECT_ROOT
    / "ABO_Audit"
    / "images"
    / "small"
)

SPLITS = ("train", "validation", "test")

SCHEMA_VERSION = "v002"
REPRESENTATION_VERSION = "v002"


# ---------------------------------------------------------------------------
# REPRESENTATION FIELDS
# ---------------------------------------------------------------------------

TEXT_FIELDS = (
    "item_name",
    "title",
    "brand",
    "bullet_point",
    "color",
    "description",
    "fabric_type",
    "finish_type",
    "item_keywords",
    "item_shape",
    "material",
    "model_name",
    "pattern",
    "product_description",
    "style",
)

STRUCTURED_FIELDS = (
    "amazon_category_path",
    "amazon_main_category",
    "feature",
    "item_keywords",
)


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# ---------------------------------------------------------------------------
# TEXT REPRESENTATION
# ---------------------------------------------------------------------------

def extract_text(v: Any) -> list[str]:
    """
    Recursively extract string values from ABO nested structures.
    """
    if isinstance(v, str):
        return [v]

    if isinstance(v, list):
        result: list[str] = []

        for item in v:
            result.extend(extract_text(item))

        return result

    if isinstance(v, dict):
        if "value" in v:
            return extract_text(v["value"])

        result: list[str] = []

        for key, value in v.items():
            if key != "language_tag":
                result.extend(extract_text(value))

        return result

    return []


def norm_text(v: Any) -> str:
    strings = extract_text(v)

    if not strings:
        return ""

    joined = " ".join(strings)

    return " ".join(joined.split()).strip()


def tokenize(v: Any) -> list[str]:
    return re.findall(r"\S+", norm_text(v))


def text_rep(record: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, list[str]] = {}
    combined: list[str] = []
    present: list[str] = []

    for field in TEXT_FIELDS:
        value = record.get(field)

        if value is None:
            continue

        text = norm_text(value)

        if not text:
            continue

        tokens = tokenize(text)

        fields[field] = tokens
        combined.extend(tokens)
        present.append(field)

    return {
        "field_tokens": fields,
        "combined_tokens": combined,
        "token_count": len(combined),
        "present_text_fields": present,
    }


# ---------------------------------------------------------------------------
# STRUCTURED REPRESENTATION
# ---------------------------------------------------------------------------

def structured_rep(record: dict[str, Any]) -> dict[str, Any]:
    presence: dict[str, bool] = {}
    types: dict[str, str] = {}

    for field in STRUCTURED_FIELDS:
        value = record.get(field)

        presence[field] = (
            value is not None
            and (
                not isinstance(value, list)
                or len(value) > 0
            )
        )

        if value is None:
            types[field] = "missing"

        elif isinstance(value, list):
            types[field] = "list"

        elif isinstance(value, dict):
            types[field] = "dict"

        else:
            types[field] = type(value).__name__

    return {
        "field_presence": presence,
        "field_types": types,
    }


# ---------------------------------------------------------------------------
# IMAGE REPRESENTATION
# ---------------------------------------------------------------------------

def normalize_relative_path(relative_path: Any) -> str | None:
    """
    Normalize the documented ABO relative image path.

    Example:
        8c/8ccb5859.jpg
    becomes:
        8c\\8ccb5859.jpg

    No path guessing is performed.
    """

    if not isinstance(relative_path, str):
        return None

    value = relative_path.strip()

    if not value:
        return None

    return value.replace("/", "\\")


def physical_image_exists(relative_path: Any) -> bool:
    """
    Check whether the documented relative image path exists physically.

    Only the frozen/documented relative_path is used.

    No downloading.
    No path guessing.
    No alternative filename search.
    """

    normalized = normalize_relative_path(relative_path)

    if normalized is None:
        return False

    candidate = IMAGE_ROOT.joinpath(*normalized.split("\\"))

    # Prevent accidental path traversal.
    try:
        candidate.resolve().relative_to(IMAGE_ROOT.resolve())
    except ValueError:
        return False

    return (
        candidate.is_file()
        and candidate.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    )


def image_ref(v: Any) -> dict[str, Any]:
    """
    Build image metadata using actual filesystem availability.

    This is the critical B3 v002 change.
    """

    if not isinstance(v, dict):
        return {
            "image_id": None,
            "relative_path": None,
            "metadata_available": False,
            "physical_file_available": False,
            "usable_for_local_image_model": False,
        }

    image_id = v.get("image_id")
    relative_path = v.get("relative_path")

    image_id = (
        None
        if image_id is None
        else str(image_id)
    )

    metadata_available = (
        v.get("metadata_available") is True
        or image_id is not None
    )

    physical = physical_image_exists(relative_path)

    return {
        "image_id": image_id,
        "relative_path": relative_path,
        "metadata_available": metadata_available,
        "physical_file_available": physical,
        "usable_for_local_image_model": physical,
    }


def image_rep(record: dict[str, Any]) -> dict[str, Any]:
    main = image_ref(
        record.get("main_image")
    )

    raw_others = record.get("other_images", [])

    if not isinstance(raw_others, list):
        raw_others = []

    others = [
        image_ref(image)
        for image in raw_others
    ]

    references = []

    if main["image_id"] is not None:
        references.append(main)

    references.extend(
        image
        for image in others
        if image["image_id"] is not None
    )

    physical_count = sum(
        1
        for image in references
        if image["physical_file_available"]
    )

    missing_count = (
        len(references)
        - physical_count
    )

    return {
        "main": main,
        "other_images": others,

        "reference_count": len(references),

        "physical_reference_count": physical_count,

        "missing_physical_reference_count": missing_count,

        "has_main_image_reference": (
            main["image_id"] is not None
        ),

        "has_locally_available_main_image": (
            main["usable_for_local_image_model"]
        ),
    }


# ---------------------------------------------------------------------------
# BUILD RECORD
# ---------------------------------------------------------------------------

def build(
    record: dict[str, Any],
    split: str,
) -> dict[str, Any]:

    for field in (
        "record_id",
        "item_id",
    ):
        if field not in record:
            raise ValueError(
                f"Missing required B2 identity field: {field}"
            )

    record_split = (
        record.get("split")
        or record.get(
            "b2_provenance",
            {},
        ).get("source_split")
    )

    if (
        record_split
        and record_split != split
    ):
        raise ValueError(
            f"Split mismatch: expected "
            f"'{split}', got '{record_split}'"
        )

    # Shallow copy of B2 record.
    # B2 itself is NEVER modified.
    output = dict(record)

    output["split"] = split
    output["frozen_split"] = split

    output["b3_representation"] = {
        "representation_version": (
            REPRESENTATION_VERSION
        ),

        "text": text_rep(record),

        "structured": structured_rep(record),

        "image": image_rep(record),
    }

    return output


# ---------------------------------------------------------------------------
# PROCESS SPLIT
# ---------------------------------------------------------------------------

def process_split(split: str) -> dict[str, Any]:

    source = (
        INPUT_DIR
        / f"{split}.jsonl"
    )

    destination = (
        OUTPUT_DIR
        / f"{split}.jsonl"
    )

    if not source.is_file():
        raise FileNotFoundError(
            f"Input file not found: {source}"
        )

    stats: dict[str, Any] = {
        "source_records": 0,
        "output_records": 0,

        "malformed_records": 0,

        "duplicate_record_ids": 0,
        "unique_record_ids": 0,

        "title_tokens": 0,
        "combined_text_tokens": 0,

        "empty_token_sequences": 0,

        "records_with_main_image_reference": 0,

        "records_with_physical_main_image": 0,

        "total_image_references": 0,

        "image_references_with_physical_files": 0,

        "image_references_missing_physical_files": 0,

        "text_field_presence": {},
    }

    record_ids: set[str] = set()

    field_counter: Counter[str] = Counter()

    with source.open(
        "r",
        encoding="utf-8",
    ) as infile, destination.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as outfile:

        for line_number, line in enumerate(
            infile,
            1,
        ):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Malformed JSON at "
                    f"{source}:{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at "
                    f"{source}:{line_number}"
                )

            stats["source_records"] += 1

            record_id = str(
                record.get(
                    "record_id",
                    "",
                )
            )

            if record_id in record_ids:
                stats["duplicate_record_ids"] += 1

            record_ids.add(record_id)

            output = build(
                record,
                split,
            )

            representation = (
                output["b3_representation"]
            )

            # ---------------------------------------------------------------
            # Text statistics
            # ---------------------------------------------------------------

            text = representation["text"]

            stats["title_tokens"] += len(
                text["field_tokens"].get(
                    "item_name",
                    [],
                )
            )

            stats["combined_text_tokens"] += (
                text["token_count"]
            )

            if text["token_count"] == 0:
                stats["empty_token_sequences"] += 1

            for field in text[
                "present_text_fields"
            ]:
                field_counter[field] += 1

            # ---------------------------------------------------------------
            # Image statistics
            # ---------------------------------------------------------------

            image = representation["image"]

            if image[
                "has_main_image_reference"
            ]:
                stats[
                    "records_with_main_image_reference"
                ] += 1

            if image[
                "has_locally_available_main_image"
            ]:
                stats[
                    "records_with_physical_main_image"
                ] += 1

            stats[
                "total_image_references"
            ] += image[
                "reference_count"
            ]

            stats[
                "image_references_with_physical_files"
            ] += image[
                "physical_reference_count"
            ]

            stats[
                "image_references_missing_physical_files"
            ] += image[
                "missing_physical_reference_count"
            ]

            outfile.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            stats["output_records"] += 1

    stats["unique_record_ids"] = len(
        record_ids
    )

    stats["text_field_presence"] = dict(
        field_counter
    )

    return stats


# ---------------------------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------------------------

def schema() -> dict[str, Any]:

    return {
        "dataset": (
            "Amazon Berkeley Objects (ABO)"
        ),

        "stage": "B3",

        "schema_version": SCHEMA_VERSION,

        "representation_version": (
            REPRESENTATION_VERSION
        ),

        "input_artifact": (
            "data/processed/abo/b2/"
            "{split}.jsonl"
        ),

        "output_artifact": (
            "data/representations/abo/b3/"
            "{split}.jsonl"
        ),

        "split_values": list(SPLITS),

        "representation_components": {

            "text": {
                "source_fields": list(
                    TEXT_FIELDS
                ),

                "normalization": (
                    "deterministic "
                    "whitespace normalization"
                ),

                "tokenization": (
                    "deterministic "
                    "whitespace tokenization"
                ),

                "lowercasing": False,
                "translation": False,
                "stemming": False,
                "stopword_removal": False,
                "punctuation_deletion": False,
            },

            "structured": {
                "source_fields": list(
                    STRUCTURED_FIELDS
                ),

                "encoding": (
                    "presence and native "
                    "source type metadata"
                ),

                "source_values_overwritten": False,
            },

            "image": {

                "source_fields": [
                    "image_id",
                    "relative_path",
                    "metadata_available",
                    "physical_file_available",
                ],

                "image_ids_preserved": True,

                "relative_paths_preserved": True,

                "physical_file_availability": (
                    "recomputed from actual "
                    "ABO_Audit/images/small "
                    "filesystem using the "
                    "documented relative_path"
                ),

                "usable_for_local_image_model": (
                    "true only when the "
                    "documented physical file "
                    "exists and has a supported "
                    "image extension"
                ),

                "image_download": False,

                "image_path_guessing": False,

                "synthetic_images": False,
            },
        },

        "provenance": {

            "record_id_preserved": True,

            "item_id_preserved": True,

            "split_preserved": True,

            "frozen_split_preserved": True,

            "b2_fields_preserved": True,

            "raw_data_modified": False,
        },

        "leakage_policy": (
            "Record-local deterministic "
            "representations only; no learned "
            "parameters."
        ),

        "missing_data_policy": (
            "Missing source fields and missing "
            "physical images remain missing. "
            "No values are fabricated."
        ),

        "physical_image_root": (
            "ABO_Audit/images/small"
        ),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:

    configure_logging()

    print("=" * 70)
    print("ABO B3 REPRESENTATION CONSTRUCTION v002")
    print("=" * 70)

    print(
        f"\nPhysical image root:\n"
        f"  {IMAGE_ROOT}"
    )

    if not IMAGE_ROOT.is_dir():

        print(
            "\nERROR: Physical image root does not exist."
        )

        return 1

    # Count currently extracted physical images.
    physical_images = [
        path
        for path in IMAGE_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png"}
        )
    ]

    print(
        f"Physical image files found: "
        f"{len(physical_images):,}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_stats: dict[
        str,
        dict[str, Any],
    ] = {}

    for split in SPLITS:

        print(
            f"\nProcessing B2 {split} split..."
        )

        stats = process_split(
            split
        )

        split_stats[split] = stats

        print(
            f"  Source records       : "
            f"{stats['source_records']:,}"
        )

        print(
            f"  Output records       : "
            f"{stats['output_records']:,}"
        )

        print(
            f"  Unique record IDs    : "
            f"{stats['unique_record_ids']:,}"
        )

        print(
            f"  Duplicate IDs        : "
            f"{stats['duplicate_record_ids']:,}"
        )

        print(
            f"  Main image refs      : "
            f"{stats['records_with_main_image_reference']:,}"
        )

        print(
            f"  Physical main images : "
            f"{stats['records_with_physical_main_image']:,}"
        )

        print(
            f"  Total image refs     : "
            f"{stats['total_image_references']:,}"
        )

        print(
            f"  Physical image refs  : "
            f"{stats['image_references_with_physical_files']:,}"
        )

        print(
            f"  Missing image refs   : "
            f"{stats['image_references_missing_physical_files']:,}"
        )

    # -----------------------------------------------------------------------
    # Overall statistics
    # -----------------------------------------------------------------------

    total_records = sum(
        x["output_records"]
        for x in split_stats.values()
    )

    total_main_refs = sum(
        x["records_with_main_image_reference"]
        for x in split_stats.values()
    )

    total_physical_main = sum(
        x["records_with_physical_main_image"]
        for x in split_stats.values()
    )

    total_image_refs = sum(
        x["total_image_references"]
        for x in split_stats.values()
    )

    total_physical_refs = sum(
        x["image_references_with_physical_files"]
        for x in split_stats.values()
    )

    total_missing_refs = sum(
        x["image_references_missing_physical_files"]
        for x in split_stats.values()
    )

    total_duplicates = sum(
        x["duplicate_record_ids"]
        for x in split_stats.values()
    )

    total_empty = sum(
        x["empty_token_sequences"]
        for x in split_stats.values()
    )

    main_coverage = (
        (
            total_physical_main
            / total_main_refs
        )
        * 100
        if total_main_refs
        else 0.0
    )

    image_coverage = (
        (
            total_physical_refs
            / total_image_refs
        )
        * 100
        if total_image_refs
        else 0.0
    )

    statistics = {

        "dataset": (
            "Amazon Berkeley Objects (ABO)"
        ),

        "stage": "B3",

        "schema_version": SCHEMA_VERSION,

        "representation_version": (
            REPRESENTATION_VERSION
        ),

        "physical_image_root": (
            "ABO_Audit/images/small"
        ),

        "physical_image_files_found": (
            len(physical_images)
        ),

        "splits": split_stats,

        "total_records": total_records,

        "total_main_image_references": (
            total_main_refs
        ),

        "total_physical_main_images": (
            total_physical_main
        ),

        "overall_main_image_coverage_percent": round(
            main_coverage,
            4,
        ),

        "total_image_references": (
            total_image_refs
        ),

        "total_physical_image_references": (
            total_physical_refs
        ),

        "total_missing_physical_references": (
            total_missing_refs
        ),

        "overall_image_reference_coverage_percent": round(
            image_coverage,
            4,
        ),

        "duplicate_record_ids": (
            total_duplicates
        ),

        "empty_token_sequences": (
            total_empty
        ),

        "invariants": {

            "record_count_preserved": (
                total_records == 147702
            ),

            "record_ids_preserved": (
                total_duplicates == 0
            ),

            "split_boundaries_preserved": True,

            "b2_source_fields_preserved": True,

            "missing_images_preserved": True,

            "synthetic_values_created": False,

            "raw_data_modified": False,

        },
    }

    # -----------------------------------------------------------------------
    # Write artifacts
    # -----------------------------------------------------------------------

    schema_file = (
        OUTPUT_DIR
        / "schema_v002.json"
    )

    statistics_file = (
        OUTPUT_DIR
        / "statistics_v002.json"
    )

    report_file = (
        REPORT_DIR
        / "abo_b3_representation_v002.json"
    )

    schema_file.write_text(
        json.dumps(
            schema(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    statistics_file.write_text(
        json.dumps(
            statistics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    status = (
        "PASS"
        if (
            total_records == 147702
            and total_duplicates == 0
            and total_empty == 0
            and total_main_refs == 147127
            and total_physical_main == 147127
        )
        else "FAIL"
    )

    report = {

        "report": (
            "abo_b3_representation"
        ),

        "report_version": "v002",

        "dataset": (
            "Amazon Berkeley Objects (ABO)"
        ),

        "stage": "B3",

        "status": status,

        "schema": schema(),

        "statistics": statistics,
    }

    report_file.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    print("\n" + "-" * 70)

    print("ABO B3 v002 SUMMARY")

    print("-" * 70)

    print(
        f"Physical images found     : "
        f"{len(physical_images):,}"
    )

    print(
        f"Total B3 records          : "
        f"{total_records:,}"
    )

    print(
        f"Main image references     : "
        f"{total_main_refs:,}"
    )

    print(
        f"Physical main images      : "
        f"{total_physical_main:,}"
    )

    print(
        f"Main image coverage       : "
        f"{main_coverage:.4f}%"
    )

    print(
        f"Total image references    : "
        f"{total_image_refs:,}"
    )

    print(
        f"Physical image references : "
        f"{total_physical_refs:,}"
    )

    print(
        f"Missing physical refs     : "
        f"{total_missing_refs:,}"
    )

    print(
        f"Overall image coverage    : "
        f"{image_coverage:.4f}%"
    )

    print(
        f"Duplicate record IDs      : "
        f"{total_duplicates:,}"
    )

    print(
        f"Empty token sequences     : "
        f"{total_empty:,}"
    )

    print(
        f"\nSTATUS: {status}"
    )

    print("\nGenerated:")

    print(
        f"  {schema_file}"
    )

    print(
        f"  {statistics_file}"
    )

    print(
        f"  {report_file}"
    )

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())