import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = PROJECT_ROOT / "MAVE_Audit" / "mave_final_dataset.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mave"

RECORDS_OUTPUT = OUTPUT_DIR / "listings.jsonl"
SCHEMA_OUTPUT = OUTPUT_DIR / "schema_v001.json"
STATISTICS_OUTPUT = OUTPUT_DIR / "statistics.json"

PROCESSING_VERSION = "v001"
SCHEMA_VERSION = "v001"

TEXT_FIELDS = {
    "title",
    "description",
    "brand",
    "amazon_category_path",
    "amazon_main_category",
}


def normalize_text(value: str) -> str:
    """Deterministically normalize whitespace only."""
    return " ".join(value.split())


def normalize_value(value: Any, field_name: str | None = None) -> Any:
    """Normalize approved textual fields while preserving structure."""
    if isinstance(value, str):
        return normalize_text(value) if field_name in TEXT_FIELDS else value

    if isinstance(value, list):
        return [normalize_value(item, field_name) for item in value]

    if isinstance(value, dict):
        return {
            key: normalize_value(item, key)
            for key, item in value.items()
        }

    return value


def make_record_id(asin: str) -> str:
    """Create a deterministic ID while preserving the original ASIN."""
    digest = hashlib.sha256(asin.encode("utf-8")).hexdigest()[:16]
    return f"mave:{digest}"


def process_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("MAVE record must be a JSON object")

    asin = record.get("asin")
    if not asin:
        raise ValueError("Encountered MAVE record without asin")

    processed = {
        key: normalize_value(value, key)
        for key, value in record.items()
    }

    processed["record_id"] = make_record_id(str(asin))

    processed["supervision"] = {
        "mave_information_available": (
            processed.get("mave_information_available") is True
        ),
        "label_available": bool(
            processed.get("mave_category")
            or processed.get("mave_attributes")
        ),
    }

    return processed


def write_schema() -> None:
    schema = {
        "dataset": "MAVE",
        "schema_version": SCHEMA_VERSION,
        "processing_version": PROCESSING_VERSION,
        "record_artifact": "listings.jsonl",
        "record_id": {
            "field": "record_id",
            "derivation": "sha256(asin) first 16 hexadecimal characters",
            "source_identifier_preserved": True,
        },
        "transformations": {
            "approved_text_fields": sorted(TEXT_FIELDS),
            "text": "deterministic whitespace normalization only",
            "source_fields": "preserved",
            "unknown_source_fields": "preserved",
            "mave_category": "preserved",
            "mave_attributes": "preserved",
            "metadata_source": "preserved",
            "mave_label_source": "preserved",
            "mave_information_available": "preserved",
            "supervision": (
                "derived from canonical MAVE availability information; "
                "no labels fabricated"
            ),
        },
        "missing_data_policy": (
            "Missing source values remain missing; no values are "
            "imputed or fabricated."
        ),
        "raw_data_modified": False,
    }

    SCHEMA_OUTPUT.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 70)
    print("MAVE PHASE A PREPROCESSING")
    print("=" * 70)

    if not DATASET_FILE.is_file():
        raise FileNotFoundError(
            f"MAVE dataset not found: {DATASET_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_records = 0
    malformed_records = 0
    missing_asin = 0
    duplicate_asin = 0
    asin_seen: set[str] = set()

    mave_information_available = 0
    mave_category_present = 0
    mave_attributes_present = 0
    metadata_complete = 0

    field_counts: dict[str, int] = {}

    print(f"Source: {DATASET_FILE}")
    print("Streaming canonical MAVE dataset...")

    with (
        DATASET_FILE.open("r", encoding="utf-8") as source,
        RECORDS_OUTPUT.open(
            "w", encoding="utf-8", newline="\n"
        ) as target,
    ):
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed_records += 1
                raise ValueError(
                    f"Malformed JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                malformed_records += 1
                raise ValueError(
                    f"Non-object JSON record at line {line_number}"
                )

            asin = record.get("asin")
            if not asin:
                missing_asin += 1
                raise ValueError(
                    f"Missing ASIN at line {line_number}"
                )

            asin = str(asin)
            if asin in asin_seen:
                duplicate_asin += 1
                raise ValueError(
                    f"Duplicate ASIN at line {line_number}: {asin}"
                )

            asin_seen.add(asin)

            processed = process_record(record)

            for key in processed:
                field_counts[key] = field_counts.get(key, 0) + 1

            if processed.get("mave_information_available") is True:
                mave_information_available += 1

            if processed.get("mave_category"):
                mave_category_present += 1

            if processed.get("mave_attributes"):
                mave_attributes_present += 1

            if processed.get("metadata_complete") is True:
                metadata_complete += 1

            target.write(
                json.dumps(
                    processed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            total_records += 1

            if total_records % 100_000 == 0:
                print(f"  Processed {total_records:,} records...")

    statistics = {
        "dataset": "MAVE",
        "processing_version": PROCESSING_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source_file": str(DATASET_FILE.relative_to(PROJECT_ROOT)),
        "processed_record_count": total_records,
        "unique_asins": len(asin_seen),
        "missing_asin": missing_asin,
        "duplicate_asin": duplicate_asin,
        "malformed_records": malformed_records,
        "mave_information_available": mave_information_available,
        "mave_category_present": mave_category_present,
        "mave_attributes_present": mave_attributes_present,
        "metadata_complete": metadata_complete,
        "field_counts": dict(sorted(field_counts.items())),
        "validation": {
            "record_count_positive": total_records > 0,
            "asins_present": missing_asin == 0,
            "asins_unique": duplicate_asin == 0,
            "jsonl_valid": malformed_records == 0,
        },
    }

    STATISTICS_OUTPUT.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_schema()

    print("-" * 70)
    print("SUMMARY")
    print(f"Records                    : {total_records:,}")
    print(f"Unique ASINs               : {len(asin_seen):,}")
    print(f"Missing ASIN               : {missing_asin:,}")
    print(f"Duplicate ASIN             : {duplicate_asin:,}")
    print(f"Malformed records          : {malformed_records:,}")
    print(
        f"MAVE information available : "
        f"{mave_information_available:,}"
    )
    print(
        f"MAVE category present      : "
        f"{mave_category_present:,}"
    )
    print(
        f"MAVE attributes present    : "
        f"{mave_attributes_present:,}"
    )
    print(
        f"Metadata complete          : "
        f"{metadata_complete:,}"
    )
    print("Outputs:")
    print(f"  {RECORDS_OUTPUT}")
    print(f"  {SCHEMA_OUTPUT}")
    print(f"  {STATISTICS_OUTPUT}")


if __name__ == "__main__":
    main()
