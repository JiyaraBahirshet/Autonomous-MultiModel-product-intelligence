import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "mave"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mave" / "b2"

REPORT_DIR = PROJECT_ROOT / "reports" / "preprocessing"
REPORT_FILE = REPORT_DIR / "mave_b2_preprocessing.json"

PROCESSING_VERSION = "v001"
SCHEMA_VERSION = "v001"

SPLITS = ("train", "validation", "test")

APPROVED_TEXT_FIELDS = {
    "title",
    "description",
    "brand",
    "amazon_category_path",
    "amazon_main_category",
}


def normalize_text(value: str) -> str:
    """Deterministically normalize whitespace without changing content."""
    return " ".join(value.split())


def make_record_id(asin: str) -> str:
    """Deterministically derive record_id using SHA256 of ASIN UTF-8 bytes."""
    digest = hashlib.sha256(asin.encode("utf-8")).hexdigest()[:16]
    return f"mave:{digest}"


def process_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Process one MAVE record and apply exact B2 structural transformations."""
    if not isinstance(record, Dict):
        raise ValueError("MAVE record must be a JSON object")

    asin = record.get("asin")
    if not asin:
        raise ValueError("Encountered MAVE record without asin")

    asin_str = str(asin)
    text_values_seen = 0
    text_values_changed = 0

    def transform(value: Any, field_name: str | None = None) -> Any:
        nonlocal text_values_seen, text_values_changed

        if isinstance(value, str):
            if field_name in APPROVED_TEXT_FIELDS:
                text_values_seen += 1
                normalized = normalize_text(value)
                if normalized != value:
                    text_values_changed += 1
                return normalized
            return value

        if isinstance(value, list):
            return [transform(item, field_name) for item in value]

        if isinstance(value, dict):
            return {key: transform(item, key) for key, item in value.items()}

        return value

    # Preserve all source fields with text normalization on approved fields
    processed = {key: transform(value, key) for key, value in record.items()}

    # Add canonical record_id
    processed["record_id"] = make_record_id(asin_str)

    # Add canonical supervision block
    processed["supervision"] = {
        "mave_information_available": record.get("mave_information_available") is True,
        "label_available": bool(
            record.get("mave_category") or record.get("mave_attributes")
        ),
    }

    # Add canonical provenance block
    processed["b2_provenance"] = {
        "processing_version": PROCESSING_VERSION,
        "source_asin": asin_str,
    }

    return processed, {
        "text_values_seen": text_values_seen,
        "text_values_changed": text_values_changed,
    }


def process_split(split: str) -> Dict[str, Any]:
    source_path = SPLIT_DIR / f"{split}.jsonl"
    output_path = OUTPUT_DIR / f"{split}.jsonl"

    if not source_path.is_file():
        raise FileNotFoundError(f"MAVE split not found: {source_path}")

    total_records = 0
    malformed_records = 0
    missing_asin = 0
    duplicate_asins = 0

    mave_information_available = 0
    mave_category_present = 0
    mave_attributes_present = 0
    metadata_complete = 0

    text_values_seen = 0
    text_values_changed = 0

    metadata_source_counts: Dict[str, int] = {}
    mave_label_source_counts: Dict[str, int] = {}
    field_counts: Dict[str, int] = {}

    asin_seen: set[str] = set()

    print()
    print(f"Processing frozen {split} split...")

    with (
        source_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8", newline="\n") as target,
    ):
        for line_number, line in enumerate(source, start=1):
            line_str = line.strip()

            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as exc:
                malformed_records += 1
                raise ValueError(
                    f"Malformed JSON in {source_path} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                malformed_records += 1
                raise ValueError(
                    f"Non-object JSON record in {source_path} at line {line_number}"
                )

            asin = record.get("asin")
            if not asin:
                missing_asin += 1
                raise ValueError(
                    f"Missing ASIN in {source_path} at line {line_number}"
                )

            asin_str = str(asin)

            if asin_str in asin_seen:
                duplicate_asins += 1
                raise ValueError(
                    f"Duplicate ASIN in frozen {split} split at line {line_number}: {asin_str}"
                )

            asin_seen.add(asin_str)

            processed, counters = process_record(record)

            text_values_seen += counters["text_values_seen"]
            text_values_changed += counters["text_values_changed"]

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

            metadata_source = processed.get("metadata_source")
            if metadata_source is not None:
                metadata_source_str = str(metadata_source)
                metadata_source_counts[metadata_source_str] = (
                    metadata_source_counts.get(metadata_source_str, 0) + 1
                )

            mave_label_source = processed.get("mave_label_source")
            if mave_label_source is not None:
                mave_label_source_str = str(mave_label_source)
                mave_label_source_counts[mave_label_source_str] = (
                    mave_label_source_counts.get(mave_label_source_str, 0) + 1
                )

            target.write(
                json.dumps(processed, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

            total_records += 1

            if total_records % 100_000 == 0:
                print(f"  Records processed : {total_records:,}")

    return {
        "split": split,
        "source_artifact": str(source_path.relative_to(PROJECT_ROOT)),
        "output_artifact": str(output_path.relative_to(PROJECT_ROOT)),
        "records_processed": total_records,
        "unique_asins": len(asin_seen),
        "missing_asin": missing_asin,
        "duplicate_asins": duplicate_asins,
        "malformed_records": malformed_records,
        "mave_information_available": mave_information_available,
        "mave_category_present": mave_category_present,
        "mave_attributes_present": mave_attributes_present,
        "metadata_complete": metadata_complete,
        "text_values_seen": text_values_seen,
        "text_values_changed": text_values_changed,
        "metadata_source_counts": dict(sorted(metadata_source_counts.items())),
        "mave_label_source_counts": dict(sorted(mave_label_source_counts.items())),
        "field_counts": dict(sorted(field_counts.items())),
    }


def write_schema() -> None:
    schema = {
        "dataset": "MAVE",
        "schema_version": SCHEMA_VERSION,
        "processing_version": PROCESSING_VERSION,
        "stage": "B2",
        "input_directory": "data/splits/mave",
        "output_directory": "data/processed/mave/b2",
        "record_artifacts": [
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
        ],
        "transformations": {
            "approved_text_fields": sorted(APPROVED_TEXT_FIELDS),
            "text": "deterministic whitespace normalization only",
            "source_fields": "preserved",
            "unknown_source_fields": "preserved",
            "mave_category": "preserved",
            "mave_attributes": "preserved",
            "metadata_source": "preserved",
            "mave_label_source": "preserved",
            "mave_information_available": "preserved",
            "metadata_complete": "preserved",
        },
        "missing_data_policy": (
            "Missing source values remain unchanged; no values are imputed or fabricated."
        ),
        "learned_statistics": False,
        "train_only_fitting_required": False,
        "raw_or_frozen_split_modified": False,
    }

    (OUTPUT_DIR / "schema_v001.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 70)
    print("MAVE B2 PREPROCESSING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    split_statistics = {split: process_split(split) for split in SPLITS}

    total_records = sum(
        item["records_processed"] for item in split_statistics.values()
    )
    total_unique_asins = sum(
        item["unique_asins"] for item in split_statistics.values()
    )
    total_missing_asin = sum(
        item["missing_asin"] for item in split_statistics.values()
    )
    total_duplicate_asins = sum(
        item["duplicate_asins"] for item in split_statistics.values()
    )
    total_malformed = sum(
        item["malformed_records"] for item in split_statistics.values()
    )
    total_mave_information = sum(
        item["mave_information_available"] for item in split_statistics.values()
    )
    total_mave_category = sum(
        item["mave_category_present"] for item in split_statistics.values()
    )
    total_mave_attributes = sum(
        item["mave_attributes_present"] for item in split_statistics.values()
    )
    total_metadata_complete = sum(
        item["metadata_complete"] for item in split_statistics.values()
    )
    total_text_values_seen = sum(
        item["text_values_seen"] for item in split_statistics.values()
    )
    total_text_values_changed = sum(
        item["text_values_changed"] for item in split_statistics.values()
    )

    statistics = {
        "dataset": "MAVE",
        "stage": "B2",
        "processing_version": PROCESSING_VERSION,
        "schema_version": SCHEMA_VERSION,
        "input_directory": "data/splits/mave",
        "output_directory": "data/processed/mave/b2",
        "split_statistics": split_statistics,
        "summary": {
            "total_records": total_records,
            "total_unique_asins": total_unique_asins,
            "missing_asin": total_missing_asin,
            "duplicate_asins": total_duplicate_asins,
            "malformed_records": total_malformed,
            "mave_information_available": total_mave_information,
            "mave_category_present": total_mave_category,
            "mave_attributes_present": total_mave_attributes,
            "metadata_complete": total_metadata_complete,
            "text_values_seen": total_text_values_seen,
            "text_values_changed": total_text_values_changed,
        },
        "validation": {
            "record_count_positive": total_records > 0,
            "asins_present": total_missing_asin == 0,
            "asins_unique_within_splits": total_duplicate_asins == 0,
            "jsonl_valid": total_malformed == 0,
            "all_three_splits_processed": all(
                split_statistics[split]["records_processed"] > 0 for split in SPLITS
            ),
        },
    }

    (OUTPUT_DIR / "statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_schema()

    REPORT_FILE.write_text(
        json.dumps(
            {
                "report": "mave_b2_preprocessing",
                "report_version": "v001",
                "dataset": "MAVE",
                "stage": "B2",
                "processing_version": PROCESSING_VERSION,
                "statistics": statistics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("-" * 70)
    print("MAVE B2 SUMMARY")
    print("-" * 70)
    for split in SPLITS:
        item = split_statistics[split]
        print(f"{split.capitalize():12} : {item['records_processed']:,} records")

    print(f"\nTotal records: {total_records:,}")


if __name__ == "__main__":
    main()