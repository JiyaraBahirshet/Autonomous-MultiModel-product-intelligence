import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mave"
)

LISTINGS_PATH = (
    PROCESSED_DIR
    / "listings.jsonl"
)

SCHEMA_PATH = (
    PROCESSED_DIR
    / "schema_v001.json"
)

STATISTICS_PATH = (
    PROCESSED_DIR
    / "statistics.json"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "validation"
)

REPORT_PATH = (
    REPORT_DIR
    / "mave_processed_validation.json"
)


REQUIRED_FIELDS = {
    "asin",
    "title",
    "description",
    "brand",
    "price",
    "feature",
    "amazon_category_path",
    "amazon_main_category",
    "imageURL",
    "imageURLHighRes",
    "mave_category",
    "mave_attributes",
    "metadata_source",
    "mave_label_source",
    "metadata_complete",
    "mave_information_available",
}


def main():
    print("=" * 70)
    print("MAVE PROCESSED DATA VALIDATION")
    print("=" * 70)

    if not LISTINGS_PATH.is_file():
        raise FileNotFoundError(
            f"Processed listings not found: {LISTINGS_PATH}"
        )

    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"Processed schema not found: {SCHEMA_PATH}"
        )

    if not STATISTICS_PATH.is_file():
        raise FileNotFoundError(
            f"Processed statistics not found: {STATISTICS_PATH}"
        )

    total = 0
    malformed = 0
    missing_asin = 0
    duplicate_asin = 0

    mave_information_available = 0
    mave_category_present = 0
    mave_attributes_present = 0
    metadata_complete = 0

    missing_required_fields = 0
    records_with_extra_fields = 0

    asin_seen = set()

    observed_fields = set()
    missing_field_counts = {}

    print()
    print("Validating processed listings...")
    print(f"Dataset: {LISTINGS_PATH}")

    with LISTINGS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            if not isinstance(record, dict):
                malformed += 1
                continue

            total += 1

            observed_fields.update(record.keys())

            missing_fields = (
                REQUIRED_FIELDS - set(record.keys())
            )

            if missing_fields:
                missing_required_fields += 1

                for field in missing_fields:
                    missing_field_counts[field] = (
                        missing_field_counts.get(field, 0)
                        + 1
                    )

            if set(record.keys()) - REQUIRED_FIELDS:
                records_with_extra_fields += 1

            asin = record.get("asin")

            if not asin:
                missing_asin += 1
            elif asin in asin_seen:
                duplicate_asin += 1
            else:
                asin_seen.add(asin)

            if record.get(
                "mave_information_available"
            ) is True:
                mave_information_available += 1

            if record.get("mave_category"):
                mave_category_present += 1

            if record.get("mave_attributes"):
                mave_attributes_present += 1

            if record.get("metadata_complete") is True:
                metadata_complete += 1

    print()
    print("Validating schema artifact...")

    try:
        schema = json.loads(
            SCHEMA_PATH.read_text(
                encoding="utf-8"
            )
        )
        schema_valid = isinstance(schema, dict)
    except (json.JSONDecodeError, OSError):
        schema_valid = False

    print()
    print("Validating statistics artifact...")

    try:
        statistics = json.loads(
            STATISTICS_PATH.read_text(
                encoding="utf-8"
            )
        )
        statistics_valid = isinstance(
            statistics,
            dict,
        )
    except (json.JSONDecodeError, OSError):
        statistics_valid = False

    statistics_consistent = False

    if statistics_valid:
        statistics_consistent = (
            statistics.get("processed_record_count") == total
            and statistics.get("unique_asins")
            == len(asin_seen)
            and statistics.get(
                "mave_information_available"
            )
            == mave_information_available
            and statistics.get(
                "mave_category_present"
            )
            == mave_category_present
            and statistics.get(
                "mave_attributes_present"
            )
            == mave_attributes_present
            and statistics.get(
                "metadata_complete"
            )
            == metadata_complete
        )

    validation = {
        "jsonl_valid": malformed == 0,
        "asins_present": missing_asin == 0,
        "asins_unique": duplicate_asin == 0,
        "required_fields_present": (
            missing_required_fields == 0
        ),
        "schema_valid": schema_valid,
        "statistics_valid": statistics_valid,
        "statistics_consistent": statistics_consistent,
    }

    overall_valid = all(validation.values())

    report = {
        "validator": "mave_processed_validation",
        "validator_version": "v001",
        "dataset": "MAVE",
        "processed_dataset": str(
            LISTINGS_PATH.relative_to(PROJECT_ROOT)
        ),

        "record_count": total,
        "unique_asins": len(asin_seen),

        "missing_asin": missing_asin,
        "duplicate_asin": duplicate_asin,
        "malformed": malformed,

        "mave_information_available": (
            mave_information_available
        ),
        "mave_category_present": (
            mave_category_present
        ),
        "mave_attributes_present": (
            mave_attributes_present
        ),
        "metadata_complete": metadata_complete,

        "missing_required_fields": (
            missing_required_fields
        ),
        "missing_required_field_counts": (
            missing_field_counts
        ),
        "records_with_extra_fields": (
            records_with_extra_fields
        ),

        "observed_fields": sorted(
            observed_fields
        ),

        "required_fields": sorted(
            REQUIRED_FIELDS
        ),

        "artifacts": {
            "listings": LISTINGS_PATH.is_file(),
            "schema": SCHEMA_PATH.is_file(),
            "statistics": STATISTICS_PATH.is_file(),
        },

        "validation": validation,
        "overall_valid": overall_valid,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print("SUMMARY")
    print("-" * 70)

    print(f"Records                    : {total:,}")
    print(f"Unique ASINs               : {len(asin_seen):,}")
    print(f"Missing ASIN               : {missing_asin:,}")
    print(f"Duplicate ASIN             : {duplicate_asin:,}")
    print(f"Malformed records          : {malformed:,}")
    print(
        "MAVE information available : "
        f"{mave_information_available:,}"
    )
    print(
        "MAVE category present      : "
        f"{mave_category_present:,}"
    )
    print(
        "MAVE attributes present    : "
        f"{mave_attributes_present:,}"
    )
    print(
        f"Metadata complete          : {metadata_complete:,}"
    )
    print(
        "Records missing required fields: "
        f"{missing_required_fields:,}"
    )
    print()
    print("VALIDATION")
    print("-" * 70)

    for name, result in validation.items():
        print(
            f"{name:30} : "
            f"{'PASS' if result else 'FAIL'}"
        )

    print()
    print(
        "OVERALL STATUS             : "
        f"{'PASS' if overall_valid else 'FAIL'}"
    )

    print()
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
