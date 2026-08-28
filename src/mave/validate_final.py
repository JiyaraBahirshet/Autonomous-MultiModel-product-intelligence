import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    PROJECT_ROOT
    / "mave_audit"
    / "mave_final_dataset.jsonl"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "validation"
)

REPORT_PATH = (
    REPORT_DIR
    / "mave_final_validation.json"
)


def main():
    if not DATASET.is_file():
        raise FileNotFoundError(
            f"MAVE dataset not found: {DATASET}"
        )

    total = 0
    malformed = 0
    missing_asin = 0
    duplicate_asin = 0
    asin_seen = set()

    mave_available = 0
    metadata_complete = 0
    mave_category_present = 0
    mave_attributes_present = 0

    field_counts = Counter()

    print("=" * 70)
    print("MAVE FINAL DATASET VALIDATION")
    print("=" * 70)
    print(f"Dataset       : {DATASET}")
    print()
    print("Scanning dataset...")

    with DATASET.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, 1):
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

            for key in record:
                field_counts[key] += 1

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
                mave_available += 1

            if record.get(
                "metadata_complete"
            ) is True:
                metadata_complete += 1

            if record.get("mave_category"):
                mave_category_present += 1

            if record.get("mave_attributes"):
                mave_attributes_present += 1

    report = {
        "validator": "mave_final_validation",
        "validator_version": "v001",
        "dataset": "MAVE",
        "dataset_file": str(
            DATASET.relative_to(PROJECT_ROOT)
        ),

        "record_count": total,
        "unique_asins": len(asin_seen),
        "missing_asin": missing_asin,
        "duplicate_asin": duplicate_asin,
        "malformed": malformed,

        "mave_information_available": mave_available,
        "metadata_complete": metadata_complete,
        "mave_category_present": mave_category_present,
        "mave_attributes_present": mave_attributes_present,

        "field_counts": dict(
            sorted(field_counts.items())
        ),

        "validation": {
            "jsonl_valid": malformed == 0,
            "asins_present": missing_asin == 0,
            "asins_unique": duplicate_asin == 0,
        },
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
    print("SUMMARY")
    print("-" * 70)
    print(f"Records                 : {total:,}")
    print(f"Unique ASINs            : {len(asin_seen):,}")
    print(f"Missing ASIN            : {missing_asin:,}")
    print(f"Duplicate ASIN          : {duplicate_asin:,}")
    print(f"Malformed JSONL         : {malformed:,}")
    print(f"MAVE information avail. : {mave_available:,}")
    print(f"Metadata complete       : {metadata_complete:,}")
    print(f"MAVE category present   : {mave_category_present:,}")
    print(f"MAVE attributes present : {mave_attributes_present:,}")
    print()
    print("FIELDS")
    print("-" * 70)

    for field, count in sorted(
        field_counts.items()
    ):
        print(f"{field:30} {count:,}")

    print()
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()