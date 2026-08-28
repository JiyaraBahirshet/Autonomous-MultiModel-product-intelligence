import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "abo"

LISTINGS_FILE = PROCESSED_DIR / "listings.jsonl"
IMAGE_INVENTORY_FILE = PROCESSED_DIR / "image_inventory.jsonl"
SCHEMA_FILE = PROCESSED_DIR / "schema_v001.json"
STATISTICS_FILE = PROCESSED_DIR / "statistics.json"

REPORT_DIR = PROJECT_ROOT / "reports" / "validation"
REPORT_FILE = REPORT_DIR / "abo_processed_validation.json"


EXPECTED_LISTING_COUNT = 147702
EXPECTED_IMAGE_METADATA_COUNT = 398212
EXPECTED_PHYSICAL_IMAGE_COUNT = 495
EXPECTED_PHYSICAL_IMAGE_MISSING_COUNT = 397717
EXPECTED_TOTAL_IMAGE_REFERENCES = 710653
EXPECTED_REFERENCES_WITH_PHYSICAL_FILES = 1793
EXPECTED_REFERENCES_MISSING_PHYSICAL_FILES = 708860
EXPECTED_MISSING_MAIN_IMAGE_ID = 575


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def validate_listings() -> dict:
    total_records = 0
    malformed_lines = 0

    missing_item_id = 0
    duplicate_item_ids = set()
    item_ids = set()

    missing_record_id = 0
    duplicate_record_ids = set()
    record_ids = set()

    missing_main_image_id = 0
    invalid_main_image_structure = 0

    invalid_other_images_structure = 0
    invalid_image_reference_records = 0

    total_image_references = 0
    image_references_with_physical_files = 0
    image_references_missing_physical_files = 0

    referenced_image_ids = set()

    with LISTINGS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            if not isinstance(record, dict):
                malformed_lines += 1
                continue

            total_records += 1

            # --------------------------------------------------
            # Core processed-record identity
            # --------------------------------------------------

            record_id = record.get("record_id")

            if not record_id:
                missing_record_id += 1
            else:
                record_id = str(record_id)

                if record_id in record_ids:
                    duplicate_record_ids.add(record_id)
                else:
                    record_ids.add(record_id)

            item_id = record.get("item_id")

            if not item_id:
                missing_item_id += 1
            else:
                item_id = str(item_id)

                if item_id in item_ids:
                    duplicate_item_ids.add(item_id)
                else:
                    item_ids.add(item_id)

            # --------------------------------------------------
            # Processed main_image representation
            # --------------------------------------------------

            if "main_image" not in record:
                missing_main_image_id += 1
                invalid_main_image_structure += 1

            else:
                main_image = record.get("main_image")

                if main_image is None:
                    missing_main_image_id += 1

                elif not isinstance(main_image, dict):
                    invalid_main_image_structure += 1

                else:
                    image_id = main_image.get("image_id")

                    if not image_id:
                        invalid_image_reference_records += 1
                    else:
                        image_id = str(image_id)
                        referenced_image_ids.add(image_id)

                        total_image_references += 1

                        if main_image.get(
                            "physical_file_available"
                        ) is True:
                            image_references_with_physical_files += 1

                        elif main_image.get(
                            "physical_file_available"
                        ) is False:
                            image_references_missing_physical_files += 1

                        else:
                            invalid_image_reference_records += 1

            # --------------------------------------------------
            # Processed other_images representation
            # --------------------------------------------------

            other_images = record.get("other_images")

            if not isinstance(other_images, list):
                invalid_other_images_structure += 1
                continue

            for image_ref in other_images:

                if not isinstance(image_ref, dict):
                    invalid_image_reference_records += 1
                    continue

                image_id = image_ref.get("image_id")

                if not image_id:
                    invalid_image_reference_records += 1
                    continue

                image_id = str(image_id)

                referenced_image_ids.add(image_id)
                total_image_references += 1

                if image_ref.get(
                    "physical_file_available"
                ) is True:
                    image_references_with_physical_files += 1

                elif image_ref.get(
                    "physical_file_available"
                ) is False:
                    image_references_missing_physical_files += 1

                else:
                    invalid_image_reference_records += 1

    return {
        "total_records": total_records,
        "malformed_lines": malformed_lines,

        "missing_record_id": missing_record_id,
        "unique_record_ids": len(record_ids),
        "duplicate_record_ids": len(duplicate_record_ids),

        "missing_item_id": missing_item_id,
        "unique_item_ids": len(item_ids),
        "duplicate_item_ids": len(duplicate_item_ids),

        "missing_main_image_id": missing_main_image_id,
        "invalid_main_image_structure": invalid_main_image_structure,
        "invalid_other_images_structure": invalid_other_images_structure,
        "invalid_image_reference_records": (
            invalid_image_reference_records
        ),

        "total_image_references": total_image_references,
        "unique_referenced_image_ids": len(
            referenced_image_ids
        ),
        "image_references_with_physical_files": (
            image_references_with_physical_files
        ),
        "image_references_missing_physical_files": (
            image_references_missing_physical_files
        ),
    }


def validate_image_inventory() -> dict:
    total_records = 0
    malformed_lines = 0

    image_ids = set()
    duplicate_image_ids = set()

    physical_available = 0
    physical_missing = 0

    invalid_records = 0

    with IMAGE_INVENTORY_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            if not isinstance(record, dict):
                malformed_lines += 1
                continue

            total_records += 1

            image_id = record.get("image_id")

            if not image_id:
                invalid_records += 1
            else:
                image_id = str(image_id)

                if image_id in image_ids:
                    duplicate_image_ids.add(image_id)
                else:
                    image_ids.add(image_id)

            physical = record.get(
                "physical_file_available"
            )

            if physical is True:
                physical_available += 1

            elif physical is False:
                physical_missing += 1

            else:
                invalid_records += 1

    return {
        "total_records": total_records,
        "malformed_lines": malformed_lines,
        "unique_image_ids": len(image_ids),
        "duplicate_image_ids": len(duplicate_image_ids),
        "physical_images_available": physical_available,
        "physical_images_missing": physical_missing,
        "invalid_records": invalid_records,
    }


def main() -> int:
    print("=" * 70)
    print("ABO PROCESSED ARTIFACT VALIDATION")
    print("=" * 70)

    required_files = [
        LISTINGS_FILE,
        IMAGE_INVENTORY_FILE,
        SCHEMA_FILE,
        STATISTICS_FILE,
    ]

    for path in required_files:
        if not path.is_file():
            print(
                f"[FAIL] Required artifact not found: {path}"
            )
            return 1

    print()
    print("Validating listings.jsonl...")
    listings = validate_listings()

    print("Validating image_inventory.jsonl...")
    images = validate_image_inventory()

    schema = load_json(SCHEMA_FILE)
    statistics = load_json(STATISTICS_FILE)

    # ----------------------------------------------------------
    # Validation checks
    # ----------------------------------------------------------

    checks = {
        "listing_record_count": (
            listings["total_records"]
            == EXPECTED_LISTING_COUNT
        ),

        "listing_jsonl_valid": (
            listings["malformed_lines"] == 0
        ),

        "record_ids_present": (
            listings["missing_record_id"] == 0
        ),

        "record_ids_unique": (
            listings["duplicate_record_ids"] == 0
        ),

        "item_ids_present": (
            listings["missing_item_id"] == 0
        ),

        "processed_main_image_structure_valid": (
            listings["invalid_main_image_structure"] == 0
        ),

        "processed_other_images_structure_valid": (
            listings["invalid_other_images_structure"] == 0
        ),

        "processed_image_references_valid": (
            listings["invalid_image_reference_records"] == 0
        ),

        "missing_main_image_matches_statistics": (
            listings["missing_main_image_id"]
            == statistics.get("missing_main_image_id")
            == EXPECTED_MISSING_MAIN_IMAGE_ID
        ),

        "image_inventory_jsonl_valid": (
            images["malformed_lines"] == 0
        ),

        "image_inventory_records_valid": (
            images["invalid_records"] == 0
        ),

        "image_metadata_count": (
            images["total_records"]
            == statistics.get("image_metadata_records")
            == EXPECTED_IMAGE_METADATA_COUNT
        ),

        "image_ids_unique": (
            images["duplicate_image_ids"] == 0
        ),

        "physical_image_count_matches_statistics": (
            images["physical_images_available"]
            == statistics.get(
                "physical_images_available"
            )
            == EXPECTED_PHYSICAL_IMAGE_COUNT
        ),

        "physical_missing_count_matches_statistics": (
            images["physical_images_missing"]
            == statistics.get(
                "physical_images_missing"
            )
            == EXPECTED_PHYSICAL_IMAGE_MISSING_COUNT
        ),

        "image_reference_count_matches_statistics": (
            listings["total_image_references"]
            == statistics.get(
                "total_image_references"
            )
            == EXPECTED_TOTAL_IMAGE_REFERENCES
        ),

        "physical_reference_count_matches_statistics": (
            listings["image_references_with_physical_files"]
            == statistics.get(
                "image_references_with_physical_files"
            )
            == EXPECTED_REFERENCES_WITH_PHYSICAL_FILES
        ),

        "missing_physical_reference_count_matches_statistics": (
            listings["image_references_missing_physical_files"]
            == statistics.get(
                "image_references_missing_physical_files"
            )
            == EXPECTED_REFERENCES_MISSING_PHYSICAL_FILES
        ),

        "reference_counts_reconcile": (
            listings["image_references_with_physical_files"]
            + listings[
                "image_references_missing_physical_files"
            ]
            == listings["total_image_references"]
        ),

        "statistics_reference_counts_reconcile": (
            statistics.get(
                "image_references_with_physical_files",
                0,
            )
            + statistics.get(
                "image_references_missing_physical_files",
                0,
            )
            == statistics.get(
                "total_image_references",
                -1,
            )
        ),

        "image_inventory_counts_reconcile": (
            statistics.get(
                "physical_images_available",
                0,
            )
            + statistics.get(
                "physical_images_missing",
                0,
            )
            == statistics.get(
                "image_metadata_records",
                -1,
            )
        ),

        "schema_dataset_correct": (
            schema.get("dataset")
            == "Amazon Berkeley Objects (ABO)"
        ),

        "schema_version_correct": (
            schema.get("schema_version") == "v001"
        ),

        "processing_version_correct": (
            schema.get("processing_version") == "v001"
        ),

        "raw_data_not_modified": (
            schema.get("raw_data_modified") is False
        ),
    }

    all_checks_pass = all(checks.values())

    report = {
        "validator": "abo_processed_validation",
        "validator_version": "v002",
        "dataset": "Amazon Berkeley Objects (ABO)",

        "artifacts": {
            "listings": str(LISTINGS_FILE),
            "image_inventory": str(IMAGE_INVENTORY_FILE),
            "schema": str(SCHEMA_FILE),
            "statistics": str(STATISTICS_FILE),
        },

        "listings": listings,
        "image_inventory": images,

        "expected": {
            "listing_records": EXPECTED_LISTING_COUNT,
            "image_metadata_records": (
                EXPECTED_IMAGE_METADATA_COUNT
            ),
            "physical_images_available": (
                EXPECTED_PHYSICAL_IMAGE_COUNT
            ),
            "physical_images_missing": (
                EXPECTED_PHYSICAL_IMAGE_MISSING_COUNT
            ),
            "total_image_references": (
                EXPECTED_TOTAL_IMAGE_REFERENCES
            ),
            "references_with_physical_files": (
                EXPECTED_REFERENCES_WITH_PHYSICAL_FILES
            ),
            "references_missing_physical_files": (
                EXPECTED_REFERENCES_MISSING_PHYSICAL_FILES
            ),
            "missing_main_image_id": (
                EXPECTED_MISSING_MAIN_IMAGE_ID
            ),
        },

        "checks": checks,

        "validation": {
            "all_checks_pass": all_checks_pass,
            "duplicate_record_ids": (
                listings["duplicate_record_ids"]
            ),
            "duplicate_item_ids": (
                listings["duplicate_item_ids"]
            ),
            "duplicate_image_ids": (
                images["duplicate_image_ids"]
            ),
        },
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
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

    print(
        f"Listing records          : "
        f"{listings['total_records']:,}"
    )

    print(
        f"Unique record IDs        : "
        f"{listings['unique_record_ids']:,}"
    )

    print(
        f"Duplicate record IDs     : "
        f"{listings['duplicate_record_ids']:,}"
    )

    print(
        f"Unique item IDs          : "
        f"{listings['unique_item_ids']:,}"
    )

    print(
        f"Duplicate item IDs       : "
        f"{listings['duplicate_item_ids']:,}"
    )

    print(
        f"Missing main image      : "
        f"{listings['missing_main_image_id']:,}"
    )

    print(
        f"Image inventory records : "
        f"{images['total_records']:,}"
    )

    print(
        f"Unique image IDs        : "
        f"{images['unique_image_ids']:,}"
    )

    print(
        f"Duplicate image IDs     : "
        f"{images['duplicate_image_ids']:,}"
    )

    print(
        f"Physical images         : "
        f"{images['physical_images_available']:,}"
    )

    print(
        f"Missing physical images: "
        f"{images['physical_images_missing']:,}"
    )

    print(
        f"Image references        : "
        f"{listings['total_image_references']:,}"
    )

    print(
        f"References with files   : "
        f"{listings['image_references_with_physical_files']:,}"
    )

    print(
        f"References without files: "
        f"{listings['image_references_missing_physical_files']:,}"
    )

    print()
    print(
        "VALIDATION STATUS       : "
        f"{'PASS' if all_checks_pass else 'FAIL'}"
    )

    print()
    print(
        f"Report written to: "
        f"{REPORT_FILE}"
    )

    return 0 if all_checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())