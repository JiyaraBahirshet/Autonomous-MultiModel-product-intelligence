import csv
import gzip
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "ABO_Audit" / "listings" / "metadata"
IMAGE_METADATA_PATH = (
    PROJECT_ROOT / "ABO_Audit" / "images" / "metadata" / "images.csv.gz"
)
IMAGE_DIR = PROJECT_ROOT / "ABO_Audit" / "images" / "small"

REPORT_DIR = PROJECT_ROOT / "reports" / "validation"
REPORT_PATH = REPORT_DIR / "abo_raw_validation.json"


def load_image_metadata(path: Path) -> dict[str, str]:
    """Load ABO image_id -> relative image path mapping."""

    mapping = {}

    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        expected_fields = {"image_id", "height", "width", "path"}

        if not expected_fields.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Unexpected image metadata columns: {reader.fieldnames}"
            )

        for row in reader:
            image_id = row.get("image_id")
            image_path = row.get("path")

            if not image_id or not image_path:
                continue

            mapping[str(image_id)] = str(image_path)

    return mapping


def normalize_relative_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def main():
    print("=" * 70)
    print("ABO RAW INPUT VALIDATION")
    print("=" * 70)

    metadata_files = sorted(METADATA_DIR.glob("*.json.gz"))

    if not metadata_files:
        raise FileNotFoundError(
            f"No metadata files found: {METADATA_DIR}"
        )

    if not IMAGE_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"ABO image metadata file not found: {IMAGE_METADATA_PATH}"
        )

    print(f"Metadata directory : {METADATA_DIR}")
    print(f"Image metadata     : {IMAGE_METADATA_PATH}")
    print(f"Image directory    : {IMAGE_DIR}")
    print(f"Metadata files     : {len(metadata_files)}")

    # ------------------------------------------------------------
    # 1. Load authoritative ABO image_id -> path mapping
    # ------------------------------------------------------------

    print()
    print("Loading ABO image metadata...")

    image_id_to_path = load_image_metadata(IMAGE_METADATA_PATH)

    print(
        f"  Image metadata mappings: "
        f"{len(image_id_to_path):,}"
    )

    # ------------------------------------------------------------
    # 2. Scan listing metadata
    # ------------------------------------------------------------

    total_records = 0
    malformed_lines = 0
    empty_lines = 0
    missing_item_id = 0
    missing_main_image_id = 0

    item_listing_keys = set()
    duplicate_item_listing_keys = set()

    referenced_image_ids = set()

    field_counts = Counter()

    records_with_other_images = 0
    other_image_reference_count = 0

    per_file = []

    print()
    print("Scanning listing metadata...")

    for path in metadata_files:
        file_records = 0
        file_bad = 0

        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    empty_lines += 1
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    file_bad += 1
                    continue

                if not isinstance(record, dict):
                    malformed_lines += 1
                    file_bad += 1
                    continue

                file_records += 1
                total_records += 1

                for key in record:
                    field_counts[key] += 1

                item_id = record.get("item_id")
                domain_name = record.get("domain_name")

                if not item_id:
                    missing_item_id += 1
                else:
                    # README states that a listing is uniquely identified
                    # by (item_id, domain_name).
                    listing_key = (
                        str(item_id),
                        str(domain_name) if domain_name else None,
                    )

                    if listing_key in item_listing_keys:
                        duplicate_item_listing_keys.add(listing_key)
                    else:
                        item_listing_keys.add(listing_key)

                main_image_id = record.get("main_image_id")

                if not main_image_id:
                    missing_main_image_id += 1
                else:
                    referenced_image_ids.add(str(main_image_id))

                other_images = record.get("other_image_id")

                if other_images:
                    records_with_other_images += 1

                    if isinstance(other_images, list):
                        other_image_reference_count += len(other_images)

                        for image_id in other_images:
                            if image_id:
                                referenced_image_ids.add(str(image_id))

        per_file.append(
            {
                "file": path.name,
                "records": file_records,
                "malformed_lines": file_bad,
                "size_bytes": path.stat().st_size,
            }
        )

        print(
            f"  {path.name}: "
            f"{file_records:,} records"
        )

    # ------------------------------------------------------------
    # 3. Resolve referenced image IDs through images.csv.gz
    # ------------------------------------------------------------

    print()
    print("Resolving referenced image IDs...")

    referenced_ids_with_metadata = (
        referenced_image_ids & image_id_to_path.keys()
    )

    referenced_ids_without_metadata = (
        referenced_image_ids - image_id_to_path.keys()
    )

    # ------------------------------------------------------------
    # 4. Check physical extracted image files
    # ------------------------------------------------------------

    print()
    print("Scanning extracted images...")

    image_files = [
        p
        for p in IMAGE_DIR.rglob("*")
        if p.is_file()
    ]

    extracted_relative_paths = set()

    for image_file in image_files:
        relative_path = image_file.relative_to(IMAGE_DIR)
        extracted_relative_paths.add(
            normalize_relative_path(relative_path.as_posix())
        )

    # Resolve image IDs to their expected physical paths.
    referenced_ids_with_existing_files = set()
    referenced_ids_with_missing_files = set()

    for image_id in referenced_ids_with_metadata:
        relative_path = normalize_relative_path(
            image_id_to_path[image_id]
        )

        expected_file = IMAGE_DIR / Path(relative_path)

        if expected_file.is_file():
            referenced_ids_with_existing_files.add(image_id)
        else:
            referenced_ids_with_missing_files.add(image_id)

    # Extracted files that are not referenced by listing metadata.
    referenced_paths = {
        normalize_relative_path(image_id_to_path[image_id])
        for image_id in referenced_ids_with_metadata
    }

    extracted_unreferenced_paths = (
        extracted_relative_paths - referenced_paths
    )

    # ------------------------------------------------------------
    # 5. Image metadata integrity
    # ------------------------------------------------------------

    duplicate_image_metadata_ids = (
        len(image_id_to_path) -
        len(set(image_id_to_path.keys()))
    )

    # ------------------------------------------------------------
    # 6. Extension inventory
    # ------------------------------------------------------------

    extension_counts = Counter(
        p.suffix.lower()
        for p in image_files
    )

    # ------------------------------------------------------------
    # 7. Build report
    # ------------------------------------------------------------

    report = {
        "validator": "abo_raw_validation",
        "validator_version": "v002",
        "dataset": "Amazon Berkeley Objects (ABO)",

        "metadata_format": {
            "listing_metadata": "JSONL",
            "compression": "gzip",
            "image_metadata": "CSV",
            "image_metadata_compression": "gzip",
        },

        "metadata": {
            "file_count": len(metadata_files),
            "record_count": total_records,
            "malformed_lines": malformed_lines,
            "empty_lines": empty_lines,

            "unique_item_ids": len(
                {
                    key[0]
                    for key in item_listing_keys
                    if key[0] is not None
                }
            ),

            "unique_listing_keys": len(item_listing_keys),
            "duplicate_listing_keys": len(
                duplicate_item_listing_keys
            ),

            "missing_item_id": missing_item_id,
            "missing_main_image_id": missing_main_image_id,

            "records_with_other_images": records_with_other_images,
            "other_image_reference_count": other_image_reference_count,

            "unique_referenced_image_ids": len(
                referenced_image_ids
            ),

            "field_presence_counts": dict(
                sorted(field_counts.items())
            ),

            "files": per_file,
        },

        "image_metadata": {
            "file": str(IMAGE_METADATA_PATH),
            "unique_image_ids": len(image_id_to_path),
            "duplicate_image_metadata_ids": (
                duplicate_image_metadata_ids
            ),
            "referenced_ids_with_metadata": len(
                referenced_ids_with_metadata
            ),
            "referenced_ids_without_metadata": len(
                referenced_ids_without_metadata
            ),
            "missing_metadata_examples": sorted(
                referenced_ids_without_metadata
            )[:20],
        },

        "images": {
            "extracted_file_count": len(image_files),

            "referenced_ids_with_existing_files": len(
                referenced_ids_with_existing_files
            ),

            "referenced_ids_with_missing_files": len(
                referenced_ids_with_missing_files
            ),

            "missing_file_examples": sorted(
                referenced_ids_with_missing_files
            )[:20],

            "extracted_unreferenced_file_count": len(
                extracted_unreferenced_paths
            ),

            "extracted_unreferenced_examples": sorted(
                extracted_unreferenced_paths
            )[:20],

            "extension_counts": dict(
                sorted(extension_counts.items())
            ),
        },

        "validation": {
            "metadata_jsonl_valid": (
                malformed_lines == 0
            ),

            "item_ids_present": (
                missing_item_id == 0
            ),

            "image_metadata_available": (
                IMAGE_METADATA_PATH.exists()
            ),

            "referenced_image_ids_resolvable": (
                len(referenced_ids_without_metadata) == 0
            ),

            "image_inventory_available": (
                len(image_files) > 0
            ),
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

    # ------------------------------------------------------------
    # 8. Console summary
    # ------------------------------------------------------------

    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(
        f"Metadata records                    : "
        f"{total_records:,}"
    )

    print(
        f"Unique item IDs                     : "
        f"{report['metadata']['unique_item_ids']:,}"
    )

    print(
        f"Unique listing keys                 : "
        f"{len(item_listing_keys):,}"
    )

    print(
        f"Duplicate listing keys              : "
        f"{len(duplicate_item_listing_keys):,}"
    )

    print(
        f"Malformed JSONL lines               : "
        f"{malformed_lines:,}"
    )

    print(
        f"Missing item_id                     : "
        f"{missing_item_id:,}"
    )

    print(
        f"Missing main_image_id               : "
        f"{missing_main_image_id:,}"
    )

    print(
        f"Unique referenced image IDs         : "
        f"{len(referenced_image_ids):,}"
    )

    print(
        f"Image metadata image IDs            : "
        f"{len(image_id_to_path):,}"
    )

    print(
        f"Referenced IDs resolved by metadata : "
        f"{len(referenced_ids_with_metadata):,}"
    )

    print(
        f"Referenced IDs without metadata     : "
        f"{len(referenced_ids_without_metadata):,}"
    )

    print(
        f"Extracted image files               : "
        f"{len(image_files):,}"
    )

    print(
        f"Referenced IDs with physical files  : "
        f"{len(referenced_ids_with_existing_files):,}"
    )

    print(
        f"Referenced IDs missing physically   : "
        f"{len(referenced_ids_with_missing_files):,}"
    )

    print(
        f"Extracted unreferenced files        : "
        f"{len(extracted_unreferenced_paths):,}"
    )

    print()
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()