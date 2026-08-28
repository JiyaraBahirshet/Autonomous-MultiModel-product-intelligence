import csv
import gzip
import json
import hashlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "ABO_Audit" / "listings" / "metadata"
IMAGE_METADATA_PATH = (
    PROJECT_ROOT / "ABO_Audit" / "images" / "metadata" / "images.csv.gz"
)
IMAGE_DIR = PROJECT_ROOT / "ABO_Audit" / "images" / "small"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "abo"

LISTINGS_OUTPUT = OUTPUT_DIR / "listings.jsonl"
IMAGE_INVENTORY_OUTPUT = OUTPUT_DIR / "image_inventory.jsonl"
SCHEMA_OUTPUT = OUTPUT_DIR / "schema_v001.json"
STATISTICS_OUTPUT = OUTPUT_DIR / "statistics.json"


TEXT_FIELDS = {
    "brand",
    "item_name",
    "bullet_point",
    "color",
    "fabric_type",
    "finish_type",
    "item_keywords",
    "item_shape",
    "material",
    "model_name",
    "pattern",
    "product_description",
    "style",
}


def normalize_text(value: str) -> str:
    """Deterministically normalize whitespace without changing content."""
    return " ".join(value.split())


def normalize_value(value: Any) -> Any:
    """
    Recursively normalize textual values while preserving
    the original ABO structural representation.
    """
    if isinstance(value, str):
        return normalize_text(value)

    if isinstance(value, list):
        return [normalize_value(item) for item in value]

    if isinstance(value, dict):
        return {
            key: normalize_value(item)
            for key, item in value.items()
        }

    return value


def load_image_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load authoritative ABO image metadata."""
    mapping = {}

    with gzip.open(path, "rt", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        expected_fields = {
            "image_id",
            "height",
            "width",
            "path",
        }

        if not expected_fields.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Unexpected image metadata columns: "
                f"{reader.fieldnames}"
            )

        for row in reader:
            image_id = row.get("image_id")
            relative_path = row.get("path")

            if not image_id or not relative_path:
                continue

            mapping[str(image_id)] = {
                "image_id": str(image_id),
                "height": int(row["height"]),
                "width": int(row["width"]),
                "relative_path": relative_path.replace("\\", "/"),
            }

    return mapping


def build_physical_inventory(
    image_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Determine which image metadata entries have corresponding
    physical files in the extracted image directory.
    """
    inventory = {}

    for image_id, metadata in image_metadata.items():
        relative_path = metadata["relative_path"]
        physical_path = IMAGE_DIR / Path(relative_path)

        inventory[image_id] = {
            **metadata,
            "physical_file_available": physical_path.is_file(),
        }

    return inventory


def image_reference(
    image_id: str | None,
    image_inventory: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Build a processed image reference."""
    if not image_id:
        return None

    image_id = str(image_id)

    metadata = image_inventory.get(image_id)

    if metadata is None:
        return {
            "image_id": image_id,
            "metadata_available": False,
            "physical_file_available": False,
            "relative_path": None,
        }

    return {
        "image_id": image_id,
        "metadata_available": True,
        "physical_file_available": metadata[
            "physical_file_available"
        ],
        "relative_path": metadata["relative_path"],
        "height": metadata["height"],
        "width": metadata["width"],
    }


def make_record_id(
    split_index: int,
    item_id: str,
    domain_name: str | None,
) -> str:
    """
    Create a deterministic record identifier.

    The source identifiers remain untouched.
    """
    source = (
        f"{split_index}|"
        f"{item_id}|"
        f"{domain_name or ''}"
    )

    digest = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:16]

    return f"abo:{digest}"


def process_listing(
    record: dict[str, Any],
    split_index: int,
    image_inventory: dict[str, dict[str, Any]],
) -> dict[str, Any]:

    item_id = record.get("item_id")
    domain_name = record.get("domain_name")

    if not item_id:
        raise ValueError(
            "Encountered listing without item_id"
        )

    processed = {}

    for key, value in record.items():
        if key in {"main_image_id", "other_image_id"}:
            continue

        processed[key] = normalize_value(value)

    main_image_id = record.get("main_image_id")

    other_image_ids = record.get(
        "other_image_id",
        [],
    )

    if not isinstance(other_image_ids, list):
        other_image_ids = []

    processed["record_id"] = make_record_id(
        split_index,
        str(item_id),
        domain_name,
    )

    processed["main_image"] = image_reference(
        main_image_id,
        image_inventory,
    )

    processed["other_images"] = [
        image_reference(
            image_id,
            image_inventory,
        )
        for image_id in other_image_ids
        if image_id
    ]

    processed["image_counts"] = {
        "referenced": (
            (1 if main_image_id else 0)
            + len(other_image_ids)
        ),
        "physical_files_available": sum(
            1
            for image_ref in [
                processed["main_image"],
                *processed["other_images"],
            ]
            if image_ref
            and image_ref["physical_file_available"]
        ),
    }

    return processed


def main() -> None:
    print("=" * 70)
    print("ABO PHASE A PREPROCESSING")
    print("=" * 70)

    metadata_files = sorted(
        METADATA_DIR.glob("*.json.gz")
    )

    if not metadata_files:
        raise FileNotFoundError(
            f"No ABO metadata files found: {METADATA_DIR}"
        )

    if not IMAGE_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Image metadata not found: "
            f"{IMAGE_METADATA_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Loading image metadata...")

    image_metadata = load_image_metadata(
        IMAGE_METADATA_PATH
    )

    print(
        f"  Image metadata records: "
        f"{len(image_metadata):,}"
    )

    print()
    print("Building image inventory...")

    image_inventory = build_physical_inventory(
        image_metadata
    )

    physical_image_count = sum(
        1
        for item in image_inventory.values()
        if item["physical_file_available"]
    )

    print(
        f"  Physical images available: "
        f"{physical_image_count:,}"
    )

    print()
    print("Processing listing metadata...")

    total_records = 0
    missing_main_image_id = 0
    total_image_references = 0
    image_references_with_files = 0

    split_index = 0

    with LISTINGS_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as output:

        for metadata_file in metadata_files:

            print(
                f"  Processing {metadata_file.name}..."
            )

            with gzip.open(
                metadata_file,
                "rt",
                encoding="utf-8",
            ) as file:

                for line in file:
                    line = line.strip()

                    if not line:
                        continue

                    record = json.loads(line)

                    processed = process_listing(
                        record,
                        split_index,
                        image_inventory,
                    )

                    if not record.get("main_image_id"):
                        missing_main_image_id += 1

                    total_image_references += (
                        processed["image_counts"][
                            "referenced"
                        ]
                    )

                    image_references_with_files += (
                        processed["image_counts"][
                            "physical_files_available"
                        ]
                    )

                    output.write(
                        json.dumps(
                            processed,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total_records += 1
                    split_index += 1

    print()
    print("Building image inventory artifact...")

    with IMAGE_INVENTORY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as output:

        for image_id in sorted(image_inventory):
            record = image_inventory[image_id]

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    statistics = {
        "dataset": "Amazon Berkeley Objects (ABO)",
        "processing_version": "v001",
        "total_listing_records": total_records,
        "listing_metadata_files": len(metadata_files),
        "missing_main_image_id": missing_main_image_id,
        "image_metadata_records": len(image_metadata),
        "physical_images_available": physical_image_count,
        "physical_images_missing": (
            len(image_metadata)
            - physical_image_count
        ),
        "total_image_references": total_image_references,
        "image_references_with_physical_files": (
            image_references_with_files
        ),
        "image_references_missing_physical_files": (
            total_image_references
            - image_references_with_files
        ),
    }

    STATISTICS_OUTPUT.write_text(
        json.dumps(
            statistics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = {
        "dataset": "Amazon Berkeley Objects (ABO)",
        "schema_version": "v001",
        "processing_version": "v001",
        "record_artifact": "listings.jsonl",
        "image_artifact": "image_inventory.jsonl",
        "transformations": {
            "text": (
                "deterministic whitespace normalization"
            ),
            "source_identifiers": "preserved",
            "nested_structures": (
                "preserved recursively"
            ),
            "image_references": (
                "resolved through authoritative image metadata"
            ),
            "physical_image_status": (
                "explicitly recorded"
            ),
        },
        "missing_data_policy": (
            "Missing source fields are preserved as missing; "
            "no values are fabricated."
        ),
        "raw_data_modified": False,
    }

    SCHEMA_OUTPUT.write_text(
        json.dumps(
            schema,
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
        f"Listing records              : "
        f"{total_records:,}"
    )

    print(
        f"Missing main_image_id        : "
        f"{missing_main_image_id:,}"
    )

    print(
        f"Image metadata records       : "
        f"{len(image_metadata):,}"
    )

    print(
        f"Physical images available    : "
        f"{physical_image_count:,}"
    )

    print(
        f"Physical images missing      : "
        f"{len(image_metadata) - physical_image_count:,}"
    )

    print()
    print("Outputs:")
    print(f"  {LISTINGS_OUTPUT}")
    print(f"  {IMAGE_INVENTORY_OUTPUT}")
    print(f"  {SCHEMA_OUTPUT}")
    print(f"  {STATISTICS_OUTPUT}")


if __name__ == "__main__":
    main()