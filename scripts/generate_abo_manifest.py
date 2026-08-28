import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR = PROJECT_ROOT / "ABO_Audit" / "listings" / "metadata"
IMAGE_METADATA = PROJECT_ROOT / "ABO_Audit" / "images" / "metadata" / "images.csv.gz"
IMAGE_DIR = PROJECT_ROOT / "ABO_Audit" / "images" / "small"

REPORT_PATH = PROJECT_ROOT / "reports" / "validation" / "abo_raw_validation.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "manifests" / "abo" / "manifest_v001.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("ABO DATASET MANIFEST GENERATION")
    print("=" * 70)

    for path in [
        METADATA_DIR,
        IMAGE_METADATA,
        IMAGE_DIR,
        REPORT_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    validation = json.loads(
        REPORT_PATH.read_text(encoding="utf-8-sig")
    )

    metadata = validation["metadata"]
    images = validation["images"]

    metadata_files = sorted(METADATA_DIR.glob("*.json.gz"))

    print(f"Metadata files: {len(metadata_files)}")
    print("Calculating checksums...")

    files = []

    for path in metadata_files:
        files.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    image_metadata_info = {
        "path": str(IMAGE_METADATA.relative_to(PROJECT_ROOT)),
        "size_bytes": IMAGE_METADATA.stat().st_size,
        "sha256": sha256(IMAGE_METADATA),
    }

    archive = (
        PROJECT_ROOT
        / "ABO_Audit"
        / "downloads"
        / "abo-images-small.tar"
    )

    archive_info = None

    if archive.exists():
        archive_info = {
            "path": str(archive.relative_to(PROJECT_ROOT)),
            "size_bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        }

    manifest = {
        "manifest_name": "abo_dataset_manifest",
        "manifest_version": "v001",

        "dataset": {
            "name": "Amazon Berkeley Objects (ABO)",
            "role": "multimodal_product_intelligence",
            "listing_key": [
                "item_id",
                "domain_name"
            ]
        },

        "source": {
            "type": "local_dataset",

            "listing_metadata": {
                "directory": str(
                    METADATA_DIR.relative_to(PROJECT_ROOT)
                ),
                "file_count": len(metadata_files),
                "files": files,
            },

            "image_metadata": image_metadata_info,

            "small_image_archive": archive_info,
        },

        "raw_structure": {
            "listing_metadata": {
                "format": "NDJSON",
                "compression": "gzip",
                "record_count": metadata["record_count"],
                "unique_item_ids": metadata["unique_item_ids"],
                "unique_listing_keys": metadata["unique_listing_keys"],
                "duplicate_listing_keys": metadata["duplicate_listing_keys"],
            },

            "image_metadata": {
                "format": "CSV",
                "compression": "gzip",
                "image_id_count": validation["image_metadata"][
                    "unique_image_ids"
                ],
            },
        },

        "images": {
            "unique_referenced_image_ids": metadata[
                "unique_referenced_image_ids"
            ],

            "referenced_ids_with_metadata": metadata[
                "unique_referenced_image_ids"
            ],

            "referenced_ids_without_metadata": 0,

            "extracted_file_count": images[
                "extracted_file_count"
            ],

            "referenced_ids_with_physical_files": images[
                "referenced_ids_with_existing_files"
            ],

            "referenced_ids_missing_physically": images[
                "referenced_ids_with_missing_files"
            ],

            "extracted_unreferenced_files": images[
                "extracted_unreferenced_file_count"
            ],
        },

        "validation": {
            "report": str(
                REPORT_PATH.relative_to(PROJECT_ROOT)
            ),

            "malformed_jsonl_lines": metadata[
                "malformed_lines"
            ],

            "missing_item_id": metadata[
                "missing_item_id"
            ],

            "missing_main_image_id": metadata[
                "missing_main_image_id"
            ],

            "duplicate_listing_keys": metadata[
                "duplicate_listing_keys"
            ],

            "referenced_ids_without_metadata": 0,
        },

        "processing": {
            "processing_version": "v001",
            "status": "raw_validated_not_processed",
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Listing records    : {metadata['record_count']:,}")
    print(f"Unique listing keys: {metadata['unique_listing_keys']:,}")
    print(f"Referenced images  : {metadata['unique_referenced_image_ids']:,}")
    print(f"Physical images    : {images['extracted_file_count']:,}")
    print()
    print(f"Manifest written to: {OUTPUT_PATH}")
    print()
    print("ABO manifest generation completed successfully.")


if __name__ == "__main__":
    main()

