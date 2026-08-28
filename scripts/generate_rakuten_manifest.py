import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "rakuten-data-challenge"
    / "rdc-catalog-train.tsv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "rakuten-data-challenge"
    / "rdc-catalog-test.tsv"
)

SCHEMA_FILE = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "rakuten"
    / "schema_v001.json"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "rakuten_raw_validation.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "rakuten"
    / "manifest_v001.json"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def file_metadata(path: Path) -> dict:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    print("=" * 70)
    print("RAKUTEN DATASET MANIFEST GENERATION")
    print("=" * 70)

    required_files = [
        TRAIN_FILE,
        TEST_FILE,
        SCHEMA_FILE,
        VALIDATION_FILE,
    ]

    for path in required_files:
        if not path.is_file():
            print(f"[FAIL] Required file not found: {path}")
            return 1

    validation = json.loads(
        VALIDATION_FILE.read_text(encoding="utf-8")
    )

    manifest = {
        "manifest_name": "rakuten_dataset_manifest",
        "manifest_version": "v001",

        "dataset": {
            "name": "Rakuten 2018",
            "role": "hierarchical_categorization",
        },

        "source": {
            "type": "local_dataset",
            "train": file_metadata(TRAIN_FILE),
            "test": file_metadata(TEST_FILE),
        },

        "raw_structure": {
            "train": {
                "format": "TSV",
                "header_present": False,
                "columns": [
                    "title",
                    "category_id_path",
                ],
                "record_count": validation["train"]["total_rows"],
            },
            "test": {
                "format": "TSV",
                "header_present": True,
                "columns": [
                    "Title",
                    "CategoryIdPath",
                ],
                "record_count": validation["test"]["total_data_rows"],
            },
        },

        "hierarchy": {
            "separator": ">",
            "identifier_type": "anonymized_numeric_category_id",
            "root_category_count": 14,
            "observed_depth_range": [
                1,
                8
            ],
            "train_unique_complete_paths": validation[
                "train"
            ]["unique_complete_category_paths"],
            "test_unique_complete_paths": validation[
                "test"
            ]["unique_complete_category_paths"],
        },

        "validation": {
            "validation_report": str(
                VALIDATION_FILE.relative_to(PROJECT_ROOT)
            ),
            "malformed_train_rows": validation[
                "train"
            ]["malformed_rows"],
            "malformed_test_rows": validation[
                "test"
            ]["malformed_rows"],
            "empty_train_titles": validation[
                "train"
            ]["empty_titles"],
            "empty_test_titles": validation[
                "test"
            ]["empty_titles"],
            "empty_train_category_paths": validation[
                "train"
            ]["empty_category_paths"],
            "empty_test_category_paths": validation[
                "test"
            ]["empty_category_paths"],
            "invalid_train_category_paths": validation[
                "train"
            ]["invalid_category_paths"],
            "invalid_test_category_paths": validation[
                "test"
            ]["invalid_category_paths"],
        },

        "schema": {
            "path": str(
                SCHEMA_FILE.relative_to(PROJECT_ROOT)
            ),
            "version": "v001",
            "sha256": sha256_file(SCHEMA_FILE),
        },

        "processing": {
            "processing_version": "v001",
            "status": "raw_validated_not_processed",
        },

        "generated": {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Manifest generated successfully.")
    print(f"Output: {OUTPUT_FILE}")
    print()
    print("Train SHA-256:")
    print(manifest["source"]["train"]["sha256"])
    print()
    print("Test SHA-256:")
    print(manifest["source"]["test"]["sha256"])
    print()
    print("Schema SHA-256:")
    print(manifest["schema"]["sha256"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

