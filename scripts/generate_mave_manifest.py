import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = PROJECT_ROOT / "mave_audit" / "mave_final_dataset.jsonl"
VALIDATION = PROJECT_ROOT / "reports" / "validation" / "mave_final_validation.json"
OUTPUT = PROJECT_ROOT / "data" / "manifests" / "mave" / "manifest_v001.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    print("=" * 70)
    print("MAVE DATASET MANIFEST GENERATION")
    print("=" * 70)

    size = DATASET.stat().st_size

    print(f"Dataset size : {size:,} bytes")
    print("Calculating checksum...")

    checksum = sha256(DATASET)

    report = {}
    if VALIDATION.exists():
        report = json.loads(
            VALIDATION.read_text(encoding="utf-8-sig")
        )

    manifest = {
        "manifest_name": "mave_dataset_manifest",
        "manifest_version": "v001",

        "dataset": {
            "name": "MAVE",
            "role": "multimodal_product_intelligence",
            "artifact": "historically_reconstructed_final_dataset"
        },

        "source": {
            "type": "local_dataset",
            "path": "mave_audit\\mave_final_dataset.jsonl",
            "size_bytes": size,
            "sha256": checksum
        },

        "raw_structure": {
            "format": "JSONL",
            "compression": "none",
            "record_count": report.get("record_count"),
            "unique_asins": report.get("unique_asins"),
            "fields": [
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
                "mave_information_available"
            ]
        },

        "mave_information": {
            "records_with_mave_information":
                report.get("mave_information_available"),
            "records_with_mave_category":
                report.get("mave_category_present"),
            "records_with_mave_attributes":
                report.get("mave_attributes_present")
        },

        "validation": {
            "report": "reports\\validation\\mave_final_validation.json",
            "malformed_jsonl": report.get("malformed"),
            "missing_asin": report.get("missing_asin"),
            "duplicate_asin": report.get("duplicate_asin")
        },

        "processing": {
            "processing_version": "v001",
            "status": "validated_not_processed"
        },

        "generated": {
            "timestamp_utc":
                datetime.now(timezone.utc).isoformat()
        }
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Records      : {report.get('record_count'):,}")
    print(f"Unique ASINs : {report.get('unique_asins'):,}")
    print(f"SHA256       : {checksum}")
    print()
    print(f"Manifest written to: {OUTPUT}")


if __name__ == "__main__":
    main()
