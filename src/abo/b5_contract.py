from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

B3_DIR = PROJECT_ROOT / "data" / "representations" / "abo" / "b3"
B4_DIR = PROJECT_ROOT / "data" / "models" / "abo" / "b4.2"
REPORT_DIR = PROJECT_ROOT / "reports" / "fusion" / "abo"

B3_STATISTICS = B3_DIR / "statistics_v002.json"
B3_SCHEMA = B3_DIR / "schema_v002.json"

B3_SPLITS = {
    "train": B3_DIR / "train.jsonl",
    "validation": B3_DIR / "validation.jsonl",
    "test": B3_DIR / "test.jsonl",
}

TEXT_MODEL = B4_DIR / "text_model.joblib"
IMAGE_MODEL = B4_DIR / "image_model.joblib"
B4_STATISTICS = B4_DIR / "statistics.json"

B3_VERSION = "v002"
B4_VERSION = "v002"
B5_VERSION = "v001"

TARGET_FIELD = "product_type"

EXPECTED_SPLIT_COUNTS = {
    "train": 70284,
    "validation": 69996,
    "test": 7422,
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fail(message: str):
    print(f"[B5 CONTRACT] FAIL: {message}")
    raise SystemExit(1)


def check_file(path: Path, description: str):
    if not path.is_file():
        fail(f"missing {description}: {path}")


def main():
    print("=" * 72)
    print("B5.1 TASK & EVALUATION CONTRACT")
    print("=" * 72)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"B3 version   : {B3_VERSION}")
    print(f"B4.2 version : {B4_VERSION}")
    print(f"B5 version   : {B5_VERSION}")
    print(f"Target       : {TARGET_FIELD}")
    print()

    # ------------------------------------------------------------------
    # Required artifacts
    # ------------------------------------------------------------------
    check_file(B3_STATISTICS, "B3 statistics")
    check_file(B3_SCHEMA, "B3 schema")

    for split, path in B3_SPLITS.items():
        check_file(path, f"B3 {split} split")

    check_file(TEXT_MODEL, "B4.2 text model")
    check_file(IMAGE_MODEL, "B4.2 image model")
    check_file(B4_STATISTICS, "B4.2 statistics")

    print("[PASS] Required B3/B4.2 artifacts exist.")

    # ------------------------------------------------------------------
    # B3 statistics validation
    # ------------------------------------------------------------------
    b3_stats = load_json(B3_STATISTICS)

    representation_version = (
        b3_stats.get("representation_version")
        or b3_stats.get("version")
    )

    if representation_version != B3_VERSION:
        fail(
            f"B3 representation version mismatch: "
            f"expected {B3_VERSION}, got {representation_version!r}"
        )

    print("[PASS] B3 statistics identify representation version v002.")

    # ------------------------------------------------------------------
    # B3 schema validation
    # ------------------------------------------------------------------
    b3_schema = load_json(B3_SCHEMA)

    schema_version = (
        b3_schema.get("schema_version")
        or b3_schema.get("version")
    )

    if schema_version is not None and schema_version != B3_VERSION:
        fail(
            f"B3 schema version mismatch: "
            f"expected {B3_VERSION}, got {schema_version!r}"
        )

    print("[PASS] B3 schema is compatible with v002.")

    # ------------------------------------------------------------------
    # B4.2 model metadata validation
    # ------------------------------------------------------------------
    import joblib

    text_bundle = joblib.load(TEXT_MODEL)
    image_bundle = joblib.load(IMAGE_MODEL)

    if not isinstance(text_bundle, dict):
        fail("B4.2 text model artifact is not a dictionary bundle.")

    if not isinstance(image_bundle, dict):
        fail("B4.2 image model artifact is not a dictionary bundle.")

    required_text_keys = {
        "vectorizer",
        "label_encoder",
        "classifier",
        "target_field",
        "target_definition",
        "representation_version",
        "b4_version",
    }

    required_image_keys = {
        "label_encoder",
        "classifier",
        "target_field",
        "target_definition",
        "image_size",
        "representation_version",
        "b4_version",
        "training_epochs",
        "training_completed",
    }

    missing_text = sorted(required_text_keys - set(text_bundle))
    missing_image = sorted(required_image_keys - set(image_bundle))

    if missing_text:
        fail(f"text model missing keys: {missing_text}")

    if missing_image:
        fail(f"image model missing keys: {missing_image}")

    if text_bundle["representation_version"] != B3_VERSION:
        fail(
            "text model representation version mismatch: "
            f"{text_bundle['representation_version']!r}"
        )

    if image_bundle["representation_version"] != B3_VERSION:
        fail(
            "image model representation version mismatch: "
            f"{image_bundle['representation_version']!r}"
        )

    if text_bundle["b4_version"] != B4_VERSION:
        fail(
            "text model B4 version mismatch: "
            f"{text_bundle['b4_version']!r}"
        )

    if image_bundle["b4_version"] != B4_VERSION:
        fail(
            "image model B4 version mismatch: "
            f"{image_bundle['b4_version']!r}"
        )

    if text_bundle["target_field"] != TARGET_FIELD:
        fail(
            "text model target mismatch: "
            f"{text_bundle['target_field']!r}"
        )

    if image_bundle["target_field"] != TARGET_FIELD:
        fail(
            "image model target mismatch: "
            f"{image_bundle['target_field']!r}"
        )

    if not image_bundle["training_completed"]:
        fail("B4.2 image model is marked training_completed=false.")

    text_classes = list(text_bundle["label_encoder"].classes_)
    image_classes = list(image_bundle["label_encoder"].classes_)

    common_classes = sorted(
        set(text_classes).intersection(image_classes),
        key=str,
    )

    text_only_classes = sorted(
        set(text_classes) - set(image_classes),
        key=str,
    )

    image_only_classes = sorted(
        set(image_classes) - set(text_classes),
        key=str,
    )

    print(
        f"[PASS] B4.2 text class space : {len(text_classes)} classes"
    )
    print(
        f"[PASS] B4.2 image class space: {len(image_classes)} classes"
    )
    print(
        f"[INFO] Common class space     : {len(common_classes)} classes"
    )
    print(
        f"[INFO] Text-only classes      : {len(text_only_classes)}"
    )
    print(
        f"[INFO] Image-only classes     : {len(image_only_classes)}"
    )

    # ------------------------------------------------------------------
    # Frozen split contract
    # ------------------------------------------------------------------
    print()
    print("[INFO] Frozen split counts expected:")
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        print(f"       {split}: {expected}")

    # ------------------------------------------------------------------
    # Contract definition
    # ------------------------------------------------------------------
    contract = {
        "report": "abo_b5_contract",
        "b5_version": B5_VERSION,
        "stage": "B5.1",
        "status": "PASS",
        "dataset": "ABO",
        "target_field": TARGET_FIELD,
        "target_definition": "product_type",
        "b3_representation_version": B3_VERSION,
        "b4_version": B4_VERSION,
        "split_policy": {
            "use_existing_frozen_splits": True,
            "regenerate_splits": False,
            "expected_counts": EXPECTED_SPLIT_COUNTS,
        },
        "input_artifacts": {
            "b3_statistics": str(B3_STATISTICS.relative_to(PROJECT_ROOT)),
            "b3_schema": str(B3_SCHEMA.relative_to(PROJECT_ROOT)),
            "b4_text_model": str(TEXT_MODEL.relative_to(PROJECT_ROOT)),
            "b4_image_model": str(IMAGE_MODEL.relative_to(PROJECT_ROOT)),
            "b4_statistics": str(B4_STATISTICS.relative_to(PROJECT_ROOT)),
        },
        "b4_class_spaces": {
            "text_class_count": len(text_classes),
            "image_class_count": len(image_classes),
            "common_class_count": len(common_classes),
            "text_only_class_count": len(text_only_classes),
            "image_only_class_count": len(image_only_classes),
            "text_only_classes": [str(x) for x in text_only_classes],
            "image_only_classes": [str(x) for x in image_only_classes],
        },
        "eligibility_policy": {
            "same_record_required": True,
            "same_frozen_split_required": True,
            "usable_b3_text_required": True,
            "usable_b3_main_image_required": True,
            "valid_target_required": True,
            "target_must_exist_in_text_model_class_space": True,
            "target_must_exist_in_image_model_class_space": True,
            "class_alignment_by_label": True,
            "class_alignment_by_integer_index": False,
        },
        "fusion_policy": {
            "fusion_performed": False,
            "reserved_for": "B5.4",
        },
        "comparison_policy": {
            "common_paired_population_required": True,
            "full_population_metrics_must_not_be_compared_directly": True,
        },
        "integrity_constraints": [
            "B3 v002 only",
            "Frozen Phase A splits only",
            "No B2 modification",
            "No B3 modification",
            "No split regeneration",
            "No image downloading",
            "No path guessing",
            "No synthetic images",
            "No Rakuten integration",
            "No MAVE integration",
            "No B4.2 retraining",
            "No fusion in B5.1",
            "No fusion in B5.2",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output = REPORT_DIR / "abo_b5_contract_v001.json"

    with output.open("w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)

    print()
    print(f"[PASS] Contract written to:")
    print(f"       {output}")
    print()
    print("=" * 72)
    print("B5.1 CONTRACT PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
