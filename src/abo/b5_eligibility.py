from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

B3_DIR = PROJECT_ROOT / "data" / "representations" / "abo" / "b3"
B4_DIR = PROJECT_ROOT / "data" / "models" / "abo" / "b4.2"

REPORT_DIR = PROJECT_ROOT / "reports" / "fusion" / "abo"
B5_DATA_DIR = PROJECT_ROOT / "data" / "models" / "abo" / "b5"

B3_VERSION = "v002"
B4_VERSION = "v002"
B5_VERSION = "v001"

TARGET_FIELD = "product_type"

SPLITS = ("train", "validation", "test")

EXPECTED_COUNTS = {
    "train": 70284,
    "validation": 69996,
    "test": 7422,
}

TEXT_MODEL = B4_DIR / "text_model.joblib"
IMAGE_MODEL = B4_DIR / "image_model.joblib"


def fail(message: str):
    print(f"[B5.2] FAIL: {message}")
    raise SystemExit(1)


def normalize_label(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    value = str(value).strip()
    return value if value else None

def get_target_label(record):
    """
    Extract the actual product_type label from the B3 v002 structure.

    Example B3 value:
        [{"value": "SHOES"}]

    B4.2 label space:
        "SHOES"
    """
    value = record.get(TARGET_FIELD)

    if value is None:
        return None

    # Canonical B3 v002 representation:
    # [{"value": "SHOES"}]
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue

            label = item.get("value")

            if label is None:
                continue

            label = str(label).strip()

            if label:
                return label

        return None

    # Defensive handling
    if isinstance(value, dict):
        label = value.get("value")

        if label is None:
            return None

        label = str(label).strip()

        return label if label else None

    # Defensive handling if already flattened
    if isinstance(value, str):
        label = value.strip()
        return label if label else None

    return None


def get_text_representation(record):
    """
    Read the B3 v002 text representation exactly as produced by B3.

    B3 v002 stores combined_tokens as a LIST of tokens, not as a string.

    Eligibility requires:
      - b3_representation exists
      - text exists
      - combined_tokens is a list
      - combined_tokens is non-empty
    """
    b3 = record.get("b3_representation")

    if not isinstance(b3, dict):
        return None

    text = b3.get("text")

    if not isinstance(text, dict):
        return None

    combined_tokens = text.get("combined_tokens")

    if not isinstance(combined_tokens, list):
        return None

    if not combined_tokens:
        return None

    return combined_tokens


def get_image_representation(record):
    """
    Read the B3 v002 main-image representation.

    Eligibility requires:
      - image representation exists
      - main image exists
      - image_id exists
      - relative_path exists
      - physical_file_available is True
      - usable_for_local_image_model is True
    """
    b3 = record.get("b3_representation")

    if not isinstance(b3, dict):
        return None

    image = b3.get("image")

    if not isinstance(image, dict):
        return None

    main = image.get("main")

    if not isinstance(main, dict):
        return None

    image_id = main.get("image_id")
    relative_path = main.get("relative_path")
    physical_available = main.get("physical_file_available")
    usable = main.get("usable_for_local_image_model")

    if not image_id:
        return None

    if not relative_path:
        return None

    if physical_available is not True:
        return None

    if usable is not True:
        return None

    return main


def check_physical_image(main_image):
    """
    Verify the exact B3 v002 relative path against the known
    extracted ABO image root.

    No searching.
    No path guessing.
    No downloading.
    No substitution.
    """
    relative_path = main_image.get("relative_path")

    if not isinstance(relative_path, str):
        return False

    image_root = (
        PROJECT_ROOT
        / "ABO_Audit"
        / "images"
        / "small"
    )

    candidate = image_root / Path(relative_path)

    try:
        candidate.resolve().relative_to(image_root.resolve())
    except ValueError:
        return False

    return candidate.is_file()


def load_model_bundle(path: Path, name: str):
    if not path.is_file():
        fail(f"missing {name}: {path}")

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        fail(f"{name} is not a dictionary artifact.")

    return bundle


def validate_model_bundle(bundle, name):
    if bundle.get("representation_version") != B3_VERSION:
        fail(
            f"{name} representation_version mismatch: "
            f"{bundle.get('representation_version')!r}"
        )

    if bundle.get("b4_version") != B4_VERSION:
        fail(
            f"{name} b4_version mismatch: "
            f"{bundle.get('b4_version')!r}"
        )

    if bundle.get("target_field") != TARGET_FIELD:
        fail(
            f"{name} target_field mismatch: "
            f"{bundle.get('target_field')!r}"
        )

    label_encoder = bundle.get("label_encoder")

    if label_encoder is None or not hasattr(label_encoder, "classes_"):
        fail(f"{name} has no usable LabelEncoder.")

    classifier = bundle.get("classifier")

    if classifier is None or not hasattr(classifier, "classes_"):
        fail(f"{name} has no usable classifier.")

    if len(label_encoder.classes_) != len(classifier.classes_):
        fail(
            f"{name} label_encoder/classifier class-count mismatch: "
            f"{len(label_encoder.classes_)} vs "
            f"{len(classifier.classes_)}"
        )


def analyze_split(
    split: str,
    text_classes: set[str],
    image_classes: set[str],
):
    path = B3_DIR / f"{split}.jsonl"

    if not path.is_file():
        fail(f"missing B3 split: {path}")

    expected = EXPECTED_COUNTS[split]

    total_records = 0

    split_field_correct = 0
    split_field_missing = 0
    split_field_mismatch = 0

    valid_target = 0
    invalid_target = 0

    usable_text = 0
    unusable_text = 0

    usable_image = 0
    unusable_image = 0

    paired_b3 = 0
    paired_with_common_target = 0

    target_in_text_only = 0
    target_in_image_only = 0
    target_in_neither = 0

    record_ids_seen = set()
    duplicate_record_ids = 0
    missing_record_ids = 0

    target_counter = Counter()
    eligible_target_counter = Counter()

    reason_counter = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(
                    f"{split}: invalid JSON at line "
                    f"{line_number}: {exc}"
                )

            total_records += 1

            # ----------------------------------------------------------
            # Record ID integrity
            # ----------------------------------------------------------
            record_id = record.get("record_id")

            if record_id is None:
                missing_record_ids += 1
                reason_counter["missing_record_id"] += 1
            else:
                record_id = str(record_id)

                if record_id in record_ids_seen:
                    duplicate_record_ids += 1

                record_ids_seen.add(record_id)

            # ----------------------------------------------------------
            # Existing split-field integrity
            #
            # IMPORTANT:
            # We do NOT require a "frozen_split" field.
            # Frozen means the Phase A artifact itself is being reused.
            # ----------------------------------------------------------
            record_split = record.get("split")

            if record_split is None:
                split_field_missing += 1
                reason_counter["missing_split_field"] += 1
            elif str(record_split) == split:
                split_field_correct += 1
            else:
                split_field_mismatch += 1
                reason_counter["split_field_mismatch"] += 1

            # ----------------------------------------------------------
            # Target
            # ----------------------------------------------------------
            target = get_target_label(record)

            if target is None:
                invalid_target += 1
                reason_counter["invalid_target"] += 1
            else:
                valid_target += 1
                target_counter[target] += 1

            # ----------------------------------------------------------
            # B3 text
            # ----------------------------------------------------------
            text_representation = get_text_representation(record)

            if text_representation is not None:
                usable_text += 1
            else:
                unusable_text += 1
                reason_counter["unusable_text"] += 1

            # ----------------------------------------------------------
            # B3 image
            # ----------------------------------------------------------
            image_representation = get_image_representation(record)

            image_usable = False

            if image_representation is not None:
                image_usable = check_physical_image(
                    image_representation
                )

            if image_usable:
                usable_image += 1
            else:
                unusable_image += 1
                reason_counter["unusable_main_image"] += 1

            # ----------------------------------------------------------
            # Paired B5 population
            # ----------------------------------------------------------
            if text_representation is not None and image_usable:
                paired_b3 += 1

                if target is None:
                    reason_counter["paired_invalid_target"] += 1
                    continue

                in_text = target in text_classes
                in_image = target in image_classes

                if in_text and in_image:
                    paired_with_common_target += 1
                    eligible_target_counter[target] += 1

                elif in_text and not in_image:
                    target_in_text_only += 1
                    reason_counter[
                        "target_absent_from_image_model"
                    ] += 1

                elif not in_text and in_image:
                    target_in_image_only += 1
                    reason_counter[
                        "target_absent_from_text_model"
                    ] += 1

                else:
                    target_in_neither += 1
                    reason_counter[
                        "target_absent_from_both_models"
                    ] += 1

    # --------------------------------------------------------------
    # Required Phase A split counts
    # --------------------------------------------------------------
    if total_records != expected:
        fail(
            f"{split}: expected {expected} records but found "
            f"{total_records}"
        )

    # --------------------------------------------------------------
    # Existing split field must agree with file
    # --------------------------------------------------------------
    if split_field_missing:
        fail(
            f"{split}: {split_field_missing} records are missing "
            "the existing 'split' field."
        )

    if split_field_mismatch:
        fail(
            f"{split}: {split_field_mismatch} records have a "
            "split field different from the expected split."
        )

    # --------------------------------------------------------------
    # Record-ID integrity within split
    # --------------------------------------------------------------
    if missing_record_ids:
        fail(
            f"{split}: {missing_record_ids} records have no "
            "record_id."
        )

    if duplicate_record_ids:
        fail(
            f"{split}: {duplicate_record_ids} duplicate record IDs "
            "detected."
        )

    return {
        "split": split,
        "b3_record_count": total_records,
        "expected_record_count": expected,
        "split_field_correct_count": split_field_correct,
        "split_field_missing_count": split_field_missing,
        "split_field_mismatch_count": split_field_mismatch,
        "valid_target_count": valid_target,
        "invalid_target_count": invalid_target,
        "usable_b3_text_count": usable_text,
        "unusable_b3_text_count": unusable_text,
        "usable_b3_main_image_count": usable_image,
        "unusable_b3_main_image_count": unusable_image,
        "paired_b3_text_and_image_count": paired_b3,
        "paired_common_target_count": paired_with_common_target,
        "paired_target_absent_from_image_model_count": (
            target_in_text_only
        ),
        "paired_target_absent_from_text_model_count": (
            target_in_image_only
        ),
        "paired_target_absent_from_both_models_count": (
            target_in_neither
        ),
        "paired_population_percent_of_split": (
            100.0 * paired_b3 / total_records
            if total_records
            else 0.0
        ),
        "common_target_population_percent_of_paired": (
            100.0 * paired_with_common_target / paired_b3
            if paired_b3
            else 0.0
        ),
        "target_class_count_observed": len(target_counter),
        "eligible_target_class_count": len(eligible_target_counter),
        "target_class_counts": {
            str(k): int(v)
            for k, v in target_counter.items()
        },
        "eligible_target_class_counts": {
            str(k): int(v)
            for k, v in eligible_target_counter.items()
        },
        "reason_counts": dict(reason_counter),
    }


def collect_common_eligible_ids(
    split: str,
    common_classes: set[str],
):
    path = B3_DIR / f"{split}.jsonl"

    ids = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            text_rep = get_text_representation(record)
            image_rep = get_image_representation(record)

            if text_rep is None:
                continue

            if image_rep is None:
                continue

            if not check_physical_image(image_rep):
                continue

            target = get_target_label(record)

            if target is None:
                continue

            if target not in common_classes:
                continue

            record_id = record.get("record_id")

            if record_id is not None:
                ids.add(str(record_id))

    return ids


def main():
    print("=" * 72)
    print("B5.2 PAIRED MULTIMODAL ELIGIBILITY AUDIT")
    print("=" * 72)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"B3 version   : {B3_VERSION}")
    print(f"B4 version   : {B4_VERSION}")
    print(f"B5 version   : {B5_VERSION}")
    print(f"Target       : {TARGET_FIELD}")
    print()

    # --------------------------------------------------------------
    # Load frozen B4.2 artifacts
    # --------------------------------------------------------------
    text_bundle = load_model_bundle(
        TEXT_MODEL,
        "B4.2 text model",
    )

    image_bundle = load_model_bundle(
        IMAGE_MODEL,
        "B4.2 image model",
    )

    validate_model_bundle(
        text_bundle,
        "B4.2 text model",
    )

    validate_model_bundle(
        image_bundle,
        "B4.2 image model",
    )

    text_classes = {
        normalize_label(x)
        for x in text_bundle["label_encoder"].classes_
    }

    image_classes = {
        normalize_label(x)
        for x in image_bundle["label_encoder"].classes_
    }

    common_classes = text_classes.intersection(image_classes)

    text_only_classes = text_classes - image_classes
    image_only_classes = image_classes - text_classes

    print(
        f"[INFO] Text model classes : {len(text_classes)}"
    )
    print(
        f"[INFO] Image model classes: {len(image_classes)}"
    )
    print(
        f"[INFO] Common classes     : {len(common_classes)}"
    )
    print(
        f"[INFO] Text-only classes  : {len(text_only_classes)}"
    )
    print(
        f"[INFO] Image-only classes : {len(image_only_classes)}"
    )
    print()

    # --------------------------------------------------------------
    # Audit each frozen B3 split
    # --------------------------------------------------------------
    split_results = {}

    for split in SPLITS:
        print(f"[INFO] Auditing {split}...")

        result = analyze_split(
            split=split,
            text_classes=text_classes,
            image_classes=image_classes,
        )

        split_results[split] = result

        print(
            f"       B3 records        : "
            f"{result['b3_record_count']}"
        )

        print(
            f"       split field valid : "
            f"{result['split_field_correct_count']}"
        )

        print(
            f"       usable text       : "
            f"{result['usable_b3_text_count']}"
        )

        print(
            f"       usable main image : "
            f"{result['usable_b3_main_image_count']}"
        )

        print(
            f"       paired            : "
            f"{result['paired_b3_text_and_image_count']}"
        )

        print(
            f"       common-target     : "
            f"{result['paired_common_target_count']}"
        )

        print()

    # --------------------------------------------------------------
    # Cross-split leakage audit on final B5 common population
    # --------------------------------------------------------------
    split_ids = {}

    for split in SPLITS:
        print(
            f"[INFO] Building common eligible ID set: {split}"
        )

        split_ids[split] = collect_common_eligible_ids(
            split,
            common_classes,
        )

        print(
            f"       IDs: {len(split_ids[split])}"
        )

    train_val = (
        split_ids["train"]
        .intersection(split_ids["validation"])
    )

    train_test = (
        split_ids["train"]
        .intersection(split_ids["test"])
    )

    val_test = (
        split_ids["validation"]
        .intersection(split_ids["test"])
    )

    cross_split_leakage = {
        "train_validation": len(train_val),
        "train_test": len(train_test),
        "validation_test": len(val_test),
    }

    if any(cross_split_leakage.values()):
        fail(
            "cross-split record-ID leakage detected in B5 "
            f"common eligible population: "
            f"{cross_split_leakage}"
        )

    print()
    print(
        "[PASS] No cross-split record-ID leakage "
        "in B5 common eligible population."
    )

    # --------------------------------------------------------------
    # Integrity status
    # --------------------------------------------------------------
    integrity_pass = True

    for split in SPLITS:
        result = split_results[split]

        if result["b3_record_count"] != EXPECTED_COUNTS[split]:
            integrity_pass = False

        if result["split_field_correct_count"] != EXPECTED_COUNTS[split]:
            integrity_pass = False

        if result["split_field_missing_count"] != 0:
            integrity_pass = False

        if result["split_field_mismatch_count"] != 0:
            integrity_pass = False

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------
    report = {
        "report": "abo_b5.2_paired_multimodal_eligibility",
        "b5_version": B5_VERSION,
        "stage": "B5.2",
        "status": "PASS" if integrity_pass else "FAIL",
        "dataset": "ABO",
        "target_field": TARGET_FIELD,
        "target_definition": "product_type",
        "b3_representation_version": B3_VERSION,
        "b4_version": B4_VERSION,

        "frozen_split_policy": {
            "meaning": (
                "Use the existing Phase A split artifacts "
                "without regeneration."
            ),
            "record_level_frozen_split_field_required": False,
            "record_split_field_validated_against_filename": True,
            "split_regeneration": False,
        },

        "class_alignment": {
            "method": "actual_product_type_label_intersection",
            "text_class_count": len(text_classes),
            "image_class_count": len(image_classes),
            "common_class_count": len(common_classes),
            "text_only_class_count": len(text_only_classes),
            "image_only_class_count": len(image_only_classes),
            "text_only_classes": sorted(
                [str(x) for x in text_only_classes]
            ),
            "image_only_classes": sorted(
                [str(x) for x in image_only_classes]
            ),
        },

        "eligibility_definition": {
            "same_record": True,
            "same_frozen_split": True,
            "existing_split_field_matches_file": True,
            "b3_text_combined_tokens_nonempty": True,
            "b3_main_image_reference_present": True,
            "b3_main_image_physical_file_available": True,
            "b3_main_image_usable_for_local_image_model": True,
            "exact_b3_relative_path_exists_locally": True,
            "valid_product_type": True,
            "product_type_in_text_model_class_space": True,
            "product_type_in_image_model_class_space": True,
        },

        "split_results": split_results,

        "final_population": {
            split: {
                "paired_b3_text_and_image_count": result[
                    "paired_b3_text_and_image_count"
                ],
                "paired_common_target_count": result[
                    "paired_common_target_count"
                ],
            }
            for split, result in split_results.items()
        },

        "cross_split_record_id_leakage": cross_split_leakage,

        "integrity": {
            "expected_split_counts_preserved": True,
            "existing_split_field_consistent": True,
            "cross_split_leakage_free": True,
            "b2_modified": False,
            "b3_modified": False,
            "b4_retrained": False,
            "images_downloaded": False,
            "paths_guessed": False,
            "synthetic_images_used": False,
            "fusion_performed": False,
        },

        "next_stage": {
            "stage": "B5.3",
            "description": (
                "Prepare aligned text/image prediction spaces "
                "for the common paired evaluation population."
            ),
            "training": False,
        },
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    B5_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        REPORT_DIR
        / "abo_b5.2_eligibility_v001.json"
    )

    with output.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("B5.2 RESULT")
    print("=" * 72)

    for split in SPLITS:
        result = split_results[split]

        print(
            f"{split:12s} "
            f"records={result['b3_record_count']:>6d}  "
            f"paired={result['paired_b3_text_and_image_count']:>6d}  "
            f"common-target={result['paired_common_target_count']:>6d}"
        )

    print()
    print(
        f"Common class space: {len(common_classes)} classes"
    )

    print(
        "Cross-split leakage: "
        f"{sum(cross_split_leakage.values())}"
    )

    print()
    print("[PASS] Report written to:")
    print(f"       {output}")
    print()
    print("=" * 72)
    print("B5.2 ELIGIBILITY AUDIT PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
