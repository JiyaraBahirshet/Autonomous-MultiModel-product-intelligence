from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

B3_DIR = PROJECT_ROOT / "data" / "representations" / "abo" / "b3"
B4_DIR = PROJECT_ROOT / "data" / "models" / "abo" / "b4.2"

B5_REPORT_DIR = PROJECT_ROOT / "reports" / "fusion" / "abo"
B5_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "abo" / "b5"

B5_CONTRACT = B5_REPORT_DIR / "abo_b5_contract_v001.json"
B5_ELIGIBILITY = B5_REPORT_DIR / "abo_b5.2_eligibility_v001.json"

TEXT_MODEL = B4_DIR / "text_model.joblib"
IMAGE_MODEL = B4_DIR / "image_model.joblib"
B4_STATS = B4_DIR / "statistics.json"

B3_STATS = B3_DIR / "statistics_v002.json"

REPRESENTATION_VERSION = "v002"
B4_VERSION = "v002"
B5_VERSION = "v001"

TARGET_FIELD = "product_type"
IMAGE_SIZE = (64, 64)

EXPECTED_SPLITS = {
    "train": 70284,
    "validation": 69996,
    "test": 7422,
}

EXPECTED_COMMON_COUNTS = {
    "train": 69823,
    "validation": 69867,
    "test": 7346,
}

EXPECTED_TEXT_CLASS_COUNT = 553
EXPECTED_IMAGE_CLASS_COUNT = 549
EXPECTED_COMMON_CLASS_COUNT = 549
EXPECTED_TEXT_ONLY_CLASS_COUNT = 4
EXPECTED_IMAGE_ONLY_CLASS_COUNT = 0


def fail(message: str) -> None:
    raise RuntimeError(f"B5.3 FAILED: {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"Missing JSON artifact: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON artifact {path}: {exc}")

    if not isinstance(data, dict):
        fail(f"JSON artifact is not an object: {path}")

    return data


def get_target_label(record: dict[str, Any]) -> str | None:
    value = record.get(TARGET_FIELD)

    if value is None:
        return None

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

    if isinstance(value, dict):
        label = value.get("value")

        if label is None:
            return None

        label = str(label).strip()

        return label if label else None

    if isinstance(value, str):
        value = value.strip()

        return value if value else None

    return None


def get_text_tokens(record: dict[str, Any]) -> list[str]:
    b3 = record.get("b3_representation")

    if not isinstance(b3, dict):
        return []

    text = b3.get("text")

    if not isinstance(text, dict):
        return []

    tokens = text.get("combined_tokens")

    if not isinstance(tokens, list):
        return []

    return [
        str(token)
        for token in tokens
        if token is not None and str(token).strip()
    ]


def get_main_image_info(
    record: dict[str, Any],
) -> dict[str, Any] | None:

    b3 = record.get("b3_representation")

    if not isinstance(b3, dict):
        return None

    image = b3.get("image")

    if not isinstance(image, dict):
        return None

    main = image.get("main")

    if not isinstance(main, dict):
        return None

    return main


def resolve_image_path(
    main_image: dict[str, Any],
) -> Path | None:

    if not main_image.get("physical_file_available"):
        return None

    if not main_image.get("usable_for_local_image_model"):
        return None

    relative_path = main_image.get("relative_path")

    if not relative_path:
        return None

    image_root = (
        PROJECT_ROOT
        / "ABO_Audit"
        / "images"
        / "small"
    )

    path = image_root / str(relative_path)

    if not path.exists():
        return None

    if not path.is_file():
        return None

    return path


def load_models() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:

    if not TEXT_MODEL.exists():
        fail(f"Missing text model: {TEXT_MODEL}")

    if not IMAGE_MODEL.exists():
        fail(f"Missing image model: {IMAGE_MODEL}")

    try:
        text_bundle = joblib.load(TEXT_MODEL)
    except Exception as exc:
        fail(f"Could not load text B4.2 model: {exc}")

    try:
        image_bundle = joblib.load(IMAGE_MODEL)
    except Exception as exc:
        fail(f"Could not load image B4.2 model: {exc}")

    if not isinstance(text_bundle, dict):
        fail(
            "Text B4.2 artifact is not a model bundle dictionary."
        )

    if not isinstance(image_bundle, dict):
        fail(
            "Image B4.2 artifact is not a model bundle dictionary."
        )

    required_text = {
        "vectorizer",
        "label_encoder",
        "classifier",
        "target_field",
        "representation_version",
        "b4_version",
    }

    required_image = {
        "label_encoder",
        "classifier",
        "target_field",
        "representation_version",
        "b4_version",
    }

    missing_text = required_text - set(text_bundle)

    if missing_text:
        fail(
            "Text B4.2 bundle is missing required keys: "
            + str(sorted(missing_text))
        )

    missing_image = required_image - set(image_bundle)

    if missing_image:
        fail(
            "Image B4.2 bundle is missing required keys: "
            + str(sorted(missing_image))
        )

    if (
        text_bundle["representation_version"]
        != REPRESENTATION_VERSION
    ):
        fail("Text model is not B3 v002.")

    if (
        image_bundle["representation_version"]
        != REPRESENTATION_VERSION
    ):
        fail("Image model is not B3 v002.")

    if text_bundle["b4_version"] != B4_VERSION:
        fail("Text model is not B4.2 v002.")

    if image_bundle["b4_version"] != B4_VERSION:
        fail("Image model is not B4.2 v002.")

    if text_bundle["target_field"] != TARGET_FIELD:
        fail("Text model target field mismatch.")

    if image_bundle["target_field"] != TARGET_FIELD:
        fail("Image model target field mismatch.")

    return text_bundle, image_bundle


def model_class_labels(
    bundle: dict[str, Any],
) -> list[str]:

    encoder = bundle["label_encoder"]
    classifier = bundle["classifier"]

    if not hasattr(classifier, "classes_"):
        fail("Model classifier has no classes_ attribute.")

    classes = classifier.classes_

    try:
        labels = encoder.inverse_transform(classes)
    except Exception as exc:
        fail(
            "Could not map classifier class indices "
            f"back to product_type labels: {exc}"
        )

    labels = [str(label) for label in labels]

    if len(labels) != len(set(labels)):
        fail(
            "Model class labels are not unique."
        )

    return labels


def validate_class_alignment(
    text_bundle: dict[str, Any],
    image_bundle: dict[str, Any],
    eligibility: dict[str, Any],
) -> list[str]:

    if "class_alignment" not in eligibility:
        fail(
            "B5.2 eligibility report is missing "
            "'class_alignment'."
        )

    expected = eligibility["class_alignment"]

    if not isinstance(expected, dict):
        fail(
            "B5.2 'class_alignment' is not an object."
        )

    text_labels = model_class_labels(text_bundle)
    image_labels = model_class_labels(image_bundle)

    text_set = set(text_labels)
    image_set = set(image_labels)

    common = sorted(text_set & image_set)

    expected_text_count = expected.get(
        "text_class_count"
    )

    expected_image_count = expected.get(
        "image_class_count"
    )

    expected_common_count = expected.get(
        "common_class_count"
    )

    if expected_text_count is None:
        fail(
            "B5.2 report missing text_class_count."
        )

    if expected_image_count is None:
        fail(
            "B5.2 report missing image_class_count."
        )

    if expected_common_count is None:
        fail(
            "B5.2 report missing common_class_count."
        )

    if len(text_labels) != expected_text_count:
        fail(
            f"Text class count mismatch: "
            f"{len(text_labels)} != "
            f"{expected_text_count}"
        )

    if len(image_labels) != expected_image_count:
        fail(
            f"Image class count mismatch: "
            f"{len(image_labels)} != "
            f"{expected_image_count}"
        )

    if len(common) != expected_common_count:
        fail(
            f"Common class count mismatch: "
            f"{len(common)} != "
            f"{expected_common_count}"
        )

    expected_text_only = set(
        expected.get(
            "text_only_classes",
            [],
        )
    )

    expected_image_only = set(
        expected.get(
            "image_only_classes",
            [],
        )
    )

    actual_text_only = text_set - image_set
    actual_image_only = image_set - text_set

    if actual_text_only != expected_text_only:
        fail(
            "Text-only class set mismatch."
        )

    if actual_image_only != expected_image_only:
        fail(
            "Image-only class set mismatch."
        )

    if text_set & image_set != set(common):
        fail(
            "Class intersection validation failed."
        )

    if len(text_labels) != EXPECTED_TEXT_CLASS_COUNT:
        fail(
            f"Expected {EXPECTED_TEXT_CLASS_COUNT} text classes, "
            f"got {len(text_labels)}."
        )

    if len(image_labels) != EXPECTED_IMAGE_CLASS_COUNT:
        fail(
            f"Expected {EXPECTED_IMAGE_CLASS_COUNT} image classes, "
            f"got {len(image_labels)}."
        )

    if len(common) != EXPECTED_COMMON_CLASS_COUNT:
        fail(
            f"Expected {EXPECTED_COMMON_CLASS_COUNT} common classes, "
            f"got {len(common)}."
        )

    if (
        len(actual_text_only)
        != EXPECTED_TEXT_ONLY_CLASS_COUNT
    ):
        fail(
            "Unexpected number of text-only classes: "
            f"{len(actual_text_only)} != "
            f"{EXPECTED_TEXT_ONLY_CLASS_COUNT}"
        )

    if (
        len(actual_image_only)
        != EXPECTED_IMAGE_ONLY_CLASS_COUNT
    ):
        fail(
            "Unexpected number of image-only classes: "
            f"{len(actual_image_only)} != "
            f"{EXPECTED_IMAGE_ONLY_CLASS_COUNT}"
        )

    return common


def build_probability_column_map(
    bundle: dict[str, Any],
    common_classes: list[str],
) -> list[int]:

    labels = model_class_labels(bundle)

    index_by_label = {
        label: index
        for index, label in enumerate(labels)
    }

    missing = [
        label
        for label in common_classes
        if label not in index_by_label
    ]

    if missing:
        fail(
            "Common classes missing from model: "
            + str(missing[:10])
        )

    columns = [
        index_by_label[label]
        for label in common_classes
    ]

    if len(columns) != len(common_classes):
        fail(
            "Probability column mapping length mismatch."
        )

    if len(set(columns)) != len(columns):
        fail(
            "Probability column mapping contains duplicates."
        )

    return columns


def preprocess_image(
    path: Path,
) -> np.ndarray:

    try:
        with Image.open(path) as image:
            image = image.convert("RGB")

            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR,
            )

            array = np.asarray(
                image,
                dtype=np.float32,
            )

    except Exception as exc:
        fail(
            f"Could not preprocess image {path}: {exc}"
        )

    if array.shape != (
        IMAGE_SIZE[1],
        IMAGE_SIZE[0],
        3,
    ):
        fail(
            f"Unexpected image array shape for {path}: "
            f"{array.shape}"
        )

    array /= 255.0

    if not np.all(np.isfinite(array)):
        fail(
            f"Non-finite pixel value detected in {path}."
        )

    return array.reshape(1, -1)


def probability_vector(
    probabilities: np.ndarray,
    columns: list[int],
) -> np.ndarray:

    if not isinstance(probabilities, np.ndarray):
        fail(
            "predict_proba did not return a NumPy array."
        )

    if probabilities.ndim != 2:
        fail(
            "predict_proba returned unexpected dimensions: "
            f"{probabilities.shape}"
        )

    if probabilities.shape[0] != 1:
        fail(
            "Expected exactly one prediction row, got "
            f"{probabilities.shape[0]}."
        )

    if not columns:
        fail(
            "Probability column mapping is empty."
        )

    if max(columns) >= probabilities.shape[1]:
        fail(
            "Probability column mapping exceeds "
            "predict_proba output width."
        )

    vector = probabilities[0, columns].astype(
        np.float32,
        copy=True,
    )

    if vector.shape != (len(columns),):
        fail(
            "Projected probability vector has unexpected shape: "
            f"{vector.shape}"
        )

    if not np.all(np.isfinite(vector)):
        fail(
            "Non-finite probability detected."
        )

    if np.any(vector < -1e-6):
        fail(
            "Negative probability detected."
        )

    if np.any(vector > 1.000001):
        fail(
            "Probability greater than 1 detected."
        )

    common_mass = float(vector.sum())

    if (
        not np.isfinite(common_mass)
        or common_mass <= 0.0
    ):
        fail(
            "Common-class probability mass is zero "
            "or non-finite."
        )

    # B5 operates in the common 549-class product_type
    # prediction space. The text model has 553 original
    # classes, of which 4 are text-only. Therefore the
    # projected text probabilities must be renormalized
    # over the 549 common classes before fusion.
    vector /= common_mass

    if not np.all(np.isfinite(vector)):
        fail(
            "Non-finite value after common-class normalization."
        )

    if np.any(vector < -1e-6):
        fail(
            "Negative probability after normalization."
        )

    if not np.allclose(
        vector.sum(),
        1.0,
        atol=1e-5,
    ):
        fail(
            "Common-class probabilities failed normalization."
        )

    return vector


def read_b3_records(split: str):
    path = B3_DIR / f"{split}.jsonl"

    if not path.exists():
        fail(
            f"Missing B3 split file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {exc}"
                )

            if not isinstance(record, dict):
                fail(
                    f"{path}, line {line_number}: "
                    "record is not a JSON object."
                )

            yield record


def is_common_eligible(
    record: dict[str, Any],
    split: str,
    common_class_set: set[str],
) -> tuple[
    bool,
    str | None,
    Path | None,
]:

    if record.get("split") != split:
        return False, None, None

    target = get_target_label(record)

    if target is None:
        return False, None, None

    if target not in common_class_set:
        return False, target, None

    tokens = get_text_tokens(record)

    if not tokens:
        return False, target, None

    main_image = get_main_image_info(record)

    if main_image is None:
        return False, target, None

    if not main_image.get(
        "physical_file_available"
    ):
        return False, target, None

    if not main_image.get(
        "usable_for_local_image_model"
    ):
        return False, target, None

    image_path = resolve_image_path(
        main_image
    )

    if image_path is None:
        return False, target, None

    return True, target, image_path


def process_split(
    split: str,
    common_classes: list[str],
    text_bundle: dict[str, Any],
    image_bundle: dict[str, Any],
) -> dict[str, Any]:

    common_class_set = set(common_classes)

    text_classifier = text_bundle["classifier"]
    text_vectorizer = text_bundle["vectorizer"]

    image_classifier = image_bundle["classifier"]

    text_columns = build_probability_column_map(
        text_bundle,
        common_classes,
    )

    image_columns = build_probability_column_map(
        image_bundle,
        common_classes,
    )

    record_ids: list[str] = []
    target_labels: list[str] = []

    text_probabilities: list[np.ndarray] = []
    image_probabilities: list[np.ndarray] = []

    total_records = 0
    split_valid = 0
    eligible = 0

    for record in read_b3_records(split):

        total_records += 1

        if record.get("split") == split:
            split_valid += 1

        ok, target, image_path = is_common_eligible(
            record,
            split,
            common_class_set,
        )

        if not ok:
            continue

        record_id = record.get("record_id")

        if record_id is None:
            fail(
                f"{split}: eligible record has no record_id."
            )

        tokens = get_text_tokens(record)

        text_input = " ".join(tokens)

        try:
            text_features = text_vectorizer.transform(
                [text_input]
            )
        except Exception as exc:
            fail(
                f"{split}: text vectorization failed "
                f"for record {record_id}: {exc}"
            )

        try:
            text_raw = text_classifier.predict_proba(
                text_features
            )
        except Exception as exc:
            fail(
                f"{split}: text prediction failed "
                f"for record {record_id}: {exc}"
            )

        image_input = preprocess_image(
            image_path
        )

        try:
            image_raw = image_classifier.predict_proba(
                image_input
            )
        except Exception as exc:
            fail(
                f"{split}: image prediction failed "
                f"for record {record_id}: {exc}"
            )

        text_vector = probability_vector(
            text_raw,
            text_columns,
        )

        image_vector = probability_vector(
            image_raw,
            image_columns,
        )

        record_ids.append(
            str(record_id)
        )

        target_labels.append(
            target
        )

        text_probabilities.append(
            text_vector
        )

        image_probabilities.append(
            image_vector
        )

        eligible += 1

        if eligible % 5000 == 0:
            print(
                f"[{split}] prepared "
                f"{eligible:,} common-target predictions..."
            )

    if total_records != EXPECTED_SPLITS[split]:
        fail(
            f"{split}: B3 record count "
            f"{total_records:,} != expected "
            f"{EXPECTED_SPLITS[split]:,}"
        )

    if split_valid != EXPECTED_SPLITS[split]:
        fail(
            f"{split}: split-field-valid count "
            f"{split_valid:,} != expected "
            f"{EXPECTED_SPLITS[split]:,}"
        )

    if eligible != EXPECTED_COMMON_COUNTS[split]:
        fail(
            f"{split}: common eligible count "
            f"{eligible:,} != expected "
            f"{EXPECTED_COMMON_COUNTS[split]:,}"
        )

    if not text_probabilities:
        fail(
            f"{split}: no text probability vectors generated."
        )

    if not image_probabilities:
        fail(
            f"{split}: no image probability vectors generated."
        )

    text_matrix = np.vstack(
        text_probabilities
    ).astype(
        np.float32,
        copy=False,
    )

    image_matrix = np.vstack(
        image_probabilities
    ).astype(
        np.float32,
        copy=False,
    )

    ids = np.asarray(
        record_ids,
        dtype=str,
    )

    targets = np.asarray(
        target_labels,
        dtype=str,
    )

    expected_shape = (
        eligible,
        len(common_classes),
    )

    if text_matrix.shape != expected_shape:
        fail(
            f"{split}: text probability shape mismatch: "
            f"{text_matrix.shape} != {expected_shape}"
        )

    if image_matrix.shape != expected_shape:
        fail(
            f"{split}: image probability shape mismatch: "
            f"{image_matrix.shape} != {expected_shape}"
        )

    if len(ids) != eligible:
        fail(
            f"{split}: record ID count mismatch."
        )

    if len(targets) != eligible:
        fail(
            f"{split}: target count mismatch."
        )

    if len(np.unique(ids)) != len(ids):
        fail(
            f"{split}: duplicate record IDs detected."
        )

    if not np.all(np.isfinite(text_matrix)):
        fail(
            f"{split}: non-finite text probabilities."
        )

    if not np.all(np.isfinite(image_matrix)):
        fail(
            f"{split}: non-finite image probabilities."
        )

    if not np.allclose(
        text_matrix.sum(axis=1),
        1.0,
        atol=1e-4,
    ):
        fail(
            f"{split}: text probabilities do not sum to 1."
        )

    if not np.allclose(
        image_matrix.sum(axis=1),
        1.0,
        atol=1e-4,
    ):
        fail(
            f"{split}: image probabilities do not sum to 1."
        )

    if np.any(text_matrix < -1e-6):
        fail(
            f"{split}: negative text probability detected."
        )

    if np.any(image_matrix < -1e-6):
        fail(
            f"{split}: negative image probability detected."
        )

    return {
        "split": split,
        "total_records": total_records,
        "split_valid_records": split_valid,
        "common_target_records": eligible,
        "record_ids": ids,
        "target_labels": targets,
        "text_probabilities": text_matrix,
        "image_probabilities": image_matrix,
    }


def check_cross_split_leakage(
    results: dict[str, dict[str, Any]],
) -> int:

    seen: dict[str, str] = {}

    leakage = 0

    for split, result in results.items():

        for record_id in result["record_ids"]:

            previous = seen.get(record_id)

            if previous is not None and previous != split:
                leakage += 1

            seen[record_id] = split

    return leakage


def check_paired_record_alignment(
    result: dict[str, Any],
) -> None:

    ids = result["record_ids"]
    targets = result["target_labels"]

    if len(ids) != len(targets):
        fail(
            f"{result['split']}: "
            "record/target length mismatch."
        )

    if (
        result["text_probabilities"].shape[0]
        != len(ids)
    ):
        fail(
            f"{result['split']}: "
            "text prediction row mismatch."
        )

    if (
        result["image_probabilities"].shape[0]
        != len(ids)
    ):
        fail(
            f"{result['split']}: "
            "image prediction row mismatch."
        )


def save_split(
    split: str,
    result: dict[str, Any],
) -> None:

    output_dir = (
        B5_MODEL_DIR
        / "predictions_v001"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{split}.npz"
    )

    np.savez_compressed(
        output_path,
        record_ids=result["record_ids"],
        target_labels=result["target_labels"],
        text_probabilities=result[
            "text_probabilities"
        ],
        image_probabilities=result[
            "image_probabilities"
        ],
    )

    print(
        f"Saved: {output_path}"
    )


def main() -> None:

    print("=" * 70)
    print(
        "ABO B5.3 — Prediction Preparation & Class Alignment"
    )
    print("=" * 70)

    contract = load_json(
        B5_CONTRACT
    )

    eligibility = load_json(
        B5_ELIGIBILITY
    )

    b3_stats = load_json(
        B3_STATS
    )

    if contract.get("status") != "PASS":
        fail(
            "B5.1 contract is not PASS."
        )

    if eligibility.get("status") != "PASS":
        fail(
            "B5.2 eligibility report is not PASS."
        )

    if (
        contract.get(
            "b3_representation_version"
        )
        != REPRESENTATION_VERSION
    ):
        fail(
            "B5.1 contract does not reference B3 v002."
        )

    if (
        eligibility.get(
            "b3_representation_version"
        )
        != REPRESENTATION_VERSION
    ):
        fail(
            "B5.2 report does not reference B3 v002."
        )

    if contract.get(
        "b4_version"
    ) != B4_VERSION:
        fail(
            "B5.1 contract does not reference B4.2 v002."
        )

    if eligibility.get(
        "b4_version"
    ) != B4_VERSION:
        fail(
            "B5.2 report does not reference B4.2 v002."
        )

    if b3_stats.get(
        "representation_version"
    ) != REPRESENTATION_VERSION:
        fail(
            "B3 statistics are not v002."
        )

    text_bundle, image_bundle = load_models()

    common_classes = validate_class_alignment(
        text_bundle,
        image_bundle,
        eligibility,
    )

    if len(common_classes) != EXPECTED_COMMON_CLASS_COUNT:
        fail(
            f"Expected "
            f"{EXPECTED_COMMON_CLASS_COUNT} common classes, "
            f"got {len(common_classes)}."
        )

    text_labels = model_class_labels(
        text_bundle
    )

    image_labels = model_class_labels(
        image_bundle
    )

    print()
    print(
        f"B3 representation: {REPRESENTATION_VERSION}"
    )
    print(
        f"B4.2 version:       {B4_VERSION}"
    )
    print(
        f"B5.3 version:       {B5_VERSION}"
    )
    print(
        f"Text classes:       {len(text_labels)}"
    )
    print(
        f"Image classes:      {len(image_labels)}"
    )
    print(
        f"Common classes:     {len(common_classes)}"
    )
    print(
        f"Text-only classes:  "
        f"{len(set(text_labels) - set(image_labels))}"
    )
    print(
        f"Image-only classes: "
        f"{len(set(image_labels) - set(text_labels))}"
    )
    print()

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    for split in (
        "train",
        "validation",
        "test",
    ):

        print(
            f"Processing {split}..."
        )

        result = process_split(
            split,
            common_classes,
            text_bundle,
            image_bundle,
        )

        check_paired_record_alignment(
            result
        )

        results[split] = result

        save_split(
            split,
            result,
        )

        print(
            f"{split}: "
            f"{result['common_target_records']:,} "
            "predictions prepared."
        )

    leakage = check_cross_split_leakage(
        results
    )

    if leakage != 0:
        fail(
            "Cross-split prediction leakage detected: "
            f"{leakage}"
        )

    output_dir = (
        B5_MODEL_DIR
        / "predictions_v001"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "prediction_version": B5_VERSION,
        "stage": "B5.3",
        "dataset": "ABO",
        "b3_representation_version": (
            REPRESENTATION_VERSION
        ),
        "b4_version": B4_VERSION,
        "target_field": TARGET_FIELD,
        "probability_space": {
            "definition": (
                "Common product_type classes shared by "
                "the frozen B4.2 text and image models"
            ),
            "class_count": len(common_classes),
            "text_original_class_count": len(
                text_labels
            ),
            "image_original_class_count": len(
                image_labels
            ),
            "text_only_classes_excluded": len(
                set(text_labels)
                - set(image_labels)
            ),
            "image_only_classes_excluded": len(
                set(image_labels)
                - set(text_labels)
            ),
            "common_class_probability_normalization": True,
            "normalization_method": (
                "Renormalize projected probabilities "
                "over the common product_type classes"
            ),
        },
        "common_classes": common_classes,
        "probability_dtype": "float32",
        "image_preprocessing": {
            "color_mode": "RGB",
            "resize": [
                64,
                64,
            ],
            "resampling": "bilinear",
            "scaling": "/255",
            "flatten": True,
        },
        "split_counts": {
            split: result[
                "common_target_records"
            ]
            for split, result in results.items()
        },
        "cross_split_prediction_record_id_leakage": (
            leakage
        ),
        "source_contracts": {
            "b5.1": str(
                B5_CONTRACT.relative_to(
                    PROJECT_ROOT
                )
            ),
            "b5.2": str(
                B5_ELIGIBILITY.relative_to(
                    PROJECT_ROOT
                )
            ),
            "b3_statistics": str(
                B3_STATS.relative_to(
                    PROJECT_ROOT
                )
            ),
            "b4_statistics": str(
                B4_STATS.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    report = {
        "report": (
            "abo_b5.3_prediction_preparation"
        ),
        "b5_version": B5_VERSION,
        "stage": "B5.3",
        "status": "PASS",
        "dataset": "ABO",
        "target_field": TARGET_FIELD,
        "b3_representation_version": (
            REPRESENTATION_VERSION
        ),
        "b4_version": B4_VERSION,
        "common_class_count": len(
            common_classes
        ),
        "class_alignment": {
            "actual_label_alignment": True,
            "probability_columns_aligned_by_product_type": (
                True
            ),
            "text_class_count": len(
                text_labels
            ),
            "image_class_count": len(
                image_labels
            ),
            "common_class_count": len(
                common_classes
            ),
            "text_only_class_count": len(
                set(text_labels)
                - set(image_labels)
            ),
            "image_only_class_count": len(
                set(image_labels)
                - set(text_labels)
            ),
            "common_class_probability_normalization": (
                True
            ),
            "normalization_method": (
                "Renormalize projected probabilities "
                "over the common product_type classes"
            ),
        },
        "split_results": {
            split: {
                "b3_record_count": result[
                    "total_records"
                ],
                "split_field_valid_record_count": result[
                    "split_valid_records"
                ],
                "prediction_record_count": result[
                    "common_target_records"
                ],
                "text_probability_shape": list(
                    result[
                        "text_probabilities"
                    ].shape
                ),
                "image_probability_shape": list(
                    result[
                        "image_probabilities"
                    ].shape
                ),
                "unique_record_ids": int(
                    len(
                        np.unique(
                            result["record_ids"]
                        )
                    )
                ),
                "text_probability_finite": bool(
                    np.all(
                        np.isfinite(
                            result[
                                "text_probabilities"
                            ]
                        )
                    )
                ),
                "image_probability_finite": bool(
                    np.all(
                        np.isfinite(
                            result[
                                "image_probabilities"
                            ]
                        )
                    )
                ),
                "text_probability_rows_sum_to_one": bool(
                    np.allclose(
                        result[
                            "text_probabilities"
                        ].sum(axis=1),
                        1.0,
                        atol=1e-4,
                    )
                ),
                "image_probability_rows_sum_to_one": bool(
                    np.allclose(
                        result[
                            "image_probabilities"
                        ].sum(axis=1),
                        1.0,
                        atol=1e-4,
                    )
                ),
            }
            for split, result in results.items()
        },
        "integrity": {
            "cross_split_record_id_leakage": leakage,
            "model_retraining": False,
            "split_regeneration": False,
            "b3_modification": False,
            "b4.2_modification": False,
            "image_download": False,
            "image_path_guessing": False,
            "synthetic_images": False,
            "fusion_performed": False,
        },
        "artifacts": {
            "prediction_directory": str(
                output_dir.relative_to(
                    PROJECT_ROOT
                )
            ),
            "metadata": str(
                metadata_path.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    B5_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        B5_REPORT_DIR
        / "abo_b5.3_prediction_prep_v001.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print(
        "B5.3 PREDICTION PREPARATION PASS"
    )
    print("=" * 70)
    print(
        f"B3:               "
        f"{REPRESENTATION_VERSION}"
    )
    print(
        f"B4.2:             "
        f"{B4_VERSION}"
    )
    print(
        f"Text classes:     "
        f"{len(text_labels)}"
    )
    print(
        f"Image classes:    "
        f"{len(image_labels)}"
    )
    print(
        f"Common classes:   "
        f"{len(common_classes)}"
    )
    print(
        f"Train:            "
        f"{results['train']['common_target_records']:,}"
    )
    print(
        f"Validation:       "
        f"{results['validation']['common_target_records']:,}"
    )
    print(
        f"Test:             "
        f"{results['test']['common_target_records']:,}"
    )
    print(
        f"Cross-split leak: "
        f"{leakage}"
    )
    print(
        f"Report:           "
        f"{report_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()