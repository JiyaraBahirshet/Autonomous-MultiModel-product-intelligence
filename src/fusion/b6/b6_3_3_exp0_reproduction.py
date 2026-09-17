from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# ============================================================================
# PROJECT / PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

B3_DIR = PROJECT_ROOT / "data" / "representations" / "abo" / "b3"
MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "b4.2"
    / "text_model.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "fusion"
    / "b6"
    / "b6_3_3_exp0_reproduction_v001.json"
)

SPLITS = ("train", "validation", "test")

REPRESENTATION_VERSION = "v002"
B4_VERSION = "v002"
SEED = 20260827
TARGET_FIELD = "product_type"


# ============================================================================
# EXACT B4.2 HELPERS
# ============================================================================

def first_scalar(value: Any) -> str:
    """
    Exact B4.2 target extraction logic.
    """
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        for item in value:
            result = first_scalar(item)
            if result:
                return result
        return ""

    if isinstance(value, dict):
        if "value" in value:
            return first_scalar(value["value"])
        return ""

    return ""


def target_from_record(record: dict[str, Any]) -> str:
    return first_scalar(record.get(TARGET_FIELD))


def text_from_record(record: dict[str, Any]) -> str:
    """
    Exact B4.2 B3-v002 text extraction.
    """
    representation = record.get("b3_representation")

    if not isinstance(representation, dict):
        return ""

    text = representation.get("text")

    if not isinstance(text, dict):
        return ""

    combined_tokens = text.get("combined_tokens")

    if not isinstance(combined_tokens, list):
        return ""

    return " ".join(str(token) for token in combined_tokens)


# ============================================================================
# B3 INPUT VALIDATION
# ============================================================================

def validate_b3_representation_version(
    record: dict[str, Any],
    path: Path,
    line_no: int,
) -> None:

    representation = record.get("b3_representation")

    if not isinstance(representation, dict):
        raise ValueError(
            f"{path}:{line_no}: missing b3_representation."
        )

    version = representation.get("representation_version")

    if version != REPRESENTATION_VERSION:
        raise ValueError(
            f"{path}:{line_no}: expected B3 representation "
            f"{REPRESENTATION_VERSION!r}, got {version!r}."
        )


def load_records(split: str) -> list[dict[str, Any]]:
    """
    Load the exact B4.2 B3-v002 input artifact.
    """
    path = B3_DIR / f"{split}.jsonl"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing B3 split: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as handle:

        for line_no, line in enumerate(handle, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_no}: record must be a JSON object."
                )

            actual_split = record.get("split")
            frozen_split = record.get("frozen_split")

            if actual_split != split:
                raise ValueError(
                    f"{path}:{line_no}: split invariant violated. "
                    f"Expected {split!r}, got {actual_split!r}."
                )

            if frozen_split != split:
                raise ValueError(
                    f"{path}:{line_no}: frozen_split invariant violated. "
                    f"Expected {split!r}, got {frozen_split!r}."
                )

            if "record_id" not in record:
                raise ValueError(
                    f"{path}:{line_no}: missing record_id."
                )

            if "item_id" not in record:
                raise ValueError(
                    f"{path}:{line_no}: missing item_id."
                )

            validate_b3_representation_version(
                record,
                path,
                line_no,
            )

            records.append(record)

    if not records:
        raise ValueError(
            f"B3 split is empty: {path}"
        )

    return records


# ============================================================================
# EXACT B4.2 EVALUATION
# ============================================================================

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 72)
    print("B6.3.3 EXP0 — FROZEN B4.2 TEXT BASELINE REPRODUCTION")
    print("=" * 72)

    # ------------------------------------------------------------------------
    # LOAD FROZEN B4.2 ARTIFACT
    # ------------------------------------------------------------------------

    print("LOADING_FROZEN_B4_2_MODEL")
    print(f"MODEL_PATH={MODEL_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Frozen B4.2 text model not found: {MODEL_PATH}"
        )

    model_artifact = joblib.load(MODEL_PATH)

    if not isinstance(model_artifact, dict):
        raise ValueError(
            "Unexpected B4.2 model artifact type. "
            f"Expected dict, got {type(model_artifact)}."
        )

    required_keys = {
        "vectorizer",
        "label_encoder",
        "classifier",
        "target_field",
        "target_definition",
        "representation_version",
        "b4_version",
    }

    missing_keys = required_keys - set(model_artifact.keys())

    if missing_keys:
        raise ValueError(
            "Frozen B4.2 model artifact is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    vectorizer = model_artifact["vectorizer"]
    label_encoder = model_artifact["label_encoder"]
    classifier = model_artifact["classifier"]

    # ------------------------------------------------------------------------
    # VERIFY FROZEN ARTIFACT METADATA
    # ------------------------------------------------------------------------

    if model_artifact["target_field"] != TARGET_FIELD:
        raise ValueError(
            f"Target field mismatch: "
            f"{model_artifact['target_field']!r}"
        )

    if model_artifact["target_definition"] != (
        "first non-empty product_type value"
    ):
        raise ValueError(
            "Target definition mismatch."
        )

    if model_artifact["representation_version"] != (
        REPRESENTATION_VERSION
    ):
        raise ValueError(
            "Representation version mismatch."
        )

    if model_artifact["b4_version"] != B4_VERSION:
        raise ValueError(
            "B4 version mismatch."
        )

    train_classes = int(len(label_encoder.classes_))
    vocabulary_size = int(len(vectorizer.vocabulary_))

    print(
        f"B4_VERSION={model_artifact['b4_version']}"
    )
    print(
        f"REPRESENTATION_VERSION="
        f"{model_artifact['representation_version']}"
    )
    print(
        f"TARGET_FIELD={model_artifact['target_field']}"
    )
    print(
        f"TARGET_DEFINITION="
        f"{model_artifact['target_definition']}"
    )
    print(
        f"TRAIN_CLASSES={train_classes}"
    )
    print(
        f"VOCABULARY_SIZE={vocabulary_size}"
    )
    print(
        f"SEED={SEED}"
    )

    if train_classes != 553:
        raise ValueError(
            f"Expected 553 frozen B4.2 classes, "
            f"got {train_classes}."
        )

    if vocabulary_size != 256395:
        raise ValueError(
            f"Expected 256395 frozen B4.2 vocabulary entries, "
            f"got {vocabulary_size}."
        )

    # ------------------------------------------------------------------------
    # VERIFY FROZEN CLASSIFIER
    # ------------------------------------------------------------------------

    if not hasattr(classifier, "predict"):
        raise ValueError(
            "Frozen classifier does not provide predict()."
        )

    if not hasattr(classifier, "classes_"):
        raise ValueError(
            "Frozen classifier does not expose classes_."
        )

    classifier_classes = int(len(classifier.classes_))

    if classifier_classes != train_classes:
        raise ValueError(
            "Classifier / LabelEncoder class-count mismatch: "
            f"classifier={classifier_classes}, "
            f"label_encoder={train_classes}"
        )

    # ------------------------------------------------------------------------
    # LOAD B3-V002 SPLITS
    # ------------------------------------------------------------------------

    data: dict[str, list[dict[str, Any]]] = {}

    for split in SPLITS:

        print(
            f"LOADING_{split.upper()}"
        )

        data[split] = load_records(split)

        print(
            f"{split.upper()}_RECORDS={len(data[split])}"
        )

    # ------------------------------------------------------------------------
    # SPLIT INTEGRITY
    # ------------------------------------------------------------------------

    all_ids: list[str] = []

    for split in SPLITS:

        ids = [
            str(record["record_id"])
            for record in data[split]
        ]

        duplicate_count = (
            len(ids) - len(set(ids))
        )

        if duplicate_count != 0:
            raise ValueError(
                f"Duplicate record IDs in {split}: "
                f"{duplicate_count}"
            )

        all_ids.extend(ids)

    cross_split_leak = (
        len(all_ids) - len(set(all_ids))
    )

    if cross_split_leak != 0:
        raise ValueError(
            "Cross-split record-ID leakage detected: "
            f"{cross_split_leak}"
        )

    print(
        "CROSS_SPLIT_RECORD_ID_LEAKAGE=0"
    )

    # ------------------------------------------------------------------------
    # EXACT B4.2 TEXT/TARGET ELIGIBILITY
    # ------------------------------------------------------------------------

    pairs: dict[
        str,
        list[tuple[str, str, str]]
    ] = {}

    for split in SPLITS:

        split_pairs: list[
            tuple[str, str, str]
        ] = []

        for record in data[split]:

            target = target_from_record(record)
            text = text_from_record(record)

            if not target:
                continue

            if not text:
                continue

            split_pairs.append(
                (
                    str(record["record_id"]),
                    text,
                    target,
                )
            )

        pairs[split] = split_pairs

        print(
            f"{split.upper()}_ELIGIBLE="
            f"{len(split_pairs)}"
        )

    # Frozen B4.2 expected populations.
    expected_eligible = {
        "train": 70284,
        "validation": 69996,
        "test": 7422,
    }

    for split, expected in expected_eligible.items():

        actual = len(pairs[split])

        if actual != expected:
            raise ValueError(
                f"{split} eligible-record count mismatch: "
                f"expected {expected}, got {actual}"
            )

    # ------------------------------------------------------------------------
    # EXACT B4.2 KNOWN-CLASS LOGIC
    #
    # Original B4.2:
    #
    # y_train = encoder.fit_transform(train_targets)
    # known_classes = set(str(value) for value in encoder.classes_)
    #
    # Here the encoder is already frozen inside the saved artifact.
    # No fit occurs.
    # ------------------------------------------------------------------------

    known_classes = set(
        str(value)
        for value in label_encoder.classes_
    )

    if len(known_classes) != 553:
        raise ValueError(
            f"Expected 553 known target classes, "
            f"got {len(known_classes)}."
        )

    validation_targets = [
        target
        for _, _, target in pairs["validation"]
    ]

    test_targets = [
        target
        for _, _, target in pairs["test"]
    ]

    validation_mask = np.asarray(
        [
            target in known_classes
            for target in validation_targets
        ],
        dtype=bool,
    )

    test_mask = np.asarray(
        [
            target in known_classes
            for target in test_targets
        ],
        dtype=bool,
    )

    if not validation_mask.any():
        raise ValueError(
            "No validation targets were observed in training."
        )

    if not test_mask.any():
        raise ValueError(
            "No test targets were observed in training."
        )

    validation_unseen = int(
        (~validation_mask).sum()
    )

    test_unseen = int(
        (~test_mask).sum()
    )

    print(
        f"VALIDATION_UNSEEN_TARGET_RECORDS="
        f"{validation_unseen}"
    )

    print(
        f"TEST_UNSEEN_TARGET_RECORDS="
        f"{test_unseen}"
    )

    if validation_unseen != 59:
        raise ValueError(
            f"Expected 59 validation unseen targets, "
            f"got {validation_unseen}"
        )

    if test_unseen != 30:
        raise ValueError(
            f"Expected 30 test unseen targets, "
            f"got {test_unseen}"
        )

    # ------------------------------------------------------------------------
    # NO RETRAINING
    # ------------------------------------------------------------------------

    print("FROZEN_MODEL_INFERENCE_ONLY=True")
    print("RETRAINING=False")
    print("VECTORIZER_FIT=False")
    print("LABEL_ENCODER_FIT=False")
    print("CLASSIFIER_FIT=False")

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    validation_texts = [
        text
        for _, text, _ in pairs["validation"]
    ]

    validation_known_texts = [
        text
        for text, keep in zip(
            validation_texts,
            validation_mask,
        )
        if keep
    ]

    y_validation = np.asarray(
        [
            target
            for target, keep in zip(
                validation_targets,
                validation_mask,
            )
            if keep
        ],
        dtype=object,
    )

    print("VALIDATION_INFERENCE")

    # IMPORTANT:
    # Use the frozen vectorizer and classifier exactly as saved.
    X_validation = vectorizer.transform(
        validation_known_texts
    )

    y_validation_encoded = label_encoder.transform(
        y_validation
    )

    validation_predictions_encoded = classifier.predict(
        X_validation
    )

    validation_predictions = label_encoder.inverse_transform(
        validation_predictions_encoded
    )

    validation_metrics = evaluate(
        y_validation_encoded,
        validation_predictions_encoded,
    )

    print(
        f"VALIDATION_ACCURACY="
        f"{validation_metrics['accuracy']}"
    )

    print(
        f"VALIDATION_MACRO_F1="
        f"{validation_metrics['macro_f1']}"
    )

    print(
        f"VALIDATION_WEIGHTED_F1="
        f"{validation_metrics['weighted_f1']}"
    )

    # ------------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------------

    test_texts = [
        text
        for _, text, _ in pairs["test"]
    ]

    test_known_texts = [
        text
        for text, keep in zip(
            test_texts,
            test_mask,
        )
        if keep
    ]

    y_test = np.asarray(
        [
            target
            for target, keep in zip(
                test_targets,
                test_mask,
            )
            if keep
        ],
        dtype=object,
    )

    print("TEST_INFERENCE")

    X_test = vectorizer.transform(
        test_known_texts
    )

    y_test_encoded = label_encoder.transform(
        y_test
    )

    test_predictions_encoded = classifier.predict(
        X_test
    )

    test_predictions = label_encoder.inverse_transform(
        test_predictions_encoded
    )

    test_metrics = evaluate(
        y_test_encoded,
        test_predictions_encoded,
    )

    print(
        f"TEST_ACCURACY="
        f"{test_metrics['accuracy']}"
    )

    print(
        f"TEST_MACRO_F1="
        f"{test_metrics['macro_f1']}"
    )

    print(
        f"TEST_WEIGHTED_F1="
        f"{test_metrics['weighted_f1']}"
    )

    # ------------------------------------------------------------------------
    # FROZEN EXPECTED VALUES
    # ------------------------------------------------------------------------

    expected_validation = {
        "accuracy": 0.9082173956560905,
        "macro_f1": 0.08524528962807053,
        "weighted_f1": 0.8980940672973457,
    }

    expected_test = {
        "accuracy": 0.5684523809523809,
        "macro_f1": 0.09387840495350011,
        "weighted_f1": 0.5050464332987978,
    }

    def assert_metric_match(
        actual: dict[str, float],
        expected: dict[str, float],
        split: str,
    ) -> None:

        tolerance = 1e-12

        for metric, expected_value in expected.items():

            actual_value = actual[metric]

            if abs(
                actual_value - expected_value
            ) > tolerance:

                raise ValueError(
                    f"{split} {metric} mismatch: "
                    f"expected {expected_value}, "
                    f"got {actual_value}"
                )

    assert_metric_match(
        validation_metrics,
        expected_validation,
        "validation",
    )

    assert_metric_match(
        test_metrics,
        expected_test,
        "test",
    )

    print(
        "FROZEN_METRIC_REPRODUCTION=PASS"
    )

    # ------------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------------

    report = {
        "report": "b6_3_3_exp0_reproduction",
        "report_version": "v001",
        "stage": "B6.3.3",
        "experiment": "Exp0",
        "status": "PASS",
        "purpose": (
            "Exact reproduction of the frozen ABO B4.2 "
            "text baseline without retraining."
        ),
        "dataset": "Amazon Berkeley Objects (ABO)",
        "task": "product_type classification",
        "representation_version": REPRESENTATION_VERSION,
        "b4_version": B4_VERSION,
        "seed": SEED,
        "model": {
            "path": str(
                MODEL_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "artifact_type": "dict",
            "frozen": True,
            "retraining": False,
            "vectorizer": (
                "frozen B4.2 vectorizer"
            ),
            "label_encoder": (
                "frozen B4.2 LabelEncoder"
            ),
            "classifier": (
                "frozen B4.2 SGDClassifier"
            ),
            "train_classes": train_classes,
            "vocabulary_size": vocabulary_size,
        },
        "input_contract": {
            "source": (
                "data/representations/abo/b3/"
                "{split}.jsonl"
            ),
            "splits": list(SPLITS),
            "representation_version": (
                REPRESENTATION_VERSION
            ),
            "target_field": TARGET_FIELD,
            "target_definition": (
                "first non-empty product_type value"
            ),
            "text_source": (
                "b3_representation.text.combined_tokens"
            ),
        },
        "eligible_records": {
            split: len(pairs[split])
            for split in SPLITS
        },
        "train_classes": train_classes,
        "validation_unseen_target_records": (
            validation_unseen
        ),
        "test_unseen_target_records": (
            test_unseen
        ),
        "validation": validation_metrics,
        "test": test_metrics,
        "expected_frozen_validation": (
            expected_validation
        ),
        "expected_frozen_test": expected_test,
        "integrity": {
            "cross_split_record_id_leakage": 0,
            "retraining": False,
            "vectorizer_refit": False,
            "label_encoder_refit": False,
            "classifier_refit": False,
            "split_regeneration": False,
            "synthetic_data": False,
            "image_download": False,
            "path_guessing": False,
            "b5_fusion_used": False,
            "rakuten_signal_used": False,
            "mave_signal_used": False,
        },
    }

    REPORT_PATH.parent.mkdir(
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

    print(
        f"REPORT_SAVED={REPORT_PATH}"
    )

    print("=" * 72)
    print("B6.3.3 EXP0 REPRODUCTION=PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()