from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]

B4_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "abo" / "b4.2" / "text_model.joblib"

TRANSFER_DIR = (
    PROJECT_ROOT
    / "data"
    / "representations"
    / "abo"
    / "b6_3_3_transfer"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "fusion" / "b6"

REPORT_PATH = (
    REPORT_DIR
    / "b6_3_3_exp1_gate1_validation_v001.json"
)

SEED = 20260827

TRAIN_COUNT = 70284
VALIDATION_COUNT = 69996
TEST_COUNT = 7422

TRANSFER_DIMENSION = 3008
TARGET_CLASS_COUNT = 549

GAMMAS = (0.1, 0.5, 1.0)

TARGET_FIELD = "product_type"

# Frozen B5 common-class eligibility boundary.
# These four classes are text-only and are excluded from the
# B6.3.3 549-class target population. They are never remapped.
EXCLUDED_TARGET_CLASSES = frozenset({
    "HAIRBAND",
    "PUNCHING_BAG",
    "SALWAR_SUIT_SET",
    "TREADMILL",
})


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_scalar(value) -> str:
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


def target_from_record(record: dict) -> str:
    return first_scalar(record.get(TARGET_FIELD))


def text_from_record(record: dict) -> str:
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


def load_b3_records(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            records.append(record)

    return records


def locate_b3_json(split: str) -> Path:
    candidates = [
        PROJECT_ROOT
        / "data"
        / "processed"
        / "abo"
        / f"{split}.jsonl",
        PROJECT_ROOT
        / "data"
        / "processed"
        / "abo"
        / f"{split}.json",
        PROJECT_ROOT
        / "data"
        / "representations"
        / "abo"
        / "b3"
        / f"{split}.jsonl",
        PROJECT_ROOT
        / "data"
        / "representations"
        / "abo"
        / "b3"
        / f"{split}.json",
    ]

    existing = [path for path in candidates if path.exists()]

    if len(existing) != 1:
        raise FileNotFoundError(
            "Could not uniquely locate ABO B3 split file for "
            f"{split!r}. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )

    return existing[0]


def load_transfer(split: str) -> tuple[np.ndarray, list[str], dict]:
    npz_path = TRANSFER_DIR / f"{split}.npz"
    metadata_path = TRANSFER_DIR / f"{split}_metadata.json"

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with np.load(npz_path, allow_pickle=False) as archive:
        keys = list(archive.keys())

        if "transferred_signal" not in archive:
            raise ValueError(
                f"Validated transfer artifact is missing "
                f"transferred_signal. Keys={keys}"
            )

        if "record_ids" not in archive:
            raise ValueError(
                f"Validated transfer artifact is missing "
                f"record_ids. Keys={keys}"
            )

        features = np.asarray(
            archive["transferred_signal"]
        )

        record_ids_array = np.asarray(
            archive["record_ids"]
        )

    record_ids = [
        str(value)
        for value in record_ids_array.tolist()
    ]

    metadata = load_json(metadata_path)

    return features, record_ids, metadata

def validate_transfer(
    split: str,
    expected_count: int,
) -> tuple[np.ndarray, list[str], dict]:

    features, record_ids, metadata = load_transfer(split)

    if features.shape != (expected_count, TRANSFER_DIMENSION):
        raise ValueError(
            f"{split} transfer shape mismatch: "
            f"expected {(expected_count, TRANSFER_DIMENSION)}, "
            f"got {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(f"{split} transfer features contain non-finite values.")

    if np.any(features < 0):
        raise ValueError(f"{split} transfer features contain negative values.")

    if len(record_ids) != expected_count:
        raise ValueError(
            f"{split} transfer record-ID count mismatch: "
            f"expected {expected_count}, got {len(record_ids)}"
        )

    if len(set(record_ids)) != len(record_ids):
        raise ValueError(f"{split} transfer record IDs are not unique.")

    return features.astype(np.float32, copy=False), record_ids, metadata


def validate_b3_order(
    split: str,
    records: list[dict],
    transfer_record_ids: list[str],
    expected_count: int,
) -> None:

    if len(records) != expected_count:
        raise ValueError(
            f"{split} B3 record count mismatch: "
            f"expected {expected_count}, got {len(records)}"
        )

    b3_ids = [
        str(record.get("record_id", ""))
        for record in records
    ]

    if any(not record_id for record_id in b3_ids):
        raise ValueError(f"{split} contains a missing B3 record_id.")

    if len(set(b3_ids)) != len(b3_ids):
        raise ValueError(f"{split} B3 record IDs are not unique.")

    if b3_ids != transfer_record_ids:
        raise ValueError(
            f"{split} B3 ordering does not exactly match the "
            "validated transfer artifact ordering."
        )


def calculate_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:

    all_class_indices = np.arange(
        TARGET_CLASS_COUNT,
        dtype=np.int64,
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, prediction)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                labels=all_class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                prediction,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def filter_common_target_records(
    records: list[dict],
) -> tuple[list[dict], np.ndarray, dict]:
    """
    Apply the frozen B5 549-class target boundary without remapping.
    """

    targets = [
        target_from_record(record)
        for record in records
    ]

    keep_mask = np.asarray(
        [
            target not in EXCLUDED_TARGET_CLASSES
            for target in targets
        ],
        dtype=bool,
    )

    excluded_by_target = {
        target: int(sum(value == target for value in targets))
        for target in sorted(EXCLUDED_TARGET_CLASSES)
    }

    filtered_records = [
        record
        for record, keep in zip(records, keep_mask)
        if keep
    ]

    return filtered_records, keep_mask, {
        "original_count": len(records),
        "filtered_count": len(filtered_records),
        "excluded_count": int((~keep_mask).sum()),
        "excluded_by_target": excluded_by_target,
    }


def validate_targets(
    train_records: list[dict],
    validation_records: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict]:

    train_targets = [
        target_from_record(record)
        for record in train_records
    ]

    validation_targets = [
        target_from_record(record)
        for record in validation_records
    ]

    if any(not value for value in train_targets):
        raise ValueError("ABO train contains an empty product_type target.")

    if any(not value for value in validation_targets):
        raise ValueError(
            "ABO validation contains an empty product_type target."
        )

    known_classes = sorted(set(train_targets))

    if set(known_classes) & EXCLUDED_TARGET_CLASSES:
        raise ValueError(
            "Excluded target class remains in filtered training population."
        )

    if len(known_classes) != TARGET_CLASS_COUNT:
        raise ValueError(
            f"Expected {TARGET_CLASS_COUNT} ABO target classes after "
            f"frozen exclusion, found {len(known_classes)}."
        )

    class_to_index = {
        value: index
        for index, value in enumerate(known_classes)
    }

    validation_known_mask = np.asarray(
        [
            target in class_to_index
            for target in validation_targets
        ],
        dtype=bool,
    )

    y_train = np.asarray(
        [class_to_index[value] for value in train_targets],
        dtype=np.int64,
    )

    y_validation = np.asarray(
        [
            class_to_index[value]
            for value, known in zip(
                validation_targets,
                validation_known_mask,
            )
            if known
        ],
        dtype=np.int64,
    )

    unseen_targets = sorted(
        {
            target
            for target, known in zip(
                validation_targets,
                validation_known_mask,
            )
            if not known
        }
    )

    return y_train, y_validation, {
        "class_count": len(known_classes),
        "class_order": known_classes,
        "validation_total_records": len(validation_records),
        "validation_known_class_records": int(
            validation_known_mask.sum()
        ),
        "validation_unseen_target_records": int(
            (~validation_known_mask).sum()
        ),
        "validation_unseen_targets": unseen_targets,
        "validation_known_mask": validation_known_mask,
    }

def build_text_features(
    model: dict,
    train_records: list[dict],
    validation_records: list[dict],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:

    vectorizer = model["vectorizer"]

    train_texts = [
        text_from_record(record)
        for record in train_records
    ]

    validation_texts = [
        text_from_record(record)
        for record in validation_records
    ]

    if any(not text for text in train_texts):
        raise ValueError("ABO train contains an empty B3 text representation.")

    if any(not text for text in validation_texts):
        raise ValueError(
            "ABO validation contains an empty B3 text representation."
        )

    train_matrix = vectorizer.transform(train_texts)
    validation_matrix = vectorizer.transform(validation_texts)

    if train_matrix.shape[0] != len(train_records):
        raise ValueError(
            "ABO train text row count does not match the "
            "provided train population."
        )

    if validation_matrix.shape[0] != len(validation_records):
        raise ValueError(
            "ABO validation text row count does not match the "
            "provided validation population."
        )

    return (
        train_matrix.tocsr(),
        validation_matrix.tocsr(),
        {
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "vectorizer_fit_during_exp1": False,
        },
    )


def make_permutation(
    count: int,
    seed: int,
) -> np.ndarray:

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(count)

    if permutation.shape != (count,):
        raise RuntimeError("Invalid permutation shape.")

    if not np.array_equal(
        np.sort(permutation),
        np.arange(count),
    ):
        raise RuntimeError("Permutation is not a complete row permutation.")

    return permutation


def run_exp1_gamma(
    gamma: float,
    X_text_train: sparse.csr_matrix,
    X_text_validation: sparse.csr_matrix,
    transfer_train_permuted: np.ndarray,
    transfer_validation: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[SGDClassifier, dict[str, float]]:

    scaled_train_transfer = (
        gamma * transfer_train_permuted
    ).astype(np.float32, copy=False)

    scaled_validation_transfer = (
        gamma * transfer_validation
    ).astype(np.float32, copy=False)

    X_train = sparse.hstack(
        [
            X_text_train,
            sparse.csr_matrix(scaled_train_transfer),
        ],
        format="csr",
    )

    X_validation = sparse.hstack(
        [
            X_text_validation,
            sparse.csr_matrix(scaled_validation_transfer),
        ],
        format="csr",
    )

    classifier = SGDClassifier(
        loss="log_loss",
        max_iter=30,
        random_state=SEED,
        n_jobs=1,
        tol=1e-3,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    predictions = classifier.predict(X_validation)

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    return classifier, metrics


def main() -> None:

    print("=" * 72)
    print("B6.3.3 EXP1 — GATE 1 VALIDATION")
    print("=" * 72)

    print("LOADING_FROZEN_B4_2_MODEL")
    frozen_model = joblib.load(B4_MODEL_PATH)

    if not isinstance(frozen_model, dict):
        raise TypeError(
            "Frozen B4.2 model must be a dict artifact."
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

    missing_keys = required_keys.difference(frozen_model.keys())

    if missing_keys:
        raise ValueError(
            f"Frozen B4.2 artifact is missing keys: {sorted(missing_keys)}"
        )

    print(
        f"B4_VERSION={frozen_model['b4_version']}"
    )
    print(
        f"REPRESENTATION_VERSION="
        f"{frozen_model['representation_version']}"
    )
    print(
        f"TARGET_FIELD={frozen_model['target_field']}"
    )
    print(
        f"TARGET_DEFINITION="
        f"{frozen_model['target_definition']}"
    )

    if frozen_model["target_field"] != TARGET_FIELD:
        raise ValueError("Frozen B4.2 target field mismatch.")

    frozen_label_encoder = frozen_model["label_encoder"]

    print(
        f"FROZEN_B4_2_CLASSES="
        f"{len(frozen_label_encoder.classes_)}"
    )

    print("LOADING_ABO_B3")

    train_path = locate_b3_json("train")
    validation_path = locate_b3_json("validation")

    train_records = load_b3_records(train_path)
    validation_records = load_b3_records(validation_path)

    print(f"TRAIN_RECORDS_ORIGINAL={len(train_records)}")
    print(f"VALIDATION_RECORDS_ORIGINAL={len(validation_records)}")

    if len(train_records) != TRAIN_COUNT:
        raise ValueError("Unexpected ABO train count.")

    if len(validation_records) != VALIDATION_COUNT:
        raise ValueError("Unexpected ABO validation count.")

    print("LOADING_VALIDATED_TRANSFER_ARTIFACTS")

    train_transfer, train_transfer_ids, train_transfer_metadata = (
        validate_transfer("train", TRAIN_COUNT)
    )

    validation_transfer, validation_transfer_ids, validation_transfer_metadata = (
        validate_transfer("validation", VALIDATION_COUNT)
    )

    print(
        f"TRAIN_TRANSFER_SHAPE_ORIGINAL={train_transfer.shape}"
    )
    print(
        f"VALIDATION_TRANSFER_SHAPE_ORIGINAL={validation_transfer.shape}"
    )

    print("VALIDATING_EXACT_B3_TRANSFER_ORDER")

    validate_b3_order(
        "train",
        train_records,
        train_transfer_ids,
        TRAIN_COUNT,
    )

    validate_b3_order(
        "validation",
        validation_records,
        validation_transfer_ids,
        VALIDATION_COUNT,
    )

    print("B3_TRANSFER_ORDER_MATCH=PASS")

    print("APPLYING_FROZEN_549_CLASS_TARGET_BOUNDARY")

    filtered_train_records, train_mask, train_filter_info = (
        filter_common_target_records(train_records)
    )

    filtered_validation_records, validation_mask, validation_filter_info = (
        filter_common_target_records(validation_records)
    )

    print(
        f"TRAIN_EXCLUDED_TARGET_RECORDS="
        f"{train_filter_info['excluded_count']}"
    )

    print(
        f"VALIDATION_EXCLUDED_TARGET_RECORDS="
        f"{validation_filter_info['excluded_count']}"
    )

    print(
        f"TRAIN_FILTERED_RECORDS="
        f"{len(filtered_train_records)}"
    )

    print(
        f"VALIDATION_FILTERED_RECORDS="
        f"{len(filtered_validation_records)}"
    )

    if len(filtered_train_records) != 70258:
        raise ValueError("Unexpected B6.3.3 filtered train count.")

    if len(filtered_validation_records) != 69995:
        raise ValueError("Unexpected B6.3.3 filtered validation count.")

    train_transfer = train_transfer[train_mask]
    validation_transfer = validation_transfer[validation_mask]

    train_records = filtered_train_records
    validation_records = filtered_validation_records

    print(
        f"TRAIN_TRANSFER_SHAPE_FILTERED="
        f"{train_transfer.shape}"
    )

    print(
        f"VALIDATION_TRANSFER_SHAPE_FILTERED="
        f"{validation_transfer.shape}"
    )

    if train_transfer.shape[0] != len(train_records):
        raise ValueError("Filtered train transfer count mismatch.")

    if validation_transfer.shape[0] != len(validation_records):
        raise ValueError("Filtered validation transfer count mismatch.")

    print("FILTERED_TRANSFER_ALIGNMENT=PASS")

    print("VALIDATING_TARGETS")

    y_train, y_validation, target_info = validate_targets(
        train_records,
        validation_records,
    )

    print(
        f"ABO_TARGET_CLASSES={target_info['class_count']}"
    )

    print(
        f"VALIDATION_KNOWN_CLASS_RECORDS="
        f"{target_info['validation_known_class_records']}"
    )

    print(
        f"VALIDATION_UNSEEN_TARGET_RECORDS="
        f"{target_info['validation_unseen_target_records']}"
    )

    print("BUILDING_FROZEN_B4_2_TEXT_FEATURES")

    X_text_train, X_text_validation, text_info = build_text_features(
        frozen_model,
        train_records,
        validation_records,
    )

    validation_known_mask = target_info["validation_known_mask"]

    X_text_validation_known = X_text_validation[
        validation_known_mask
    ]

    validation_transfer_known = validation_transfer[
        validation_known_mask
    ]

    print(
        f"VALIDATION_METRIC_FEATURE_ROWS="
        f"{X_text_validation_known.shape[0]}"
    )

    print(
        f"TEXT_TRAIN_SHAPE={X_text_train.shape}"
    )
    print(
        f"TEXT_VALIDATION_SHAPE={X_text_validation.shape}"
    )
    print(
        f"TEXT_VOCABULARY_SIZE={text_info['vocabulary_size']}"
    )
    print("VECTORIZER_FIT_DURING_EXP1=False")

    print("CREATING_WITHIN_TRAIN_ROW_PERMUTATION")

    permutation = make_permutation(
        len(train_records),
        SEED,
    )

    permuted_train_transfer = train_transfer[permutation]

    print(f"PERMUTATION_SEED={SEED}")
    print(
        f"PERMUTATION_LENGTH={len(permutation)}"
    )
    print("PERMUTATION_SCOPE=ABO_TRAIN_ONLY")
    print("VALIDATION_TRANSFER_PERMUTED=False")
    print("TEST_TRANSFER_LOADED=False")

    if np.array_equal(
        permutation,
        np.arange(len(train_records)),
    ):
        raise RuntimeError(
            "Generated permutation is the identity permutation. "
            "Regenerate using the same deterministic RNG contract."
        )

    results = {}

    for gamma in GAMMAS:

        print("-" * 72)
        print(f"EXP1_GAMMA={gamma}")
        print("-" * 72)

        classifier, metrics = run_exp1_gamma(
            gamma=gamma,
            X_text_train=X_text_train,
            X_text_validation=X_text_validation_known,
            transfer_train_permuted=permuted_train_transfer,
            transfer_validation=validation_transfer_known,
            y_train=y_train,
            y_validation=y_validation,
        )

        results[str(gamma)] = {
            "gamma": gamma,
            "metrics": metrics,
            "classifier": {
                "type": "SGDClassifier",
                "loss": "log_loss",
                "max_iter": 30,
                "random_state": SEED,
                "n_jobs": 1,
                "tol": 1e-3,
            },
            "train_rows": len(train_records),
            "validation_rows": target_info[
                "validation_known_class_records"
            ],
            "combined_dimension": int(
                X_text_train.shape[1] + TRANSFER_DIMENSION
            ),
        }

        print(
            f"VALIDATION_ACCURACY={metrics['accuracy']}"
        )
        print(
            f"VALIDATION_MACRO_F1={metrics['macro_f1']}"
        )
        print(
            f"VALIDATION_WEIGHTED_F1={metrics['weighted_f1']}"
        )

        del classifier

    selected_gamma = max(
        GAMMAS,
        key=lambda value: (
            results[str(value)]["metrics"]["macro_f1"],
            results[str(value)]["metrics"]["weighted_f1"],
            results[str(value)]["metrics"]["accuracy"],
        ),
    )

    print("=" * 72)
    print("EXP1_VALIDATION_SELECTION")
    print("=" * 72)
    print(f"SELECTED_GAMMA={selected_gamma}")
    print("SELECTION_SPLIT=ABO_VALIDATION")
    print("TEST_USED_FOR_SELECTION=False")

    report = {
        "experiment": "B6.3.3 Exp1",
        "experiment_type": "negative_control",
        "status": "GATE_1_PASS",
        "objective": (
            "Evaluate ABO text plus within-train row-permuted "
            "Rakuten transfer signal."
        ),
        "seed": SEED,
        "source": {
            "dataset": "Rakuten",
            "representation_version": "B3_v001",
            "source_model": (
                "Frozen B6.3.3 Rakuten TF-IDF + SGD"
            ),
            "source_classes": 3008,
            "source_model_inference_only": True,
            "validation_used": False,
            "test_used": False,
        },
        "target": {
            "dataset": "ABO",
            "representation_version": "B3_v002",
            "target_field": TARGET_FIELD,
            "target_classes": TARGET_CLASS_COUNT,
            "original_train_count": TRAIN_COUNT,
            "original_validation_count": VALIDATION_COUNT,
            "original_test_count": TEST_COUNT,
            "filtered_train_count": len(train_records),
            "filtered_validation_count": len(validation_records),
            "validation_metric_count": target_info[
                "validation_known_class_records"
            ],
            "validation_unseen_target_records": target_info[
                "validation_unseen_target_records"
            ],
            "excluded_target_classes": sorted(
                EXCLUDED_TARGET_CLASSES
            ),
        },
        "transfer_artifacts": {
            "train": str(
                TRANSFER_DIR / "train.npz"
            ),
            "validation": str(
                TRANSFER_DIR / "validation.npz"
            ),
            "test": str(
                TRANSFER_DIR / "test.npz"
            ),
            "dimension": TRANSFER_DIMENSION,
            "train_order_validated": True,
            "validation_order_validated": True,
        },
        "permutation": {
            "scope": "ABO_train_only",
            "seed": SEED,
            "row_permutation": True,
            "validation_permuted": False,
            "test_loaded": False,
            "identity_permutation": False,
        },
        "integration": {
            "method": "feature_concatenation",
            "formula": "[X_text | gamma * X_transfer]",
            "text_features": int(X_text_train.shape[1]),
            "transfer_features": TRANSFER_DIMENSION,
            "combined_features": int(
                X_text_train.shape[1] + TRANSFER_DIMENSION
            ),
            "feature_scaling_identical_across_exp1_exp2": True,
        },
        "downstream_classifier": {
            "type": "SGDClassifier",
            "loss": "log_loss",
            "max_iter": 30,
            "random_state": SEED,
            "n_jobs": 1,
            "tol": 1e-3,
        },
        "gamma_candidates": list(GAMMAS),
        "gamma_selection": {
            "split": "ABO_validation",
            "selected_gamma": selected_gamma,
        },
        "results": results,
        "frozen_baseline_reference": {
            "experiment": "Exp0",
            "b4_version": "v002",
            "representation_version": "v002",
            "retraining": False,
        },
        "test_evaluation": {
            "performed": False,
            "reason": "Gate 1 only",
        },
        "leakage_controls": {
            "record_joins": False,
            "asin_matching": False,
            "product_id_matching": False,
            "category_id_mapping": False,
            "semantic_label_crosswalk": False,
            "synthetic_pairing": False,
            "external_enrichment": False,
            "image_download": False,
            "path_guessing": False,
            "split_regeneration": False,
            "rakuten_validation_test_for_tuning": False,
            "abo_test_for_tuning": False,
            "b5_predictions_used": False,
            "b5_weights_used": False,
            "rakuten_source_model_modified": False,
        },
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print(f"REPORT_SAVED={REPORT_PATH}")

    print("=" * 72)
    print("B6.3.3 EXP1 GATE 1=PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
