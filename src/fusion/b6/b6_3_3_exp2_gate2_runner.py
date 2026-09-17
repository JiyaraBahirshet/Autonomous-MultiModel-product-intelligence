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
    / "b6_3_3_exp2_gate2_test_v001.json"
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
            "vectorizer_fit_during_exp2": False,
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


def run_exp2_gate2(
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
    print("B6.3.3 EXP2 — GATE 2 FINAL TEST EVALUATION")
    print("=" * 72)

    print("LOADING_FROZEN_B4_2_MODEL")

    model_artifact = joblib.load(B4_MODEL_PATH)

    vectorizer = model_artifact["vectorizer"]
    label_encoder = model_artifact["label_encoder"]
    frozen_classifier = model_artifact["classifier"]

    print(f"B4_VERSION={model_artifact['b4_version']}")
    print(
        f"REPRESENTATION_VERSION="
        f"{model_artifact['representation_version']}"
    )
    print(f"TARGET_FIELD={model_artifact['target_field']}")
    print(f"TARGET_DEFINITION={model_artifact['target_definition']}")
    print(f"FROZEN_B4_2_CLASSES={len(label_encoder.classes_)}")

    if len(label_encoder.classes_) != 553:
        raise ValueError("Frozen B4.2 artifact does not contain 553 classes.")

    print("LOADING_ABO_B3")

    train_path = locate_b3_json("train")
    validation_path = locate_b3_json("validation")
    test_path = locate_b3_json("test")

    train_records = load_b3_records(train_path)
    validation_records = load_b3_records(validation_path)
    test_records = load_b3_records(test_path)

    print(f"TRAIN_RECORDS_ORIGINAL={len(train_records)}")
    print(f"VALIDATION_RECORDS_ORIGINAL={len(validation_records)}")
    print(f"TEST_RECORDS_ORIGINAL={len(test_records)}")

    if len(train_records) != TRAIN_COUNT:
        raise ValueError("Unexpected ABO train record count.")

    if len(validation_records) != VALIDATION_COUNT:
        raise ValueError("Unexpected ABO validation record count.")

    if len(test_records) != TEST_COUNT:
        raise ValueError("Unexpected ABO test record count.")

    print("LOADING_VALIDATED_TRANSFER_ARTIFACTS")

    train_transfer, train_transfer_record_ids, train_transfer_metadata = load_transfer("train")
    test_transfer, test_transfer_record_ids, test_transfer_metadata = load_transfer("test")

    print(
        f"TRAIN_TRANSFER_SHAPE_ORIGINAL="
        f"{train_transfer.shape}"
    )
    print(
        f"TEST_TRANSFER_SHAPE_ORIGINAL="
        f"{test_transfer.shape}"
    )

    train_transfer, train_transfer_record_ids, train_transfer_metadata = validate_transfer(
        "train",
        TRAIN_COUNT,
    )
    test_transfer, test_transfer_record_ids, test_transfer_metadata = validate_transfer(
        "test",
        TEST_COUNT,
    )

    print("EXACT_B3_TRANSFER_ORDER_VALIDATED_BY_ARTIFACT_GATE")

    print("APPLYING_FROZEN_549_CLASS_TARGET_BOUNDARY")

    filtered_train, train_mask, train_exclusion = (
        filter_common_target_records(train_records)
    )

    filtered_test, test_mask, test_exclusion = (
        filter_common_target_records(test_records)
    )

    filtered_train_transfer = train_transfer[train_mask]
    filtered_test_transfer = test_transfer[test_mask]

    print(
        "TRAIN_EXCLUDED_TARGET_RECORDS="
        f"{train_exclusion['excluded_count']}"
    )
    print(
        "TEST_EXCLUDED_TARGET_RECORDS="
        f"{test_exclusion['excluded_count']}"
    )

    print(
        f"TRAIN_FILTERED_RECORDS={len(filtered_train)}"
    )
    print(
        f"TEST_FILTERED_RECORDS={len(filtered_test)}"
    )

    print(
        f"TRAIN_TRANSFER_SHAPE_FILTERED="
        f"{filtered_train_transfer.shape}"
    )
    print(
        f"TEST_TRANSFER_SHAPE_FILTERED="
        f"{filtered_test_transfer.shape}"
    )

    if len(filtered_train) != 70258:
        raise ValueError(
            f"Expected 70258 filtered train records, "
            f"found {len(filtered_train)}."
        )

    if len(filtered_test) != 7416:
        raise ValueError(
            f"Expected 7416 filtered test records, "
            f"found {len(filtered_test)}."
        )

    if filtered_train_transfer.shape[0] != len(filtered_train):
        raise ValueError("Filtered train transfer alignment failed.")

    if filtered_test_transfer.shape[0] != len(filtered_test):
        raise ValueError("Filtered test transfer alignment failed.")

    print("FILTERED_TRANSFER_ALIGNMENT=PASS")

    print("VALIDATING_TARGETS")

    y_train, _, target_info = validate_targets(
        filtered_train,
        [],
    )

    test_targets = [
        target_from_record(record)
        for record in filtered_test
    ]

    class_to_index = {
        value: index
        for index, value in enumerate(target_info["class_order"])
    }

    test_known_mask = np.asarray(
        [
            target in class_to_index
            for target in test_targets
        ],
        dtype=bool,
    )

    y_test = np.asarray(
        [
            class_to_index[target]
            for target in np.asarray(test_targets)[test_known_mask]
        ],
        dtype=np.int64,
    )

    print(
        f"ABO_TARGET_CLASSES={target_info['class_count']}"
    )
    print(
        "TEST_KNOWN_CLASS_RECORDS="
        f"{int(test_known_mask.sum())}"
    )
    print(
        "TEST_UNSEEN_TARGET_RECORDS="
        f"{int((~test_known_mask).sum())}"
    )

    if target_info["class_count"] != TARGET_CLASS_COUNT:
        raise ValueError(
            f"Expected {TARGET_CLASS_COUNT} target classes."
        )

    print("BUILDING_FROZEN_B4_2_TEXT_FEATURES")

    train_texts = [
        text_from_record(record)
        for record in filtered_train
    ]

    test_texts = [
        text_from_record(record)
        for record in filtered_test
    ]

    X_text_train = vectorizer.transform(train_texts)
    X_text_test = vectorizer.transform(test_texts)

    print(f"TEXT_TRAIN_SHAPE={X_text_train.shape}")
    print(f"TEXT_TEST_SHAPE={X_text_test.shape}")
    print(
        f"TEXT_VOCABULARY_SIZE={len(vectorizer.vocabulary_)}"
    )
    print("VECTORIZER_FIT_DURING_EXP2_GATE2=False")

    print("USING_GENUINE_ALIGNED_TRAIN_TRANSFER")
    print("TRAIN_TRANSFER_PERMUTED=False")
    print("VALIDATION_USED_FOR_SELECTION=False")
    print("SELECTED_GAMMA=1.0")
    print("TEST_TRANSFER_LOADED=True")
    print("FROZEN_GAMMA=1.0")

    gamma = 1.0

    scaled_train_transfer = (
        gamma * filtered_train_transfer
    ).astype(np.float32, copy=False)

    scaled_test_transfer = (
        gamma * filtered_test_transfer[test_known_mask]
    ).astype(np.float32, copy=False)

    X_train = sparse.hstack(
        [
            X_text_train,
            sparse.csr_matrix(scaled_train_transfer),
        ],
        format="csr",
    )

    X_test = sparse.hstack(
        [
            X_text_test[test_known_mask],
            sparse.csr_matrix(scaled_test_transfer),
        ],
        format="csr",
    )

    print(
        f"COMBINED_TRAIN_SHAPE={X_train.shape}"
    )
    print(
        f"COMBINED_TEST_SHAPE={X_test.shape}"
    )

    classifier = SGDClassifier(
        loss="log_loss",
        max_iter=30,
        random_state=SEED,
        n_jobs=1,
        tol=1e-3,
    )

    print("TRAINING_EXP2_GATE2_MODEL")

    classifier.fit(
        X_train,
        y_train,
    )

    predictions = classifier.predict(X_test)

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print("-" * 72)
    print("EXP2_GATE2_FINAL_TEST")
    print("-" * 72)
    print(f"TEST_ACCURACY={metrics['accuracy']}")
    print(f"TEST_MACRO_F1={metrics['macro_f1']}")
    print(f"TEST_WEIGHTED_F1={metrics['weighted_f1']}")

    report = {
        "experiment": "B6.3.3 Exp2",
        "experiment_type": "genuine_aligned_transfer",
        "gate": "Gate 2",
        "status": "GATE_2_PASS",
        "objective": (
            "Final test evaluation of ABO text plus within-train "
            "row-permuted Rakuten transfer signal using the gamma "
            "selected during Exp2 Gate 1."
        ),
        "selection": {
            "split": "ABO_validation",
            "selected_gamma": 1.0,
            "test_used_for_selection": False,
            "gamma_search_during_gate2": False,
        },
        "target": {
            "dataset": "ABO",
            "representation_version": "B3_v002",
            "target_field": TARGET_FIELD,
            "target_classes": TARGET_CLASS_COUNT,
            "original_train_count": TRAIN_COUNT,
            "original_validation_count": VALIDATION_COUNT,
            "original_test_count": TEST_COUNT,
            "filtered_train_count": len(filtered_train),
            "filtered_test_count": len(filtered_test),
            "test_metric_count": int(test_known_mask.sum()),
            "test_unseen_target_records": int(
                (~test_known_mask).sum()
            ),
            "excluded_target_classes": sorted(
                EXCLUDED_TARGET_CLASSES
            ),
        },
        "permutation": {
            "scope": "none",
            "seed": None,
            "row_permutation": False,
            "validation_permuted": False,
            "test_permuted": False,
            "test_loaded": True,
            "identity_permutation": False,
        },
        "gamma": {
            "value": gamma,
            "selection_split": "ABO_validation",
            "frozen_before_test": True,
        },
        "metrics": metrics,
        "classifier": {
            "type": "SGDClassifier",
            "loss": "log_loss",
            "max_iter": 30,
            "random_state": SEED,
            "n_jobs": 1,
            "tol": 1e-3,
        },
        "dimensions": {
            "text_dimension": int(X_text_train.shape[1]),
            "transfer_dimension": TRANSFER_DIMENSION,
            "combined_dimension": int(X_train.shape[1]),
        },
        "leakage": {
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
        "integrity": {
            "vectorizer_fit_during_gate2": False,
            "label_encoder_fit_during_gate2": False,
            "source_model_modified": False,
            "test_evaluated_once": True,
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
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"REPORT_SAVED={REPORT_PATH}")

    print("=" * 72)
    print("B6.3.3 EXP2 GATE 2=PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
