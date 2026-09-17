"""
ABO B4.2 — Independent Text-only and Image-only Baselines

Frozen methodological contract
------------------------------
- Input: frozen ABO B3 representations only.
- Required B3 representation version: v002.
- Target: product_type, first non-empty value.
- Text baseline: TF-IDF over B3 combined_tokens + linear SGD classifier.
- Image baseline: B3 v002 documented main image only + deterministic
  RGB -> 64x64 bilinear resize -> float32 [0,1] -> flatten
  + incremental SGDClassifier(log_loss).
- Text and image experiments are independent.
- No multimodal fusion is performed in B4.2; fusion is reserved for B5.
- No image downloading, path guessing, alternate-file search, or synthetic data.
- Frozen B2/B3 data and frozen split assignments are never modified.
- Validation/test targets unseen during image training are reported separately.
- Image training supports safe epoch-level checkpoint/resume.

Important B3 v002 schema rule
-----------------------------
The image eligibility metadata is stored under:
record["b3_representation"]["image"]["main"]

NOT under:
record["main_image"]

The latter caused the previous B4.2 implementation to use the wrong subset.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

B3_DIR = PROJECT_ROOT / "data" / "representations" / "abo" / "b3"

MODEL_DIR = PROJECT_ROOT / "data" / "models" / "abo" / "b4.2"

REPORT_DIR = PROJECT_ROOT / "reports" / "baseline" / "abo"

IMAGE_ROOT = PROJECT_ROOT / "ABO_Audit" / "images" / "small"

CHECKPOINT_DIR = MODEL_DIR / "checkpoints" / "image"

CHECKPOINT_PATH = CHECKPOINT_DIR / "latest_image_checkpoint.joblib"
CHECKPOINT_META_PATH = CHECKPOINT_DIR / "latest_image_checkpoint.json"

TEXT_MODEL_PATH = MODEL_DIR / "text_model.joblib"
IMAGE_MODEL_PATH = MODEL_DIR / "image_model.joblib"

REPORT_PATH = REPORT_DIR / "abo_b4.2_report.json"
STATISTICS_PATH = MODEL_DIR / "statistics.json"


# ============================================================================
# FROZEN EXPERIMENT CONTRACT
# ============================================================================

SPLITS = ("train", "validation", "test")

REPRESENTATION_VERSION = "v002"
B4_VERSION = "v002"

SEED = 20260827

TARGET_FIELD = "product_type"


# ============================================================================
# TEXT CONFIGURATION
# ============================================================================

TEXT_MIN_DF = 2
TEXT_MAX_FEATURES = 300_000
TEXT_MAX_ITER = 30


# ============================================================================
# IMAGE CONFIGURATION
# ============================================================================

IMAGE_SIZE = (64, 64)
IMAGE_BATCH_SIZE = 256
IMAGE_EPOCHS = 30


# ============================================================================
# ATOMIC FILE HELPERS
# ============================================================================

def atomic_joblib_dump(payload: Any, path: Path) -> None:
    """Atomically replace a joblib file so an interrupted write does not
    destroy the previous valid checkpoint/model."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    tmp_path = Path(tmp_name)

    try:
        joblib.dump(payload, tmp_path, compress=3)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    tmp_path = Path(tmp_name)

    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_json_dump(path, payload)


# ============================================================================
# BASIC HELPERS
# ============================================================================

def json_loads(line: str) -> dict[str, Any]:
    obj = json.loads(line)

    if not isinstance(obj, dict):
        raise ValueError("ABO B3 record must be a JSON object.")

    return obj


def first_scalar(value: Any) -> str:
    """
    Extract the first meaningful scalar without inventing or transforming
    labels.
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
# B3 VERSION / INPUT VALIDATION
# ============================================================================

def validate_b3_representation_version(
    record: dict[str, Any],
    path: Path,
    line_no: int,
) -> None:
    """
    B3 v002 is stored inside b3_representation.representation_version.
    """
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
    path = B3_DIR / f"{split}.jsonl"

    if not path.is_file():
        raise FileNotFoundError(f"Missing B3 split: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            record = json_loads(line)

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
        raise ValueError(f"B3 split is empty: {path}")

    return records


def validate_split_integrity(
    data: dict[str, list[dict[str, Any]]],
) -> tuple[int, int]:
    duplicate_ids = 0
    all_ids: list[str] = []

    for split in SPLITS:
        ids = [str(record["record_id"]) for record in data[split]]

        duplicate_ids += len(ids) - len(set(ids))
        all_ids.extend(ids)

    cross_split_leak = len(all_ids) - len(set(all_ids))

    if duplicate_ids != 0:
        raise ValueError(
            f"Duplicate record IDs detected: {duplicate_ids}"
        )

    if cross_split_leak != 0:
        raise ValueError(
            f"Cross-split record-ID leakage detected: "
            f"{cross_split_leak}"
        )

    return duplicate_ids, cross_split_leak


# ============================================================================
# METRICS
# ============================================================================

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
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
# IMAGE PATH RESOLUTION — B3 v002
# ============================================================================

def main_image_path(record: dict[str, Any]) -> Path | None:
    """
    Resolve ONLY:

        record["b3_representation"]["image"]["main"]

    Eligibility requires:
      - main metadata exists,
      - physical_file_available == True,
      - usable_for_local_image_model == True,
      - documented relative_path exists,
      - resolved file exists locally,
      - resolved file remains under IMAGE_ROOT,
      - supported extension.

    No path guessing.
    No alternate filename search.
    No downloading.
    No synthetic images.
    """
    representation = record.get("b3_representation")

    if not isinstance(representation, dict):
        return None

    image = representation.get("image")

    if not isinstance(image, dict):
        return None

    main = image.get("main")

    if not isinstance(main, dict):
        return None

    if main.get("physical_file_available") is not True:
        return None

    if main.get("usable_for_local_image_model") is not True:
        return None

    relative_path = main.get("relative_path")

    if not isinstance(relative_path, str):
        return None

    relative_path = relative_path.strip()

    if not relative_path:
        return None

    normalized = relative_path.replace("/", "\\")

    candidate = IMAGE_ROOT.joinpath(*normalized.split("\\"))

    try:
        resolved_root = IMAGE_ROOT.resolve()
        resolved_candidate = candidate.resolve()
        resolved_candidate.relative_to(resolved_root)
    except (ValueError, OSError):
        return None

    if not resolved_candidate.is_file():
        return None

    if resolved_candidate.suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
    }:
        return None

    return resolved_candidate


def image_records(
    records: list[dict[str, Any]],
) -> Iterator[tuple[Path, str]]:
    for record in records:
        target = target_from_record(record)

        if not target:
            continue

        path = main_image_path(record)

        if path is None:
            continue

        yield path, target


def collect_image_labels(
    records: list[dict[str, Any]],
) -> list[str]:
    return [
        target
        for _, target in image_records(records)
    ]


# ============================================================================
# IMAGE FEATURE EXTRACTION
# ============================================================================

def load_image_features(path: Path) -> np.ndarray:
    """
    Deterministic image representation:
        disk image
        -> RGB
        -> 64x64 bilinear resize
        -> float32
        -> divide by 255
        -> flatten
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")

        resized = rgb.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )

        array = np.asarray(
            resized,
            dtype=np.float32,
        )

    array /= 255.0

    return array.reshape(-1)


def image_feature_batches(
    records: list[dict[str, Any]],
    label_encoder: LabelEncoder,
    known_only: bool,
    batch_size: int = IMAGE_BATCH_SIZE,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    feature_batch: list[np.ndarray] = []
    target_batch: list[int] = []

    class_to_id = {
        str(label): int(index)
        for index, label in enumerate(label_encoder.classes_)
    }

    for path, target in image_records(records):
        target_id = class_to_id.get(target)

        if target_id is None:
            if known_only:
                continue
            raise ValueError(
                "Encountered image target outside label encoder."
            )

        try:
            features = load_image_features(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read image: {path}"
            ) from exc

        feature_batch.append(features)
        target_batch.append(target_id)

        if len(feature_batch) >= batch_size:
            yield (
                np.asarray(feature_batch, dtype=np.float32),
                np.asarray(target_batch, dtype=np.int64),
            )
            feature_batch.clear()
            target_batch.clear()

    if feature_batch:
        yield (
            np.asarray(feature_batch, dtype=np.float32),
            np.asarray(target_batch, dtype=np.int64),
        )


# ============================================================================
# IMAGE CHECKPOINTING
# ============================================================================

def checkpoint_config() -> dict[str, Any]:
    return {
        "representation_version": REPRESENTATION_VERSION,
        "b4_version": B4_VERSION,
        "seed": SEED,
        "target_field": TARGET_FIELD,
        "image_size": list(IMAGE_SIZE),
        "image_batch_size": IMAGE_BATCH_SIZE,
        "image_epochs": IMAGE_EPOCHS,
        "shuffle": False,
        "training_method": "SGDClassifier.partial_fit",
        "feature_representation": (
            "RGB -> 64x64 bilinear resize -> float32 -> "
            "[0,1] normalization -> flatten"
        ),
    }


def save_image_checkpoint(
    classifier: SGDClassifier,
    encoder: LabelEncoder,
    completed_epoch: int,
    train_eligible_records: int,
) -> None:
    payload = {
        "checkpoint_type": "abo_b4.2_image_epoch_checkpoint",
        "checkpoint_version": "v001",
        "completed_epoch": int(completed_epoch),
        "classifier": classifier,
        "label_encoder": encoder,
        "train_eligible_records": int(train_eligible_records),
        "config": checkpoint_config(),
    }

    atomic_joblib_dump(
        payload,
        CHECKPOINT_PATH,
    )

    metadata = {
        "checkpoint_type": "abo_b4.2_image_epoch_checkpoint",
        "checkpoint_version": "v001",
        "completed_epoch": int(completed_epoch),
        "train_eligible_records": int(train_eligible_records),
        "config": checkpoint_config(),
        "checkpoint_path": str(CHECKPOINT_PATH),
    }

    atomic_json_dump(
        CHECKPOINT_META_PATH,
        metadata,
    )


def load_image_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"No image checkpoint exists at {CHECKPOINT_PATH}"
        )

    checkpoint = joblib.load(CHECKPOINT_PATH)

    if not isinstance(checkpoint, dict):
        raise ValueError("Image checkpoint is not a dictionary.")

    if checkpoint.get("checkpoint_type") != (
        "abo_b4.2_image_epoch_checkpoint"
    ):
        raise ValueError("Unexpected image checkpoint type.")

    if checkpoint.get("config") != checkpoint_config():
        raise ValueError(
            "Image checkpoint configuration does not match the current "
            "B4.2 configuration. Refusing unsafe resume."
        )

    if not isinstance(
        checkpoint.get("classifier"),
        SGDClassifier,
    ):
        raise ValueError("Checkpoint classifier is invalid.")

    if not isinstance(
        checkpoint.get("label_encoder"),
        LabelEncoder,
    ):
        raise ValueError("Checkpoint label encoder is invalid.")

    return checkpoint


# ============================================================================
# IMAGE TRAINING
# ============================================================================

def train_image_classifier(
    train_records: list[dict[str, Any]],
    encoder: LabelEncoder,
    epochs_this_run: int,
    resume: bool,
) -> tuple[SGDClassifier, int, bool]:
    """
    Train for at most epochs_this_run additional epochs.

    Checkpoint is written after EVERY COMPLETED EPOCH.

    If the process is interrupted:
      - all completed epochs remain in latest_image_checkpoint.joblib;
      - at most the currently running epoch is lost;
      - rerun with --resume to continue from the last completed epoch.

    Returns:
        classifier,
        completed_epoch,
        training_complete
    """
    train_eligible = len(
        collect_image_labels(train_records)
    )

    if train_eligible == 0:
        raise ValueError(
            "No eligible physical-image training records."
        )

    if resume:
        checkpoint = load_image_checkpoint()

        classifier = checkpoint["classifier"]
        checkpoint_encoder = checkpoint["label_encoder"]
        completed_epoch = int(checkpoint["completed_epoch"])

        if not np.array_equal(
            checkpoint_encoder.classes_,
            encoder.classes_,
        ):
            raise ValueError(
                "Checkpoint label space does not match current "
                "B3-derived training label space. Refusing resume."
            )

        if int(checkpoint["train_eligible_records"]) != train_eligible:
            raise ValueError(
                "Checkpoint training eligibility count does not match "
                "current frozen B3 data. Refusing resume."
            )

        if completed_epoch >= IMAGE_EPOCHS:
            print(
                f"  Checkpoint already has {completed_epoch}/{IMAGE_EPOCHS} "
                "completed epochs."
            )
            return classifier, completed_epoch, True

        print(
            f"  Resuming image training from epoch "
            f"{completed_epoch + 1}/{IMAGE_EPOCHS}."
        )

    else:
        classifier = SGDClassifier(
            loss="log_loss",
            random_state=SEED,
            n_jobs=1,
            tol=None,
        )

        completed_epoch = 0

        print(
            "  Starting image training from epoch 1."
        )

    target_epoch = min(
        IMAGE_EPOCHS,
        completed_epoch + epochs_this_run,
    )

    classes = np.arange(
        len(encoder.classes_),
        dtype=np.int64,
    )

    while completed_epoch < target_epoch:
        epoch = completed_epoch + 1

        batch_count = 0
        record_count = 0

        for X_batch, y_batch in image_feature_batches(
            train_records,
            encoder,
            known_only=True,
            batch_size=IMAGE_BATCH_SIZE,
        ):
            if not hasattr(classifier, "classes_"):
                classifier.partial_fit(
                    X_batch,
                    y_batch,
                    classes=classes,
                )
            else:
                classifier.partial_fit(
                    X_batch,
                    y_batch,
                )

            batch_count += 1
            record_count += len(y_batch)

        if batch_count == 0:
            raise ValueError(
                f"Image epoch {epoch} processed zero batches."
            )

        completed_epoch = epoch

        save_image_checkpoint(
            classifier=classifier,
            encoder=encoder,
            completed_epoch=completed_epoch,
            train_eligible_records=train_eligible,
        )

        print(
            f"  Completed image epoch "
            f"{completed_epoch}/{IMAGE_EPOCHS} "
            f"({record_count:,} records, {batch_count:,} batches)."
        )

    training_complete = completed_epoch >= IMAGE_EPOCHS

    return classifier, completed_epoch, training_complete


# ============================================================================
# IMAGE EVALUATION
# ============================================================================

def predict_image_records(
    records: list[dict[str, Any]],
    encoder: LabelEncoder,
    classifier: SGDClassifier,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Evaluate in bounded-memory batches.

    Returns:
        y_true,
        y_pred,
        total_eligible_records,
        unseen_target_records
    """
    total_eligible = 0
    unseen_targets = 0

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    known_classes = set(
        str(value)
        for value in encoder.classes_
    )

    class_to_id = {
        str(label): int(index)
        for index, label in enumerate(encoder.classes_)
    }

    for path, target in image_records(records):
        total_eligible += 1

        if target not in known_classes:
            unseen_targets += 1
            continue

        try:
            features = load_image_features(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read image: {path}"
            ) from exc

        X = features.reshape(1, -1)

        y_true_parts.append(
            np.asarray(
                [class_to_id[target]],
                dtype=np.int64,
            )
        )

        y_pred_parts.append(
            classifier.predict(X).astype(np.int64)
        )

    if not y_true_parts:
        raise ValueError(
            "No evaluation image records have targets observed in training."
        )

    return (
        np.concatenate(y_true_parts),
        np.concatenate(y_pred_parts),
        total_eligible,
        unseen_targets,
    )


# ============================================================================
# TEXT BASELINE
# ============================================================================

def run_text_baseline(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> dict[str, Any]:
    def make_pairs(
        records: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []

        for record in records:
            text = text_from_record(record)
            target = target_from_record(record)

            if text and target:
                pairs.append((text, target))

        return pairs

    train_pairs = make_pairs(train)
    validation_pairs = make_pairs(validation)
    test_pairs = make_pairs(test)

    if not train_pairs:
        raise ValueError("No eligible ABO text training records.")

    if not validation_pairs:
        raise ValueError("No eligible ABO text validation records.")

    if not test_pairs:
        raise ValueError("No eligible ABO text test records.")

    train_text, train_targets = zip(*train_pairs)
    validation_text, validation_targets = zip(*validation_pairs)
    test_text, test_targets = zip(*test_pairs)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=TEXT_MIN_DF,
        max_features=TEXT_MAX_FEATURES,
        token_pattern=r"(?u)\S+",
        dtype=np.float32,
    )

    X_train = vectorizer.fit_transform(train_text)
    X_validation = vectorizer.transform(validation_text)
    X_test = vectorizer.transform(test_text)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_targets)

    known_classes = set(
        str(value)
        for value in encoder.classes_
    )

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

    classifier = SGDClassifier(
        loss="log_loss",
        max_iter=TEXT_MAX_ITER,
        random_state=SEED,
        n_jobs=1,
        tol=1e-3,
    )

    classifier.fit(X_train, y_train)

    y_validation = encoder.transform(
        np.asarray(validation_targets)[validation_mask]
    )

    validation_predictions = classifier.predict(
        X_validation[validation_mask]
    )

    y_test = encoder.transform(
        np.asarray(test_targets)[test_mask]
    )

    test_predictions = classifier.predict(
        X_test[test_mask]
    )

    validation_metrics = evaluate(
        y_validation,
        validation_predictions,
    )

    test_metrics = evaluate(
        y_test,
        test_predictions,
    )

    atomic_joblib_dump(
        {
            "vectorizer": vectorizer,
            "label_encoder": encoder,
            "classifier": classifier,
            "target_field": TARGET_FIELD,
            "target_definition": (
                "first non-empty product_type value"
            ),
            "representation_version": REPRESENTATION_VERSION,
            "b4_version": B4_VERSION,
        },
        TEXT_MODEL_PATH,
    )

    return {
        "status": "PASS",
        "modality": "text_only",
        "target_field": TARGET_FIELD,
        "target_definition": (
            "first non-empty product_type value"
        ),
        "representation_version": REPRESENTATION_VERSION,
        "train_eligible_records": len(train_pairs),
        "validation_eligible_records": len(validation_pairs),
        "test_eligible_records": len(test_pairs),
        "train_classes": int(len(encoder.classes_)),
        "train_vocabulary": int(len(vectorizer.vocabulary_)),
        "validation_unseen_target_records": int(
            (~validation_mask).sum()
        ),
        "test_unseen_target_records": int(
            (~test_mask).sum()
        ),
        "validation": validation_metrics,
        "test": test_metrics,
        "configuration": {
            "vectorizer": "TF-IDF",
            "min_df": TEXT_MIN_DF,
            "max_features": TEXT_MAX_FEATURES,
            "classifier": "SGDClassifier(log_loss)",
            "max_iter": TEXT_MAX_ITER,
            "random_seed": SEED,
            "fit_scope": "training split only",
        },
    }


# ============================================================================
# IMAGE BASELINE
# ============================================================================

def run_image_baseline(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    epochs_this_run: int,
    resume: bool,
) -> dict[str, Any]:
    train_image_labels = collect_image_labels(train)
    validation_image_labels = collect_image_labels(validation)
    test_image_labels = collect_image_labels(test)

    if not train_image_labels:
        raise ValueError(
            "No eligible physical-image training records."
        )

    if not validation_image_labels:
        raise ValueError(
            "No eligible physical-image validation records."
        )

    if not test_image_labels:
        raise ValueError(
            "No eligible physical-image test records."
        )

    encoder = LabelEncoder()
    encoder.fit(train_image_labels)

    print(
        f"  Image training classes: "
        f"{len(encoder.classes_):,}"
    )

    classifier, completed_epoch, training_complete = (
        train_image_classifier(
            train_records=train,
            encoder=encoder,
            epochs_this_run=epochs_this_run,
            resume=resume,
        )
    )

    if not training_complete:
        progress = {
            "status": "IN_PROGRESS",
            "modality": "image_only",
            "representation_version": REPRESENTATION_VERSION,
            "b4_version": B4_VERSION,
            "completed_epochs": completed_epoch,
            "total_epochs": IMAGE_EPOCHS,
            "next_command": (
                "python src/abo/b4_baseline.py "
                "--modality image --resume "
                f"--epochs-per-run {epochs_this_run}"
            ),
        }

        save_json(
            CHECKPOINT_DIR / "progress.json",
            progress,
        )

        print(
            "\nImage training interval finished."
        )
        print(
            f"Checkpoint preserved at:\n  {CHECKPOINT_PATH}"
        )
        print(
            f"Progress: {completed_epoch}/{IMAGE_EPOCHS} epochs."
        )
        print(
            "No validation/test evaluation was performed because "
            "the final image model is not yet complete."
        )

        return progress

    # Final model is fixed only after all 30 epochs.
    validation_result = predict_image_records(
        validation,
        encoder,
        classifier,
    )

    (
        y_validation,
        validation_predictions,
        validation_eligible,
        validation_unseen,
    ) = validation_result

    validation_metrics = evaluate(
        y_validation,
        validation_predictions,
    )

    test_result = predict_image_records(
        test,
        encoder,
        classifier,
    )

    (
        y_test,
        test_predictions,
        test_eligible,
        test_unseen,
    ) = test_result

    test_metrics = evaluate(
        y_test,
        test_predictions,
    )

    atomic_joblib_dump(
        {
            "label_encoder": encoder,
            "classifier": classifier,
            "target_field": TARGET_FIELD,
            "target_definition": (
                "first non-empty product_type value"
            ),
            "image_size": IMAGE_SIZE,
            "representation_version": REPRESENTATION_VERSION,
            "b4_version": B4_VERSION,
            "training_epochs": IMAGE_EPOCHS,
            "training_completed": True,
        },
        IMAGE_MODEL_PATH,
    )

    return {
        "status": "PASS",
        "modality": "image_only",
        "target_field": TARGET_FIELD,
        "target_definition": (
            "first non-empty product_type value"
        ),
        "representation_version": REPRESENTATION_VERSION,
        "train_eligible_physical_image_records": len(
            train_image_labels
        ),
        "validation_eligible_physical_image_records": (
            validation_eligible
        ),
        "test_eligible_physical_image_records": (
            test_eligible
        ),
        "train_classes": int(len(encoder.classes_)),
        "validation_unseen_target_records": validation_unseen,
        "test_unseen_target_records": test_unseen,
        "validation": validation_metrics,
        "test": test_metrics,
        "configuration": {
            "image_input": (
                "B3 v002 documented physical main image only"
            ),
            "image_size": list(IMAGE_SIZE),
            "feature_representation": (
                "RGB, deterministic 64x64 bilinear resize, "
                "float32 normalization to [0,1], flatten"
            ),
            "classifier": "SGDClassifier(log_loss)",
            "training_method": "incremental partial_fit",
            "epochs": IMAGE_EPOCHS,
            "batch_size": IMAGE_BATCH_SIZE,
            "random_seed": SEED,
            "shuffle": False,
            "memory_strategy": "streaming bounded-memory batches",
            "checkpointing": "after every completed epoch",
            "resume_supported": True,
        },
        "physical_image_policy": {
            "download": False,
            "path_guessing": False,
            "synthetic_images": False,
            "b3_physical_file_required": True,
            "b3_local_image_model_eligibility_required": True,
            "filesystem_existence_rechecked": True,
            "documented_relative_path_only": True,
            "eligibility_source": (
                "record.b3_representation.image.main"
            ),
        },
    }


# ============================================================================
# REPORT
# ============================================================================

def build_final_report(
    text_result: dict[str, Any] | None,
    image_result: dict[str, Any] | None,
    duplicate_ids: int,
    cross_split_leak: int,
    modality: str,
) -> dict[str, Any]:
    return {
        "report": "abo_b4.2_baseline",
        "report_version": B4_VERSION,
        "dataset": "Amazon Berkeley Objects (ABO)",
        "stage": "B4.2",
        "status": "PASS",
        "execution_modality": modality,
        "target_contract": {
            "target_field": TARGET_FIELD,
            "definition": "first non-empty product_type value",
            "target_frozen_before_training": True,
            "target_source": "frozen B3 record field",
        },
        "input_contract": {
            "source": "data/representations/abo/b3/{split}.jsonl",
            "representation_version": REPRESENTATION_VERSION,
            "splits": list(SPLITS),
        },
        "split_integrity": {
            "duplicate_record_ids": duplicate_ids,
            "cross_split_record_id_leakage": cross_split_leak,
            "split_assignments_regenerated": False,
        },
        "baselines": {
            "text_only": text_result,
            "image_only": image_result,
        },
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "b4_seed": SEED,
            "b5_b6_information_used": False,
            "raw_or_frozen_data_modified": False,
        },
        "fusion": {
            "performed": False,
            "reserved_for": "B5",
        },
        "data_policy": {
            "b3_modified": False,
            "b2_modified": False,
            "splits_regenerated": False,
            "image_download": False,
            "image_path_guessing": False,
            "synthetic_images": False,
        },
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ABO B4.2 independent text-only and image-only baselines "
            "with resumable image training."
        )
    )

    parser.add_argument(
        "--modality",
        choices=("text", "image", "all"),
        default="all",
        help=(
            "Run text baseline, image baseline, or both. "
            "Use image for interval/resume training."
        ),
    )

    parser.add_argument(
        "--epochs-per-run",
        type=int,
        default=1,
        help=(
            "Maximum number of NEW image epochs to run in this invocation. "
            "Default: 1. Ignored for text baseline."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume image training from the latest completed-epoch "
            "checkpoint."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    args = parse_args()

    if args.epochs_per_run < 1:
        raise ValueError("--epochs-per-run must be >= 1.")

    if args.modality != "image" and args.resume:
        raise ValueError(
            "--resume is only valid with --modality image."
        )

    print("=" * 72)
    print("ABO B4.2 — INDEPENDENT TEXT-ONLY + IMAGE-ONLY BASELINES")
    print("=" * 72)

    print(f"B3 representation version : {REPRESENTATION_VERSION}")
    print(f"B4.2 version              : {B4_VERSION}")
    print(f"Random seed               : {SEED}")
    print(f"Selected modality         : {args.modality}")

    if args.modality in ("image", "all"):
        print(f"Image root                : {IMAGE_ROOT}")
        print(f"Image checkpoint         : {CHECKPOINT_PATH}")

    if not B3_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing B3 directory: {B3_DIR}"
        )

    if args.modality in ("image", "all"):
        if not IMAGE_ROOT.is_dir():
            raise FileNotFoundError(
                f"Missing physical image root: {IMAGE_ROOT}"
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading frozen B3 splits...")
    data = {
        split: load_records(split)
        for split in SPLITS
    }

    for split in SPLITS:
        print(
            f"  {split:12s}: {len(data[split]):,}"
        )

    print("\nChecking record-ID integrity...")
    duplicate_ids, cross_split_leak = validate_split_integrity(data)

    print("  Duplicate record IDs : 0")
    print("  Cross-split leakage  : 0")

    text_result: dict[str, Any] | None = None
    image_result: dict[str, Any] | None = None

    if args.modality in ("text", "all"):
        print("\nRunning TEXT-ONLY baseline...")
        text_result = run_text_baseline(
            data["train"],
            data["validation"],
            data["test"],
        )

        print(
            f"  Validation accuracy : "
            f"{text_result['validation']['accuracy']:.6f}"
        )
        print(
            f"  Validation Macro F1 : "
            f"{text_result['validation']['macro_f1']:.6f}"
        )
        print(
            f"  Test accuracy       : "
            f"{text_result['test']['accuracy']:.6f}"
        )
        print(
            f"  Test Macro F1       : "
            f"{text_result['test']['macro_f1']:.6f}"
        )

    if args.modality in ("image", "all"):
        if args.modality == "all":
            print(
                "\nNOTE: --modality all performs only "
                f"{args.epochs_per_run} new image epoch(s) in this run."
            )

        print("\nRunning IMAGE-ONLY baseline...")

        image_result = run_image_baseline(
            data["train"],
            data["validation"],
            data["test"],
            epochs_this_run=args.epochs_per_run,
            resume=args.resume,
        )

        if image_result["status"] == "IN_PROGRESS":
            print("\n" + "-" * 72)
            print("B4.2 IMAGE TRAINING STATUS: IN PROGRESS")
            print("-" * 72)
            print(
                f"Completed epochs: "
                f"{image_result['completed_epochs']}/"
                f"{image_result['total_epochs']}"
            )
            print(
                "The checkpoint is safe to reuse."
            )
            print(
                "Run again with --modality image --resume "
                "--epochs-per-run N"
            )
            return 0

        print(
            "  Physical-image eligible records:"
            f" train={image_result['train_eligible_physical_image_records']:,},"
            f" validation={image_result['validation_eligible_physical_image_records']:,},"
            f" test={image_result['test_eligible_physical_image_records']:,}"
        )

        print(
            f"  Validation accuracy : "
            f"{image_result['validation']['accuracy']:.6f}"
        )
        print(
            f"  Validation Macro F1 : "
            f"{image_result['validation']['macro_f1']:.6f}"
        )
        print(
            f"  Test accuracy       : "
            f"{image_result['test']['accuracy']:.6f}"
        )
        print(
            f"  Test Macro F1       : "
            f"{image_result['test']['macro_f1']:.6f}"
        )

    # Final report is written only when every selected modality is complete.
    report = build_final_report(
        text_result=text_result,
        image_result=image_result,
        duplicate_ids=duplicate_ids,
        cross_split_leak=cross_split_leak,
        modality=args.modality,
    )

    save_json(REPORT_PATH, report)
    save_json(STATISTICS_PATH, report)

    print("\n" + "-" * 72)
    print("B4.2 ABO SUMMARY")
    print("-" * 72)

    if text_result is not None:
        print("Text-only baseline : PASS")

    if image_result is not None:
        print("Image-only baseline: PASS")

    print(
        f"Record-ID leakage  : {cross_split_leak}"
    )
    print(
        f"B3 version         : {REPRESENTATION_VERSION}"
    )
    print(
        "Fusion             : NOT PERFORMED (reserved for B5)"
    )
    print("\nSTATUS: PASS")

    print("\nArtifacts:")
    print(f"  Models : {MODEL_DIR}")
    print(f"  Report : {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
