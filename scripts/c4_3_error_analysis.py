from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# C4.3 — FROZEN C-BASE TEST ERROR ANALYSIS
# ============================================================

SEED = 20260827
DATASET = "ABO"
PHASE = "C4.3"

NUM_CLASSES = 549
TEST_POPULATION = 7346

ROOT = Path(".")
C4_DIR = ROOT / "data" / "models" / "abo" / "phase_c" / "c4"
TEST_EMBED_DIR = C4_DIR / "test_embeddings"
EVAL_DIR = C4_DIR / "evaluation"

PREDICTIONS_PATH = EVAL_DIR / "cbase_test_predictions.npz"
RESULTS_PATH = EVAL_DIR / "cbase_test_results.json"
C4_1_MANIFEST_PATH = (
    TEST_EMBED_DIR / "c4_test_embedding_manifest.json"
)

OUTPUT_DIR = EVAL_DIR / "c4_3_error_analysis"

TRUE_LABELS_PATH = TEST_EMBED_DIR / "test_labels.npy"
RECORD_IDS_PATH = TEST_EMBED_DIR / "test_record_ids.json"

# Frozen B4.2 image baseline artifact used for the Phase C
# 549-class ordering.
B4_IMAGE_PATH = (
    ROOT / "data" / "models" / "abo" / "b4.2" / "image_model.joblib"
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def assert_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {description} does not exist: {path}"
        )


def load_frozen_b4_classes(path: Path) -> list[str]:
    """
    Read-only loading of the frozen B4.2 image model.

    C4.1/C4.2 established that the artifact stores the class
    ordering in:
        bundle["label_encoder"].classes_
    """
    import joblib

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise RuntimeError(
            f"Unexpected B4.2 image artifact type: {type(bundle)}"
        )

    if "label_encoder" not in bundle:
        raise RuntimeError(
            "B4.2 image artifact does not contain "
            "'label_encoder'."
        )

    classes = bundle["label_encoder"].classes_

    classes = [str(x) for x in classes]

    if len(classes) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} frozen image classes, "
            f"found {len(classes)}."
        )

    if len(set(classes)) != len(classes):
        raise RuntimeError(
            "Frozen B4.2 class ordering contains duplicate labels."
        )

    return classes


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 72)
    print("C4.3 — FROZEN C-BASE TEST ERROR ANALYSIS")
    print("=" * 72)
    print(f"Phase                  : {PHASE}")
    print(f"Dataset                : {DATASET}")
    print(f"Test population        : {TEST_POPULATION}")
    print(f"Classes                : {NUM_CLASSES}")
    print(f"Seed                   : {SEED}")
    print("Training               : NONE")
    print("Model selection        : NONE")
    print("Hyperparameter tuning  : NONE")
    print("")

    # --------------------------------------------------------
    # Verify required artifacts
    # --------------------------------------------------------

    assert_exists(PREDICTIONS_PATH, "C4.2 predictions")
    assert_exists(RESULTS_PATH, "C4.2 results")
    assert_exists(C4_1_MANIFEST_PATH, "C4.1 test manifest")
    assert_exists(TRUE_LABELS_PATH, "C4.1 test labels")
    assert_exists(RECORD_IDS_PATH, "C4.1 test record IDs")
    assert_exists(B4_IMAGE_PATH, "frozen B4.2 image model")

    # --------------------------------------------------------
    # Load C4.2 results
    # --------------------------------------------------------

    c4_2_results = load_json(RESULTS_PATH)

    if c4_2_results.get("phase") != "C4.2":
        raise RuntimeError(
            "Unexpected C4.2 results phase."
        )

    if c4_2_results.get("test_usage") != "inference_only":
        raise RuntimeError(
            "C4.2 artifact does not declare inference-only test usage."
        )

    if c4_2_results.get("model_selection_from_test") is not False:
        raise RuntimeError(
            "C4.2 indicates model selection from test."
        )

    if c4_2_results.get("hyperparameter_selection_from_test") is not False:
        raise RuntimeError(
            "C4.2 indicates hyperparameter selection from test."
        )

    if c4_2_results.get("training_on_test") is not False:
        raise RuntimeError(
            "C4.2 indicates training on test."
        )

    if c4_2_results.get("b3_modified") is not False:
        raise RuntimeError(
            "C4.2 indicates B3 was modified."
        )

    if c4_2_results.get("c3_modified") is not False:
        raise RuntimeError(
            "C4.2 indicates C3 was modified."
        )

    if c4_2_results.get("test_population") != TEST_POPULATION:
        raise RuntimeError(
            "C4.2 test population does not match locked population."
        )

    # --------------------------------------------------------
    # Load C4.1 manifest
    # --------------------------------------------------------

    c4_1_manifest = load_json(C4_1_MANIFEST_PATH)

    print("C4.1 MANIFEST")
    print(
        f"Manifest path          : {C4_1_MANIFEST_PATH}"
    )
    print(
        f"Manifest SHA-256       : "
        f"{sha256_file(C4_1_MANIFEST_PATH)}"
    )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    prediction_bundle = np.load(
        PREDICTIONS_PATH,
        allow_pickle=False,
    )

    if prediction_bundle.files != ["predictions"]:
        raise RuntimeError(
            f"Unexpected prediction keys: "
            f"{prediction_bundle.files}"
        )

    predictions = np.asarray(
        prediction_bundle["predictions"],
        dtype=np.int64,
    )

    if predictions.shape != (TEST_POPULATION,):
        raise RuntimeError(
            f"Unexpected prediction shape: {predictions.shape}"
        )

    if not np.all(
        (predictions >= 0)
        & (predictions < NUM_CLASSES)
    ):
        raise RuntimeError(
            "Predictions contain class IDs outside 0..548."
        )

    # --------------------------------------------------------
    # Load true labels
    # --------------------------------------------------------

    true_labels = np.load(
        TRUE_LABELS_PATH,
        allow_pickle=False,
    ).astype(np.int64, copy=False)

    if true_labels.shape != (TEST_POPULATION,):
        raise RuntimeError(
            f"Unexpected label shape: {true_labels.shape}"
        )

    if not np.all(
        (true_labels >= 0)
        & (true_labels < NUM_CLASSES)
    ):
        raise RuntimeError(
            "True labels contain class IDs outside 0..548."
        )

    # --------------------------------------------------------
    # Load record IDs
    # --------------------------------------------------------

    record_ids = load_json(RECORD_IDS_PATH)

    if not isinstance(record_ids, list):
        raise RuntimeError(
            "test_record_ids.json is not a list."
        )

    if len(record_ids) != TEST_POPULATION:
        raise RuntimeError(
            f"Expected {TEST_POPULATION} record IDs, "
            f"found {len(record_ids)}."
        )

    if len(set(record_ids)) != TEST_POPULATION:
        raise RuntimeError(
            "Test record IDs are not unique."
        )

    # --------------------------------------------------------
    # Load frozen class ordering
    # --------------------------------------------------------

    class_names = load_frozen_b4_classes(B4_IMAGE_PATH)

    # --------------------------------------------------------
    # Independent metric verification
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    reported_metrics = c4_2_results["metrics"]

    print("")
    print("C4.2 METRIC RECOMPUTATION")
    print(f"Accuracy               : {accuracy:.6f}")
    print(f"Macro-F1               : {macro_f1:.6f}")
    print(f"Weighted-F1            : {weighted_f1:.6f}")

    tolerance = 1e-12

    if abs(accuracy - reported_metrics["accuracy"]) > tolerance:
        raise RuntimeError(
            "C4.2 Accuracy does not reproduce independently."
        )

    if abs(macro_f1 - reported_metrics["macro_f1"]) > tolerance:
        raise RuntimeError(
            "C4.2 Macro-F1 does not reproduce independently."
        )

    if abs(weighted_f1 - reported_metrics["weighted_f1"]) > tolerance:
        raise RuntimeError(
            "C4.2 Weighted-F1 does not reproduce independently."
        )

    print("PASS: C4.2 metrics reproduced exactly.")

    # --------------------------------------------------------
    # Per-class report
    # --------------------------------------------------------

    labels = np.arange(NUM_CLASSES)

    report = classification_report(
        true_labels,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=labels,
    )

    # --------------------------------------------------------
    # Per-class statistics
    # --------------------------------------------------------

    class_rows = []

    for class_id, class_name in enumerate(class_names):
        support = int(np.sum(true_labels == class_id))
        predicted_count = int(
            np.sum(predictions == class_id)
        )
        correct = int(cm[class_id, class_id])
        incorrect = support - correct

        stats = report[class_name]

        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "predicted_count": predicted_count,
                "correct": correct,
                "incorrect": incorrect,
                "precision": float(stats["precision"]),
                "recall": float(stats["recall"]),
                "f1": float(stats["f1-score"]),
            }
        )

    # --------------------------------------------------------
    # Confusion pairs
    # --------------------------------------------------------

    confusion_pairs = []

    for true_id in range(NUM_CLASSES):
        for pred_id in range(NUM_CLASSES):
            if true_id == pred_id:
                continue

            count = int(cm[true_id, pred_id])

            if count <= 0:
                continue

            confusion_pairs.append(
                {
                    "true_class_id": true_id,
                    "true_class": class_names[true_id],
                    "predicted_class_id": pred_id,
                    "predicted_class": class_names[pred_id],
                    "count": count,
                }
            )

    confusion_pairs.sort(
        key=lambda x: (
            -x["count"],
            x["true_class_id"],
            x["predicted_class_id"],
        )
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    true_support_total = int(
        np.bincount(
            true_labels,
            minlength=NUM_CLASSES,
        ).sum()
    )

    prediction_total = int(
        np.bincount(
            predictions,
            minlength=NUM_CLASSES,
        ).sum()
    )

    zero_prediction_classes = [
        {
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "support": row["support"],
        }
        for row in class_rows
        if row["predicted_count"] == 0
    ]

    zero_test_support_classes = [
        {
            "class_id": row["class_id"],
            "class_name": row["class_name"],
        }
        for row in class_rows
        if row["support"] == 0
    ]

    # --------------------------------------------------------
    # Correct / incorrect record-level analysis
    # --------------------------------------------------------

    correct_mask = true_labels == predictions

    correct_count = int(np.sum(correct_mask))
    incorrect_count = int(np.sum(~correct_mask))

    incorrect_records = []

    for idx in np.flatnonzero(~correct_mask):
        true_id = int(true_labels[idx])
        pred_id = int(predictions[idx])

        incorrect_records.append(
            {
                "record_id": str(record_ids[idx]),
                "true_class_id": true_id,
                "true_class": class_names[true_id],
                "predicted_class_id": pred_id,
                "predicted_class": class_names[pred_id],
            }
        )

    # --------------------------------------------------------
    # Support buckets
    # --------------------------------------------------------

    def support_bucket(n: int) -> str:
        if n == 0:
            return "0"
        if n <= 5:
            return "1-5"
        if n <= 10:
            return "6-10"
        if n <= 25:
            return "11-25"
        if n <= 50:
            return "26-50"
        if n <= 100:
            return "51-100"
        if n <= 500:
            return "101-500"
        return "501+"

    support_bucket_summary = {}

    for row in class_rows:
        bucket = support_bucket(row["support"])

        if bucket not in support_bucket_summary:
            support_bucket_summary[bucket] = {
                "classes": 0,
                "total_support": 0,
                "correct": 0,
                "incorrect": 0,
            }

        entry = support_bucket_summary[bucket]

        entry["classes"] += 1
        entry["total_support"] += row["support"]
        entry["correct"] += row["correct"]
        entry["incorrect"] += row["incorrect"]

    for bucket, entry in support_bucket_summary.items():
        if entry["total_support"] > 0:
            entry["accuracy"] = (
                entry["correct"]
                / entry["total_support"]
            )
        else:
            entry["accuracy"] = None

    # --------------------------------------------------------
    # Aggregate summary
    # --------------------------------------------------------

    summary = {
        "phase": PHASE,
        "dataset": DATASET,
        "target_field": "product_type",
        "num_classes": NUM_CLASSES,
        "test_population": TEST_POPULATION,
        "seed": SEED,
        "test_usage": "frozen C4.2 prediction error analysis only",
        "training": False,
        "model_selection": False,
        "hyperparameter_selection": False,
        "c4_2_metrics_recomputed": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "prediction_count": int(prediction_total),
        "true_label_count": int(true_support_total),
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "error_rate": float(incorrect_count / TEST_POPULATION),
        "zero_prediction_class_count": len(
            zero_prediction_classes
        ),
        "zero_test_support_class_count": len(
            zero_test_support_classes
        ),
        "nonzero_confusion_pair_count": len(
            confusion_pairs
        ),
        "top_confusion_pairs": confusion_pairs[:50],
        "top_low_f1_classes": sorted(
            class_rows,
            key=lambda x: (
                x["f1"],
                -x["support"],
                x["class_id"],
            ),
        )[:50],
        "top_high_support_classes": sorted(
            class_rows,
            key=lambda x: (
                -x["support"],
                x["class_id"],
            ),
        )[:50],
        "zero_prediction_classes": zero_prediction_classes,
        "support_bucket_summary": support_bucket_summary,
    }

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_write_json(
        OUTPUT_DIR / "c4_3_summary.json",
        summary,
    )

    atomic_write_json(
        OUTPUT_DIR / "c4_3_per_class.json",
        {
            "phase": PHASE,
            "rows": class_rows,
        },
    )

    atomic_write_json(
        OUTPUT_DIR / "c4_3_confusion_pairs.json",
        {
            "phase": PHASE,
            "nonzero_pairs": confusion_pairs,
        },
    )

    atomic_write_json(
        OUTPUT_DIR / "c4_3_incorrect_records.json",
        {
            "phase": PHASE,
            "count": len(incorrect_records),
            "records": incorrect_records,
        },
    )

    atomic_write_json(
        OUTPUT_DIR / "c4_3_support_buckets.json",
        {
            "phase": PHASE,
            "buckets": support_bucket_summary,
        },
    )

    np.save(
        OUTPUT_DIR / "c4_3_confusion_matrix.npy",
        cm,
    )

    manifest = {
        "phase": PHASE,
        "dataset": DATASET,
        "target_field": "product_type",
        "num_classes": NUM_CLASSES,
        "test_population": TEST_POPULATION,
        "seed": SEED,
        "prediction_source": str(
            PREDICTIONS_PATH
        ),
        "prediction_sha256": sha256_file(
            PREDICTIONS_PATH
        ),
        "results_source": str(
            RESULTS_PATH
        ),
        "results_sha256": sha256_file(
            RESULTS_PATH
        ),
        "c4_1_manifest": str(
            C4_1_MANIFEST_PATH
        ),
        "c4_1_manifest_sha256": sha256_file(
            C4_1_MANIFEST_PATH
        ),
        "b4_2_image_model": str(
            B4_IMAGE_PATH
        ),
        "b4_2_image_model_sha256": sha256_file(
            B4_IMAGE_PATH
        ),
        "true_labels_source": str(
            TRUE_LABELS_PATH
        ),
        "record_ids_source": str(
            RECORD_IDS_PATH
        ),
        "training": False,
        "model_selection": False,
        "hyperparameter_selection": False,
        "test_usage": "post-hoc error analysis only",
        "b3_modified": False,
        "c3_modified": False,
        "outputs": {
            "summary": str(
                OUTPUT_DIR / "c4_3_summary.json"
            ),
            "per_class": str(
                OUTPUT_DIR / "c4_3_per_class.json"
            ),
            "confusion_pairs": str(
                OUTPUT_DIR / "c4_3_confusion_pairs.json"
            ),
            "incorrect_records": str(
                OUTPUT_DIR / "c4_3_incorrect_records.json"
            ),
            "support_buckets": str(
                OUTPUT_DIR / "c4_3_support_buckets.json"
            ),
            "confusion_matrix": str(
                OUTPUT_DIR / "c4_3_confusion_matrix.npy"
            ),
        },
    }

    atomic_write_json(
        OUTPUT_DIR / "c4_3_error_analysis_manifest.json",
        manifest,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("")
    print("=" * 72)
    print("C4.3 ERROR ANALYSIS SUMMARY")
    print("=" * 72)

    print(
        f"Test records            : {TEST_POPULATION}"
    )
    print(
        f"Correct predictions     : {correct_count}"
    )
    print(
        f"Incorrect predictions   : {incorrect_count}"
    )
    print(
        f"Error rate              : "
        f"{incorrect_count / TEST_POPULATION:.6f}"
    )

    print("")
    print("RECOMPUTED C4.2 METRICS")
    print(
        f"Accuracy                : {accuracy:.6f}"
    )
    print(
        f"Macro-F1                : {macro_f1:.6f}"
    )
    print(
        f"Weighted-F1             : {weighted_f1:.6f}"
    )

    print("")
    print(
        f"Classes with zero predictions "
        f": {len(zero_prediction_classes)}"
    )
    print(
        f"Classes with zero test support "
        f": {len(zero_test_support_classes)}"
    )
    print(
        f"Nonzero confusion pairs   "
        f": {len(confusion_pairs)}"
    )

    print("")
    print("TOP 10 CONFUSION PAIRS")
    print("-" * 72)

    for row in confusion_pairs[:10]:
        print(
            f"{row['count']:5d} | "
            f"{row['true_class']} -> "
            f"{row['predicted_class']}"
        )

    print("")
    print("TOP 10 LOWEST-F1 CLASSES")
    print("-" * 72)

    for row in sorted(
        class_rows,
        key=lambda x: (
            x["f1"],
            -x["support"],
            x["class_id"],
        ),
    )[:10]:
        print(
            f"F1={row['f1']:.6f} | "
            f"support={row['support']:5d} | "
            f"precision={row['precision']:.6f} | "
            f"recall={row['recall']:.6f} | "
            f"{row['class_name']}"
        )

    print("")
    print("OUTPUTS")
    print(
        f"Directory: {OUTPUT_DIR}"
    )
    print(
        "C4.3 COMPLETE."
    )


if __name__ == "__main__":
    main()