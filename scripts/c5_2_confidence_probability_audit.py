
"""
Phase C5.2 — Confidence / Probability Behavior Audit

Purpose
-------
Analyze the behavior of frozen C5.1 predictive signals against observed
correctness on the frozen 7,346-record ABO Phase C test population.

This is a DESCRIPTIVE reliability-readiness analysis.

It does NOT:
- train a model;
- modify the C-Base checkpoint;
- select a model;
- tune hyperparameters;
- select an operational confidence threshold;
- perform calibration;
- perform temperature scaling;
- implement selective prediction;
- implement abstention;
- implement human-review routing;
- implement open-set detection.

Authoritative inputs
--------------------
C5.1 frozen signal artifacts:
    data/models/abo/phase_c/c5/c5_1_reliability_signal_audit/

Expected artifacts:
    c5_1_probabilities.npy
    c5_1_predictions.npy
    c5_1_top1_probability.npy
    c5_1_top2_probability.npy
    c5_1_top1_top2_margin.npy
    c5_1_predictive_entropy.npy
    c5_1_correct.npy
    c5_1_signal_inventory.json
    c5_1_signal_inventory_manifest.json

Additional authoritative labels / IDs are loaded from C4.1:
    data/models/abo/phase_c/c4/test_embeddings/test_labels.npy
    data/models/abo/phase_c/c4/test_embeddings/test_record_ids.json

Primary frozen benchmark
------------------------
Dataset: ABO
Target: product_type
Classes: 549
Test population: 7,346
Seed: 20260827

Scientific status
-----------------
Raw probabilities / margins / entropy:
    VERIFIED as model-generated signals.

Calibration:
    UNKNOWN

Selective prediction:
    UNKNOWN

Abstention:
    UNKNOWN

Human-review routing:
    UNKNOWN

Open-set detection:
    UNKNOWN
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

C5_1_DIR = (
    ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c5"
    / "c5_1_reliability_signal_audit"
)

C4_1_DIR = (
    ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c4"
    / "test_embeddings"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c5"
    / "c5_2_confidence_probability_audit"
)

NUM_CLASSES = 549
EXPECTED_TEST_RECORDS = 7_346
SEED = 20260827

# Descriptive confidence bins.
#
# These bins are fixed for reporting only.
# They are NOT operational thresholds and are NOT optimized.
CONFIDENCE_BINS = [
    (0.0, 0.1, "[0.0,0.1)"),
    (0.1, 0.2, "[0.1,0.2)"),
    (0.2, 0.3, "[0.2,0.3)"),
    (0.3, 0.4, "[0.3,0.4)"),
    (0.4, 0.5, "[0.4,0.5)"),
    (0.5, 0.6, "[0.5,0.6)"),
    (0.6, 0.7, "[0.6,0.7)"),
    (0.7, 0.8, "[0.7,0.8)"),
    (0.8, 0.9, "[0.8,0.9)"),
    (0.9, 1.0, "[0.9,1.0]"),
]

# Quantiles used only for descriptive reporting.
QUANTILES = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]

# High-confidence diagnostic level.
#
# This is deliberately NOT an operational threshold.
# It is simply a descriptive bucket for identifying potentially interesting
# high-confidence errors.
HIGH_CONFIDENCE_DIAGNOSTIC_LEVEL = 0.90

# Low-confidence diagnostic level.
#
# Again, this is descriptive only and is not proposed as an abstention
# threshold.
LOW_CONFIDENCE_DIAGNOSTIC_LEVEL = 0.50


# ============================================================================
# UTILITIES
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_status(label: str, value: Any) -> None:
    print(f"{label:<38}: {value}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)

    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())

    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    temporary.replace(path)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {description} not found:\n{path}"
        )

    if not path.is_file():
        raise RuntimeError(
            f"Expected {description} to be a file:\n{path}"
        )

    print_status(f"PASS | {description}", path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_npy(path: Path, description: str) -> np.ndarray:
    require_file(path, description)

    array = np.load(path, allow_pickle=False)

    print_status(
        f"Loaded {description}",
        f"shape={array.shape}, dtype={array.dtype}",
    )

    return array


def finite_or_fail(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        bad_count = int(np.size(array) - np.count_nonzero(np.isfinite(array)))

        raise RuntimeError(
            f"{name} contains {bad_count} non-finite values."
        )

    print_status(f"PASS | {name} finite", True)


def safe_mean(array: np.ndarray) -> float | None:
    if array.size == 0:
        return None

    return float(np.mean(array))


def safe_median(array: np.ndarray) -> float | None:
    if array.size == 0:
        return None

    return float(np.median(array))


def safe_std(array: np.ndarray) -> float | None:
    if array.size == 0:
        return None

    return float(np.std(array))


def descriptive_statistics(array: np.ndarray) -> dict[str, Any]:
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "quantiles": {},
        }

    quantile_values = np.quantile(array, QUANTILES)

    quantiles = {
        f"{q:.2f}": float(value)
        for q, value in zip(QUANTILES, quantile_values)
    }

    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "quantiles": quantiles,
    }


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    value = float(value)

    if not math.isfinite(value):
        return None

    return value


# ============================================================================
# ARTIFACT PATHS
# ============================================================================

PROBABILITIES_PATH = C5_1_DIR / "c5_1_probabilities.npy"
PREDICTIONS_PATH = C5_1_DIR / "c5_1_predictions.npy"
TOP1_PROBABILITY_PATH = C5_1_DIR / "c5_1_top1_probability.npy"
TOP2_PROBABILITY_PATH = C5_1_DIR / "c5_1_top2_probability.npy"
MARGIN_PATH = C5_1_DIR / "c5_1_top1_top2_margin.npy"
ENTROPY_PATH = C5_1_DIR / "c5_1_predictive_entropy.npy"
CORRECT_PATH = C5_1_DIR / "c5_1_correct.npy"

SIGNAL_INVENTORY_PATH = C5_1_DIR / "c5_1_signal_inventory.json"
SIGNAL_MANIFEST_PATH = C5_1_DIR / "c5_1_signal_inventory_manifest.json"

LABELS_PATH = C4_1_DIR / "test_labels.npy"
RECORD_IDS_PATH = C4_1_DIR / "test_record_ids.json"
C4_1_MANIFEST_PATH = C4_1_DIR / "c4_test_embedding_manifest.json"


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print_header("C5.2 — CONFIDENCE / PROBABILITY BEHAVIOR AUDIT")

    print_status("Phase", "C5.2")
    print_status("Purpose", "Descriptive reliability-readiness analysis")
    print_status("Training", "NONE")
    print_status("Model selection", "NONE")
    print_status("Hyperparameter tuning", "NONE")
    print_status("Threshold optimization", "NONE")
    print_status("Calibration fitting", "NONE")
    print_status("Seed", SEED)

    # ------------------------------------------------------------------------
    # 1. ARTIFACT AVAILABILITY
    # ------------------------------------------------------------------------

    print_header("ARTIFACT AVAILABILITY")

    required_c5_1_files = [
        (PROBABILITIES_PATH, "C5.1 probabilities"),
        (PREDICTIONS_PATH, "C5.1 predictions"),
        (TOP1_PROBABILITY_PATH, "C5.1 top-1 probabilities"),
        (TOP2_PROBABILITY_PATH, "C5.1 top-2 probabilities"),
        (MARGIN_PATH, "C5.1 top-1/top-2 margins"),
        (ENTROPY_PATH, "C5.1 predictive entropy"),
        (CORRECT_PATH, "C5.1 correctness"),
        (SIGNAL_INVENTORY_PATH, "C5.1 signal inventory"),
        (SIGNAL_MANIFEST_PATH, "C5.1 signal manifest"),
    ]

    for path, description in required_c5_1_files:
        require_file(path, description)

    require_file(LABELS_PATH, "C4.1 test labels")
    require_file(RECORD_IDS_PATH, "C4.1 test record IDs")
    require_file(C4_1_MANIFEST_PATH, "C4.1 test embedding manifest")

    # ------------------------------------------------------------------------
    # 2. LOAD C5.1 SIGNALS
    # ------------------------------------------------------------------------

    print_header("LOAD FROZEN C5.1 SIGNALS")

    probabilities = load_npy(
        PROBABILITIES_PATH,
        "C5.1 probabilities",
    )

    predictions = load_npy(
        PREDICTIONS_PATH,
        "C5.1 predictions",
    )

    top1_probability = load_npy(
        TOP1_PROBABILITY_PATH,
        "C5.1 top-1 probabilities",
    )

    top2_probability = load_npy(
        TOP2_PROBABILITY_PATH,
        "C5.1 top-2 probabilities",
    )

    margins = load_npy(
        MARGIN_PATH,
        "C5.1 top-1/top-2 margins",
    )

    entropy = load_npy(
        ENTROPY_PATH,
        "C5.1 predictive entropy",
    )

    correct = load_npy(
        CORRECT_PATH,
        "C5.1 correctness",
    )

    labels = load_npy(
        LABELS_PATH,
        "C4.1 test labels",
    )

    with RECORD_IDS_PATH.open("r", encoding="utf-8") as handle:
        record_ids = json.load(handle)

    signal_inventory = load_json(SIGNAL_INVENTORY_PATH)
    signal_manifest = load_json(SIGNAL_MANIFEST_PATH)
    c4_1_manifest = load_json(C4_1_MANIFEST_PATH)

    # ------------------------------------------------------------------------
    # 3. BASIC SHAPE / IDENTITY VALIDATION
    # ------------------------------------------------------------------------

    print_header("SIGNAL SHAPE AND IDENTITY VALIDATION")

    if probabilities.shape != (
        EXPECTED_TEST_RECORDS,
        NUM_CLASSES,
    ):
        raise RuntimeError(
            "Unexpected probability matrix shape: "
            f"{probabilities.shape}; expected "
            f"({EXPECTED_TEST_RECORDS}, {NUM_CLASSES})."
        )

    if predictions.shape != (EXPECTED_TEST_RECORDS,):
        raise RuntimeError(
            f"Unexpected prediction shape: {predictions.shape}."
        )

    expected_vector_shapes = {
        "top1_probability": top1_probability,
        "top2_probability": top2_probability,
        "margin": margins,
        "entropy": entropy,
        "correct": correct,
        "labels": labels,
    }

    for name, array in expected_vector_shapes.items():
        if array.shape != (EXPECTED_TEST_RECORDS,):
            raise RuntimeError(
                f"Unexpected {name} shape: {array.shape}; expected "
                f"({EXPECTED_TEST_RECORDS},)."
            )

    if len(record_ids) != EXPECTED_TEST_RECORDS:
        raise RuntimeError(
            f"Unexpected record-ID count: {len(record_ids)}."
        )

    if len(set(record_ids)) != EXPECTED_TEST_RECORDS:
        raise RuntimeError(
            "Test record IDs are not unique."
        )

    print_status("PASS | Test population", EXPECTED_TEST_RECORDS)
    print_status("PASS | Probability shape", probabilities.shape)
    print_status("PASS | Prediction shape", predictions.shape)
    print_status("PASS | Labels shape", labels.shape)
    print_status("PASS | Record IDs unique", True)

    # ------------------------------------------------------------------------
    # 4. FINITE / RANGE VALIDATION
    # ------------------------------------------------------------------------

    print_header("SIGNAL INTEGRITY")

    finite_or_fail("Probabilities", probabilities)
    finite_or_fail("Top-1 probability", top1_probability)
    finite_or_fail("Top-2 probability", top2_probability)
    finite_or_fail("Top-1/top-2 margin", margins)
    finite_or_fail("Predictive entropy", entropy)

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Predictions contain non-finite values.")

    if not np.all(np.isfinite(labels)):
        raise RuntimeError("Labels contain non-finite values.")

    if not np.all(np.isfinite(correct)):
        raise RuntimeError("Correctness contains non-finite values.")

    if not np.all((predictions >= 0) & (predictions < NUM_CLASSES)):
        raise RuntimeError("Predictions contain class IDs outside 0..548.")

    if not np.all((labels >= 0) & (labels < NUM_CLASSES)):
        raise RuntimeError("Labels contain class IDs outside 0..548.")

    probability_row_sums = probabilities.sum(axis=1)

    max_probability_sum_error = float(
        np.max(np.abs(probability_row_sums - 1.0))
    )

    if max_probability_sum_error > 1e-5:
        raise RuntimeError(
            "Probability rows do not sum to 1 within tolerance. "
            f"Maximum error: {max_probability_sum_error}"
        )

    if not np.all((top1_probability >= 0.0) & (top1_probability <= 1.0)):
        raise RuntimeError("Top-1 probabilities outside [0,1].")

    if not np.all((top2_probability >= 0.0) & (top2_probability <= 1.0)):
        raise RuntimeError("Top-2 probabilities outside [0,1].")

    if not np.all(margins >= -1e-7):
        raise RuntimeError("Negative top-1/top-2 margins detected.")

    if not np.all(entropy >= -1e-7):
        raise RuntimeError("Negative predictive entropy detected.")

    reconstructed_predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(predictions.dtype)

    prediction_mismatch_count = int(
        np.count_nonzero(
            reconstructed_predictions != predictions
        )
    )

    if prediction_mismatch_count != 0:
        raise RuntimeError(
            "Predictions do not match argmax(probabilities). "
            f"Mismatches: {prediction_mismatch_count}"
        )

    reconstructed_correct = (
        predictions == labels
    ).astype(correct.dtype)

    correctness_mismatch_count = int(
        np.count_nonzero(
            reconstructed_correct != correct
        )
    )

    if correctness_mismatch_count != 0:
        raise RuntimeError(
            "Correctness vector does not match predictions vs labels. "
            f"Mismatches: {correctness_mismatch_count}"
        )

    print_status("PASS | Probability rows sum to 1", True)
    print_status(
        "Maximum probability-sum error",
        max_probability_sum_error,
    )
    print_status(
        "PASS | Predictions match probability argmax",
        True,
    )
    print_status(
        "PASS | Correctness matches labels",
        True,
    )

    # ------------------------------------------------------------------------
    # 5. C4.2 METRIC REPRODUCTION
    # ------------------------------------------------------------------------

    print_header("C4.2 METRIC REPRODUCTION")

    accuracy = float(
        accuracy_score(
            labels,
            predictions,
        )
    )

    macro_f1 = float(
        f1_score(
            labels,
            predictions,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = float(
        f1_score(
            labels,
            predictions,
            labels=np.arange(NUM_CLASSES),
            average="weighted",
            zero_division=0,
        )
    )

    expected_accuracy = 0.786823
    expected_macro_f1 = 0.341346
    expected_weighted_f1 = 0.782435

    metric_tolerance = 1e-6

    if abs(accuracy - expected_accuracy) > metric_tolerance:
        raise RuntimeError(
            f"Accuracy mismatch: {accuracy} vs expected "
            f"{expected_accuracy}"
        )

    if abs(macro_f1 - expected_macro_f1) > metric_tolerance:
        raise RuntimeError(
            f"Macro-F1 mismatch: {macro_f1} vs expected "
            f"{expected_macro_f1}"
        )

    if abs(weighted_f1 - expected_weighted_f1) > metric_tolerance:
        raise RuntimeError(
            f"Weighted-F1 mismatch: {weighted_f1} vs expected "
            f"{expected_weighted_f1}"
        )

    print_status("Accuracy", f"{accuracy:.6f}")
    print_status("Macro-F1", f"{macro_f1:.6f}")
    print_status("Weighted-F1", f"{weighted_f1:.6f}")
    print_status(
        "PASS | C4.2 metrics reproduced",
        True,
    )

    # ------------------------------------------------------------------------
    # 6. OVERALL SIGNAL STATISTICS
    # ------------------------------------------------------------------------

    print_header("OVERALL SIGNAL STATISTICS")

    overall_statistics = {
        "top1_probability": descriptive_statistics(
            top1_probability.astype(np.float64)
        ),
        "top2_probability": descriptive_statistics(
            top2_probability.astype(np.float64)
        ),
        "top1_top2_margin": descriptive_statistics(
            margins.astype(np.float64)
        ),
        "predictive_entropy": descriptive_statistics(
            entropy.astype(np.float64)
        ),
    }

    for signal_name, stats in overall_statistics.items():
        print()
        print(f"{signal_name}:")
        print(f"  count  : {stats['count']}")
        print(f"  mean   : {stats['mean']}")
        print(f"  median : {stats['median']}")
        print(f"  min    : {stats['min']}")
        print(f"  max    : {stats['max']}")

    # ------------------------------------------------------------------------
    # 7. CORRECT VS INCORRECT ANALYSIS
    # ------------------------------------------------------------------------

    print_header("CORRECT VS INCORRECT CONFIDENCE BEHAVIOR")

    correct_mask = correct.astype(bool)
    incorrect_mask = ~correct_mask

    correct_count = int(np.count_nonzero(correct_mask))
    incorrect_count = int(np.count_nonzero(incorrect_mask))

    if correct_count + incorrect_count != EXPECTED_TEST_RECORDS:
        raise RuntimeError(
            "Correct + incorrect counts do not equal test population."
        )

    signal_arrays = {
        "top1_probability": top1_probability.astype(np.float64),
        "top2_probability": top2_probability.astype(np.float64),
        "top1_top2_margin": margins.astype(np.float64),
        "predictive_entropy": entropy.astype(np.float64),
    }

    correct_incorrect_statistics: dict[str, Any] = {
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "correct_fraction": float(
            correct_count / EXPECTED_TEST_RECORDS
        ),
        "incorrect_fraction": float(
            incorrect_count / EXPECTED_TEST_RECORDS
        ),
        "signals": {},
    }

    for signal_name, values in signal_arrays.items():
        correct_values = values[correct_mask]
        incorrect_values = values[incorrect_mask]

        correct_stats = descriptive_statistics(correct_values)
        incorrect_stats = descriptive_statistics(incorrect_values)

        mean_difference = None

        if (
            correct_stats["mean"] is not None
            and incorrect_stats["mean"] is not None
        ):
            mean_difference = float(
                correct_stats["mean"]
                - incorrect_stats["mean"]
            )

        correct_incorrect_statistics["signals"][signal_name] = {
            "correct": correct_stats,
            "incorrect": incorrect_stats,
            "correct_mean_minus_incorrect_mean": mean_difference,
        }

        print()
        print(signal_name)
        print(
            f"  correct mean    : "
            f"{correct_stats['mean']}"
        )
        print(
            f"  incorrect mean  : "
            f"{incorrect_stats['mean']}"
        )
        print(
            f"  mean difference : "
            f"{mean_difference}"
        )

    # ------------------------------------------------------------------------
    # 8. CONFIDENCE BINS
    # ------------------------------------------------------------------------

    print_header("TOP-1 PROBABILITY DESCRIPTIVE BINS")

    confidence_bins: list[dict[str, Any]] = []

    for lower, upper, label in CONFIDENCE_BINS:
        if upper < 1.0:
            mask = (
                (top1_probability >= lower)
                & (top1_probability < upper)
            )
        else:
            mask = (
                (top1_probability >= lower)
                & (top1_probability <= upper)
            )

        count = int(np.count_nonzero(mask))

        if count == 0:
            row = {
                "bin": label,
                "lower_bound": lower,
                "upper_bound": upper,
                "count": 0,
                "fraction": 0.0,
                "observed_accuracy": None,
                "mean_top1_probability": None,
                "mean_top2_probability": None,
                "mean_margin": None,
                "mean_entropy": None,
                "correct_count": 0,
                "incorrect_count": 0,
            }
        else:
            bin_correct = correct_mask[mask]

            row = {
                "bin": label,
                "lower_bound": lower,
                "upper_bound": upper,
                "count": count,
                "fraction": float(
                    count / EXPECTED_TEST_RECORDS
                ),
                "observed_accuracy": float(
                    np.mean(bin_correct)
                ),
                "mean_top1_probability": float(
                    np.mean(top1_probability[mask])
                ),
                "mean_top2_probability": float(
                    np.mean(top2_probability[mask])
                ),
                "mean_margin": float(
                    np.mean(margins[mask])
                ),
                "mean_entropy": float(
                    np.mean(entropy[mask])
                ),
                "correct_count": int(
                    np.count_nonzero(bin_correct)
                ),
                "incorrect_count": int(
                    np.count_nonzero(~bin_correct)
                ),
            }

        confidence_bins.append(row)

        print(
            f"{label:<12} "
            f"count={row['count']:<5} "
            f"accuracy={row['observed_accuracy']}"
        )

    total_binned = sum(
        int(row["count"])
        for row in confidence_bins
    )

    if total_binned != EXPECTED_TEST_RECORDS:
        raise RuntimeError(
            "Confidence bins do not cover the complete test population. "
            f"Binned={total_binned}, expected={EXPECTED_TEST_RECORDS}"
        )

    print_status(
        "PASS | Confidence bins cover all records",
        total_binned,
    )

    # ------------------------------------------------------------------------
    # 9. HIGH-CONFIDENCE ERRORS
    # ------------------------------------------------------------------------

    print_header("HIGH-CONFIDENCE ERROR DIAGNOSTIC")

    high_confidence_error_mask = (
        (~correct_mask)
        & (
            top1_probability
            >= HIGH_CONFIDENCE_DIAGNOSTIC_LEVEL
        )
    )

    high_confidence_error_indices = np.flatnonzero(
        high_confidence_error_mask
    )

    high_confidence_errors: list[dict[str, Any]] = []

    for index in high_confidence_error_indices:
        i = int(index)

        high_confidence_errors.append(
            {
                "index": i,
                "record_id": record_ids[i],
                "true_class": int(labels[i]),
                "predicted_class": int(predictions[i]),
                "top1_probability": float(top1_probability[i]),
                "top2_probability": float(top2_probability[i]),
                "top1_top2_margin": float(margins[i]),
                "predictive_entropy": float(entropy[i]),
            }
        )

    high_confidence_errors.sort(
        key=lambda row: row["top1_probability"],
        reverse=True,
    )

    print_status(
        "Diagnostic level",
        HIGH_CONFIDENCE_DIAGNOSTIC_LEVEL,
    )
    print_status(
        "High-confidence errors",
        len(high_confidence_errors),
    )

    if high_confidence_errors:
        print()
        print("Top high-confidence errors:")

        for row in high_confidence_errors[:20]:
            print(
                f"  {row['record_id']} | "
                f"true={row['true_class']} | "
                f"pred={row['predicted_class']} | "
                f"p1={row['top1_probability']:.6f} | "
                f"margin={row['top1_top2_margin']:.6f}"
            )

    # ------------------------------------------------------------------------
    # 10. LOW-CONFIDENCE CORRECT DIAGNOSTIC
    # ------------------------------------------------------------------------

    print_header("LOW-CONFIDENCE CORRECT DIAGNOSTIC")

    low_confidence_correct_mask = (
        correct_mask
        & (
            top1_probability
            < LOW_CONFIDENCE_DIAGNOSTIC_LEVEL
        )
    )

    low_confidence_correct_indices = np.flatnonzero(
        low_confidence_correct_mask
    )

    low_confidence_correct: list[dict[str, Any]] = []

    for index in low_confidence_correct_indices:
        i = int(index)

        low_confidence_correct.append(
            {
                "index": i,
                "record_id": record_ids[i],
                "true_class": int(labels[i]),
                "predicted_class": int(predictions[i]),
                "top1_probability": float(top1_probability[i]),
                "top2_probability": float(top2_probability[i]),
                "top1_top2_margin": float(margins[i]),
                "predictive_entropy": float(entropy[i]),
            }
        )

    low_confidence_correct.sort(
        key=lambda row: row["top1_probability"]
    )

    print_status(
        "Diagnostic level",
        LOW_CONFIDENCE_DIAGNOSTIC_LEVEL,
    )
    print_status(
        "Low-confidence correct predictions",
        len(low_confidence_correct),
    )

    if low_confidence_correct:
        print()
        print("Lowest-confidence correct predictions:")

        for row in low_confidence_correct[:20]:
            print(
                f"  {row['record_id']} | "
                f"class={row['predicted_class']} | "
                f"p1={row['top1_probability']:.6f} | "
                f"margin={row['top1_top2_margin']:.6f}"
            )

    # ------------------------------------------------------------------------
    # 11. CONFIDENCE / CORRECTNESS SEPARATION SUMMARY
    # ------------------------------------------------------------------------

    print_header("CONFIDENCE / CORRECTNESS SEPARATION SUMMARY")

    correct_top1_mean = safe_mean(
        top1_probability[correct_mask]
    )
    incorrect_top1_mean = safe_mean(
        top1_probability[incorrect_mask]
    )

    correct_margin_mean = safe_mean(
        margins[correct_mask]
    )
    incorrect_margin_mean = safe_mean(
        margins[incorrect_mask]
    )

    correct_entropy_mean = safe_mean(
        entropy[correct_mask]
    )
    incorrect_entropy_mean = safe_mean(
        entropy[incorrect_mask]
    )

    separation_summary = {
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "top1_probability": {
            "correct_mean": correct_top1_mean,
            "incorrect_mean": incorrect_top1_mean,
            "mean_difference_correct_minus_incorrect": (
                None
                if correct_top1_mean is None
                or incorrect_top1_mean is None
                else float(
                    correct_top1_mean
                    - incorrect_top1_mean
                )
            ),
        },
        "top1_top2_margin": {
            "correct_mean": correct_margin_mean,
            "incorrect_mean": incorrect_margin_mean,
            "mean_difference_correct_minus_incorrect": (
                None
                if correct_margin_mean is None
                or incorrect_margin_mean is None
                else float(
                    correct_margin_mean
                    - incorrect_margin_mean
                )
            ),
        },
        "predictive_entropy": {
            "correct_mean": correct_entropy_mean,
            "incorrect_mean": incorrect_entropy_mean,
            "mean_difference_correct_minus_incorrect": (
                None
                if correct_entropy_mean is None
                or incorrect_entropy_mean is None
                else float(
                    correct_entropy_mean
                    - incorrect_entropy_mean
                )
            ),
        },
        "diagnostic_high_confidence_error_count": len(
            high_confidence_errors
        ),
        "diagnostic_low_confidence_correct_count": len(
            low_confidence_correct
        ),
    }

    print(
        "Top-1 probability mean difference "
        "(correct - incorrect): "
        f"{separation_summary['top1_probability']['mean_difference_correct_minus_incorrect']}"
    )

    print(
        "Margin mean difference "
        "(correct - incorrect): "
        f"{separation_summary['top1_top2_margin']['mean_difference_correct_minus_incorrect']}"
    )

    print(
        "Entropy mean difference "
        "(correct - incorrect): "
        f"{separation_summary['predictive_entropy']['mean_difference_correct_minus_incorrect']}"
    )

    # ------------------------------------------------------------------------
    # 12. SIGNAL PROVENANCE HASHES
    # ------------------------------------------------------------------------

    print_header("SIGNAL PROVENANCE")

    input_hashes = {
        "c5_1_probabilities_sha256": sha256_file(
            PROBABILITIES_PATH
        ),
        "c5_1_predictions_sha256": sha256_file(
            PREDICTIONS_PATH
        ),
        "c5_1_top1_probability_sha256": sha256_file(
            TOP1_PROBABILITY_PATH
        ),
        "c5_1_top2_probability_sha256": sha256_file(
            TOP2_PROBABILITY_PATH
        ),
        "c5_1_margin_sha256": sha256_file(
            MARGIN_PATH
        ),
        "c5_1_entropy_sha256": sha256_file(
            ENTROPY_PATH
        ),
        "c5_1_correct_sha256": sha256_file(
            CORRECT_PATH
        ),
        "c4_1_labels_sha256": sha256_file(
            LABELS_PATH
        ),
        "c4_1_record_ids_sha256": sha256_file(
            RECORD_IDS_PATH
        ),
    }

    for name, digest in input_hashes.items():
        print_status(name, digest)

    # ------------------------------------------------------------------------
    # 13. SUMMARY ARTIFACT
    # ------------------------------------------------------------------------

    print_header("BUILD C5.2 SUMMARY")

    summary = {
        "phase": "C5.2",
        "stage": "Confidence / Probability Behavior Audit",
        "dataset": "ABO",
        "task": "product_type classification",
        "num_classes": NUM_CLASSES,
        "test_population": EXPECTED_TEST_RECORDS,
        "seed": SEED,

        "purpose": (
            "Descriptive analysis of frozen C5.1 predictive "
            "probability, margin, and entropy signals against "
            "observed correctness."
        ),

        "input_source": {
            "stage": "C5.1",
            "directory": str(C5_1_DIR),
            "signals": [
                "logits",
                "probabilities",
                "predictions",
                "top1_probability",
                "top2_probability",
                "top1_top2_margin",
                "predictive_entropy",
                "correct",
                "joint_representation",
            ],
        },

        "c4_2_metrics_reproduced": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        },

        "overall_signal_statistics": overall_statistics,

        "correct_incorrect_statistics": (
            correct_incorrect_statistics
        ),

        "confidence_bins": confidence_bins,

        "diagnostic_levels": {
            "high_confidence_error_level": (
                HIGH_CONFIDENCE_DIAGNOSTIC_LEVEL
            ),
            "low_confidence_correct_level": (
                LOW_CONFIDENCE_DIAGNOSTIC_LEVEL
            ),
            "interpretation": (
                "These levels are descriptive diagnostic buckets only. "
                "They are not operational thresholds and were not "
                "optimized on the test set."
            ),
        },

        "high_confidence_errors": {
            "count": len(high_confidence_errors),
            "records": high_confidence_errors,
        },

        "low_confidence_correct": {
            "count": len(low_confidence_correct),
            "records": low_confidence_correct,
        },

        "confidence_correctness_separation": (
            separation_summary
        ),

        "scientific_status": {
            "raw_predictive_probabilities": "VERIFIED",
            "raw_top1_confidence_score": "VERIFIED",
            "margin_signal": "VERIFIED",
            "predictive_entropy_signal": "VERIFIED",
            "calibration": "UNKNOWN",
            "selective_prediction": "UNKNOWN",
            "abstention": "UNKNOWN",
            "human_review_routing": "UNKNOWN",
            "open_set_detection": "UNKNOWN",
        },

        "test_usage": {
            "test_access": "FINAL_INFERENCE_AND_DESCRIPTIVE_ANALYSIS",
            "training_on_test": False,
            "model_selection_from_test": False,
            "hyperparameter_selection_from_test": False,
            "threshold_optimization_on_test": False,
            "calibration_fitting_on_test": False,
        },

        "frozen_artifacts_modified": {
            "b3": False,
            "b4": False,
            "b5": False,
            "c3": False,
            "c4_1": False,
            "c4_2": False,
            "c5_1": False,
        },

        "input_hashes": input_hashes,
    }

    # ------------------------------------------------------------------------
    # 14. WRITE OUTPUTS
    # ------------------------------------------------------------------------

    print_header("SAVE C5.2 ARTIFACTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "c5_2_summary.json"
    bins_path = OUTPUT_DIR / "c5_2_confidence_bins.json"
    correct_incorrect_path = (
        OUTPUT_DIR
        / "c5_2_correct_incorrect_summary.json"
    )
    high_confidence_path = (
        OUTPUT_DIR
        / "c5_2_high_confidence_errors.json"
    )
    low_confidence_path = (
        OUTPUT_DIR
        / "c5_2_low_confidence_correct.json"
    )
    manifest_path = OUTPUT_DIR / "c5_2_audit_manifest.json"

    atomic_write_json(
        summary_path,
        summary,
    )

    atomic_write_json(
        bins_path,
        {
            "phase": "C5.2",
            "test_population": EXPECTED_TEST_RECORDS,
            "bins": confidence_bins,
        },
    )

    atomic_write_json(
        correct_incorrect_path,
        correct_incorrect_statistics,
    )

    atomic_write_json(
        high_confidence_path,
        {
            "phase": "C5.2",
            "diagnostic_level": (
                HIGH_CONFIDENCE_DIAGNOSTIC_LEVEL
            ),
            "operational_threshold": False,
            "count": len(high_confidence_errors),
            "records": high_confidence_errors,
        },
    )

    atomic_write_json(
        low_confidence_path,
        {
            "phase": "C5.2",
            "diagnostic_level": (
                LOW_CONFIDENCE_DIAGNOSTIC_LEVEL
            ),
            "operational_threshold": False,
            "count": len(low_confidence_correct),
            "records": low_confidence_correct,
        },
    )

    manifest = {
        "phase": "C5.2",
        "stage": "Confidence / Probability Behavior Audit",
        "dataset": "ABO",
        "target": "product_type",
        "num_classes": NUM_CLASSES,
        "test_population": EXPECTED_TEST_RECORDS,
        "seed": SEED,

        "source_stage": "C5.1",

        "source_files": {
            "c5_1_probabilities": str(PROBABILITIES_PATH),
            "c5_1_predictions": str(PREDICTIONS_PATH),
            "c5_1_top1_probability": str(
                TOP1_PROBABILITY_PATH
            ),
            "c5_1_top2_probability": str(
                TOP2_PROBABILITY_PATH
            ),
            "c5_1_margin": str(MARGIN_PATH),
            "c5_1_entropy": str(ENTROPY_PATH),
            "c5_1_correct": str(CORRECT_PATH),
            "c4_1_labels": str(LABELS_PATH),
            "c4_1_record_ids": str(RECORD_IDS_PATH),
        },

        "output_files": {
            "summary": str(summary_path),
            "confidence_bins": str(bins_path),
            "correct_incorrect_summary": str(
                correct_incorrect_path
            ),
            "high_confidence_errors": str(
                high_confidence_path
            ),
            "low_confidence_correct": str(
                low_confidence_path
            ),
            "manifest": str(manifest_path),
        },

        "analysis_scope": [
            "overall probability distribution",
            "correct vs incorrect signal behavior",
            "descriptive confidence bins",
            "high-confidence error diagnostics",
            "low-confidence correct diagnostics",
            "margin behavior",
            "predictive entropy behavior",
        ],

        "explicit_non_claims": [
            "calibration",
            "selective prediction",
            "abstention",
            "human-review routing",
            "open-set detection",
            "operational confidence threshold",
        ],

        "diagnostic_thresholds": {
            "high_confidence_error_level": (
                HIGH_CONFIDENCE_DIAGNOSTIC_LEVEL
            ),
            "low_confidence_correct_level": (
                LOW_CONFIDENCE_DIAGNOSTIC_LEVEL
            ),
            "optimized": False,
            "operational": False,
        },

        "test_usage": {
            "training": False,
            "model_selection": False,
            "hyperparameter_tuning": False,
            "threshold_optimization": False,
            "calibration_fitting": False,
            "descriptive_analysis": True,
        },

        "integrity": {
            "c5_1_modified": False,
            "c4_1_modified": False,
            "c4_2_modified": False,
            "c3_modified": False,
            "b5_modified": False,
            "b4_modified": False,
            "b3_modified": False,
        },

        "reproduction": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "script": str(Path(__file__).resolve()),
        },

        "input_hashes": input_hashes,

        "status": "C5.2 COMPLETE",
    }

    atomic_write_json(
        manifest_path,
        manifest,
    )

    # ------------------------------------------------------------------------
    # 15. FINAL STATUS
    # ------------------------------------------------------------------------

    print_header("C5.2 FINAL STATUS")

    print("C5.2 COMPLETE")
    print()

    print_status(
        "Test records",
        EXPECTED_TEST_RECORDS,
    )
    print_status(
        "Accuracy",
        f"{accuracy:.6f}",
    )
    print_status(
        "Macro-F1",
        f"{macro_f1:.6f}",
    )
    print_status(
        "Weighted-F1",
        f"{weighted_f1:.6f}",
    )

    print()
    print("Scientific status:")
    print("  Raw probability behavior : VERIFIED")
    print("  Margin behavior          : VERIFIED")
    print("  Entropy behavior         : VERIFIED")
    print("  Calibration              : UNKNOWN")
    print("  Selective prediction     : UNKNOWN")
    print("  Abstention               : UNKNOWN")
    print("  Human-review routing     : UNKNOWN")
    print("  Open-set detection       : UNKNOWN")

    print()
    print("Integrity:")
    print("  Training                : NO")
    print("  Model selection         : NO")
    print("  Hyperparameter tuning   : NO")
    print("  Threshold optimization  : NO")
    print("  Calibration fitting     : NO")
    print("  C5.1 modified           : NO")
    print("  C4.2 modified           : NO")
    print("  C3 modified             : NO")
    print("  B5 modified             : NO")
    print("  B4 modified             : NO")
    print("  B3 modified             : NO")

    print()
    print("Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("C5.2 audit artifacts:")
    print(f"  {summary_path.name}")
    print(f"  {bins_path.name}")
    print(f"  {correct_incorrect_path.name}")
    print(f"  {high_confidence_path.name}")
    print(f"  {low_confidence_path.name}")
    print(f"  {manifest_path.name}")


if __name__ == "__main__":
    main()
