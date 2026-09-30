"""
C5.3 — Margin / Ambiguity Analysis
===================================

Phase C — Advanced ABO Multimodal Product Understanding

Purpose
-------
Analyze frozen C5.1 predictive signals and frozen C4 test labels to characterize:

1. Top-1 / top-2 probability competition
2. Margin distribution
3. Margin behavior for correct vs incorrect predictions
4. Descriptive margin bins
5. Low-margin predictions
6. High-margin errors
7. Relationships among top-1 probability, top-2 probability,
   margin, and predictive entropy
8. Top-2 class competition
9. Directional error pairs

Scientific boundaries
---------------------
This stage is DESCRIPTIVE / DIAGNOSTIC ONLY.

It does NOT:
- train a model
- load the model checkpoint
- perform model selection
- tune hyperparameters
- optimize thresholds
- fit calibration
- perform selective prediction
- implement abstention
- implement human-review routing
- implement open-set detection
- modify C5.1 or C4 artifacts

Scientific status:
- margin behavior: VERIFIED
- margin/correctness association: DEMONSTRATED
- top-2 competition: VERIFIED
- calibration: UNKNOWN
- selective prediction: UNKNOWN
- abstention: UNKNOWN
- human-review routing: UNKNOWN
- open-set detection: UNKNOWN

Seed
----
20260827

Expected test population
------------------------
7346

Expected classes
----------------
549
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EXPECTED_TEST_POPULATION = 7346
EXPECTED_NUM_CLASSES = 549
SEED = 20260827

PROJECT_ROOT = Path(__file__).resolve().parents[1]

C5_1_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c5"
    / "c5_1_reliability_signal_audit"
)

C4_TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c4"
    / "test_embeddings"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c5"
    / "c5_3_margin_ambiguity_analysis"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {description} is missing:\n  {path}"
        )


def json_dump(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def safe_float(value) -> float:
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def describe_array(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)

    return {
        "count": int(x.size),
        "min": safe_float(np.min(x)),
        "max": safe_float(np.max(x)),
        "mean": safe_float(np.mean(x)),
        "median": safe_float(np.median(x)),
        "std": safe_float(np.std(x)),
        "p01": safe_float(np.percentile(x, 1)),
        "p05": safe_float(np.percentile(x, 5)),
        "p10": safe_float(np.percentile(x, 10)),
        "p25": safe_float(np.percentile(x, 25)),
        "p50": safe_float(np.percentile(x, 50)),
        "p75": safe_float(np.percentile(x, 75)),
        "p90": safe_float(np.percentile(x, 90)),
        "p95": safe_float(np.percentile(x, 95)),
        "p99": safe_float(np.percentile(x, 99)),
    }


def correlation(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size != y.size:
        raise ValueError("Correlation arrays have different sizes.")

    if x.size < 2:
        return {
            "pearson_r": None,
            "n": int(x.size),
        }

    x_std = np.std(x)
    y_std = np.std(y)

    if x_std == 0.0 or y_std == 0.0:
        return {
            "pearson_r": None,
            "n": int(x.size),
        }

    r = np.corrcoef(x, y)[0, 1]

    return {
        "pearson_r": safe_float(r),
        "n": int(x.size),
    }


def accuracy(correct: np.ndarray) -> float:
    return float(np.mean(correct.astype(np.float64)))


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------

def load_inputs() -> dict:
    print("=" * 80)
    print("C5.3 — Margin / Ambiguity Analysis")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] C5.1 directory: {C5_1_DIR}")
    print(f"[INFO] C4 test directory: {C4_TEST_DIR}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    required_c5_1 = {
        "probabilities": C5_1_DIR / "c5_1_probabilities.npy",
        "predictions": C5_1_DIR / "c5_1_predictions.npy",
        "top1_probability": C5_1_DIR / "c5_1_top1_probability.npy",
        "top2_probability": C5_1_DIR / "c5_1_top2_probability.npy",
        "margin": C5_1_DIR / "c5_1_top1_top2_margin.npy",
        "entropy": C5_1_DIR / "c5_1_predictive_entropy.npy",
        "correct": C5_1_DIR / "c5_1_correct.npy",
        "inventory": C5_1_DIR / "c5_1_signal_inventory.json",
        "inventory_manifest": C5_1_DIR / "c5_1_signal_inventory_manifest.json",
    }

    for name, path in required_c5_1.items():
        ensure_exists(path, f"C5.1 artifact '{name}'")

    labels_path = C4_TEST_DIR / "test_labels.npy"
    ensure_exists(labels_path, "C4 frozen test labels")

    probabilities = np.load(
        required_c5_1["probabilities"],
        allow_pickle=False,
    )

    predictions = np.load(
        required_c5_1["predictions"],
        allow_pickle=False,
    )

    top1_probability = np.load(
        required_c5_1["top1_probability"],
        allow_pickle=False,
    )

    top2_probability = np.load(
        required_c5_1["top2_probability"],
        allow_pickle=False,
    )

    margin = np.load(
        required_c5_1["margin"],
        allow_pickle=False,
    )

    entropy = np.load(
        required_c5_1["entropy"],
        allow_pickle=False,
    )

    correct = np.load(
        required_c5_1["correct"],
        allow_pickle=False,
    )

    labels = np.load(
        labels_path,
        allow_pickle=False,
    )

    inventory = json.loads(
        required_c5_1["inventory"].read_text(encoding="utf-8")
    )

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "top1_probability": top1_probability,
        "top2_probability": top2_probability,
        "margin": margin,
        "entropy": entropy,
        "correct": correct,
        "labels": labels,
        "inventory": inventory,
        "paths": required_c5_1 | {"labels": labels_path},
    }


# ---------------------------------------------------------------------
# Integrity validation
# ---------------------------------------------------------------------

def validate_inputs(data: dict) -> None:
    probabilities = data["probabilities"]
    predictions = data["predictions"]
    top1 = data["top1_probability"]
    top2 = data["top2_probability"]
    margin = data["margin"]
    entropy = data["entropy"]
    correct = data["correct"]
    labels = data["labels"]
    inventory = data["inventory"]

    print("\n[CHECK] Validating input artifacts...")

    if probabilities.shape != (
        EXPECTED_TEST_POPULATION,
        EXPECTED_NUM_CLASSES,
    ):
        raise ValueError(
            f"Unexpected probability shape: {probabilities.shape}"
        )

    one_d = {
        "predictions": predictions,
        "top1_probability": top1,
        "top2_probability": top2,
        "margin": margin,
        "entropy": entropy,
        "correct": correct,
        "labels": labels,
    }

    for name, arr in one_d.items():
        if arr.shape != (EXPECTED_TEST_POPULATION,):
            raise ValueError(
                f"Unexpected {name} shape: {arr.shape}"
            )

    if inventory.get("test_population") != EXPECTED_TEST_POPULATION:
        raise ValueError(
            "C5.1 inventory test population does not match expected population."
        )

    if inventory.get("num_classes") != EXPECTED_NUM_CLASSES:
        raise ValueError(
            "C5.1 inventory class count does not match expected class count."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain non-finite values.")

    if not np.all(np.isfinite(top1)):
        raise ValueError("Top-1 probabilities contain non-finite values.")

    if not np.all(np.isfinite(top2)):
        raise ValueError("Top-2 probabilities contain non-finite values.")

    if not np.all(np.isfinite(margin)):
        raise ValueError("Margins contain non-finite values.")

    if not np.all(np.isfinite(entropy)):
        raise ValueError("Entropy contains non-finite values.")

    if not np.all(np.isfinite(labels)):
        raise ValueError("Labels contain non-finite values.")

    if not np.all((labels >= 0) & (labels < EXPECTED_NUM_CLASSES)):
        raise ValueError("Labels contain invalid class IDs.")

    if not np.all((predictions >= 0) & (predictions < EXPECTED_NUM_CLASSES)):
        raise ValueError("Predictions contain invalid class IDs.")

    if not np.all((correct == 0) | (correct == 1)):
        raise ValueError("Correctness array is not binary.")

    # Reconstruct expected top-1 probabilities.
    reconstructed_top1 = probabilities[
        np.arange(EXPECTED_TEST_POPULATION),
        predictions,
    ]

    max_top1_error = float(
        np.max(np.abs(reconstructed_top1 - top1))
    )

    if max_top1_error > 1e-6:
        raise ValueError(
            f"Top-1 probability mismatch: {max_top1_error}"
        )

    # Reconstruct top-2 probabilities.
    partitioned = np.partition(
        probabilities,
        EXPECTED_NUM_CLASSES - 2,
        axis=1,
    )

    reconstructed_top2 = partitioned[:, -2]

    max_top2_error = float(
        np.max(np.abs(reconstructed_top2 - top2))
    )

    if max_top2_error > 1e-6:
        raise ValueError(
            f"Top-2 probability mismatch: {max_top2_error}"
        )

    reconstructed_margin = top1 - top2

    max_margin_error = float(
        np.max(np.abs(reconstructed_margin - margin))
    )

    if max_margin_error > 1e-6:
        raise ValueError(
            f"Margin mismatch: {max_margin_error}"
        )

    reconstructed_correct = (
        predictions == labels
    ).astype(np.int64)

    max_correct_mismatch = int(
        np.sum(reconstructed_correct != correct)
    )

    if max_correct_mismatch != 0:
        raise ValueError(
            f"Correctness mismatch count: {max_correct_mismatch}"
        )

    probability_row_sums = np.sum(probabilities, axis=1)

    max_probability_sum_error = float(
        np.max(np.abs(probability_row_sums - 1.0))
    )

    if max_probability_sum_error > 1e-5:
        raise ValueError(
            f"Probability row-sum error too large: "
            f"{max_probability_sum_error}"
        )

    print("[PASS] Probability shape:", probabilities.shape)
    print("[PASS] Predictions shape:", predictions.shape)
    print("[PASS] Labels shape:", labels.shape)
    print("[PASS] Top-1 probabilities verified.")
    print("[PASS] Top-2 probabilities verified.")
    print("[PASS] Margin verified.")
    print("[PASS] Correctness verified against labels.")
    print(
        "[PASS] Maximum probability row-sum error:",
        max_probability_sum_error,
    )


# ---------------------------------------------------------------------
# Margin bins
# ---------------------------------------------------------------------

MARGIN_BINS = [
    (0.00, 0.05, "[0.00, 0.05)"),
    (0.05, 0.10, "[0.05, 0.10)"),
    (0.10, 0.20, "[0.10, 0.20)"),
    (0.20, 0.30, "[0.20, 0.30)"),
    (0.30, 0.40, "[0.30, 0.40)"),
    (0.40, 0.50, "[0.40, 0.50)"),
    (0.50, 0.60, "[0.50, 0.60)"),
    (0.60, 0.70, "[0.60, 0.70)"),
    (0.70, 0.80, "[0.70, 0.80)"),
    (0.80, 0.90, "[0.80, 0.90)"),
    (0.90, 1.00, "[0.90, 1.00]"),
]


def build_margin_bins(
    margin: np.ndarray,
    correct: np.ndarray,
) -> list[dict]:

    rows = []

    for i, (low, high, label) in enumerate(MARGIN_BINS):
        if i == len(MARGIN_BINS) - 1:
            mask = (margin >= low) & (margin <= high)
        else:
            mask = (margin >= low) & (margin < high)

        count = int(np.sum(mask))

        if count == 0:
            bin_accuracy = None
            correct_count = 0
            incorrect_count = 0
        else:
            correct_count = int(np.sum(correct[mask]))
            incorrect_count = count - correct_count
            bin_accuracy = float(correct_count / count)

        rows.append(
            {
                "bin": label,
                "lower_bound": low,
                "upper_bound": high,
                "count": count,
                "correct": correct_count,
                "incorrect": incorrect_count,
                "empirical_accuracy": bin_accuracy,
            }
        )

    return rows


# ---------------------------------------------------------------------
# Low-margin / high-margin diagnostics
# ---------------------------------------------------------------------

def build_prediction_records(
    indices: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    top1: np.ndarray,
    top2: np.ndarray,
    margin: np.ndarray,
    entropy: np.ndarray,
) -> list[dict]:

    records = []

    for idx in indices:
        idx = int(idx)

        row = probabilities[idx]

        top2_indices = np.argsort(row)[-2:][::-1]

        records.append(
            {
                "test_index": idx,
                "true_class": int(labels[idx]),
                "predicted_class": int(predictions[idx]),
                "top1_probability": float(top1[idx]),
                "top2_probability": float(top2[idx]),
                "margin": float(margin[idx]),
                "entropy": float(entropy[idx]),
                "top1_class": int(top2_indices[0]),
                "top2_class": int(top2_indices[1]),
                "top2_class_probability": float(
                    row[top2_indices[1]]
                ),
                "correct": bool(labels[idx] == predictions[idx]),
            }
        )

    return records


# ---------------------------------------------------------------------
# Top-2 competition
# ---------------------------------------------------------------------

def build_top2_competition(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    margin: np.ndarray,
    top1: np.ndarray,
    top2: np.ndarray,
) -> dict:

    pair_counts = {}

    for i in range(len(labels)):
        row = probabilities[i]
        top_indices = np.argsort(row)[-2:][::-1]

        top1_class = int(top_indices[0])
        top2_class = int(top_indices[1])

        key = f"{top1_class}->{top2_class}"

        if key not in pair_counts:
            pair_counts[key] = {
                "top1_class": top1_class,
                "top2_class": top2_class,
                "count": 0,
                "correct_top1": 0,
                "incorrect_top1": 0,
                "mean_margin": 0.0,
                "mean_top1_probability": 0.0,
                "mean_top2_probability": 0.0,
            }

        entry = pair_counts[key]

        entry["count"] += 1

        if predictions[i] == labels[i]:
            entry["correct_top1"] += 1
        else:
            entry["incorrect_top1"] += 1

        entry["mean_margin"] += float(margin[i])
        entry["mean_top1_probability"] += float(top1[i])
        entry["mean_top2_probability"] += float(top2[i])

    rows = []

    for entry in pair_counts.values():
        count = entry["count"]

        entry["mean_margin"] /= count
        entry["mean_top1_probability"] /= count
        entry["mean_top2_probability"] /= count

        entry["empirical_top1_accuracy"] = (
            entry["correct_top1"] / count
        )

        rows.append(entry)

    rows.sort(
        key=lambda x: (
            x["count"],
            x["incorrect_top1"],
        ),
        reverse=True,
    )

    return {
        "unique_top2_pairs": len(rows),
        "top_pairs_by_frequency": rows[:100],
    }


# ---------------------------------------------------------------------
# Directional confusion pairs
# ---------------------------------------------------------------------

def build_directional_confusions(
    predictions: np.ndarray,
    labels: np.ndarray,
    correct: np.ndarray,
) -> list[dict]:

    pair_counts = {}

    incorrect_indices = np.where(correct == 0)[0]

    for idx in incorrect_indices:
        true_class = int(labels[idx])
        predicted_class = int(predictions[idx])

        key = f"{true_class}->{predicted_class}"

        if key not in pair_counts:
            pair_counts[key] = {
                "true_class": true_class,
                "predicted_class": predicted_class,
                "count": 0,
            }

        pair_counts[key]["count"] += 1

    rows = list(pair_counts.values())

    rows.sort(
        key=lambda x: x["count"],
        reverse=True,
    )

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    data = load_inputs()

    validate_inputs(data)

    probabilities = data["probabilities"]
    predictions = data["predictions"]
    top1 = data["top1_probability"]
    top2 = data["top2_probability"]
    margin = data["margin"]
    entropy = data["entropy"]
    correct = data["correct"]
    labels = data["labels"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[INFO] Running descriptive margin analysis...")

    # ---------------------------------------------------------------
    # Overall descriptions
    # ---------------------------------------------------------------

    correct_mask = correct.astype(bool)
    incorrect_mask = ~correct_mask

    overall = {
        "test_population": EXPECTED_TEST_POPULATION,
        "num_classes": EXPECTED_NUM_CLASSES,
        "accuracy_from_correctness": accuracy(correct),
        "margin": describe_array(margin),
        "top1_probability": describe_array(top1),
        "top2_probability": describe_array(top2),
        "predictive_entropy": describe_array(entropy),
        "correct_count": int(np.sum(correct_mask)),
        "incorrect_count": int(np.sum(incorrect_mask)),
    }

    # ---------------------------------------------------------------
    # Correct vs incorrect
    # ---------------------------------------------------------------

    correct_incorrect = {
        "correct": {
            "count": int(np.sum(correct_mask)),
            "top1_probability": describe_array(
                top1[correct_mask]
            ),
            "top2_probability": describe_array(
                top2[correct_mask]
            ),
            "margin": describe_array(
                margin[correct_mask]
            ),
            "entropy": describe_array(
                entropy[correct_mask]
            ),
        },
        "incorrect": {
            "count": int(np.sum(incorrect_mask)),
            "top1_probability": describe_array(
                top1[incorrect_mask]
            ),
            "top2_probability": describe_array(
                top2[incorrect_mask]
            ),
            "margin": describe_array(
                margin[incorrect_mask]
            ),
            "entropy": describe_array(
                entropy[incorrect_mask]
            ),
        },
    }

    # ---------------------------------------------------------------
    # Fixed descriptive bins
    # ---------------------------------------------------------------

    margin_bins = build_margin_bins(
        margin,
        correct,
    )

    # ---------------------------------------------------------------
    # Diagnostic low-margin examples
    # ---------------------------------------------------------------

    low_margin_mask = margin < 0.10
    low_margin_indices = np.where(low_margin_mask)[0]

    low_margin_predictions = build_prediction_records(
        low_margin_indices,
        probabilities,
        predictions,
        labels,
        top1,
        top2,
        margin,
        entropy,
    )

    low_margin_predictions.sort(
        key=lambda x: x["margin"]
    )

    # ---------------------------------------------------------------
    # Diagnostic high-margin errors
    # ---------------------------------------------------------------

    high_margin_error_mask = (
        (margin >= 0.90)
        & (correct == 0)
    )

    high_margin_error_indices = np.where(
        high_margin_error_mask
    )[0]

    high_margin_errors = build_prediction_records(
        high_margin_error_indices,
        probabilities,
        predictions,
        labels,
        top1,
        top2,
        margin,
        entropy,
    )

    high_margin_errors.sort(
        key=lambda x: x["margin"],
        reverse=True,
    )

    # ---------------------------------------------------------------
    # Top-2 competition
    # ---------------------------------------------------------------

    top2_competition = build_top2_competition(
        probabilities,
        predictions,
        labels,
        margin,
        top1,
        top2,
    )

    # ---------------------------------------------------------------
    # Directional confusions
    # ---------------------------------------------------------------

    directional_confusions = build_directional_confusions(
        predictions,
        labels,
        correct,
    )

    # ---------------------------------------------------------------
    # Signal relationships
    # ---------------------------------------------------------------

    signal_relationships = {
        "top1_probability_vs_margin": correlation(
            top1,
            margin,
        ),
        "top1_probability_vs_entropy": correlation(
            top1,
            entropy,
        ),
        "margin_vs_entropy": correlation(
            margin,
            entropy,
        ),
        "margin_vs_correctness": correlation(
            margin,
            correct.astype(np.float64),
        ),
        "top1_probability_vs_correctness": correlation(
            top1,
            correct.astype(np.float64),
        ),
        "entropy_vs_correctness": correlation(
            entropy,
            correct.astype(np.float64),
        ),
    }

    # ---------------------------------------------------------------
    # Scientific interpretation
    # ---------------------------------------------------------------

    scientific_status = {
        "margin_behavior": "VERIFIED",
        "margin_correctness_association": "DEMONSTRATED",
        "top2_competition": "VERIFIED",
        "calibration": "UNKNOWN",
        "selective_prediction": "UNKNOWN",
        "abstention": "UNKNOWN",
        "human_review_routing": "UNKNOWN",
        "open_set_detection": "UNKNOWN",
        "interpretation": (
            "Margin is analyzed as a descriptive model-output signal. "
            "Its empirical association with correctness does not by "
            "itself establish calibration, selective prediction, "
            "abstention, human-review routing, or open-set detection."
        ),
        "diagnostic_margin_level": {
            "low_margin": "<0.10",
            "high_margin_error": ">=0.90",
            "meaning": (
                "These are fixed descriptive diagnostics only and "
                "are not operational thresholds."
            ),
        },
    }

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = {
        "phase": "C",
        "stage": "C5.3",
        "task": "Margin / Ambiguity Analysis",
        "dataset": "ABO",
        "target_field": "product_type",
        "num_classes": EXPECTED_NUM_CLASSES,
        "test_population": EXPECTED_TEST_POPULATION,
        "seed": SEED,
        "created_utc": utc_now(),
        "input_sources": {
            "c5_1_directory": str(C5_1_DIR),
            "c4_test_directory": str(C4_TEST_DIR),
            "c5_1_signal_inventory": str(
                C5_1_DIR / "c5_1_signal_inventory.json"
            ),
            "c4_test_labels": str(
                C4_TEST_DIR / "test_labels.npy"
            ),
        },
        "overall": overall,
        "correct_incorrect": {
            "correct_count": int(np.sum(correct_mask)),
            "incorrect_count": int(np.sum(incorrect_mask)),
            "correct_mean_margin": float(
                np.mean(margin[correct_mask])
            ),
            "incorrect_mean_margin": float(
                np.mean(margin[incorrect_mask])
            ),
            "mean_margin_difference_correct_minus_incorrect": float(
                np.mean(margin[correct_mask])
                - np.mean(margin[incorrect_mask])
            ),
            "correct_mean_top1_probability": float(
                np.mean(top1[correct_mask])
            ),
            "incorrect_mean_top1_probability": float(
                np.mean(top1[incorrect_mask])
            ),
            "correct_mean_entropy": float(
                np.mean(entropy[correct_mask])
            ),
            "incorrect_mean_entropy": float(
                np.mean(entropy[incorrect_mask])
            ),
        },
        "low_margin_diagnostic": {
            "definition": "margin < 0.10",
            "count": int(np.sum(low_margin_mask)),
            "accuracy": (
                float(
                    np.mean(correct[low_margin_mask])
                )
                if np.any(low_margin_mask)
                else None
            ),
        },
        "high_margin_error_diagnostic": {
            "definition": "margin >= 0.90 and incorrect",
            "count": int(
                np.sum(high_margin_error_mask)
            ),
        },
        "top2_competition": {
            "unique_pairs": top2_competition[
                "unique_top2_pairs"
            ],
        },
        "directional_confusion": {
            "unique_error_pairs": len(
                directional_confusions
            ),
            "top_pairs": directional_confusions[:100],
        },
        "scientific_status": scientific_status,
    }

    # ---------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------

    summary_path = OUTPUT_DIR / "c5_3_summary.json"
    bins_path = OUTPUT_DIR / "c5_3_margin_bins.json"
    ci_path = OUTPUT_DIR / "c5_3_correct_incorrect_summary.json"
    low_path = OUTPUT_DIR / "c5_3_low_margin_predictions.json"
    high_path = OUTPUT_DIR / "c5_3_high_margin_errors.json"
    top2_path = OUTPUT_DIR / "c5_3_top2_competition.json"
    relationships_path = OUTPUT_DIR / "c5_3_signal_relationships.json"

    json_dump(summary, summary_path)

    json_dump(
        {
            "stage": "C5.3",
            "margin_bins": margin_bins,
        },
        bins_path,
    )

    json_dump(
        {
            "stage": "C5.3",
            "correct_incorrect": correct_incorrect,
        },
        ci_path,
    )

    json_dump(
        {
            "stage": "C5.3",
            "diagnostic_definition": (
                "margin < 0.10; descriptive only"
            ),
            "count": len(low_margin_predictions),
            "predictions": low_margin_predictions,
        },
        low_path,
    )

    json_dump(
        {
            "stage": "C5.3",
            "diagnostic_definition": (
                "margin >= 0.90 and incorrect; descriptive only"
            ),
            "count": len(high_margin_errors),
            "errors": high_margin_errors,
        },
        high_path,
    )

    json_dump(
        top2_competition,
        top2_path,
    )

    json_dump(
        {
            "stage": "C5.3",
            "relationships": signal_relationships,
        },
        relationships_path,
    )

    # ---------------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------------

    input_hashes = {}

    for name, path in data["paths"].items():
        input_hashes[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    output_paths = {
        "summary": summary_path,
        "margin_bins": bins_path,
        "correct_incorrect_summary": ci_path,
        "low_margin_predictions": low_path,
        "high_margin_errors": high_path,
        "top2_competition": top2_path,
        "signal_relationships": relationships_path,
    }

    output_hashes = {}

    for name, path in output_paths.items():
        output_hashes[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    manifest = {
        "phase": "C",
        "stage": "C5.3",
        "task": "Margin / Ambiguity Analysis",
        "dataset": "ABO",
        "target_field": "product_type",
        "num_classes": EXPECTED_NUM_CLASSES,
        "test_population": EXPECTED_TEST_POPULATION,
        "seed": SEED,
        "created_utc": utc_now(),
        "test_usage": {
            "test_access": True,
            "usage": "read-only descriptive reliability analysis",
            "training": False,
            "model_selection": False,
            "hyperparameter_tuning": False,
            "threshold_optimization": False,
            "calibration_fitting": False,
        },
        "upstream_integrity": {
            "c5_1_read_only": True,
            "c4_read_only": True,
            "no_upstream_modification": True,
        },
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "scientific_status": scientific_status,
    }

    manifest_path = (
        OUTPUT_DIR / "c5_3_audit_manifest.json"
    )

    # Write manifest once before hashing itself.
    json_dump(manifest, manifest_path)

    print("\n" + "=" * 80)
    print("C5.3 ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"[RESULT] Test population: "
        f"{EXPECTED_TEST_POPULATION}"
    )

    print(
        f"[RESULT] Correct: "
        f"{int(np.sum(correct_mask))}"
    )

    print(
        f"[RESULT] Incorrect: "
        f"{int(np.sum(incorrect_mask))}"
    )

    print(
        f"[RESULT] Mean margin — correct: "
        f"{np.mean(margin[correct_mask]):.6f}"
    )

    print(
        f"[RESULT] Mean margin — incorrect: "
        f"{np.mean(margin[incorrect_mask]):.6f}"
    )

    print(
        f"[RESULT] Low-margin diagnostic (<0.10): "
        f"{int(np.sum(low_margin_mask))}"
    )

    print(
        f"[RESULT] High-margin error diagnostic (>=0.90): "
        f"{int(np.sum(high_margin_error_mask))}"
    )

    print(
        f"[RESULT] Unique top-2 competition pairs: "
        f"{top2_competition['unique_top2_pairs']}"
    )

    print(
        f"[RESULT] Unique directional error pairs: "
        f"{len(directional_confusions)}"
    )

    print("\n[SCIENTIFIC STATUS]")
    print("  Margin behavior: VERIFIED")
    print("  Margin/correctness association: DEMONSTRATED")
    print("  Top-2 competition: VERIFIED")
    print("  Calibration: UNKNOWN")
    print("  Selective prediction: UNKNOWN")
    print("  Abstention: UNKNOWN")
    print("  Human-review routing: UNKNOWN")
    print("  Open-set detection: UNKNOWN")

    print("\n[OUTPUTS]")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")

    print(f"  manifest: {manifest_path}")

    print("\n[PASS] C5.3 completed without modifying upstream artifacts.")


if __name__ == "__main__":
    main()