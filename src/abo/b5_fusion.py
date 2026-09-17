"""
B5.4 — ABO Multimodal Fusion

Purpose
-------
Fuse the already-generated B5.3 text-only and image-only probability
predictions for ABO product_type classification.

Fusion:
    P_fused = alpha * P_text + (1 - alpha) * P_image

Protocol
--------
1. Validate the completed B5.3 prediction artifacts.
2. Load the B5.3 VALIDATION predictions.
3. Search the predefined alpha grid on VALIDATION only.
4. Select alpha using:
       primary   = highest validation macro F1
       tie-break = highest validation accuracy
       tie-break = highest validation weighted F1
       tie-break = smallest alpha
5. Freeze the selected alpha.
6. Release validation probability arrays.
7. Load B5.3 TEST predictions.
8. Evaluate:
       - image-only endpoint (alpha=0)
       - text-only endpoint (alpha=1)
       - frozen selected fusion alpha
9. Write the B5.4 report and frozen fusion metadata.

Important
---------
B5.4 is a late-probability fusion stage. It does NOT:
- read physical images
- regenerate B3 representations
- retrain B4.2 models
- regenerate splits
- modify B2/B3/B4.2
- download images
- guess image paths
- use synthetic data
- integrate Rakuten or MAVE
- tune alpha using the test set

Source contracts
----------------
B5.4 consumes only the completed B5.3 prediction artifacts and
their corresponding validation report.

B5.3 is LOCKED at v001 prediction preparation over B3 v002 / B4.2 v002.
"""


from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

PRED_DIR = (
    ROOT
    / "data"
    / "models"
    / "abo"
    / "b5"
    / "predictions_v001"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "fusion"
    / "abo"
)

B5_3_REPORT_PATH = (
    REPORT_DIR
    / "abo_b5.3_prediction_prep_v001.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "abo_b5.4_fusion_v001.json"
)

FUSION_DIR = (
    ROOT
    / "data"
    / "models"
    / "abo"
    / "b5"
    / "fusion_v001"
)

FUSION_METADATA_PATH = (
    FUSION_DIR
    / "metadata.json"
)

B5_3_METADATA_PATH = (
    PRED_DIR
    / "metadata.json"
)


# ============================================================================
# LOCKED PROJECT CONTRACT
# ============================================================================

DATASET = "ABO"

TARGET_FIELD = "product_type"

B3_VERSION = "v002"

B4_VERSION = "v002"

B5_VERSION = "v001"

B5_4_VERSION = "v001"

PREDICTION_VERSION = "v001"

EXPECTED_CLASS_COUNT = 549

EXPECTED_TEXT_CLASS_COUNT = 553

EXPECTED_IMAGE_CLASS_COUNT = 549

EXPECTED_COUNTS = {
    "train": 69_823,
    "validation": 69_867,
    "test": 7_346,
}


# ============================================================================
# FUSION SEARCH GRID
# ============================================================================

# alpha = 0.0 -> image-only
# alpha = 1.0 -> text-only

ALPHA_GRID = [
    i / 10.0
    for i in range(11)
]


# Probability arrays are evaluated in chunks.
EVAL_CHUNK_SIZE = 8_192


# B5.4 itself is deterministic and does not use randomness.
RANDOM_SEED = 20260827


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def fail(message: str) -> None:
    raise RuntimeError(
        f"B5.4 FAILURE: {message}"
    )


def load_json(
    path: Path,
) -> dict[str, Any]:

    if not path.exists():
        fail(
            f"Required JSON artifact not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            value = json.load(handle)

    except Exception as exc:
        fail(
            f"Could not read JSON {path}: {exc}"
        )

    if not isinstance(value, dict):
        fail(
            f"Expected JSON object in {path}"
        )

    return value


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def require_key(
    obj: dict[str, Any],
    key: str,
    context: str,
) -> Any:

    if key not in obj:
        fail(
            f"{context} is missing required field "
            f"'{key}'."
        )

    return obj[key]


def require_value(
    obj: dict[str, Any],
    key: str,
    expected: Any,
    context: str,
) -> None:

    value = require_key(
        obj,
        key,
        context,
    )

    if value != expected:
        fail(
            f"{context}.{key} mismatch. "
            f"Expected {expected!r}; "
            f"found {value!r}."
        )


# ============================================================================
# B5.3 REPORT VALIDATION
# ============================================================================

def validate_b5_3_report() -> dict[str, Any]:
    """
    Validate the actual B5.3 prediction-preparation report schema.

    Important:
    - This function follows the actual B5.3 report produced by
      b5_prediction_prep.py.
    - It does NOT invent fields that are absent from the report.
    - Version provenance is taken from B5.3 metadata, not assumed
      from a different report field.
    """

    report = load_json(B5_3_REPORT_PATH)

    # ------------------------------------------------------------------
    # Core report status
    # ------------------------------------------------------------------

    require_value(
        report,
        "status",
        "PASS",
        "B5.3 report",
    )

    # ------------------------------------------------------------------
    # Report identity
    # ------------------------------------------------------------------

    if "report" in report:
        require_value(
            report,
            "report",
            "abo_b5.3_prediction_preparation",
            "B5.3 report",
        )

    if "stage" in report:
        require_value(
            report,
            "stage",
            "B5.3",
            "B5.3 report",
        )

    if "dataset" in report:
        require_value(
            report,
            "dataset",
            "ABO",
            "B5.3 report",
        )

    if "target_field" in report:
        require_value(
            report,
            "target_field",
            TARGET_FIELD,
            "B5.3 report",
        )

    # ------------------------------------------------------------------
    # Class alignment
    #
    # Actual B5.3 schema:
    #
    # class_alignment:
    #   actual_label_alignment
    #   probability_columns_aligned_by_product_type
    #   text_class_count
    #   image_class_count
    #   common_class_count
    #   text_only_class_count
    #   image_only_class_count
    #   common_class_probability_normalization
    #   normalization_method
    # ------------------------------------------------------------------

    class_alignment = require_key(
        report,
        "class_alignment",
        "B5.3 report",
    )

    if not isinstance(class_alignment, dict):
        fail(
            "B5.3 report.class_alignment "
            "must be an object."
        )

    require_value(
        class_alignment,
        "actual_label_alignment",
        True,
        "B5.3 report.class_alignment",
    )

    require_value(
        class_alignment,
        "probability_columns_aligned_by_product_type",
        True,
        "B5.3 report.class_alignment",
    )

    require_value(
        class_alignment,
        "common_class_probability_normalization",
        True,
        "B5.3 report.class_alignment",
    )

    # ------------------------------------------------------------------
    # Exact known class counts
    # ------------------------------------------------------------------

    if "text_class_count" in class_alignment:
        require_value(
            class_alignment,
            "text_class_count",
            553,
            "B5.3 report.class_alignment",
        )

    if "image_class_count" in class_alignment:
        require_value(
            class_alignment,
            "image_class_count",
            549,
            "B5.3 report.class_alignment",
        )

    if "common_class_count" in class_alignment:
        require_value(
            class_alignment,
            "common_class_count",
            549,
            "B5.3 report.class_alignment",
        )

    if "text_only_class_count" in class_alignment:
        require_value(
            class_alignment,
            "text_only_class_count",
            4,
            "B5.3 report.class_alignment",
        )

    if "image_only_class_count" in class_alignment:
        require_value(
            class_alignment,
            "image_only_class_count",
            0,
            "B5.3 report.class_alignment",
        )

    # ------------------------------------------------------------------
    # Normalization method
    # ------------------------------------------------------------------

    if "normalization_method" in class_alignment:
        require_value(
            class_alignment,
            "normalization_method",
            (
                "Renormalize projected probabilities "
                "over the common product_type classes"
            ),
            "B5.3 report.class_alignment",
        )

    # ------------------------------------------------------------------
    # Integrity
    #
    # Actual B5.3 schema uses:
    #   cross_split_record_id_leakage
    #   model_retraining
    #   split_regeneration
    #   b3_modification
    #   b4.2_modification
    #   image_download
    #   image_path_guessing
    #   synthetic_images
    #   fusion_performed
    # ------------------------------------------------------------------

    integrity = require_key(
        report,
        "integrity",
        "B5.3 report",
    )

    if not isinstance(integrity, dict):
        fail(
            "B5.3 report.integrity "
            "must be an object."
        )

    require_value(
        integrity,
        "cross_split_record_id_leakage",
        0,
        "B5.3 report.integrity",
    )

    require_value(
        integrity,
        "model_retraining",
        False,
        "B5.3 report.integrity",
    )

    require_value(
        integrity,
        "split_regeneration",
        False,
        "B5.3 report.integrity",
    )

    require_value(
        integrity,
        "b3_modification",
        False,
        "B5.3 report.integrity",
    )

    require_value(
        integrity,
        "b4.2_modification",
        False,
        "B5.3 report.integrity",
    )

    require_value(
        integrity,
        "image_download",
        False,
        "B5.3 report.integrity",
    )

    require_value(
        integrity,
        "image_path_guessing",
        False,
        "B5.3 report.integrity",
    )

    require_value(
        integrity,
        "synthetic_images",
        False,
        "B5.3 report.integrity",
    )

    require_value(
        integrity,
        "fusion_performed",
        False,
        "B5.3 report.integrity",
    )

    return report
# ============================================================================
# B5.3 METADATA VALIDATION
# ============================================================================

def validate_b5_3_metadata() -> dict[str, Any]:
    """
    Validate the actual locked B5.3 metadata schema.

    Actual B5.3 schema:

        prediction_version
        stage
        dataset
        b3_representation_version
        b4_version
        target_field
        probability_space
        common_classes
        split_counts
        cross_split_prediction_record_id_leakage
        source_contracts

    No recursive lookup is used.
    """

    metadata = load_json(
        B5_3_METADATA_PATH
    )

    # ------------------------------------------------------------------
    # Top-level provenance
    # ------------------------------------------------------------------

    require_value(
        metadata,
        "prediction_version",
        PREDICTION_VERSION,
        "B5.3 metadata",
    )

    require_value(
        metadata,
        "stage",
        "B5.3",
        "B5.3 metadata",
    )

    require_value(
        metadata,
        "dataset",
        DATASET,
        "B5.3 metadata",
    )

    # IMPORTANT:
    # The actual B5.3 field is
    # "b3_representation_version", NOT "b3_version".
    require_value(
        metadata,
        "b3_representation_version",
        B3_VERSION,
        "B5.3 metadata",
    )

    require_value(
        metadata,
        "b4_version",
        B4_VERSION,
        "B5.3 metadata",
    )

    require_value(
        metadata,
        "target_field",
        TARGET_FIELD,
        "B5.3 metadata",
    )

    # ------------------------------------------------------------------
    # Probability-space contract
    # ------------------------------------------------------------------

    probability_space = require_key(
        metadata,
        "probability_space",
        "B5.3 metadata",
    )

    if not isinstance(
        probability_space,
        dict,
    ):
        fail(
            "B5.3 metadata.probability_space "
            "must be an object."
        )

    require_value(
        probability_space,
        "class_count",
        EXPECTED_CLASS_COUNT,
        "B5.3 metadata.probability_space",
    )

    require_value(
        probability_space,
        "text_original_class_count",
        EXPECTED_TEXT_CLASS_COUNT,
        "B5.3 metadata.probability_space",
    )

    require_value(
        probability_space,
        "image_original_class_count",
        EXPECTED_IMAGE_CLASS_COUNT,
        "B5.3 metadata.probability_space",
    )

    require_value(
        probability_space,
        "text_only_classes_excluded",
        EXPECTED_TEXT_CLASS_COUNT - EXPECTED_CLASS_COUNT,
        "B5.3 metadata.probability_space",
    )

    require_value(
        probability_space,
        "image_only_classes_excluded",
        EXPECTED_IMAGE_CLASS_COUNT - EXPECTED_CLASS_COUNT,
        "B5.3 metadata.probability_space",
    )

    require_value(
        probability_space,
        "common_class_probability_normalization",
        True,
        "B5.3 metadata.probability_space",
    )

    require_value(
        probability_space,
        "normalization_method",
        "Renormalize projected probabilities over the common product_type classes",
        "B5.3 metadata.probability_space",
    )

    # ------------------------------------------------------------------
    # Common class space
    # ------------------------------------------------------------------

    common_classes = require_key(
        metadata,
        "common_classes",
        "B5.3 metadata",
    )

    if not isinstance(
        common_classes,
        list,
    ):
        fail(
            "B5.3 metadata.common_classes "
            "must be a list."
        )

    if len(common_classes) != EXPECTED_CLASS_COUNT:
        fail(
            "B5.3 common class count mismatch. "
            f"Expected {EXPECTED_CLASS_COUNT}; "
            f"found {len(common_classes)}."
        )

    normalized_classes: list[str] = []

    for index, label in enumerate(
        common_classes
    ):

        if not isinstance(
            label,
            str,
        ):
            fail(
                "B5.3 common_classes contains "
                f"a non-string label at index {index}: "
                f"{label!r}"
            )

        normalized = label.strip()

        if not normalized:
            fail(
                f"B5.3 common_classes contains "
                f"an empty label at index {index}."
            )

        normalized_classes.append(
            normalized
        )

    if len(
        set(normalized_classes)
    ) != EXPECTED_CLASS_COUNT:

        fail(
            "B5.3 common_classes contains duplicate labels."
        )

    # ------------------------------------------------------------------
    # Split counts
    # ------------------------------------------------------------------

    split_counts = require_key(
        metadata,
        "split_counts",
        "B5.3 metadata",
    )

    if not isinstance(
        split_counts,
        dict,
    ):
        fail(
            "B5.3 metadata.split_counts "
            "must be an object."
        )

    for split, expected_count in EXPECTED_COUNTS.items():

        actual_count = require_key(
            split_counts,
            split,
            "B5.3 metadata.split_counts",
        )

        if actual_count != expected_count:
            fail(
                f"B5.3 metadata split count mismatch "
                f"for {split}. Expected "
                f"{expected_count:,}; found "
                f"{actual_count:,}."
            )

    # ------------------------------------------------------------------
    # Cross-split leakage
    # ------------------------------------------------------------------

    require_value(
        metadata,
        "cross_split_prediction_record_id_leakage",
        0,
        "B5.3 metadata",
    )

    # ------------------------------------------------------------------
    # Source contracts
    # ------------------------------------------------------------------

    source_contracts = require_key(
        metadata,
        "source_contracts",
        "B5.3 metadata",
    )

    if not isinstance(
        source_contracts,
        dict,
    ):
        fail(
            "B5.3 metadata.source_contracts "
            "must be an object."
        )

    required_source_contracts = {
        "b5.1",
        "b5.2",
        "b3_statistics",
        "b4_statistics",
    }

    missing_contracts = (
        required_source_contracts
        - set(source_contracts.keys())
    )

    if missing_contracts:
        fail(
            "B5.3 metadata is missing source contracts: "
            f"{sorted(missing_contracts)}"
        )

    return {
        **metadata,
        "_validated_common_classes": normalized_classes,
    }


# ============================================================================
# PREDICTION LOADING
# ============================================================================

REQUIRED_ARRAYS = {
    "record_ids",
    "target_labels",
    "text_probabilities",
    "image_probabilities",
}


def load_prediction_split(
    split: str,
) -> dict[str, np.ndarray]:

    path = (
        PRED_DIR
        / f"{split}.npz"
    )

    if not path.exists():
        fail(
            f"Missing B5.3 prediction artifact: {path}"
        )

    try:

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            missing = (
                REQUIRED_ARRAYS
                - set(data.files)
            )

            if missing:
                fail(
                    f"{split}.npz is missing "
                    f"required arrays: "
                    f"{sorted(missing)}"
                )

            arrays = {
                key: data[key]
                for key in REQUIRED_ARRAYS
            }

    except RuntimeError:
        raise

    except Exception as exc:
        fail(
            f"Could not load {path}: {exc}"
        )

    return arrays


# ============================================================================
# PREDICTION VALIDATION
# ============================================================================

def validate_prediction_arrays(
    split: str,
    arrays: dict[str, np.ndarray],
) -> None:

    if split not in EXPECTED_COUNTS:
        fail(
            f"No expected population count "
            f"defined for split '{split}'."
        )

    expected_count = EXPECTED_COUNTS[
        split
    ]

    record_ids = arrays[
        "record_ids"
    ]

    target_labels = arrays[
        "target_labels"
    ]

    text_probs = arrays[
        "text_probabilities"
    ]

    image_probs = arrays[
        "image_probabilities"
    ]

    # ------------------------------------------------------------------
    # Record IDs / target labels
    # ------------------------------------------------------------------

    if record_ids.ndim != 1:
        fail(
            f"{split}: record_ids must be 1-dimensional."
        )

    if target_labels.ndim != 1:
        fail(
            f"{split}: target_labels must be 1-dimensional."
        )

    if len(record_ids) != expected_count:
        fail(
            f"{split}: expected {expected_count:,} "
            f"record IDs, found {len(record_ids):,}."
        )

    if len(target_labels) != expected_count:
        fail(
            f"{split}: expected {expected_count:,} "
            f"target labels, found {len(target_labels):,}."
        )

    unique_count = len(
        np.unique(record_ids)
    )

    if unique_count != expected_count:
        fail(
            f"{split}: duplicate record IDs detected. "
            f"Unique={unique_count:,}, "
            f"expected={expected_count:,}."
        )

    # ------------------------------------------------------------------
    # Probability shapes
    # ------------------------------------------------------------------

    expected_shape = (
        expected_count,
        EXPECTED_CLASS_COUNT,
    )

    if text_probs.ndim != 2:
        fail(
            f"{split}: text_probabilities "
            "must be 2-dimensional."
        )

    if image_probs.ndim != 2:
        fail(
            f"{split}: image_probabilities "
            "must be 2-dimensional."
        )

    if text_probs.shape != expected_shape:
        fail(
            f"{split}: text probability shape mismatch. "
            f"Expected {expected_shape}, "
            f"found {text_probs.shape}."
        )

    if image_probs.shape != expected_shape:
        fail(
            f"{split}: image probability shape mismatch. "
            f"Expected {expected_shape}, "
            f"found {image_probs.shape}."
        )

    # ------------------------------------------------------------------
    # Probability validity
    # ------------------------------------------------------------------

    if not np.all(
        np.isfinite(text_probs)
    ):
        fail(
            f"{split}: non-finite text probabilities detected."
        )

    if not np.all(
        np.isfinite(image_probs)
    ):
        fail(
            f"{split}: non-finite image probabilities detected."
        )

    if np.any(
        text_probs < 0
    ):
        fail(
            f"{split}: negative text probabilities detected."
        )

    if np.any(
        image_probs < 0
    ):
        fail(
            f"{split}: negative image probabilities detected."
        )

    text_sums = text_probs.sum(
        axis=1
    )

    image_sums = image_probs.sum(
        axis=1
    )

    if not np.allclose(
        text_sums,
        1.0,
        atol=1e-5,
    ):
        fail(
            f"{split}: text probability rows "
            "are not normalized."
        )

    if not np.allclose(
        image_sums,
        1.0,
        atol=1e-5,
    ):
        fail(
            f"{split}: image probability rows "
            "are not normalized."
        )


# ============================================================================
# TARGET LABEL ALIGNMENT
# ============================================================================

def encode_targets(
    target_labels: np.ndarray,
    common_classes: list[str],
    split: str,
) -> np.ndarray:

    class_to_index = {
        label: index
        for index, label
        in enumerate(common_classes)
    }

    encoded = np.empty(
        len(target_labels),
        dtype=np.int32,
    )

    missing: set[str] = set()

    for index, raw_label in enumerate(
        target_labels
    ):

        label = str(
            raw_label
        ).strip()

        if label not in class_to_index:

            missing.add(label)

            encoded[index] = -1

        else:

            encoded[index] = (
                class_to_index[label]
            )

    if missing:

        examples = sorted(
            missing
        )[:10]

        fail(
            f"{split}: target labels outside "
            "the B5.3 common class space. "
            f"Examples: {examples}"
        )

    return encoded


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(
    target_indices: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:

    labels = np.arange(
        EXPECTED_CLASS_COUNT
    )

    return {
        "accuracy": float(
            accuracy_score(
                target_indices,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                target_indices,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                target_indices,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================================
# FUSION EVALUATION
# ============================================================================

def evaluate_fusion(
    text_probs: np.ndarray,
    image_probs: np.ndarray,
    target_indices: np.ndarray,
    alpha: float,
) -> dict[str, float]:

    if not 0.0 <= alpha <= 1.0:
        fail(
            f"Invalid alpha: {alpha}"
        )

    if text_probs.shape != image_probs.shape:
        fail(
            "Text/image probability shapes "
            "differ during fusion."
        )

    if text_probs.shape[0] != len(
        target_indices
    ):
        fail(
            "Probability/target row count mismatch."
        )

    predictions = np.empty(
        len(target_indices),
        dtype=np.int32,
    )

    for start in range(
        0,
        len(target_indices),
        EVAL_CHUNK_SIZE,
    ):

        end = min(
            start + EVAL_CHUNK_SIZE,
            len(target_indices),
        )

        text_chunk = text_probs[
            start:end
        ]

        image_chunk = image_probs[
            start:end
        ]

        fused_chunk = (
            alpha * text_chunk
            + (1.0 - alpha) * image_chunk
        )

        predictions[
            start:end
        ] = np.argmax(
            fused_chunk,
            axis=1,
        )

    return compute_metrics(
        target_indices,
        predictions,
    )


# ============================================================================
# ALPHA SELECTION
# ============================================================================

def select_alpha(
    validation_results: list[dict[str, Any]],
) -> dict[str, Any]:

    if not validation_results:
        fail(
            "No validation alpha results available."
        )

    ranked = sorted(
        validation_results,
        key=lambda result: (
            -result["metrics"]["macro_f1"],
            -result["metrics"]["accuracy"],
            -result["metrics"]["weighted_f1"],
            result["alpha"],
        ),
    )

    return ranked[0]


# ============================================================================
# METRIC COMPARISON
# ============================================================================

def calculate_delta(
    primary: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:

    return {
        metric: (
            primary[metric]
            - baseline[metric]
        )
        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
        )
    }


def classify_fusion_outcome(
    fused: dict[str, float],
    text: dict[str, float],
    image: dict[str, float],
) -> str:

    fused_macro = fused[
        "macro_f1"
    ]

    text_macro = text[
        "macro_f1"
    ]

    image_macro = image[
        "macro_f1"
    ]

    if (
        fused_macro > text_macro
        and fused_macro > image_macro
    ):
        return "better_than_both"

    if fused_macro > text_macro:
        return "better_than_text_only"

    if fused_macro > image_macro:
        return "better_than_image_only"

    if (
        fused_macro == text_macro
        and fused_macro == image_macro
    ):
        return "equal_to_both"

    return "not_better_than_either"


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print("=" * 72)
    print("B5.4 — ABO MULTIMODAL FUSION")
    print("=" * 72)

    print(
        f"Project root:       {ROOT}"
    )

    print(
        f"B3:                 {B3_VERSION}"
    )

    print(
        f"B4.2:               {B4_VERSION}"
    )

    print(
        f"B5:                 {B5_VERSION}"
    )

    print(
        f"B5.4:               {B5_4_VERSION}"
    )

    print(
        f"Target:             {TARGET_FIELD}"
    )

    print(
        f"B5.3 predictions:   {PRED_DIR}"
    )

    print()

    # ======================================================================
    # STEP 1 — B5.3 provenance
    # ======================================================================

    print(
        "[1/7] Validating B5.3 provenance..."
    )

    b5_3_metadata = (
        validate_b5_3_metadata()
    )

    b5_3_report = (
        validate_b5_3_report()
    )

    common_classes = (
        b5_3_metadata[
            "_validated_common_classes"
        ]
    )

    print(
        "  B5.3 metadata: PASS"
    )

    print(
        "  B5.3 report:   PASS"
    )

    print(
        f"  B3 representation: "
        f"{b5_3_metadata['b3_representation_version']}"
    )

    print(
        f"  B4.2 version:       "
        f"{b5_3_metadata['b4_version']}"
    )

    print(
        f"  Text classes:       "
        f"{b5_3_metadata['probability_space']['text_original_class_count']}"
    )

    print(
        f"  Image classes:      "
        f"{b5_3_metadata['probability_space']['image_original_class_count']}"
    )

    print(
        f"  Common classes:     "
        f"{len(common_classes)}"
    )

    # ======================================================================
    # STEP 2 — validation predictions
    # ======================================================================

    print()
    print(
        "[2/7] Loading B5.3 validation predictions..."
    )

    validation = load_prediction_split(
        "validation"
    )

    validate_prediction_arrays(
        "validation",
        validation,
    )

    print(
        f"  Validation rows: "
        f"{len(validation['record_ids']):,}"
    )

    print(
        "  Validation artifact: PASS"
    )

    # ======================================================================
    # STEP 3 — validation target alignment
    # ======================================================================

    print()
    print(
        "[3/7] Validating validation alignment..."
    )

    validation_targets = encode_targets(
        validation[
            "target_labels"
        ],
        common_classes,
        "validation",
    )

    print(
        "  Record IDs:             PASS"
    )

    print(
        "  Target labels:          PASS"
    )

    print(
        "  Text probabilities:     PASS"
    )

    print(
        "  Image probabilities:    PASS"
    )

    print(
        "  Common class space:     "
        f"{len(common_classes)}"
    )

    # ======================================================================
    # STEP 4 — validation alpha selection
    # ======================================================================

    print()
    print(
        "[4/7] Selecting alpha on VALIDATION only..."
    )

    print()
    print(
        "  P_fused = alpha * P_text "
        "+ (1-alpha) * P_image"
    )

    print(
        "  Selection priority:"
    )

    print(
        "    1. validation macro F1"
    )

    print(
        "    2. validation accuracy"
    )

    print(
        "    3. validation weighted F1"
    )

    print(
        "    4. smaller alpha"
    )

    print()

    validation_results: list[
        dict[str, Any]
    ] = []

    for alpha in ALPHA_GRID:

        metrics = evaluate_fusion(
            validation[
                "text_probabilities"
            ],
            validation[
                "image_probabilities"
            ],
            validation_targets,
            alpha,
        )

        result = {
            "alpha": round(
                alpha,
                4,
            ),

            "text_weight": round(
                alpha,
                4,
            ),

            "image_weight": round(
                1.0 - alpha,
                4,
            ),

            "metrics": metrics,
        }

        validation_results.append(
            result
        )

        print(
            f"  alpha={alpha:.1f} | "
            f"accuracy={metrics['accuracy']:.6f} | "
            f"macro_f1={metrics['macro_f1']:.6f} | "
            f"weighted_f1={metrics['weighted_f1']:.6f}"
        )

    selected = select_alpha(
        validation_results
    )

    selected_alpha = float(
        selected["alpha"]
    )

    selected_text_weight = (
        selected_alpha
    )

    selected_image_weight = (
        1.0 - selected_alpha
    )

    print()
    print(
        f"  SELECTED ALPHA: "
        f"{selected_alpha:.4f}"
    )

    print(
        f"  Text weight:    "
        f"{selected_text_weight:.4f}"
    )

    print(
        f"  Image weight:   "
        f"{selected_image_weight:.4f}"
    )

    validation_image_only = next(
        result
        for result in validation_results
        if result["alpha"] == 0.0
    )

    validation_text_only = next(
        result
        for result in validation_results
        if result["alpha"] == 1.0
    )

    validation_fused = (
        selected["metrics"]
    )

    validation_delta_vs_text = (
        calculate_delta(
            validation_fused,
            validation_text_only[
                "metrics"
            ],
        )
    )

    validation_delta_vs_image = (
        calculate_delta(
            validation_fused,
            validation_image_only[
                "metrics"
            ],
        )
    )

    # ----------------------------------------------------------------------
    # Release validation arrays before loading test.
    # ----------------------------------------------------------------------

    del validation
    del validation_targets

    gc.collect()

    print()
    print(
        "  Validation probability arrays released."
    )

    # ======================================================================
    # STEP 5 — test predictions
    # ======================================================================

    print()
    print(
        "[5/7] Loading B5.3 test predictions..."
    )

    test = load_prediction_split(
        "test"
    )

    validate_prediction_arrays(
        "test",
        test,
    )

    test_targets = encode_targets(
        test[
            "target_labels"
        ],
        common_classes,
        "test",
    )

    print(
        f"  Test rows: "
        f"{len(test['record_ids']):,}"
    )

    print(
        "  Test artifact: PASS"
    )

    print(
        "  Test target alignment: PASS"
    )

    print(
        "  B5.3 cross-split leakage: 0"
    )

    print(
        "  Test record-ID uniqueness: PASS"
    )

    # ======================================================================
    # STEP 6 — frozen test evaluation
    # ======================================================================

    print()
    print(
        "[6/7] Evaluating FROZEN alpha on TEST..."
    )

    # ----------------------------------------------------------------------
    # Image-only endpoint: alpha = 0
    # ----------------------------------------------------------------------

    test_image_only = evaluate_fusion(
        test[
            "text_probabilities"
        ],
        test[
            "image_probabilities"
        ],
        test_targets,
        0.0,
    )

    # ----------------------------------------------------------------------
    # Text-only endpoint: alpha = 1
    # ----------------------------------------------------------------------

    test_text_only = evaluate_fusion(
        test[
            "text_probabilities"
        ],
        test[
            "image_probabilities"
        ],
        test_targets,
        1.0,
    )

    # ----------------------------------------------------------------------
    # Frozen selected alpha
    # ----------------------------------------------------------------------

    test_fused = evaluate_fusion(
        test[
            "text_probabilities"
        ],
        test[
            "image_probabilities"
        ],
        test_targets,
        selected_alpha,
    )

    test_delta_vs_text = (
        calculate_delta(
            test_fused,
            test_text_only,
        )
    )

    test_delta_vs_image = (
        calculate_delta(
            test_fused,
            test_image_only,
        )
    )

    fusion_outcome = (
        classify_fusion_outcome(
            test_fused,
            test_text_only,
            test_image_only,
        )
    )

    print()
    print(
        "  TEST RESULTS"
    )

    print(
        f"  Text-only  | "
        f"accuracy={test_text_only['accuracy']:.6f} | "
        f"macro_f1={test_text_only['macro_f1']:.6f} | "
        f"weighted_f1={test_text_only['weighted_f1']:.6f}"
    )

    print(
        f"  Image-only | "
        f"accuracy={test_image_only['accuracy']:.6f} | "
        f"macro_f1={test_image_only['macro_f1']:.6f} | "
        f"weighted_f1={test_image_only['weighted_f1']:.6f}"
    )

    print(
        f"  Fusion     | "
        f"accuracy={test_fused['accuracy']:.6f} | "
        f"macro_f1={test_fused['macro_f1']:.6f} | "
        f"weighted_f1={test_fused['weighted_f1']:.6f}"
    )

    print()
    print(
        f"  Fusion outcome: "
        f"{fusion_outcome}"
    )

    # ======================================================================
    # STEP 7 — write B5.4 artifacts
    # ======================================================================

    print()
    print(
        "[7/7] Writing B5.4 artifacts..."
    )

    FUSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Fusion metadata
    # ----------------------------------------------------------------------

    fusion_metadata = {
        "fusion_version": B5_4_VERSION,

        "stage": "B5.4",

        "dataset": DATASET,

        "target_field": TARGET_FIELD,

        "b3_version": B3_VERSION,

        "b4_version": B4_VERSION,

        "b5_version": B5_VERSION,

        "prediction_version": PREDICTION_VERSION,

        "method": {
            "type": "late_probability_fusion",

            "equation": (
                "P_fused = alpha * P_text "
                "+ (1-alpha) * P_image"
            ),

            "selection_split": "validation",

            "evaluation_split": "test",

            "alpha_grid": ALPHA_GRID,

            "selection_criterion": [
                "validation_macro_f1_descending",
                "validation_accuracy_descending",
                "validation_weighted_f1_descending",
                "alpha_ascending",
            ],

            "selected_alpha": selected_alpha,

            "selected_text_weight": (
                selected_text_weight
            ),

            "selected_image_weight": (
                selected_image_weight
            ),
        },

        "class_space": {
            "class_count": len(
                common_classes
            ),

            "text_original_class_count": (
                b5_3_metadata[
                    "probability_space"
                ][
                    "text_original_class_count"
                ]
            ),

            "image_original_class_count": (
                b5_3_metadata[
                    "probability_space"
                ][
                    "image_original_class_count"
                ]
            ),

            "alignment": "product_type",

            "classes": common_classes,
        },

        "population": {
            "train_records": EXPECTED_COUNTS[
                "train"
            ],

            "validation_records": EXPECTED_COUNTS[
                "validation"
            ],

            "test_records": EXPECTED_COUNTS[
                "test"
            ],

            "common_paired_population": True,

            "population_source": "B5.2/B5.3",
        },

        "validation_selection": {
            "selected_metrics": (
                validation_fused
            ),

            "text_only": (
                validation_text_only[
                    "metrics"
                ]
            ),

            "image_only": (
                validation_image_only[
                    "metrics"
                ]
            ),

            "delta_vs_text_only": (
                validation_delta_vs_text
            ),

            "delta_vs_image_only": (
                validation_delta_vs_image
            ),

            "alpha_results": (
                validation_results
            ),
        },

        "test_evaluation": {
            "selected_alpha": selected_alpha,

            "text_only": test_text_only,

            "image_only": test_image_only,

            "fusion": test_fused,

            "delta_vs_text_only": (
                test_delta_vs_text
            ),

            "delta_vs_image_only": (
                test_delta_vs_image
            ),

            "fusion_outcome_by_macro_f1": (
                fusion_outcome
            ),
        },

        "integrity": {
            "b5_3_prediction_artifacts_used": True,

            "b3_modified": False,

            "b4_2_modified": False,

            "b4_2_retrained": False,

            "splits_regenerated": False,

            "images_read": False,

            "image_download": False,

            "path_guessing": False,

            "synthetic_data": False,

            "rakuten_used": False,

            "mave_used": False,

            "train_used_for_alpha_selection": False,

            "test_used_for_alpha_selection": False,

            "validation_used_for_alpha_selection": True,

            "validation_test_record_id_leakage": (
                b5_3_metadata[
                    "cross_split_prediction_record_id_leakage"
                ]
            ),

            "fusion_performed": True,
        },

        "source_contracts": {
            "b5_3_metadata": str(
                B5_3_METADATA_PATH
            ),

            "b5_3_report": str(
                B5_3_REPORT_PATH
            ),
        },
    }

    write_json(
        FUSION_METADATA_PATH,
        fusion_metadata,
    )

    # ----------------------------------------------------------------------
    # B5.4 report
    # ----------------------------------------------------------------------

    report = {
        "report_version": B5_4_VERSION,

        "report_name": (
            "abo_b5.4_multimodal_fusion"
        ),

        "status": "PASS",

        "dataset": DATASET,

        "stage": "B5.4",

        "b3_version": B3_VERSION,

        "b4_version": B4_VERSION,

        "b5_version": B5_VERSION,

        "target_field": TARGET_FIELD,

        "execution": {
            "seed": RANDOM_SEED,

            "images_read": False,

            "model_training": False,

            "model_retraining": False,

            "split_regeneration": False,

            "fusion_only": True,
        },

        "prediction_source": {
            "directory": str(
                PRED_DIR
            ),

            "prediction_version": (
                PREDICTION_VERSION
            ),

            "b5_3_metadata": str(
                B5_3_METADATA_PATH
            ),

            "b5_3_report": str(
                B5_3_REPORT_PATH
            ),
        },

        "class_alignment": {
            "class_count": len(
                common_classes
            ),

            "text_original_classes": (
                b5_3_metadata[
                    "probability_space"
                ][
                    "text_original_class_count"
                ]
            ),

            "image_original_classes": (
                b5_3_metadata[
                    "probability_space"
                ][
                    "image_original_class_count"
                ]
            ),

            "common_classes": len(
                common_classes
            ),

            "probability_columns_aligned_by_product_type": (
                True
            ),

            "common_probability_normalization": (
                b5_3_metadata[
                    "probability_space"
                ][
                    "common_class_probability_normalization"
                ]
            ),

            "normalization_method": (
                b5_3_metadata[
                    "probability_space"
                ][
                    "normalization_method"
                ]
            ),
        },

        "population": {
            "train": EXPECTED_COUNTS[
                "train"
            ],

            "validation": EXPECTED_COUNTS[
                "validation"
            ],

            "test": EXPECTED_COUNTS[
                "test"
            ],

            "paired_common_target_population": (
                True
            ),

            "validation_test_record_id_leakage": (
                b5_3_metadata[
                    "cross_split_prediction_record_id_leakage"
                ]
            ),
        },

        "fusion": {
            "equation": (
                "P_fused = alpha * P_text "
                "+ (1-alpha) * P_image"
            ),

            "alpha_grid": ALPHA_GRID,

            "selected_alpha": selected_alpha,

            "selected_text_weight": (
                selected_text_weight
            ),

            "selected_image_weight": (
                selected_image_weight
            ),

            "selection_split": "validation",

            "test_evaluation_alpha_frozen": (
                True
            ),

            "selection_criterion": [
                "macro_f1",
                "accuracy",
                "weighted_f1",
                "smallest_alpha",
            ],
        },

        "validation": {
            "selected": selected,

            "text_only": (
                validation_text_only
            ),

            "image_only": (
                validation_image_only
            ),

            "delta_vs_text_only": (
                validation_delta_vs_text
            ),

            "delta_vs_image_only": (
                validation_delta_vs_image
            ),

            "alpha_grid_results": (
                validation_results
            ),
        },

        "test": {
            "selected_alpha": selected_alpha,

            "text_only": (
                test_text_only
            ),

            "image_only": (
                test_image_only
            ),

            "fusion": (
                test_fused
            ),

            "delta_vs_text_only": (
                test_delta_vs_text
            ),

            "delta_vs_image_only": (
                test_delta_vs_image
            ),

            "fusion_outcome_by_macro_f1": (
                fusion_outcome
            ),
        },

        "integrity": {
            "b3_modified": False,

            "b4_2_modified": False,

            "b4_2_retrained": False,

            "splits_regenerated": False,

            "images_read": False,

            "image_download": False,

            "path_guessing": False,

            "synthetic_data": False,

            "rakuten_integration": False,

            "mave_integration": False,

            "test_tuning": False,

            "train_used_for_alpha_selection": False,

            "validation_used_for_alpha_selection": True,

            "fusion_performed": True,
        },

        "artifacts": {
            "fusion_metadata": str(
                FUSION_METADATA_PATH
            ),

            "report": str(
                REPORT_PATH
            ),
        },

        "source_contracts": {
            "b5_3_metadata": str(
                B5_3_METADATA_PATH
            ),

            "b5_3_report": str(
                B5_3_REPORT_PATH
            ),
        },
    }

    write_json(
        REPORT_PATH,
        report,
    )

    # ======================================================================
    # FINAL SUMMARY
    # ======================================================================

    print()
    print("=" * 72)
    print("B5.4 MULTIMODAL FUSION PASS")
    print("=" * 72)

    print(
        f"Selected alpha:       "
        f"{selected_alpha:.4f}"
    )

    print(
        f"Text weight:          "
        f"{selected_text_weight:.4f}"
    )

    print(
        f"Image weight:         "
        f"{selected_image_weight:.4f}"
    )

    print()
    print("TEST")

    print(
        f"Text-only accuracy:   "
        f"{test_text_only['accuracy']:.6f}"
    )

    print(
        f"Image-only accuracy:  "
        f"{test_image_only['accuracy']:.6f}"
    )

    print(
        f"Fusion accuracy:      "
        f"{test_fused['accuracy']:.6f}"
    )

    print(
        f"Text-only macro F1:   "
        f"{test_text_only['macro_f1']:.6f}"
    )

    print(
        f"Image-only macro F1:  "
        f"{test_image_only['macro_f1']:.6f}"
    )

    print(
        f"Fusion macro F1:      "
        f"{test_fused['macro_f1']:.6f}"
    )

    print()
    print(
        f"Fusion outcome:       "
        f"{fusion_outcome}"
    )

    print()
    print(
        f"Report: {REPORT_PATH}"
    )

    print(
        f"Metadata: {FUSION_METADATA_PATH}"
    )

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except Exception as exc:

        print()
        print(
            str(exc)
        )

        sys.exit(1)