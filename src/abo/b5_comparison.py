from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# ============================================================================
# B5.5 — ABO COMPARISON AGAINST B4.2
# ============================================================================
#
# Purpose
# -------
# Compare the LOCKED B4.2 text-only and image-only baselines against the
# LOCKED B5.4 multimodal fusion result on the exact same B5 common paired
# population.
#
# B5.5 does NOT:
# - retrain B4.2
# - read physical images
# - regenerate B3
# - regenerate splits
# - tune alpha
# - modify B3/B4.2/B5.3/B5.4
# - use Rakuten
# - use MAVE
#
# B5.5 consumes:
# - B5.3 prediction artifacts
# - B5.3 metadata
# - B5.4 fusion report
# - B5.4 fusion metadata
#
# Primary comparison:
# - accuracy
# - macro F1
# - weighted F1
#
# Important:
# The B5.4 alpha is already frozen.
# B5.5 does NOT select or modify alpha.
# ============================================================================


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

B5_3_METADATA_PATH = (
    PRED_DIR
    / "metadata.json"
)

B5_4_REPORT_PATH = (
    ROOT
    / "reports"
    / "fusion"
    / "abo"
    / "abo_b5.4_fusion_v001.json"
)

B5_4_METADATA_PATH = (
    ROOT
    / "data"
    / "models"
    / "abo"
    / "b5"
    / "fusion_v001"
    / "metadata.json"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "fusion"
    / "abo"
)

REPORT_PATH = (
    REPORT_DIR
    / "abo_b5.5_comparison_v001.json"
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

B5_5_VERSION = "v001"

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
# GENERAL UTILITIES
# ============================================================================

def fail(message: str) -> None:
    raise RuntimeError(
        f"B5.5 FAILURE: {message}"
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

    if not isinstance(
        value,
        dict,
    ):

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
# B5.3 METADATA VALIDATION
# ============================================================================

def validate_b5_3_metadata() -> dict[str, Any]:

    metadata = load_json(
        B5_3_METADATA_PATH
    )

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
        4,
        "B5.3 metadata.probability_space",
    )

    require_value(
        probability_space,
        "image_only_classes_excluded",
        0,
        "B5.3 metadata.probability_space",
    )

    require_value(
        probability_space,
        "common_class_probability_normalization",
        True,
        "B5.3 metadata.probability_space",
    )

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

    if len(
        set(
            str(label).strip()
            for label in common_classes
        )
    ) != EXPECTED_CLASS_COUNT:

        fail(
            "B5.3 common_classes contains duplicate labels."
        )

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

    for split, expected in EXPECTED_COUNTS.items():

        require_value(
            split_counts,
            split,
            expected,
            "B5.3 metadata.split_counts",
        )

    require_value(
        metadata,
        "cross_split_prediction_record_id_leakage",
        0,
        "B5.3 metadata",
    )

    return metadata


# ============================================================================
# B5.4 REPORT VALIDATION
# ============================================================================

def validate_b5_4_report() -> dict[str, Any]:

    report = load_json(
        B5_4_REPORT_PATH
    )

    # ------------------------------------------------------------------
    # Core identity
    # ------------------------------------------------------------------

    require_value(
        report,
        "status",
        "PASS",
        "B5.4 report",
    )

    if "dataset" in report:

        require_value(
            report,
            "dataset",
            DATASET,
            "B5.4 report",
        )

    if "stage" in report:

        require_value(
            report,
            "stage",
            "B5.4",
            "B5.4 report",
        )

    if "b3_version" in report:

        require_value(
            report,
            "b3_version",
            B3_VERSION,
            "B5.4 report",
        )

    if "b4_version" in report:

        require_value(
            report,
            "b4_version",
            B4_VERSION,
            "B5.4 report",
        )

    if "b5_version" in report:

        require_value(
            report,
            "b5_version",
            B5_VERSION,
            "B5.4 report",
        )

    if "target_field" in report:

        require_value(
            report,
            "target_field",
            TARGET_FIELD,
            "B5.4 report",
        )

    # ------------------------------------------------------------------
    # Prediction source
    # ------------------------------------------------------------------

    prediction_source = require_key(
        report,
        "prediction_source",
        "B5.4 report",
    )

    if not isinstance(
        prediction_source,
        dict,
    ):

        fail(
            "B5.4 report.prediction_source "
            "must be an object."
        )

    if "prediction_version" in prediction_source:

        require_value(
            prediction_source,
            "prediction_version",
            PREDICTION_VERSION,
            "B5.4 report.prediction_source",
        )

    # ------------------------------------------------------------------
    # Class alignment
    # ------------------------------------------------------------------

    class_alignment = require_key(
        report,
        "class_alignment",
        "B5.4 report",
    )

    if not isinstance(
        class_alignment,
        dict,
    ):

        fail(
            "B5.4 report.class_alignment "
            "must be an object."
        )

    require_value(
        class_alignment,
        "class_count",
        EXPECTED_CLASS_COUNT,
        "B5.4 report.class_alignment",
    )

    require_value(
        class_alignment,
        "text_original_classes",
        EXPECTED_TEXT_CLASS_COUNT,
        "B5.4 report.class_alignment",
    )

    require_value(
        class_alignment,
        "image_original_classes",
        EXPECTED_IMAGE_CLASS_COUNT,
        "B5.4 report.class_alignment",
    )

    require_value(
        class_alignment,
        "common_classes",
        EXPECTED_CLASS_COUNT,
        "B5.4 report.class_alignment",
    )

    require_value(
        class_alignment,
        "probability_columns_aligned_by_product_type",
        True,
        "B5.4 report.class_alignment",
    )

    require_value(
        class_alignment,
        "common_probability_normalization",
        True,
        "B5.4 report.class_alignment",
    )

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    population = require_key(
        report,
        "population",
        "B5.4 report",
    )

    if not isinstance(
        population,
        dict,
    ):

        fail(
            "B5.4 report.population "
            "must be an object."
        )

    require_value(
        population,
        "train",
        EXPECTED_COUNTS["train"],
        "B5.4 report.population",
    )

    require_value(
        population,
        "validation",
        EXPECTED_COUNTS["validation"],
        "B5.4 report.population",
    )

    require_value(
        population,
        "test",
        EXPECTED_COUNTS["test"],
        "B5.4 report.population",
    )

    require_value(
        population,
        "paired_common_target_population",
        True,
        "B5.4 report.population",
    )

    # ------------------------------------------------------------------
    # Frozen fusion
    # ------------------------------------------------------------------

    fusion = require_key(
        report,
        "fusion",
        "B5.4 report",
    )

    if not isinstance(
        fusion,
        dict,
    ):

        fail(
            "B5.4 report.fusion "
            "must be an object."
        )

    selected_alpha = require_key(
        fusion,
        "selected_alpha",
        "B5.4 report.fusion",
    )

    if not isinstance(
        selected_alpha,
        (int, float),
    ):

        fail(
            "B5.4 selected_alpha must be numeric."
        )

    if not 0.0 <= float(selected_alpha) <= 1.0:

        fail(
            f"B5.4 selected_alpha is invalid: "
            f"{selected_alpha}"
        )

    require_value(
        fusion,
        "selection_split",
        "validation",
        "B5.4 report.fusion",
    )

    require_value(
        fusion,
        "test_evaluation_alpha_frozen",
        True,
        "B5.4 report.fusion",
    )

    selection_criterion = require_key(
        fusion,
        "selection_criterion",
        "B5.4 report.fusion",
    )

    if selection_criterion != [
        "macro_f1",
        "accuracy",
        "weighted_f1",
        "smallest_alpha",
    ]:

        fail(
            "B5.4 selection criterion does not "
            "match the locked B5.4 protocol."
        )

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    integrity = require_key(
        report,
        "integrity",
        "B5.4 report",
    )

    if not isinstance(
        integrity,
        dict,
    ):

        fail(
            "B5.4 report.integrity "
            "must be an object."
        )

    require_value(
        integrity,
        "b3_modified",
        False,
        "B5.4 report.integrity",
    )

    require_value(
        integrity,
        "b4_2_modified",
        False,
        "B5.4 report.integrity",
    )

    require_value(
        integrity,
        "b4_2_retrained",
        False,
        "B5.4 report.integrity",
    )

    require_value(
        integrity,
        "splits_regenerated",
        False,
        "B5.4 report.integrity",
    )

    require_value(
        integrity,
        "images_read",
        False,
        "B5.4 report.integrity",
    )

    require_value(
        integrity,
        "test_tuning",
        False,
        "B5.4 report.integrity",
    )

    require_value(
        integrity,
        "train_used_for_alpha_selection",
        False,
        "B5.4 report.integrity",
    )

    require_value(
        integrity,
        "validation_used_for_alpha_selection",
        True,
        "B5.4 report.integrity",
    )

    return report


# ============================================================================
# B5.4 METADATA VALIDATION
# ============================================================================

def validate_b5_4_metadata() -> dict[str, Any]:

    metadata = load_json(
        B5_4_METADATA_PATH
    )

    require_value(
        metadata,
        "fusion_version",
        B5_4_VERSION,
        "B5.4 metadata",
    )

    require_value(
        metadata,
        "stage",
        "B5.4",
        "B5.4 metadata",
    )

    require_value(
        metadata,
        "dataset",
        DATASET,
        "B5.4 metadata",
    )

    require_value(
        metadata,
        "target_field",
        TARGET_FIELD,
        "B5.4 metadata",
    )

    require_value(
        metadata,
        "b3_version",
        B3_VERSION,
        "B5.4 metadata",
    )

    require_value(
        metadata,
        "b4_version",
        B4_VERSION,
        "B5.4 metadata",
    )

    require_value(
        metadata,
        "b5_version",
        B5_VERSION,
        "B5.4 metadata",
    )

    require_value(
        metadata,
        "prediction_version",
        PREDICTION_VERSION,
        "B5.4 metadata",
    )

    method = require_key(
        metadata,
        "method",
        "B5.4 metadata",
    )

    if not isinstance(
        method,
        dict,
    ):

        fail(
            "B5.4 metadata.method "
            "must be an object."
        )

    require_value(
        method,
        "type",
        "late_probability_fusion",
        "B5.4 metadata.method",
    )

    require_value(
        method,
        "selection_split",
        "validation",
        "B5.4 metadata.method",
    )

    require_value(
        method,
        "evaluation_split",
        "test",
        "B5.4 metadata.method",
    )

    selected_alpha = require_key(
        method,
        "selected_alpha",
        "B5.4 metadata.method",
    )

    if not isinstance(
        selected_alpha,
        (int, float),
    ):

        fail(
            "B5.4 metadata selected_alpha "
            "must be numeric."
        )

    if not 0.0 <= float(selected_alpha) <= 1.0:

        fail(
            f"Invalid B5.4 selected alpha: "
            f"{selected_alpha}"
        )

    require_value(
        metadata,
        "integrity",
        metadata["integrity"],
        "B5.4 metadata",
    )

    integrity = metadata[
        "integrity"
    ]

    if not isinstance(
        integrity,
        dict,
    ):

        fail(
            "B5.4 metadata.integrity "
            "must be an object."
        )

    require_value(
        integrity,
        "b3_modified",
        False,
        "B5.4 metadata.integrity",
    )

    require_value(
        integrity,
        "b4_2_modified",
        False,
        "B5.4 metadata.integrity",
    )

    require_value(
        integrity,
        "b4_2_retrained",
        False,
        "B5.4 metadata.integrity",
    )

    require_value(
        integrity,
        "splits_regenerated",
        False,
        "B5.4 metadata.integrity",
    )

    require_value(
        integrity,
        "images_read",
        False,
        "B5.4 metadata.integrity",
    )

    require_value(
        integrity,
        "test_used_for_alpha_selection",
        False,
        "B5.4 metadata.integrity",
    )

    require_value(
        integrity,
        "validation_used_for_alpha_selection",
        True,
        "B5.4 metadata.integrity",
    )

    return metadata


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
            f"Missing B5.3 prediction artifact: "
            f"{path}"
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
# PREDICTION ARTIFACT VALIDATION
# ============================================================================

def validate_prediction_split(
    split: str,
    arrays: dict[str, np.ndarray],
) -> None:

    expected = EXPECTED_COUNTS[
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

    if record_ids.ndim != 1:

        fail(
            f"{split}: record_ids must be 1-dimensional."
        )

    if target_labels.ndim != 1:

        fail(
            f"{split}: target_labels must be 1-dimensional."
        )

    if len(record_ids) != expected:

        fail(
            f"{split}: expected {expected:,} "
            f"records; found {len(record_ids):,}."
        )

    if len(target_labels) != expected:

        fail(
            f"{split}: target label count mismatch."
        )

    if len(
        np.unique(record_ids)
    ) != expected:

        fail(
            f"{split}: duplicate record IDs detected."
        )

    expected_shape = (
        expected,
        EXPECTED_CLASS_COUNT,
    )

    if text_probs.shape != expected_shape:

        fail(
            f"{split}: text probability shape mismatch. "
            f"Expected {expected_shape}; "
            f"found {text_probs.shape}."
        )

    if image_probs.shape != expected_shape:

        fail(
            f"{split}: image probability shape mismatch. "
            f"Expected {expected_shape}; "
            f"found {image_probs.shape}."
        )

    if not np.all(
        np.isfinite(text_probs)
    ):

        fail(
            f"{split}: non-finite text probabilities."
        )

    if not np.all(
        np.isfinite(image_probs)
    ):

        fail(
            f"{split}: non-finite image probabilities."
        )

    if np.any(
        text_probs < 0
    ):

        fail(
            f"{split}: negative text probabilities."
        )

    if np.any(
        image_probs < 0
    ):

        fail(
            f"{split}: negative image probabilities."
        )

    if not np.allclose(
        text_probs.sum(axis=1),
        1.0,
        atol=1e-5,
    ):

        fail(
            f"{split}: text probabilities "
            "are not normalized."
        )

    if not np.allclose(
        image_probs.sum(axis=1),
        1.0,
        atol=1e-5,
    ):

        fail(
            f"{split}: image probabilities "
            "are not normalized."
        )


# ============================================================================
# CROSS-SPLIT RECORD-ID VALIDATION
# ============================================================================

def validate_cross_split_record_ids(
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
) -> dict[str, int]:

    train_set = set(
        train_ids.tolist()
    )

    validation_set = set(
        validation_ids.tolist()
    )

    test_set = set(
        test_ids.tolist()
    )

    train_validation = len(
        train_set & validation_set
    )

    train_test = len(
        train_set & test_set
    )

    validation_test = len(
        validation_set & test_set
    )

    if train_validation != 0:

        fail(
            "Train/validation record-ID leakage detected."
        )

    if train_test != 0:

        fail(
            "Train/test record-ID leakage detected."
        )

    if validation_test != 0:

        fail(
            "Validation/test record-ID leakage detected."
        )

    return {
        "train_validation": train_validation,
        "train_test": train_test,
        "validation_test": validation_test,
    }


# ============================================================================
# TARGET ENCODING
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
            f"the common class space. "
            f"Examples: {examples}"
        )

    return encoded


# ============================================================================
# METRICS
# ============================================================================

def metrics_from_probabilities(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    labels = np.arange(
        EXPECTED_CLASS_COUNT
    )

    return {
        "accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================================
# FUSION
# ============================================================================

def fused_probabilities(
    text_probabilities: np.ndarray,
    image_probabilities: np.ndarray,
    alpha: float,
) -> np.ndarray:

    if text_probabilities.shape != image_probabilities.shape:

        fail(
            "Text and image probability shapes "
            "do not match."
        )

    fused = (
        alpha * text_probabilities
        + (1.0 - alpha)
        * image_probabilities
    )

    if not np.all(
        np.isfinite(fused)
    ):

        fail(
            "Fusion produced non-finite probabilities."
        )

    if np.any(
        fused < 0
    ):

        fail(
            "Fusion produced negative probabilities."
        )

    if not np.allclose(
        fused.sum(axis=1),
        1.0,
        atol=1e-5,
    ):

        fail(
            "Fusion probabilities are not normalized."
        )

    return fused


# ============================================================================
# METRIC DELTAS
# ============================================================================

def metric_delta(
    primary: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:

    return {
        metric: float(
            primary[metric]
            - baseline[metric]
        )
        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
        )
    }


def relative_metric_change(
    primary: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float | None]:

    result: dict[
        str,
        float | None
    ] = {}

    for metric in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ):

        denominator = baseline[
            metric
        ]

        if denominator == 0:

            result[metric] = None

        else:

            result[metric] = float(
                (
                    primary[metric]
                    - denominator
                )
                / denominator
            )

    return result


# ============================================================================
# RANKING
# ============================================================================

def rank_models(
    text_metrics: dict[str, float],
    image_metrics: dict[str, float],
    fusion_metrics: dict[str, float],
) -> dict[str, list[str]]:

    models = [
        "text_only",
        "image_only",
        "fusion",
    ]

    metric_names = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    metric_rankings: dict[
        str,
        list[str]
    ] = {}

    values = {
        "text_only": text_metrics,
        "image_only": image_metrics,
        "fusion": fusion_metrics,
    }

    for metric in metric_names:

        ordered = sorted(
            models,
            key=lambda model: (
                -values[model][metric],
                model,
            ),
        )

        metric_rankings[
            metric
        ] = ordered

    return metric_rankings


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print("=" * 72)
    print("B5.5 — ABO COMPARISON AGAINST B4.2")
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
        f"B5.5:               {B5_5_VERSION}"
    )

    print(
        f"Target:             {TARGET_FIELD}"
    )

    # ======================================================================
    # STEP 1 — SOURCE VALIDATION
    # ======================================================================

    print()
    print(
        "[1/6] Validating locked B5.3 and B5.4 artifacts..."
    )

    b5_3_metadata = (
        validate_b5_3_metadata()
    )

    b5_4_report = (
        validate_b5_4_report()
    )

    b5_4_metadata = (
        validate_b5_4_metadata()
    )

    common_classes = [
        str(label).strip()
        for label
        in b5_3_metadata[
            "common_classes"
        ]
    ]

    selected_alpha = float(
        b5_4_report[
            "fusion"
        ][
            "selected_alpha"
        ]
    )

    metadata_alpha = float(
        b5_4_metadata[
            "method"
        ][
            "selected_alpha"
        ]
    )

    if selected_alpha != metadata_alpha:

        fail(
            "B5.4 report and B5.4 metadata "
            "contain different selected alpha values."
        )

    print(
        "  B5.3 metadata: PASS"
    )

    print(
        "  B5.4 report:   PASS"
    )

    print(
        "  B5.4 metadata: PASS"
    )

    print(
        f"  B3 representation: v{B3_VERSION[1:]}"
    )

    print(
        f"  B4.2 version:       {B4_VERSION}"
    )

    print(
        f"  Common classes:     {len(common_classes)}"
    )

    print(
        f"  Frozen alpha:       {selected_alpha:.4f}"
    )

    # ======================================================================
    # STEP 2 — LOAD SAME B5 POPULATION
    # ======================================================================

    print()
    print(
        "[2/6] Loading B5.3 common-population predictions..."
    )

    train = load_prediction_split(
        "train"
    )

    validation = load_prediction_split(
        "validation"
    )

    test = load_prediction_split(
        "test"
    )

    validate_prediction_split(
        "train",
        train,
    )

    validate_prediction_split(
        "validation",
        validation,
    )

    validate_prediction_split(
        "test",
        test,
    )

    leakage = (
        validate_cross_split_record_ids(
            train["record_ids"],
            validation["record_ids"],
            test["record_ids"],
        )
    )

    print(
        f"  Train:            "
        f"{len(train['record_ids']):,}"
    )

    print(
        f"  Validation:       "
        f"{len(validation['record_ids']):,}"
    )

    print(
        f"  Test:             "
        f"{len(test['record_ids']):,}"
    )

    print(
        "  Record-ID uniqueness: PASS"
    )

    print(
        "  Cross-split leakage:  0"
    )

    # ======================================================================
    # STEP 3 — VERIFY COMMON POPULATION
    # ======================================================================

    print()
    print(
        "[3/6] Verifying common B5 comparison population..."
    )

    train_targets = encode_targets(
        train["target_labels"],
        common_classes,
        "train",
    )

    validation_targets = encode_targets(
        validation["target_labels"],
        common_classes,
        "validation",
    )

    test_targets = encode_targets(
        test["target_labels"],
        common_classes,
        "test",
    )

    # B5.3 has already established the common paired target population.
    # B5.5 therefore uses exactly those records and labels.
    print(
        "  Train target alignment:      PASS"
    )

    print(
        "  Validation target alignment: PASS"
    )

    print(
        "  Test target alignment:       PASS"
    )

    print(
        "  Common class space:          "
        f"{len(common_classes)}"
    )

    print(
        "  Comparison population:       "
        "B5.3 common paired population"
    )

    # ======================================================================
    # STEP 4 — RECOMPUTE LOCKED ENDPOINTS
    # ======================================================================

    print()
    print(
        "[4/6] Evaluating locked B4.2 endpoints..."
    )

    # ------------------------------------------------------------------
    # B4.2 text-only endpoint
    #
    # alpha = 1.0
    # ------------------------------------------------------------------

    test_text_only = metrics_from_probabilities(
        test[
            "text_probabilities"
        ],
        test_targets,
    )

    # ------------------------------------------------------------------
    # B4.2 image-only endpoint
    #
    # alpha = 0.0
    # ------------------------------------------------------------------

    test_image_only = metrics_from_probabilities(
        test[
            "image_probabilities"
        ],
        test_targets,
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

    # ======================================================================
    # STEP 5 — RECOMPUTE FROZEN B5.4 FUSION
    # ======================================================================

    print()
    print(
        "[5/6] Evaluating frozen B5.4 fusion..."
    )

    test_fused_probabilities = fused_probabilities(
        test[
            "text_probabilities"
        ],
        test[
            "image_probabilities"
        ],
        selected_alpha,
    )

    test_fusion = metrics_from_probabilities(
        test_fused_probabilities,
        test_targets,
    )

    print(
        f"  Fusion     | "
        f"accuracy={test_fusion['accuracy']:.6f} | "
        f"macro_f1={test_fusion['macro_f1']:.6f} | "
        f"weighted_f1={test_fusion['weighted_f1']:.6f}"
    )

    # ------------------------------------------------------------------
    # Compare fusion against each B4.2 endpoint.
    # ------------------------------------------------------------------

    delta_fusion_vs_text = metric_delta(
        test_fusion,
        test_text_only,
    )

    delta_fusion_vs_image = metric_delta(
        test_fusion,
        test_image_only,
    )

    relative_fusion_vs_text = (
        relative_metric_change(
            test_fusion,
            test_text_only,
        )
    )

    relative_fusion_vs_image = (
        relative_metric_change(
            test_fusion,
            test_image_only,
        )
    )

    rankings = rank_models(
        test_text_only,
        test_image_only,
        test_fusion,
    )

    # ------------------------------------------------------------------
    # Explicit metric outcomes.
    # ------------------------------------------------------------------

    metric_outcomes = {}

    for metric in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ):

        fusion_value = test_fusion[
            metric
        ]

        text_value = test_text_only[
            metric
        ]

        image_value = test_image_only[
            metric
        ]

        if (
            fusion_value > text_value
            and fusion_value > image_value
        ):

            outcome = "fusion_best"

        elif (
            fusion_value > text_value
        ):

            outcome = "fusion_better_than_text_only"

        elif (
            fusion_value > image_value
        ):

            outcome = "fusion_better_than_image_only"

        elif (
            fusion_value == text_value
            and fusion_value == image_value
        ):

            outcome = "fusion_equal_to_both"

        else:

            outcome = "fusion_not_best"

        metric_outcomes[
            metric
        ] = {
            "fusion": fusion_value,
            "text_only": text_value,
            "image_only": image_value,
            "outcome": outcome,
        }

    # ------------------------------------------------------------------
    # Important overall interpretation.
    #
    # Do NOT collapse all metrics into "better" or "worse".
    # B5.5 reports metric-specific outcomes explicitly.
    # ------------------------------------------------------------------

    if (
        metric_outcomes[
            "macro_f1"
        ][
            "outcome"
        ] == "fusion_best"
    ):

        primary_outcome = (
            "fusion_best_on_macro_f1"
        )

    else:

        primary_outcome = (
            "fusion_not_best_on_macro_f1"
        )

    # ======================================================================
    # STEP 6 — WRITE B5.5 REPORT
    # ======================================================================

    print()
    print(
        "[6/6] Writing B5.5 comparison report..."
    )

    report = {
        "report_version": B5_5_VERSION,

        "report_name": (
            "abo_b5.5_comparison_against_b4.2"
        ),

        "status": "PASS",

        "dataset": DATASET,

        "stage": "B5.5",

        "b3_version": B3_VERSION,

        "b4_version": B4_VERSION,

        "b5_version": B5_VERSION,

        "target_field": TARGET_FIELD,

        "comparison_scope": {
            "description": (
                "Comparison of locked B4.2 text-only and "
                "image-only endpoints against locked B5.4 "
                "multimodal fusion on the exact same B5 "
                "common paired population."
            ),

            "population_source": (
                "B5.3 common paired prediction population"
            ),

            "same_population_for_all_models": True,

            "same_target_labels_for_all_models": True,

            "same_common_class_space_for_all_models": True,

            "common_class_count": (
                EXPECTED_CLASS_COUNT
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

            "comparison_split": "test",

            "test_records_compared": (
                EXPECTED_COUNTS["test"]
            ),

            "paired_common_target_population": True,

            "cross_split_record_id_leakage": leakage,
        },

        "frozen_fusion": {
            "source_stage": "B5.4",

            "fusion_version": B5_4_VERSION,

            "selected_alpha": selected_alpha,

            "text_weight": selected_alpha,

            "image_weight": (
                1.0 - selected_alpha
            ),

            "alpha_was_retuned_in_b5_5": False,

            "test_used_for_alpha_selection": False,
        },

        "models_compared": {
            "text_only": {
                "source": "B4.2",
                "modality": "text",
                "model_version": B4_VERSION,
                "metrics": test_text_only,
            },

            "image_only": {
                "source": "B4.2",
                "modality": "image",
                "model_version": B4_VERSION,
                "metrics": test_image_only,
            },

            "fusion": {
                "source": "B5.4",
                "modality": "text_plus_image",
                "fusion_version": B5_4_VERSION,
                "metrics": test_fusion,
            },
        },

        "fusion_vs_text_only": {
            "absolute_delta": (
                delta_fusion_vs_text
            ),

            "relative_change": (
                relative_fusion_vs_text
            ),
        },

        "fusion_vs_image_only": {
            "absolute_delta": (
                delta_fusion_vs_image
            ),

            "relative_change": (
                relative_fusion_vs_image
            ),
        },

        "metric_outcomes": metric_outcomes,

        "rankings": rankings,

        "primary_outcome": primary_outcome,

        "interpretation": {
            "accuracy": (
                "Fusion accuracy is evaluated directly "
                "against the B4.2 text-only and image-only "
                "baselines on the same B5 comparison population."
            ),

            "macro_f1": (
                "Macro F1 is the primary class-balanced "
                "comparison metric. Fusion is reported as "
                "best only if its macro F1 exceeds both "
                "B4.2 endpoints."
            ),

            "weighted_f1": (
                "Weighted F1 is reported independently and "
                "is not substituted for accuracy or macro F1."
            ),

            "no_universal_improvement_claim": True,
        },

        "integrity": {
            "b3_modified": False,

            "b4_2_modified": False,

            "b4_2_retrained": False,

            "b5_3_modified": False,

            "b5_4_modified": False,

            "splits_regenerated": False,

            "images_read": False,

            "image_download": False,

            "path_guessing": False,

            "synthetic_data": False,

            "alpha_retuned": False,

            "test_used_for_tuning": False,

            "rakuten_used": False,

            "mave_used": False,

            "same_population_comparison": True,

            "same_class_space_comparison": True,

            "fusion_recomputed_from_locked_b5_3_predictions": True,
        },

        "source_contracts": {
            "b5_3_metadata": str(
                B5_3_METADATA_PATH
            ),

            "b5_4_report": str(
                B5_4_REPORT_PATH
            ),

            "b5_4_metadata": str(
                B5_4_METADATA_PATH
            ),
        },
    }

    write_json(
        REPORT_PATH,
        report,
    )

    # ======================================================================
    # FINAL OUTPUT
    # ======================================================================

    print()
    print("=" * 72)
    print("B5.5 COMPARISON AGAINST B4.2 PASS")
    print("=" * 72)

    print()
    print(
        "TEST — SAME B5 COMMON PAIRED POPULATION"
    )

    print(
        f"Text-only  | "
        f"accuracy={test_text_only['accuracy']:.6f} | "
        f"macro_f1={test_text_only['macro_f1']:.6f} | "
        f"weighted_f1={test_text_only['weighted_f1']:.6f}"
    )

    print(
        f"Image-only | "
        f"accuracy={test_image_only['accuracy']:.6f} | "
        f"macro_f1={test_image_only['macro_f1']:.6f} | "
        f"weighted_f1={test_image_only['weighted_f1']:.6f}"
    )

    print(
        f"Fusion     | "
        f"accuracy={test_fusion['accuracy']:.6f} | "
        f"macro_f1={test_fusion['macro_f1']:.6f} | "
        f"weighted_f1={test_fusion['weighted_f1']:.6f}"
    )

    print()
    print(
        f"Frozen alpha: {selected_alpha:.4f}"
    )

    print()
    print(
        "FUSION DELTA VS TEXT-ONLY"
    )

    print(
        f"  Accuracy:    "
        f"{delta_fusion_vs_text['accuracy']:+.6f}"
    )

    print(
        f"  Macro F1:     "
        f"{delta_fusion_vs_text['macro_f1']:+.6f}"
    )

    print(
        f"  Weighted F1:  "
        f"{delta_fusion_vs_text['weighted_f1']:+.6f}"
    )

    print()
    print(
        "FUSION DELTA VS IMAGE-ONLY"
    )

    print(
        f"  Accuracy:    "
        f"{delta_fusion_vs_image['accuracy']:+.6f}"
    )

    print(
        f"  Macro F1:     "
        f"{delta_fusion_vs_image['macro_f1']:+.6f}"
    )

    print(
        f"  Weighted F1:  "
        f"{delta_fusion_vs_image['weighted_f1']:+.6f}"
    )

    print()
    print(
        f"Macro F1 outcome: "
        f"{primary_outcome}"
    )

    print()
    print(
        f"Report: {REPORT_PATH}"
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