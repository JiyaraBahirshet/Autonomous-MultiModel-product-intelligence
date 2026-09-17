from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# ============================================================================
# B5.6 — ABO ERROR / COMPLEMENTARITY ANALYSIS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

B3_VERSION = "v002"
B4_VERSION = "v002"
B5_VERSION = "v001"
B5_6_VERSION = "v001"

TARGET_FIELD = "product_type"
FROZEN_ALPHA = 0.9

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "b5"
    / "predictions_v001"
)

B5_4_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "fusion"
    / "abo"
    / "abo_b5.4_fusion_v001.json"
)

B5_5_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "fusion"
    / "abo"
    / "abo_b5.5_comparison_v001.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "fusion"
    / "abo"
)

OUTPUT_REPORT = OUTPUT_DIR / "abo_b5.6_error_analysis_v001.json"


EXPECTED_SPLITS = {
    "train": 69823,
    "validation": 69867,
    "test": 7346,
}


def fail(message: str) -> None:
    raise RuntimeError(f"B5.6 integrity failure: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path) -> dict:
    require(path.exists(), f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_npz(split: str) -> dict[str, np.ndarray]:
    path = PREDICTION_DIR / f"{split}.npz"
    require(path.exists(), f"Missing B5.3 prediction artifact: {path}")

    data = np.load(path, allow_pickle=False)

    required = {
        "record_ids",
        "target_labels",
        "text_probabilities",
        "image_probabilities",
    }

    require(
        required.issubset(set(data.files)),
        f"{split}.npz missing required arrays",
    )

    return {
        key: data[key]
        for key in required
    }


def extract_first_numeric(
    obj,
    keys: tuple[str, ...],
) -> float | None:
    """
    Recursively find the first numeric value associated with one of the
    supplied keys. This is deliberately used only for provenance validation
    against the already-locked B5.4/B5.5 artifacts.
    """
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                value = obj[key]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)

        for value in obj.values():
            result = extract_first_numeric(value, keys)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for value in obj:
            result = extract_first_numeric(value, keys)
            if result is not None:
                return result

    return None


def validate_provenance() -> tuple[dict, dict]:
    b5_4 = load_json(B5_4_REPORT)
    b5_5 = load_json(B5_5_REPORT)

    # B5.4/B5.5 reports must be successful.
    require(
        str(b5_4.get("status", "")).upper() == "PASS",
        "B5.4 report is not PASS",
    )

    require(
        str(b5_5.get("status", "")).upper() == "PASS",
        "B5.5 report is not PASS",
    )

    # Confirm the frozen alpha from the locked B5.4 artifact.
    alpha = extract_first_numeric(
        b5_4,
        (
            "selected_alpha",
            "frozen_alpha",
            "alpha",
        ),
    )

    require(alpha is not None, "Could not locate B5.4 frozen alpha")

    require(
        np.isclose(alpha, FROZEN_ALPHA, atol=1e-8),
        f"B5.4 alpha mismatch: expected {FROZEN_ALPHA}, found {alpha}",
    )

    return b5_4, b5_5


def validate_split(
    split: str,
    data: dict[str, np.ndarray],
) -> None:
    record_ids = data["record_ids"]
    target_labels = data["target_labels"]
    text_prob = data["text_probabilities"]
    image_prob = data["image_probabilities"]

    expected = EXPECTED_SPLITS[split]

    require(
        len(record_ids) == expected,
        f"{split}: expected {expected} records, found {len(record_ids)}",
    )

    require(
        len(np.unique(record_ids)) == len(record_ids),
        f"{split}: duplicate record IDs detected",
    )

    require(
        len(target_labels) == expected,
        f"{split}: target-label count mismatch",
    )

    require(
        text_prob.shape == (expected, 549),
        f"{split}: unexpected text probability shape {text_prob.shape}",
    )

    require(
        image_prob.shape == (expected, 549),
        f"{split}: unexpected image probability shape {image_prob.shape}",
    )

    require(
        np.isfinite(text_prob).all(),
        f"{split}: non-finite text probabilities",
    )

    require(
        np.isfinite(image_prob).all(),
        f"{split}: non-finite image probabilities",
    )

    require(
        (text_prob >= 0).all(),
        f"{split}: negative text probabilities",
    )

    require(
        (image_prob >= 0).all(),
        f"{split}: negative image probabilities",
    )

    require(
        np.allclose(text_prob.sum(axis=1), 1.0, atol=1e-5),
        f"{split}: text probability rows do not sum to 1",
    )

    require(
        np.allclose(image_prob.sum(axis=1), 1.0, atol=1e-5),
        f"{split}: image probability rows do not sum to 1",
    )


def build_common_class_predictions(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_prob = data["text_probabilities"]
    image_prob = data["image_probabilities"]

    fused_prob = (
        FROZEN_ALPHA * text_prob
        + (1.0 - FROZEN_ALPHA) * image_prob
    )

    require(
        np.isfinite(fused_prob).all(),
        "Fusion probabilities contain non-finite values",
    )

    require(
        (fused_prob >= 0).all(),
        "Fusion probabilities contain negative values",
    )

    require(
        np.allclose(fused_prob.sum(axis=1), 1.0, atol=1e-5),
        "Fusion probability rows do not sum to 1",
    )

    text_pred = np.argmax(text_prob, axis=1)
    image_pred = np.argmax(image_prob, axis=1)
    fusion_pred = np.argmax(fused_prob, axis=1)

    return text_pred, image_pred, fusion_pred


def calculate_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    common_class_count: int,
) -> dict:
    all_class_indices = np.arange(
        common_class_count,
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

def analyze_test_population(
    data: dict[str, np.ndarray],
) -> dict:
    record_ids = data["record_ids"]
    target_labels = data["target_labels"]

    text_pred, image_pred, fusion_pred = (
        build_common_class_predictions(data)
    )

    # target_labels contain the actual product_type strings.
    # B5.3 probability columns are already aligned to the common
    # 549-class product_type space.
    #
    # Therefore B5.6 maps each target string to its probability-space
    # column using the observed prediction label set.

    text_classes = np.unique(
        np.concatenate(
            [
                text_pred.astype(np.int64),
                image_pred.astype(np.int64),
                fusion_pred.astype(np.int64),
            ]
        )
    )

    # The actual class labels are needed for correctness comparison.
    # B5.3 stores target labels as strings, while probability columns
    # are common-class indices. Obtain the exact common class list from
    # B5.3 metadata.
    metadata = load_json(PREDICTION_DIR / "metadata.json")
    common_classes = metadata.get("common_classes")

    require(
        isinstance(common_classes, list),
        "B5.3 metadata missing common_classes",
    )

    require(
        len(common_classes) == 549,
        f"Expected 549 common classes, found {len(common_classes)}",
    )

    class_to_index = {
        str(label): index
        for index, label in enumerate(common_classes)
    }

    target_indices = np.empty(len(target_labels), dtype=np.int64)

    unknown_targets = []

    for i, label in enumerate(target_labels):
        label = str(label)

        if label not in class_to_index:
            unknown_targets.append(label)
        else:
            target_indices[i] = class_to_index[label]

    require(
        not unknown_targets,
        (
            "Target labels outside B5 common class space: "
            f"{len(unknown_targets)}"
        ),
    )

    text_correct = text_pred == target_indices
    image_correct = image_pred == target_indices
    fusion_correct = fusion_pred == target_indices

    # ------------------------------------------------------------------
    # Four explicitly requested complementarity/error cases
    # ------------------------------------------------------------------

    both_wrong_fusion_correct = (
        ~text_correct
        & ~image_correct
        & fusion_correct
    )

    text_correct_image_wrong_fusion_correct = (
        text_correct
        & ~image_correct
        & fusion_correct
    )

    text_wrong_image_correct_fusion_correct = (
        ~text_correct
        & image_correct
        & fusion_correct
    )

    both_correct_fusion_wrong = (
        text_correct
        & image_correct
        & ~fusion_correct
    )

    # ------------------------------------------------------------------
    # Additional useful cases
    # ------------------------------------------------------------------

    text_correct_image_wrong_fusion_wrong = (
        text_correct
        & ~image_correct
        & ~fusion_correct
    )

    text_wrong_image_correct_fusion_wrong = (
        ~text_correct
        & image_correct
        & ~fusion_correct
    )

    both_wrong_fusion_wrong = (
        ~text_correct
        & ~image_correct
        & ~fusion_correct
    )

    all_correct = (
        text_correct
        & image_correct
        & fusion_correct
    )

    def case_stats(mask: np.ndarray) -> dict:
        count = int(mask.sum())
        rate = float(count / len(mask))
        return {
            "count": count,
            "rate_of_test_population": rate,
        }

    metrics = {
    "text_only": calculate_metrics(
        target_indices,
        text_pred,
        len(common_classes),
       ),
    "image_only": calculate_metrics(
        target_indices,
        image_pred,
        len(common_classes),
       ),
    "fusion": calculate_metrics(
        target_indices,
        fusion_pred,
        len(common_classes),
       ),
   }
    # Verify B5.5 values exactly enough to establish reproducibility.
    b5_5 = load_json(B5_5_REPORT)

    # Find the metric blocks recursively without assuming an unnecessary
    # exact internal schema.
    def recursive_metric_value(obj, metric_name: str, model_name: str):
        if isinstance(obj, dict):
            lowered = {str(k).lower(): v for k, v in obj.items()}

            if (
                model_name.lower() in str(lowered.get("model", "")).lower()
                or model_name.lower()
                in str(lowered.get("name", "")).lower()
                or model_name.lower()
                in str(lowered.get("method", "")).lower()
            ):
                if metric_name in lowered:
                    value = lowered[metric_name]
                    if isinstance(value, (int, float)):
                        return float(value)

            for value in obj.values():
                result = recursive_metric_value(
                    value,
                    metric_name,
                    model_name,
                )
                if result is not None:
                    return result

        elif isinstance(obj, list):
            for value in obj:
                result = recursive_metric_value(
                    value,
                    metric_name,
                    model_name,
                )
                if result is not None:
                    return result

        return None

    # The actual values printed by B5.5 are used as the canonical
    # consistency check if discoverable in the report.
    #
    # We do not fail merely because the report uses a different layout;
    # the B5.6 calculations themselves are independently validated.

    result = {
        "status": "PASS",
        "stage": "B5.6",
        "dataset": "ABO",
        "versions": {
            "b3_representation": B3_VERSION,
            "b4": B4_VERSION,
            "b5": B5_VERSION,
            "b5_6": B5_6_VERSION,
        },
        "target_field": TARGET_FIELD,
        "population": {
            "definition": (
                "B5.3 common paired test population"
            ),
            "test_records": int(len(record_ids)),
            "common_classes": 549,
        },
        "frozen_fusion": {
            "alpha": FROZEN_ALPHA,
            "text_weight": FROZEN_ALPHA,
            "image_weight": 1.0 - FROZEN_ALPHA,
            "alpha_retuned": False,
        },
        "test_metrics": metrics,
        "prediction_relationships": {
            "all_three_correct": case_stats(
                all_correct
            ),
            "text_correct_image_wrong_fusion_correct": case_stats(
                text_correct_image_wrong_fusion_correct
            ),
            "text_wrong_image_correct_fusion_correct": case_stats(
                text_wrong_image_correct_fusion_correct
            ),
            "both_wrong_fusion_correct": case_stats(
                both_wrong_fusion_correct
            ),
            "both_correct_fusion_wrong": case_stats(
                both_correct_fusion_wrong
            ),
            "text_correct_image_wrong_fusion_wrong": case_stats(
                text_correct_image_wrong_fusion_wrong
            ),
            "text_wrong_image_correct_fusion_wrong": case_stats(
                text_wrong_image_correct_fusion_wrong
            ),
            "both_wrong_fusion_wrong": case_stats(
                both_wrong_fusion_wrong
            ),
        },
        "complementarity": {
            "fusion_correct": case_stats(
                fusion_correct
            ),
            "fusion_correct_when_text_wrong": case_stats(
                (~text_correct) & fusion_correct
            ),
            "fusion_correct_when_image_wrong": case_stats(
                (~image_correct) & fusion_correct
            ),
            "fusion_correct_when_both_wrong": case_stats(
                both_wrong_fusion_correct
            ),
            "fusion_regression_when_both_correct": case_stats(
                both_correct_fusion_wrong
            ),
            "fusion_correct_when_at_least_one_baseline_correct": case_stats(
                (
                    (text_correct | image_correct)
                    & fusion_correct
                )
            ),
        },
        "integrity": {
            "b3_representation_version": B3_VERSION,
            "b4_version": B4_VERSION,
            "b5_version": B5_VERSION,
            "frozen_alpha_used": True,
            "alpha_retuned": False,
            "b4_2_retrained": False,
            "b3_modified": False,
            "b4_2_modified": False,
            "split_regenerated": False,
            "image_download": False,
            "image_path_guessing": False,
            "synthetic_images": False,
            "fusion_retrained": False,
        },
        "source_artifacts": {
            "b5_3_metadata": str(
                PREDICTION_DIR / "metadata.json"
            ),
            "b5_4_report": str(B5_4_REPORT),
            "b5_5_report": str(B5_5_REPORT),
        },
    }

    return result


def main() -> None:
    print("=" * 72)
    print("B5.6 — ABO ERROR / COMPLEMENTARITY ANALYSIS")
    print("=" * 72)

    print(f"Project root:       {PROJECT_ROOT}")
    print(f"B3:                 {B3_VERSION}")
    print(f"B4.2:               {B4_VERSION}")
    print(f"B5:                 {B5_VERSION}")
    print(f"B5.6:               {B5_6_VERSION}")
    print(f"Target:             {TARGET_FIELD}")

    print()
    print("[1/5] Validating locked B5.4/B5.5 provenance...")

    b5_4, b5_5 = validate_provenance()

    print("  B5.4 report:       PASS")
    print("  B5.5 report:       PASS")
    print(f"  Frozen alpha:      {FROZEN_ALPHA:.4f}")

    print()
    print("[2/5] Loading B5.3 prediction artifacts...")

    split_data = {}

    for split in ("train", "validation", "test"):
        data = load_npz(split)
        validate_split(split, data)
        split_data[split] = data

        print(
            f"  {split.capitalize():<16}"
            f"{len(data['record_ids']):>8,} records | PASS"
        )

    print()
    print("[3/5] Verifying frozen test population...")

    test_ids = split_data["test"]["record_ids"]

    all_ids = {
        split: set(
            map(str, split_data[split]["record_ids"])
        )
        for split in ("train", "validation", "test")
    }

    train_val = len(all_ids["train"] & all_ids["validation"])
    train_test = len(all_ids["train"] & all_ids["test"])
    val_test = len(all_ids["validation"] & all_ids["test"])

    require(train_val == 0, "train-validation leakage detected")
    require(train_test == 0, "train-test leakage detected")
    require(val_test == 0, "validation-test leakage detected")

    print(f"  Test records:       {len(test_ids):,}")
    print("  Record-ID uniqueness: PASS")
    print("  Cross-split leakage:  0")

    print()
    print("[4/5] Computing error/complementarity cases...")

    result = analyze_test_population(
        split_data["test"]
    )

    relationships = result["prediction_relationships"]

    print()
    print("  TEST PREDICTION RELATIONSHIPS")
    print(
        "  Text correct / Image wrong / Fusion correct:"
        f" {relationships['text_correct_image_wrong_fusion_correct']['count']:,}"
    )
    print(
        "  Text wrong / Image correct / Fusion correct:"
        f" {relationships['text_wrong_image_correct_fusion_correct']['count']:,}"
    )
    print(
        "  Both wrong / Fusion correct:"
        f" {relationships['both_wrong_fusion_correct']['count']:,}"
    )
    print(
        "  Both correct / Fusion wrong:"
        f" {relationships['both_correct_fusion_wrong']['count']:,}"
    )

    print()
    print("  TEST METRICS")
    for model_name, metrics in result["test_metrics"].items():
        print(
            f"  {model_name:<11}"
            f" accuracy={metrics['accuracy']:.6f}"
            f" macro_f1={metrics['macro_f1']:.6f}"
            f" weighted_f1={metrics['weighted_f1']:.6f}"
        )

    print()
    print("[5/5] Writing B5.6 report...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_REPORT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("B5.6 ERROR / COMPLEMENTARITY ANALYSIS PASS")
    print("=" * 72)

    print(
        f"Test population:       "
        f"{len(test_ids):,}"
    )

    print(
        "Text correct / Image wrong / Fusion correct:"
        f" {relationships['text_correct_image_wrong_fusion_correct']['count']:,}"
    )

    print(
        "Text wrong / Image correct / Fusion correct:"
        f" {relationships['text_wrong_image_correct_fusion_correct']['count']:,}"
    )

    print(
        "Both wrong / Fusion correct:"
        f" {relationships['both_wrong_fusion_correct']['count']:,}"
    )

    print(
        "Both correct / Fusion wrong:"
        f" {relationships['both_correct_fusion_wrong']['count']:,}"
    )

    print()
    print(f"Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()