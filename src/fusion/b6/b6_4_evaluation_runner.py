from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORT_DIR = PROJECT_ROOT / "reports" / "fusion" / "b6"

EXP1_REPORT = (
    REPORT_DIR / "b6_3_3_exp1_gate2_test_v001.json"
)

EXP2_REPORT = (
    REPORT_DIR / "b6_3_3_exp2_gate2_test_v001.json"
)

B6_3_4_REPORT = (
    REPORT_DIR / "b6_3_4_evaluation_results_v001.json"
)

OUTPUT_REPORT = (
    REPORT_DIR / "b6_4_evaluation_results_v001.json"
)

EXPECTED_GAMMA = 1.0
EXPECTED_TARGET_CLASSES = 549

EXPECTED_FILTERED_TRAIN = 70258
EXPECTED_FILTERED_TEST = 7416

EXPECTED_TEST_KNOWN = 7386
EXPECTED_TEST_UNSEEN = 30

EXPECTED_EXCLUDED_CLASSES = [
    "HAIRBAND",
    "PUNCHING_BAG",
    "SALWAR_SUIT_SET",
    "TREADMILL",
]

EXPECTED_EXP1 = {
    "accuracy": 0.5682372055239643,
    "macro_f1": 0.058720810905631214,
    "weighted_f1": 0.50588931202641,
}

EXPECTED_EXP2 = {
    "accuracy": 0.5683725968047658,
    "macro_f1": 0.05877880520618687,
    "weighted_f1": 0.5061787255110273,
}

EXPECTED_TEXT_ONLY = {
    "accuracy": 0.5679664229623612,
    "macro_f1": 0.05875117923535614,
    "weighted_f1": 0.5056574753989312,
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required report missing: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return value


def require_equal(
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        raise ValueError(
            f"{name} mismatch: "
            f"expected={expected!r}, "
            f"actual={actual!r}"
        )


def require_close(
    name: str,
    actual: float,
    expected: float,
    tolerance: float = 1e-12,
) -> None:
    difference = abs(
        float(actual) - float(expected)
    )

    if difference > tolerance:
        raise ValueError(
            f"{name} mismatch: "
            f"expected={expected!r}, "
            f"actual={actual!r}, "
            f"difference={difference!r}"
        )


def extract_gate2_metrics(
    report: dict[str, Any],
) -> dict[str, float]:
    metrics = report.get("metrics")

    if not isinstance(metrics, dict):
        raise ValueError(
            "Gate 2 report does not contain metrices."

        )

    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
    }


def extract_text_only_metrics(
    report: dict[str, Any],
) -> dict[str, float]:
    ablation = report.get(
        "matched_text_only_ablation"
    )

    if not isinstance(ablation, dict):
        raise ValueError(
            "B6.3.4 report does not contain "
            "matched_text_only_ablation."
        )

    metrics = ablation.get("metrics")

    if not isinstance(metrics, dict):
        raise ValueError(
            "matched_text_only_ablation does not "
            "contain metrics."
        )

    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
    }


def calculate_delta(
    lhs: dict[str, float],
    rhs: dict[str, float],
) -> dict[str, float]:
    return {
        metric: float(
            lhs[metric] - rhs[metric]
        )
        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
        )
    }


def main() -> None:
    print("=" * 72)
    print("B6.4 — EVALUATION")
    print("=" * 72)
    print("READ_ONLY_EVALUATION=True")
    print("NO_MODEL_TRAINING=True")
    print("NO_FROZEN_ARTIFACT_MODIFICATION=True")

    exp1 = load_json(EXP1_REPORT)
    exp2 = load_json(EXP2_REPORT)
    b6_3_4 = load_json(B6_3_4_REPORT)

    print("INPUT_REPORTS=PASS")

    # ================================================================
    # Frozen B6.3 protocol validation
    # ================================================================

    for name, report in (
        ("EXP1_GATE2", exp1),
        ("EXP2_GATE2", exp2),
    ):
        target = report.get("target")

        if not isinstance(target, dict):
            raise ValueError(
                f"{name}: missing target metadata."
            )

        require_equal(
            f"{name}_TARGET_CLASSES",
            target.get("target_classes"),
            EXPECTED_TARGET_CLASSES,
        )

        require_equal(
            f"{name}_FILTERED_TRAIN",
            target.get("filtered_train_count"),
            EXPECTED_FILTERED_TRAIN,
        )

        require_equal(
            f"{name}_FILTERED_TEST",
            target.get("filtered_test_count"),
            EXPECTED_FILTERED_TEST,
        )

        require_equal(
            f"{name}_TEST_KNOWN",
            target.get("test_metric_count"),
            EXPECTED_TEST_KNOWN,
        )

        require_equal(
            f"{name}_TEST_UNSEEN",
            target.get(
                "test_unseen_target_records"
            ),
            EXPECTED_TEST_UNSEEN,
        )

        require_equal(
            f"{name}_EXCLUDED_CLASSES",
            target.get(
                "excluded_target_classes"
            ),
            EXPECTED_EXCLUDED_CLASSES,
        )

        gamma = report.get("gamma")

        if isinstance(gamma, dict):
             gamma = gamma.get("value")

        if gamma is None:
            raise ValueError(
                f"{name}: gamma value missing from report."
            )
        require_close(
            f"{name}_GAMMA",
            float(gamma),
            EXPECTED_GAMMA,
        )

    print("FROZEN_PROTOCOL=PASS")
    print("POPULATION_BOUNDARY=PASS")

    # ================================================================
    # Frozen metrics
    # ================================================================

    exp1_metrics = extract_gate2_metrics(exp1)
    exp2_metrics = extract_gate2_metrics(exp2)
    text_only_metrics = extract_text_only_metrics(
        b6_3_4
    )

    for metric in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ):
        require_close(
            f"EXP1_{metric.upper()}",
            exp1_metrics[metric],
            EXPECTED_EXP1[metric],
        )

        require_close(
            f"EXP2_{metric.upper()}",
            exp2_metrics[metric],
            EXPECTED_EXP2[metric],
        )

        require_close(
            f"TEXT_ONLY_{metric.upper()}",
            text_only_metrics[metric],
            EXPECTED_TEXT_ONLY[metric],
        )

    print("FROZEN_METRICS=PASS")

    # ================================================================
    # Required B6.4 comparisons
    # ================================================================

    exp2_minus_exp1 = calculate_delta(
        exp2_metrics,
        exp1_metrics,
    )

    exp2_minus_text_only = calculate_delta(
        exp2_metrics,
        text_only_metrics,
    )

    text_only_minus_exp1 = calculate_delta(
        text_only_metrics,
        exp1_metrics,
    )

    print(
        "EXP2_MINUS_EXP1="
        + json.dumps(exp2_minus_exp1)
    )

    print(
        "EXP2_MINUS_TEXT_ONLY="
        + json.dumps(exp2_minus_text_only)
    )

    print(
        "TEXT_ONLY_MINUS_EXP1="
        + json.dumps(text_only_minus_exp1)
    )

    # ================================================================
    # Interpretation
    # ================================================================

    exp2_better_than_exp1 = all(
        exp2_minus_exp1[metric] > 0
        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
        )
    )

    exp2_better_than_text_only = all(
        exp2_minus_text_only[metric] > 0
        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
        )
    )

    if (
        exp2_better_than_exp1
        and exp2_better_than_text_only
    ):
        interpretation = (
            "SMALL_WEAK_POSITIVE_TRANSFER"
        )
    elif (
        exp2_better_than_exp1
        or exp2_better_than_text_only
    ):
        interpretation = "MIXED_RESULT"
    else:
        interpretation = "NO_POSITIVE_TRANSFER"

    # ================================================================
    # Scientific boundary
    # ================================================================

    scientific_boundary = (
        "representation-level cross-dataset transfer"
    )

    prohibited_claims = {
        "strong_transfer": False,
        "substantial_improvement": False,
        "statistical_significance": False,
        "universal_classifier": False,
        "label_transfer": False,
        "multimodal_superiority": False,
        "uncertainty_implementation": False,
        "selective_prediction_implementation": False,
        "open_set_implementation": False,
        "human_review_routing_implementation": False,
    }

    # ================================================================
    # Report
    # ================================================================

    result = {
        "stage": "B6.4",
        "substage": "B6.4.2",
        "version": "v001",
        "status": "PASS",
        "evaluation_type": (
            "read_only_frozen_result_evaluation"
        ),
        "read_only": True,
        "model_training": False,
        "frozen_artifact_modification": False,
        "new_gamma_search": False,
        "test_tuning": False,
        "target": {
            "dataset": "ABO",
            "task": "product_type_classification",
            "target_classes": EXPECTED_TARGET_CLASSES,
            "filtered_train_count": (
                EXPECTED_FILTERED_TRAIN
            ),
            "filtered_test_count": (
                EXPECTED_FILTERED_TEST
            ),
            "test_metric_records": (
                EXPECTED_TEST_KNOWN
            ),
            "test_unseen_target_records": (
                EXPECTED_TEST_UNSEEN
            ),
            "excluded_target_classes": (
                EXPECTED_EXCLUDED_CLASSES
            ),
        },
        "frozen_inputs": {
            "exp1_gate2": str(EXP1_REPORT),
            "exp2_gate2": str(EXP2_REPORT),
            "b6_3_4": str(B6_3_4_REPORT),
        },
        "frozen_metrics": {
            "exp1_negative_control": exp1_metrics,
            "exp2_genuine_transfer": exp2_metrics,
            "matched_text_only": text_only_metrics,
        },
        "metric_deltas": {
            "exp2_minus_exp1": exp2_minus_exp1,
            "exp2_minus_text_only": (
                exp2_minus_text_only
            ),
            "text_only_minus_exp1": (
                text_only_minus_exp1
            ),
        },
        "interpretation": interpretation,
        "scientific_boundary": scientific_boundary,
        "statistical_significance_claim": False,
        "prohibited_claims": prohibited_claims,
    }

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_REPORT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "INTERPRETATION="
        + interpretation
    )
    print(
        "STATISTICAL_SIGNIFICANCE_CLAIM=False"
    )
    print("NEW_GAMMA_SEARCH=False")
    print("TEST_TUNING=False")
    print(
        "REPORT_SAVED="
        + str(OUTPUT_REPORT)
    )
    print("=" * 72)
    print("B6.4 EVALUATION=PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
