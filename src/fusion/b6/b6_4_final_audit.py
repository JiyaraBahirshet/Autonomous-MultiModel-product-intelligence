from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports" / "fusion" / "b6"

INPUT = REPORT_DIR / "b6_4_evaluation_results_v001.json"
OUTPUT = REPORT_DIR / "b6_4_final_audit_v001.json"


def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def eq(name, actual, expected):
    if actual != expected:
        raise ValueError(
            f"{name}: expected={expected!r}, actual={actual!r}"
        )


def close(name, actual, expected, tol=1e-12):
    if abs(float(actual) - float(expected)) > tol:
        raise ValueError(
            f"{name}: expected={expected!r}, actual={actual!r}"
        )


def main():
    print("=" * 72)
    print("B6.4.3 — FINAL EVALUATION AUDIT")
    print("=" * 72)
    print("READ_ONLY_AUDIT=True")
    print("NO_MODEL_TRAINING=True")
    print("NO_FROZEN_ARTIFACT_MODIFICATION=True")

    report = load(INPUT)

    eq("stage", report["stage"], "B6.4")
    eq("substage", report["substage"], "B6.4.2")
    eq("version", report["version"], "v001")
    eq("status", report["status"], "PASS")

    eq("read_only", report["read_only"], True)
    eq("model_training", report["model_training"], False)
    eq(
        "frozen_artifact_modification",
        report["frozen_artifact_modification"],
        False,
    )
    eq("new_gamma_search", report["new_gamma_search"], False)
    eq("test_tuning", report["test_tuning"], False)

    target = report["target"]

    eq("target_dataset", target["dataset"], "ABO")
    eq(
        "target_task",
        target["task"],
        "product_type_classification",
    )
    eq("target_classes", target["target_classes"], 549)
    eq("filtered_train_count", target["filtered_train_count"], 70258)
    eq("filtered_test_count", target["filtered_test_count"], 7416)
    eq("test_metric_records", target["test_metric_records"], 7386)
    eq(
        "test_unseen_target_records",
        target["test_unseen_target_records"],
        30,
    )

    eq(
        "excluded_target_classes",
        target["excluded_target_classes"],
        [
            "HAIRBAND",
            "PUNCHING_BAG",
            "SALWAR_SUIT_SET",
            "TREADMILL",
        ],
    )

    expected = {
        "exp1_negative_control": {
            "accuracy": 0.5682372055239643,
            "macro_f1": 0.058720810905631214,
            "weighted_f1": 0.50588931202641,
        },
        "exp2_genuine_transfer": {
            "accuracy": 0.5683725968047658,
            "macro_f1": 0.05877880520618687,
            "weighted_f1": 0.5061787255110273,
        },
        "matched_text_only": {
            "accuracy": 0.5679664229623612,
            "macro_f1": 0.05875117923535614,
            "weighted_f1": 0.5056574753989312,
        },
    }

    for group, metrics in expected.items():
        actual = report["frozen_metrics"][group]
        for metric, value in metrics.items():
            close(
                f"{group}.{metric}",
                actual[metric],
                value,
            )

    print("METRICS=PASS")
    print("POPULATION=PASS")
    print("PROTOCOL_CONTROLS=PASS")

    d = report["metric_deltas"]

    close(
        "exp2_minus_exp1.accuracy",
        d["exp2_minus_exp1"]["accuracy"],
        0.0001353912808015334,
    )
    close(
        "exp2_minus_exp1.macro_f1",
        d["exp2_minus_exp1"]["macro_f1"],
        0.000057994300555656465,
    )
    close(
        "exp2_minus_exp1.weighted_f1",
        d["exp2_minus_exp1"]["weighted_f1"],
        0.00028941348461730687,
    )

    close(
        "exp2_minus_text_only.accuracy",
        d["exp2_minus_text_only"]["accuracy"],
        0.0004061738424046002,
    )
    close(
        "exp2_minus_text_only.macro_f1",
        d["exp2_minus_text_only"]["macro_f1"],
        0.000027625970830731655,
    )
    close(
        "exp2_minus_text_only.weighted_f1",
        d["exp2_minus_text_only"]["weighted_f1"],
        0.0005212501120960278,
    )

    print("DELTA_CONSISTENCY=PASS")

    eq(
        "interpretation",
        report["interpretation"],
        "SMALL_WEAK_POSITIVE_TRANSFER",
    )

    eq(
        "scientific_boundary",
        report["scientific_boundary"],
        "representation-level cross-dataset transfer",
    )

    eq(
        "statistical_significance_claim",
        report["statistical_significance_claim"],
        False,
    )

    print("SCIENTIFIC_INTERPRETATION=PASS")
    print("SIGNIFICANCE_CLAIM_CONTROL=PASS")

    result = {
        "stage": "B6.4",
        "substage": "B6.4.3",
        "version": "v001",
        "status": "PASS",
        "read_only_audit": True,
        "no_model_training": True,
        "no_frozen_artifact_modification": True,
        "checks": {
            "report_schema": "PASS",
            "protocol_controls": "PASS",
            "population": "PASS",
            "metrics": "PASS",
            "metric_deltas": "PASS",
            "scientific_interpretation": "PASS",
            "statistical_significance_control": "PASS",
        },
        "conclusion": (
            "B6.4 evaluation is internally consistent with "
            "the frozen B6.3/B6.3.4 evidence. Genuine "
            "Rakuten-derived representation-level transfer "
            "shows a small/weak positive result without "
            "statistical significance claims."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("AUDIT_REPORT_SAVED=" + str(OUTPUT))
    print("=" * 72)
    print("B6.4.3 FINAL AUDIT=PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()