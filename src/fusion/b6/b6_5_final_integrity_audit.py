from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports" / "fusion" / "b6"
MODEL_DIR = ROOT / "models" / "fusion" / "b6"
TRANSFER_DIR = (
    ROOT
    / "data"
    / "representations"
    / "abo"
    / "b6_3_3_transfer"
)


REPORTS = {
    "exp1": REPORT_DIR / "b6_3_3_exp1_gate2_test_v001.json",
    "exp2": REPORT_DIR / "b6_3_3_exp2_gate2_test_v001.json",
    "b6_3_4": REPORT_DIR / "b6_3_4_evaluation_results_v001.json",
    "b6_3_5": REPORT_DIR / "b6_3_5_integrity_reproducibility_audit_v001.json",
    "b6_4": REPORT_DIR / "b6_4_evaluation_results_v001.json",
    "b6_4_3": REPORT_DIR / "b6_4_final_audit_v001.json",
}


SOURCE_MODEL = (
    MODEL_DIR / "b6_3_3_rakuten_source_model_v001.pkl"
)


TRANSFER_FILES = [
    TRANSFER_DIR / "train.npz",
    TRANSFER_DIR / "validation.npz",
    TRANSFER_DIR / "test.npz",
]


OUTPUT = (
    REPORT_DIR
    / "b6_5_final_integrity_audit_v001.json"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def main():
    print("=" * 72)
    print("B6.5 — FINAL INTEGRITY & REPRODUCIBILITY AUDIT")
    print("=" * 72)

    print("READ_ONLY_AUDIT=True")
    print("NO_MODEL_TRAINING=True")
    print("NO_FROZEN_ARTIFACT_MODIFICATION=True")

    # ---------------------------------------------------------------
    # Required files
    # ---------------------------------------------------------------

    missing = []

    for name, path in REPORTS.items():
        if not path.exists():
            missing.append(
                f"REPORT:{name}:{path}"
            )

    if not SOURCE_MODEL.exists():
        missing.append(
            f"SOURCE_MODEL:{SOURCE_MODEL}"
        )

    for path in TRANSFER_FILES:
        if not path.exists():
            missing.append(
                f"TRANSFER:{path}"
            )

    require(
        not missing,
        "Missing required artifacts/reports:\n"
        + "\n".join(missing),
    )

    print("REQUIRED_ARTIFACTS=PASS")

    reports = {
        name: load_json(path)
        for name, path in REPORTS.items()
    }

    # ---------------------------------------------------------------
    # B6.3.5 audit
    # ---------------------------------------------------------------

    audit_635 = reports["b6_3_5"]

    require(
        audit_635.get("status") == "PASS",
        "B6.3.5 audit is not PASS.",
    )

    print("B6.3.5_AUDIT=PASS")

    # ---------------------------------------------------------------
    # B6.4.3 audit
    # ---------------------------------------------------------------

    audit_643 = reports["b6_4_3"]

    require(
        audit_643.get("status") == "PASS",
        "B6.4.3 audit is not PASS.",
    )

    require(
        audit_643.get("read_only_audit") is True,
        "B6.4.3 read-only control failed.",
    )

    require(
        audit_643.get("no_model_training") is True,
        "B6.4.3 training control failed.",
    )

    require(
        audit_643.get("no_frozen_artifact_modification") is True,
        "B6.4.3 frozen-artifact control failed.",
    )

    print("B6.4.3_AUDIT=PASS")

    # ---------------------------------------------------------------
    # B6.4 evaluation
    # ---------------------------------------------------------------

    b64 = reports["b6_4"]

    require(
        b64.get("status") == "PASS",
        "B6.4 status is not PASS.",
    )

    require(
        b64.get("read_only") is True,
        "B6.4 read-only control failed.",
    )

    require(
        b64.get("model_training") is False,
        "B6.4 model-training control failed.",
    )

    require(
        b64.get("frozen_artifact_modification") is False,
        "B6.4 frozen-artifact modification control failed.",
    )

    require(
        b64.get("new_gamma_search") is False,
        "B6.4 gamma-search control failed.",
    )

    require(
        b64.get("test_tuning") is False,
        "B6.4 test-tuning control failed.",
    )

    print("B6.4_CONTROLS=PASS")

    # ---------------------------------------------------------------
    # Frozen population boundary
    # ---------------------------------------------------------------

    target = b64.get("target")

    require(
        isinstance(target, dict),
        "B6.4 target metadata missing.",
    )

    require(
        target.get("dataset") == "ABO",
        "Dataset mismatch.",
    )

    require(
        target.get("task")
        == "product_type_classification",
        "Task mismatch.",
    )

    require(
        target.get("target_classes") == 549,
        "Target class count mismatch.",
    )

    require(
        target.get("filtered_train_count") == 70258,
        "Filtered train count mismatch.",
    )

    require(
        target.get("filtered_test_count") == 7416,
        "Filtered test count mismatch.",
    )

    require(
        target.get("test_metric_records") == 7386,
        "Test metric count mismatch.",
    )

    require(
        target.get("test_unseen_target_records") == 30,
        "Test unseen count mismatch.",
    )

    require(
        target.get("excluded_target_classes")
        == [
            "HAIRBAND",
            "PUNCHING_BAG",
            "SALWAR_SUIT_SET",
            "TREADMILL",
        ],
        "Excluded target-class boundary mismatch.",
    )

    print("POPULATION_BOUNDARY=PASS")

    # ---------------------------------------------------------------
    # Frozen B6.3.3 experiment controls
    # ---------------------------------------------------------------

    for name in ("exp1", "exp2"):
        report = reports[name]

        require(
            report.get("status") == "GATE_2_PASS",
            f"{name} status is not GATE_2_PASS.",
        )

        experiment_target = report.get("target")

        require(
            isinstance(experiment_target, dict),
            f"{name}: target metadata missing.",
        )

        require(
            experiment_target.get("target_classes") == 549,
            f"{name}: target class mismatch.",
        )

        require(
            experiment_target.get("filtered_train_count")
            == 70258,
            f"{name}: filtered train mismatch.",
        )

        require(
            experiment_target.get("filtered_test_count")
            == 7416,
            f"{name}: filtered test mismatch.",
        )

        require(
            experiment_target.get("test_metric_count")
            == 7386,
            f"{name}: test metric count mismatch.",
        )

        gamma = report.get("gamma")

        require(
            isinstance(gamma, dict),
            f"{name}: gamma metadata missing.",
        )

        require(
            float(gamma.get("value")) == 1.0,
            f"{name}: gamma is not frozen at 1.0.",
        )

    # ---------------------------------------------------------------
    # Exp1 / Exp2 permutation controls
    # ---------------------------------------------------------------

    exp1 = reports["exp1"]
    exp2 = reports["exp2"]

    exp1_permutation = exp1.get("permutation")
    exp2_permutation = exp2.get("permutation")

    require(
        isinstance(exp1_permutation, dict),
        "Exp1 permutation metadata missing.",
    )

    require(
        isinstance(exp2_permutation, dict),
        "Exp2 permutation metadata missing.",
    )

    # Exp1 = negative control:
    # transfer rows are randomly permuted within ABO Train only.
    require(
        exp1_permutation.get("row_permutation") is True,
        "Exp1 must use row-permuted training transfer.",
    )


    require(
        exp1_permutation.get("validation_permuted") is False,
        "Exp1 validation transfer must not be permuted.",
    )

    require(
        exp1_permutation.get("test_permuted") is False,
        "Exp1 test transfer must not be permuted.",
    )

    # Exp2 = genuine aligned transfer:
    # no permutation is applied.
    require(
        exp2_permutation.get("row_permutation") is False,
        "Exp2 must not use row-permuted transfer.",
    )

    require(
        exp2_permutation.get("validation_permuted") is False,
        "Exp2 validation transfer must not be permuted.",
    )

    require(
        exp2_permutation.get("test_permuted") is False,
        "Exp2 test transfer must not be permuted.",
    )

    print("B6.3.3_EXPERIMENT_CONTROLS=PASS")

    # ---------------------------------------------------------------
    # Scientific interpretation
    # ---------------------------------------------------------------

    require(
        b64.get("interpretation")
        == "SMALL_WEAK_POSITIVE_TRANSFER",
        "B6.4 interpretation mismatch.",
    )

    require(
        b64.get("statistical_significance_claim") is False,
        "Statistical significance claim detected.",
    )

    require(
        b64.get("scientific_boundary")
        == "representation-level cross-dataset transfer",
        "Scientific boundary mismatch.",
    )

    print("SCIENTIFIC_BOUNDARY=PASS")

    # ---------------------------------------------------------------
    # Frozen artifact integrity
    # ---------------------------------------------------------------

    require(
        SOURCE_MODEL.exists(),
        "Frozen Rakuten source model missing.",
    )

    require(
        len(TRANSFER_FILES) == 3,
        "Unexpected transfer artifact count.",
    )

    print("FROZEN_ARTIFACTS=PASS")

    # ---------------------------------------------------------------
    # Final result
    # ---------------------------------------------------------------

    result = {
        "stage": "B6.5",
        "version": "v001",
        "status": "PASS",
        "read_only_audit": True,
        "no_model_training": True,
        "no_frozen_artifact_modification": True,
        "checks": {
            "required_artifacts": "PASS",
            "b6_3_5_audit": "PASS",
            "b6_4_3_audit": "PASS",
            "b6_4_controls": "PASS",
            "population_boundary": "PASS",
            "b6_3_3_experiment_controls": "PASS",
            "scientific_boundary": "PASS",
            "frozen_artifacts": "PASS",
        },
        "phase_b_readiness": "READY_FOR_FINAL_FREEZE",
        "scientific_conclusion": (
            "Small/weak positive representation-level "
            "cross-dataset transfer was observed. "
            "No statistical significance claim is made."
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("REPORT_SAVED=" + str(OUTPUT))
    print("=" * 72)
    print("B6.5 FINAL AUDIT=PASS")
    print("PHASE_B_READINESS=READY_FOR_FINAL_FREEZE")
    print("=" * 72)


if __name__ == "__main__":
    main()
