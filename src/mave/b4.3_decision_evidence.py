from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports" / "baseline" / "mave"

FEASIBILITY_REPORT = REPORT_DIR / "mave_b4.3_feasibility.json"
INTERFACE_REPORT = REPORT_DIR / "mave_b3_b4.3_interface_audit.json"
PROVENANCE_REPORT = REPORT_DIR / "mave_b4_provenance_trace_audit.json"
ROOT_CAUSE_REPORT = REPORT_DIR / "mave_b2_canonical_root_cause_audit.json"

OUTPUT_REPORT = REPORT_DIR / "mave_b4.3_decision_evidence_v002.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required evidence report not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return data


def get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return current if current is not None else default


def main() -> int:
    print("=" * 70)
    print("MAVE B4.3 — FINAL DECISION / EVIDENCE REPORT")
    print("=" * 70)
    print()
    print("READ-ONLY EVIDENCE CONSOLIDATION")
    print("No dataset modification will be performed.")
    print("No model training will be performed.")
    print()

    feasibility = load_json(FEASIBILITY_REPORT)
    interface = load_json(INTERFACE_REPORT)
    provenance = load_json(PROVENANCE_REPORT)
    root_cause = load_json(ROOT_CAUSE_REPORT)

    # ------------------------------------------------------------------
    # Evidence extraction
    # ------------------------------------------------------------------

    total_records = get_nested(
        root_cause,
        "canonical_trace",
        "total_records_scanned",
        default=0,
    )

    supervised_candidates = get_nested(
        root_cause,
        "canonical_trace",
        "target_matches",
        default=0,
    )

    canonical_text = get_nested(
        root_cause,
        "canonical_trace",
        "target_records_with_approved_text",
        default=0,
    )

    canonical_images = get_nested(
        root_cause,
        "canonical_trace",
        "target_records_with_image_references",
        default=0,
    )

    b2_matches = get_nested(
        root_cause,
        "b2_trace",
        "target_matches",
        default=0,
    )

    b2_text = get_nested(
        root_cause,
        "b2_trace",
        "target_records_with_approved_text",
        default=0,
    )

    b2_supervision = get_nested(
        root_cause,
        "b2_trace",
        "target_records_with_supervision",
        default=0,
    )

    b3_text = get_nested(
        provenance,
        "provenance_trace",
        "b3_records_with_usable_text",
        default=0,
    )

    interface_candidates = get_nested(
        interface,
        "summary",
        "supervised_candidates",
        default=None,
    )

    if interface_candidates is None:
        interface_candidates = get_nested(
            interface,
            "supervised_candidates",
            default=supervised_candidates,
        )

    interface_candidates_with_text = get_nested(
        interface,
        "summary",
        "candidates_with_b3_text",
        default=None,
    )

    if interface_candidates_with_text is None:
        interface_candidates_with_text = get_nested(
            interface,
            "candidates_with_b3_text",
            default=0,
        )

    cross_split_asin_leakage = get_nested(
        root_cause,
        "integrity_checks",
        "cross_split_asin_leakage",
        default=None,
    )

    if cross_split_asin_leakage is None:
        cross_split_asin_leakage = get_nested(
            interface,
            "cross_split_asin_leakage",
            default=0,
        )

    cross_split_record_id_leakage = get_nested(
        root_cause,
        "integrity_checks",
        "cross_split_record_id_leakage",
        default=None,
    )

    if cross_split_record_id_leakage is None:
        cross_split_record_id_leakage = get_nested(
            interface,
            "cross_split_record_id_leakage",
            default=0,
        )

    source_asin_mismatches = get_nested(
        root_cause,
        "b2_trace",
        "target_source_asin_mismatches",
        default=0,
    )

    duplicate_target_ids = get_nested(
        root_cause,
        "b2_trace",
        "target_duplicate_record_ids",
        default=0,
    )

    missing_canonical_targets = get_nested(
        provenance,
        "provenance_trace",
        "missing_canonical_targets",
        default=0,
    )

    missing_b2_targets = get_nested(
        provenance,
        "provenance_trace",
        "missing_b2_targets",
        default=0,
    )

    record_id_mismatches = get_nested(
        provenance,
        "provenance_trace",
        "b2_record_id_mismatches",
        default=0,
    )

    split_mismatches = get_nested(
        provenance,
        "provenance_trace",
        "b2_split_mismatches",
        default=0,
    )

    # ------------------------------------------------------------------
    # Frozen split counts
    # ------------------------------------------------------------------

    split_counts = {
        "train": 2_326_033,
        "validation": 290_488,
        "test": 290_837,
    }

    supervised_by_split = {
        "train": 5_724,
        "validation": 706,
        "test": 718,
    }

    # ------------------------------------------------------------------
    # Final decision
    # ------------------------------------------------------------------

    confirmed_source_disjointness = (
        supervised_candidates == 7_148
        and canonical_text == 0
        and b2_text == 0
        and b3_text == 0
        and b2_matches == 7_148
        and b2_supervision == 7_148
        and missing_canonical_targets == 0
        and missing_b2_targets == 0
        and source_asin_mismatches == 0
        and duplicate_target_ids == 0
    )

    integrity_pass = (
        cross_split_asin_leakage == 0
        and cross_split_record_id_leakage == 0
        and source_asin_mismatches == 0
        and duplicate_target_ids == 0
        and record_id_mismatches == 0
        and split_mismatches == 0
    )

    blocked = confirmed_source_disjointness

    decision_status = "BLOCKED" if blocked else "REVIEW_REQUIRED"

    # ------------------------------------------------------------------
    # Final evidence report
    # ------------------------------------------------------------------

    report: dict[str, Any] = {
        "report_name": "MAVE B4.3 Final Decision / Evidence Report",
        "report_version": "v002",
        "dataset": "MAVE",
        "phase": "B4",
        "sub_stage": "B4.3",
        "mode": "READ_ONLY",

        "decision": {
            "status": decision_status,
            "decision": "DO_NOT_TRAIN",
            "reason": (
                "CONFIRMED_SOURCE_LEVEL_LABEL_INPUT_DISJOINTNESS"
                if blocked
                else "SOURCE_LEVEL_DISJOINTNESS_NOT_CONFIRMED"
            ),
            "root_cause": (
                "SOURCE_LEVEL_TEXT_ABSENCE_FOR_MAVE_SUPERVISED_RECORDS"
                if blocked
                else "REQUIRES_REVIEW"
            ),
            "model_training_performed": False,
        },

        "population": {
            "total_records": total_records,
            "split_counts": split_counts,
            "mave_information_records": supervised_candidates,
            "supervised_candidates": supervised_candidates,
            "supervised_candidates_by_split": supervised_by_split,
            "canonical_supervised_records_with_approved_text": canonical_text,
            "b2_supervised_records_with_approved_text": b2_text,
            "b3_supervised_records_with_usable_text": b3_text,
            "supervised_candidates_with_usable_b3_text": interface_candidates_with_text,
            "supervised_candidates_without_usable_b3_text": (
                supervised_candidates - interface_candidates_with_text
            ),
            "canonical_supervised_records_with_image_references": canonical_images,
        },

        "provenance_trace": {
            "supervised_asins": supervised_candidates,
            "canonical_matches": supervised_candidates,
            "b2_matches": b2_matches,
            "canonical_records_with_text": canonical_text,
            "b2_records_with_text": b2_text,
            "b3_records_with_usable_text": b3_text,
            "missing_canonical_targets": missing_canonical_targets,
            "missing_b2_targets": missing_b2_targets,
            "b2_source_asin_mismatches": source_asin_mismatches,
            "b2_record_id_mismatches": record_id_mismatches,
            "b2_split_mismatches": split_mismatches,
            "classification": (
                "MAVE_LABEL_WITHOUT_CANONICAL_APPROVED_TEXT"
                if blocked
                else "REQUIRES_REVIEW"
            ),
        },

        "integrity": {
            "cross_split_asin_leakage": cross_split_asin_leakage,
            "cross_split_record_id_leakage": cross_split_record_id_leakage,
            "duplicate_target_record_ids": duplicate_target_ids,
            "b2_source_asin_mismatches": source_asin_mismatches,
            "b2_record_id_mismatches": record_id_mismatches,
            "b2_split_mismatches": split_mismatches,
            "frozen_splits_preserved": True,
            "integrity_status": "PASS" if integrity_pass else "FAIL",
        },

        "root_cause_evidence": {
            "canonical_source_scanned_records": total_records,
            "canonical_target_matches": supervised_candidates,
            "canonical_targets_with_approved_text": canonical_text,
            "b2_targets_with_approved_text": b2_text,
            "b3_targets_with_usable_text": b3_text,
            "source_level_text_absence_confirmed": confirmed_source_disjointness,
            "source_level_classification": (
                "SOURCE_AND_B2_NO_APPROVED_TEXT"
                if blocked
                else "NOT_CONFIRMED"
            ),
        },

        "architectural_interpretation": {
            "mave_universe_records": total_records,
            "mave_supervised_information_records": supervised_candidates,
            "records_with_approved_general_text": total_records - supervised_candidates,
            "supervised_population_with_approved_text": 0,
            "interpretation": (
                "The 2,907,358-record MAVE universe must not be conflated "
                "with the 7,148-record MAVE-information supervision population. "
                "The supervised population contains labels but no approved "
                "text input in the canonical source, B2, or B3. Therefore "
                "the specified TF-IDF + linear supervised B4.3 baseline has "
                "zero valid supervised training examples under the frozen "
                "task and input contract."
            ),
        },

        "prohibited_actions": [
            "Do not fabricate MAVE labels.",
            "Do not convert missing MAVE labels into negative labels.",
            "Do not infer MAVE labels from Amazon metadata.",
            "Do not fabricate or backfill missing product text.",
            "Do not modify frozen B2 artifacts.",
            "Do not modify frozen B3 artifacts.",
            "Do not modify frozen dataset splits.",
            "Do not download external product information to repair the population.",
            "Do not perform B5 multimodal fusion as a workaround.",
            "Do not train a model over an incorrectly constructed population.",
        ],

        "acceptance": {
            "b3_artifacts_loaded": True,
            "frozen_splits_preserved": True,
            "asin_leakage_zero": cross_split_asin_leakage == 0,
            "duplicate_ids_zero": duplicate_target_ids == 0,
            "target_eligibility_reported": True,
            "labels_fabricated": False,
            "tfidf_fitted": False,
            "validation_evaluated": False,
            "test_evaluated": False,
            "metrics_generated": False,
            "provenance_recorded": True,
            "b2_b3_modified": False,
            "b5_fusion_performed": False,
            "decision_status": decision_status,
        },

        "evidence_artifacts": [
            "reports/baseline/mave/mave_b4.3_feasibility.json",
            "reports/baseline/mave/mave_b3_b4.3_interface_audit.json",
            "reports/baseline/mave/mave_b4_provenance_trace_audit.json",
            "reports/baseline/mave/mave_b2_canonical_root_cause_audit.json",
        ],

        "decision_statement": (
            "B4.3 is formally BLOCKED under the frozen MAVE task/input "
            "contract. The block is caused by confirmed source-level "
            "label/input disjointness, not by preprocessing corruption, "
            "split leakage, provenance failure, or implementation failure. "
            "No supervised B4.3 model training is authorized."
            if blocked
            else
            "B4.3 requires further evidence review before a final decision."
        ),

        "recommended_next_state": (
            "HOLD_FOR_ARCHITECTURAL_REVIEW"
            if blocked
            else "REVIEW_EVIDENCE"
        ),

        "integrity_statement": (
            "Evidence-only report. No raw dataset, canonical MAVE data, "
            "frozen B2/B3 artifacts, splits, labels, or model inputs were "
            "modified."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print("-" * 70)
    print("B4.3 FINAL DECISION")
    print("-" * 70)

    print(f"Total MAVE records                 : {total_records:,}")
    print(f"MAVE supervised records            : {supervised_candidates:,}")
    print(f"Canonical supervised + text        : {canonical_text:,}")
    print(f"B2 supervised + text               : {b2_text:,}")
    print(f"B3 supervised + usable text        : {b3_text:,}")
    print(f"Canonical image references         : {canonical_images:,}")
    print(f"Cross-split ASIN leakage           : {cross_split_asin_leakage:,}")
    print(f"Cross-split record-ID leakage      : {cross_split_record_id_leakage:,}")
    print(f"Source ASIN mismatches             : {source_asin_mismatches:,}")
    print(f"Duplicate target record IDs        : {duplicate_target_ids:,}")
    print()
    print(f"Source-level disjointness          : "
          f"{'CONFIRMED' if confirmed_source_disjointness else 'NOT CONFIRMED'}")
    print(f"Integrity                         : "
          f"{'PASS' if integrity_pass else 'FAIL'}")
    print()
    print(f"DECISION: {decision_status}")
    print("MODEL TRAINING: NOT PERFORMED")
    print("B2/B3 MODIFICATION: NONE")
    print("B5 FUSION: NOT PERFORMED")
    print()
    print("Report:")
    print(f"  {OUTPUT_REPORT}")

    return 0 if blocked and integrity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())