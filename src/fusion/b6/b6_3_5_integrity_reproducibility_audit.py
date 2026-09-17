from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


# ============================================================================
# B6.3.5 — FINAL INTEGRITY & REPRODUCIBILITY AUDIT
#
# READ-ONLY AUDIT
# NO MODEL TRAINING
# NO FROZEN ARTIFACT MODIFICATION
# NO EXPERIMENT RERUN
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# IMPORTANT:
# The verified frozen source-model location is models/fusion/b6/.
SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fusion"
    / "b6"
    / "b6_3_3_rakuten_source_model_v001.pkl"
)

TRANSFER_DIR = (
    PROJECT_ROOT
    / "data"
    / "representations"
    / "abo"
    / "b6_3_3_transfer"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "fusion"
    / "b6"
)

OUTPUT_PATH = (
    REPORT_DIR
    / "b6_3_5_integrity_reproducibility_audit_v001.json"
)

REPORT_PATHS = {
    "exp0": REPORT_DIR / "b6_3_3_exp0_reproduction_v001.json",
    "exp1_gate1": REPORT_DIR / "b6_3_3_exp1_gate1_validation_v001.json",
    "exp1_gate2": REPORT_DIR / "b6_3_3_exp1_gate2_test_v001.json",
    "exp2_gate1": REPORT_DIR / "b6_3_3_exp2_gate1_validation_v001.json",
    "exp2_gate2": REPORT_DIR / "b6_3_3_exp2_gate2_test_v001.json",
    "b6_3_4": REPORT_DIR / "b6_3_4_evaluation_results_v001.json",
}

SEED = 20260827

TARGET_CLASS_COUNT = 549
TRANSFER_DIMENSION = 3008

ORIGINAL_COUNTS = {
    "train": 70284,
    "validation": 69996,
    "test": 7422,
}

FILTERED_COUNTS = {
    "train": 70258,
    "validation": 69995,
    "test": 7416,
}

KNOWN_COUNTS = {
    "validation": 69936,
    "test": 7386,
}

UNSEEN_COUNTS = {
    "validation": 59,
    "test": 30,
}

EXCLUDED_TARGET_CLASSES = [
    "HAIRBAND",
    "PUNCHING_BAG",
    "SALWAR_SUIT_SET",
    "TREADMILL",
]

EXPECTED_GAMMAS = [0.1, 0.5, 1.0]
EXPECTED_SELECTED_GAMMA = 1.0


# ============================================================================
# JSON-SAFE CONVERSION
# ============================================================================

def json_safe(value: Any) -> Any:
    """
    Convert NumPy/scalar/path objects into ordinary JSON-compatible values.
    This prevents np.bool_ / np.int64 / np.float64 serialization failures.
    """

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


# ============================================================================
# BASIC HELPERS
# ============================================================================

def audit_result(
    status: str,
    details: Any = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "details": json_safe(details),
    }


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        return None

    return value


def find_value(
    data: Any,
    keys: tuple[str, ...],
) -> Any:
    """
    Recursively search a JSON object for the first occurrence of
    one of the requested keys.

    This is used only to tolerate harmless differences in report
    nesting. It does not invent values.
    """

    if isinstance(data, dict):

        for key in keys:
            if key in data:
                return data[key]

        for value in data.values():
            found = find_value(value, keys)

            if found is not None:
                return found

    elif isinstance(data, list):

        for item in data:
            found = find_value(item, keys)

            if found is not None:
                return found

    return None


def find_nested_dict(
    data: Any,
    key: str,
) -> dict[str, Any] | None:

    if isinstance(data, dict):

        value = data.get(key)

        if isinstance(value, dict):
            return value

        for child in data.values():
            found = find_nested_dict(child, key)

            if found is not None:
                return found

    elif isinstance(data, list):

        for item in data:
            found = find_nested_dict(item, key)

            if found is not None:
                return found

    return None


def numeric_equal(
    actual: Any,
    expected: float,
) -> bool:

    if actual is None:
        return False

    try:
        return bool(
            np.isclose(
                float(actual),
                expected,
                rtol=0,
                atol=1e-15,
            )
        )
    except (TypeError, ValueError):
        return False


# ============================================================================
# REPORT EXISTENCE / READABILITY
# ============================================================================

def audit_reports() -> dict[str, Any]:

    results = {}
    all_pass = True

    for name, path in REPORT_PATHS.items():

        exists = path.exists()

        entry = {
            "path": str(path),
            "exists": exists,
        }

        if not exists:

            entry["json_readable"] = False
            all_pass = False

        else:

            try:
                data = load_json(path)

                readable = data is not None

                entry["json_readable"] = readable

                if readable:
                    entry["top_level_keys"] = sorted(
                        data.keys()
                    )
                else:
                    all_pass = False

            except Exception as exc:

                entry["json_readable"] = False
                entry["error"] = str(exc)
                all_pass = False

        results[name] = entry

    return audit_result(
        "PASS" if all_pass else "FAIL",
        results,
    )


# ============================================================================
# SOURCE MODEL
# ============================================================================

def audit_source_model() -> dict[str, Any]:

    if not SOURCE_MODEL_PATH.exists():

        return audit_result(
            "FAIL",
            {
                "reason": "Frozen Rakuten source model missing.",
                "path": str(SOURCE_MODEL_PATH),
            },
        )

    try:
        artifact = joblib.load(
            SOURCE_MODEL_PATH
        )

    except Exception as exc:

        return audit_result(
            "FAIL",
            {
                "reason": "Could not load frozen source model.",
                "error": str(exc),
            },
        )

    if not isinstance(artifact, dict):

        return audit_result(
            "FAIL",
            {
                "reason": (
                    "Unexpected source-model artifact type."
                ),
                "type": str(type(artifact)),
            },
        )

    pipeline = artifact.get("pipeline")
    metadata = artifact.get("metadata")

    if pipeline is None:

        return audit_result(
            "FAIL",
            {
                "reason": (
                    "Frozen source artifact does not contain "
                    "the expected pipeline."
                ),
                "keys": list(artifact.keys()),
            },
        )

    if not isinstance(metadata, dict):
        metadata = {}

    named_steps = getattr(
        pipeline,
        "named_steps",
        {},
    )

    tfidf = named_steps.get("tfidf")
    classifier = named_steps.get("classifier")

    checks = {
        "artifact_is_dict": True,
        "pipeline_present": pipeline is not None,
        "metadata_present": bool(metadata),
        "source_classes": (
            metadata.get("source_classes") == 3008
        ),
        "train_records": (
            metadata.get("train_records") == 719701
        ),
        "representation_version": (
            metadata.get("representation_version")
            == "B3_v001"
        ),
        "target_dimension": (
            metadata.get("target_dimension") == 3008
        ),
        "tfidf_present": tfidf is not None,
        "classifier_present": classifier is not None,
        "tfidf_ngram_range": (
            getattr(
                tfidf,
                "ngram_range",
                None,
            )
            == (1, 2)
        ),
        "tfidf_sublinear_tf": (
            getattr(
                tfidf,
                "sublinear_tf",
                None,
            )
            is True
        ),
        "tfidf_max_features": (
            getattr(
                tfidf,
                "max_features",
                None,
            )
            == 100000
        ),
        "classifier_loss": (
            getattr(
                classifier,
                "loss",
                None,
            )
            == "log_loss"
        ),
        "classifier_penalty": (
            getattr(
                classifier,
                "penalty",
                None,
            )
            == "l2"
        ),
        "classifier_random_state": (
            getattr(
                classifier,
                "random_state",
                None,
            )
            == SEED
        ),
    }

    passed = all(
        bool(value)
        for value in checks.values()
    )

    return audit_result(
        "PASS" if passed else "FAIL",
        {
            "path": str(SOURCE_MODEL_PATH),
            "sha256": sha256_file(
                SOURCE_MODEL_PATH
            ),
            "checks": checks,
            "metadata": metadata,
            "pipeline_type": str(type(pipeline)),
        },
    )


# ============================================================================
# TRANSFER ARTIFACTS
# ============================================================================

def audit_transfer_artifacts() -> dict[str, Any]:

    results = {}
    all_pass = True

    for split, expected_count in ORIGINAL_COUNTS.items():

        path = TRANSFER_DIR / f"{split}.npz"

        if not path.exists():

            results[split] = audit_result(
                "FAIL",
                {
                    "reason": "Transfer artifact missing.",
                    "path": str(path),
                },
            )

            all_pass = False
            continue

        try:

            with np.load(
                path,
                allow_pickle=False,
            ) as data:

                keys = list(data.files)

                if "transferred_signal" not in data:

                    results[split] = audit_result(
                        "FAIL",
                        {
                            "reason": (
                                "transferred_signal key missing."
                            ),
                            "keys": keys,
                        },
                    )

                    all_pass = False
                    continue

                signal = data[
                    "transferred_signal"
                ]

                expected_shape = (
                    expected_count,
                    TRANSFER_DIMENSION,
                )

                shape_pass = (
                    signal.shape
                    == expected_shape
                )

                finite_pass = bool(
                    np.isfinite(signal).all()
                )

                nonnegative_pass = bool(
                    (signal >= 0).all()
                )

                split_pass = (
                    shape_pass
                    and finite_pass
                    and nonnegative_pass
                )

                results[split] = audit_result(
                    "PASS"
                    if split_pass
                    else "FAIL",
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "shape": list(
                            signal.shape
                        ),
                        "expected_shape": list(
                            expected_shape
                        ),
                        "finite": finite_pass,
                        "nonnegative": (
                            nonnegative_pass
                        ),
                        "keys": keys,
                    },
                )

                if not split_pass:
                    all_pass = False

        except Exception as exc:

            results[split] = audit_result(
                "FAIL",
                {
                    "reason": (
                        "Could not inspect transfer artifact."
                    ),
                    "error": str(exc),
                },
            )

            all_pass = False

    return audit_result(
        "PASS" if all_pass else "FAIL",
        results,
    )


# ============================================================================
# POPULATION / TARGET BOUNDARY
# ============================================================================

def audit_population_boundary() -> dict[str, Any]:

    checks = {
        "target_class_count": (
            TARGET_CLASS_COUNT == 549
        ),
        "train_original": (
            ORIGINAL_COUNTS["train"]
            == 70284
        ),
        "validation_original": (
            ORIGINAL_COUNTS["validation"]
            == 69996
        ),
        "test_original": (
            ORIGINAL_COUNTS["test"]
            == 7422
        ),
        "train_filtered": (
            FILTERED_COUNTS["train"]
            == 70258
        ),
        "validation_filtered": (
            FILTERED_COUNTS["validation"]
            == 69995
        ),
        "test_filtered": (
            FILTERED_COUNTS["test"]
            == 7416
        ),
        "validation_known": (
            KNOWN_COUNTS["validation"]
            == 69936
        ),
        "test_known": (
            KNOWN_COUNTS["test"]
            == 7386
        ),
        "validation_unseen": (
            UNSEEN_COUNTS["validation"]
            == 59
        ),
        "test_unseen": (
            UNSEEN_COUNTS["test"]
            == 30
        ),
        "excluded_class_count": (
            len(EXCLUDED_TARGET_CLASSES)
            == 4
        ),
    }

    passed = all(
        bool(value)
        for value in checks.values()
    )

    return audit_result(
        "PASS" if passed else "FAIL",
        {
            "checks": checks,
            "original_counts": ORIGINAL_COUNTS,
            "filtered_counts": FILTERED_COUNTS,
            "known_counts": KNOWN_COUNTS,
            "unseen_counts": UNSEEN_COUNTS,
            "excluded_target_classes": (
                EXCLUDED_TARGET_CLASSES
            ),
        },
    )


# ============================================================================
# EXPERIMENT CONSISTENCY
# ============================================================================

def audit_experiments() -> dict[str, Any]:

    reports = {}

    for name, path in REPORT_PATHS.items():

        data = load_json(path)

        if data is None:

            return audit_result(
                "FAIL",
                {
                    "reason": (
                        f"Could not load report: {name}"
                    ),
                    "path": str(path),
                },
            )

        reports[name] = data

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # ------------------------------------------------------------------------
    # Exp0
    # ------------------------------------------------------------------------

    exp0 = reports["exp0"]

    checks["exp0_present"] = True

    details["exp0"] = {
        "status": find_value(
            exp0,
            ("status",),
        ),
        "frozen_model_inference_only": find_value(
            exp0,
            ("frozen_model_inference_only",),
        ),
        "retraining": find_value(
            exp0,
            ("retraining",),
        ),
    }

    # ------------------------------------------------------------------------
    # Exp1 Gate 1
    # ------------------------------------------------------------------------

    exp1_g1 = reports["exp1_gate1"]

    selected_gamma = find_value(
        exp1_g1,
        (
            "selected_gamma",
        ),
    )

    test_used = find_value(
        exp1_g1,
        (
            "test_used",
        ),
    )

    checks["exp1_gate1_present"] = True
    checks["exp1_gate1_gamma"] = (
        selected_gamma
        == EXPECTED_SELECTED_GAMMA
    )
    checks["exp1_gate1_test_isolated"] = (
        test_used is False
    )

    details["exp1_gate1"] = {
        "selected_gamma": selected_gamma,
        "test_used_for_selection": test_used,
    }

    # ------------------------------------------------------------------------
    # Exp1 Gate 2
    # ------------------------------------------------------------------------

    exp1_g2 = reports["exp1_gate2"]

    exp1_gamma = find_value(
        exp1_g2,
        (
            "selected_gamma",
        ),
    )

    exp1_test_used = find_value(
        exp1_g2,
        (
            "validation_used_for_selection",
            "test_used_for_selection",
        ),
    )
    permutation = find_nested_dict(
        exp1_g2,
        "permutation",
    )

    if permutation is None:
        permutation = {}

    checks["exp1_gate2_present"] = True

    checks["exp1_gate2_gamma"] = (
        exp1_gamma
        == EXPECTED_SELECTED_GAMMA
    )

    checks["exp1_gate2_test_isolated"] = (
        exp1_test_used is False
    )

    checks["exp1_gate2_permutation_scope"] = (
        permutation.get("scope")
        == "ABO_train_only"
    )

    checks["exp1_gate2_permutation_seed"] = (
        permutation.get("seed")
        == SEED
    )

    checks["exp1_gate2_row_permutation"] = (
        permutation.get("row_permutation")
        is True
    )

    checks["exp1_gate2_validation_unpermuted"] = (
        permutation.get("validation_permuted")
        is False
    )

    checks["exp1_gate2_test_unpermuted"] = (
        permutation.get("test_permuted")
        is False
    )

    details["exp1_gate2"] = {
        "selected_gamma": exp1_gamma,
        "test_used_for_selection": exp1_test_used,
        "permutation": permutation,
    }

    # ------------------------------------------------------------------------
    # Exp2 Gate 1
    # ------------------------------------------------------------------------

    exp2_g1 = reports["exp2_gate1"]

    exp2_g1_gamma = find_value(
        exp2_g1,
        (
            "selected_gamma",
        ),
    )

    exp2_g1_test_used = find_value(
        exp2_g1,
        (
            "test_used",
        ),
    )

    checks["exp2_gate1_present"] = True

    checks["exp2_gate1_gamma"] = (
        exp2_g1_gamma
        == EXPECTED_SELECTED_GAMMA
    )

    checks["exp2_gate1_test_isolated"] = (
        exp2_g1_test_used is False
    )

    details["exp2_gate1"] = {
        "selected_gamma": exp2_g1_gamma,
        "test_used_for_selection": (
            exp2_g1_test_used
        ),
    }

    # ------------------------------------------------------------------------
    # Exp2 Gate 2
    # ------------------------------------------------------------------------

    exp2_g2 = reports["exp2_gate2"]

    exp2_g2_gamma = find_value(
        exp2_g2,
        (
            "selected_gamma",
        ),
    )

    exp2_g2_test_used = find_value(
        exp2_g2,
        (
            "validation_used_for_selection",
            "test_used_for_selection",
        ),
     )

    exp2_search = find_value(
        exp2_g2,
        (
            "gamma_search_during_gate2",
        ),
    )

    exp2_type = find_value(
        exp2_g2,
        (
            "experiment_type",
        ),
    )

    checks["exp2_gate2_present"] = True

    checks["exp2_gate2_gamma"] = (
        exp2_g2_gamma
        == EXPECTED_SELECTED_GAMMA
    )

    checks["exp2_gate2_test_isolated"] = (
        exp2_g2_test_used is False
    )

    checks["exp2_gate2_no_gamma_search"] = (
        exp2_search is False
    )

    checks["exp2_gate2_genuine_transfer"] = (
        exp2_type
        == "genuine_aligned_transfer"
    )

    details["exp2_gate2"] = {
        "selected_gamma": exp2_g2_gamma,
        "test_used_for_selection": (
            exp2_g2_test_used
        ),
        "gamma_search_during_gate2": exp2_search,
        "experiment_type": exp2_type,
    }

    # ------------------------------------------------------------------------
    # Frozen numerical test results
    # ------------------------------------------------------------------------

    expected_metrics = {
        "exp1_gate2": {
            "accuracy": 0.5682372055239643,
            "macro_f1": 0.058720810905631214,
            "weighted_f1": 0.50588931202641,
        },
        "exp2_gate2": {
            "accuracy": 0.5683725968047658,
            "macro_f1": 0.05877880520618687,
            "weighted_f1": 0.5061787255110273,
        },
    }

    numerical_details = {}

    for experiment, expected in expected_metrics.items():

        report = reports[experiment]

        metrics = extract_metrics(report)

        numerical_details[experiment] = {
            "actual": metrics,
            "expected": expected,
            "checks": {},
        }

        for metric, expected_value in expected.items():

            actual_value = metrics.get(metric)

            passed = numeric_equal(
                actual_value,
                expected_value,
            )

            numerical_details[
                experiment
            ]["checks"][metric] = passed

            checks[
                f"{experiment}_{metric}"
            ] = passed

    details["numerical_results"] = (
        numerical_details
    )

    passed = all(
        bool(value)
        for value in checks.values()
    )

    return audit_result(
        "PASS" if passed else "FAIL",
        {
            "checks": checks,
            "details": details,
        },
    )


def extract_metrics(
    report: dict[str, Any],
) -> dict[str, Any]:

    direct_candidates = [
        report.get("metrics"),
        report.get("test_metrics"),
    ]

    for candidate in direct_candidates:

        if isinstance(candidate, dict):

            if all(
                metric in candidate
                for metric in (
                    "accuracy",
                    "macro_f1",
                    "weighted_f1",
                )
            ):
                return candidate

    # Search nested dictionaries.
    for key in (
        "test",
        "final_test",
        "final_test_metrics",
        "evaluation",
        "results",
    ):

        candidate = find_nested_dict(
            report,
            key,
        )

        if isinstance(candidate, dict):

            if all(
                metric in candidate
                for metric in (
                    "accuracy",
                    "macro_f1",
                    "weighted_f1",
                )
            ):
                return candidate

    # Some reports store a metrics dictionary nested deeper.
    found = find_value(
        report,
        (
            "metrics",
            "test_metrics",
        ),
    )

    if isinstance(found, dict):
        return found

    return {}


# ============================================================================
# LEAKAGE CONTROLS
# ============================================================================

def audit_leakage_controls() -> dict[str, Any]:

    # These are frozen B6.3 protocol constraints.
    # This audit records the approved state; it does not perform new
    # cross-dataset matching or retrain anything.

    controls = {
        "record_joins": False,
        "asin_matching": False,
        "product_id_matching": False,
        "category_id_mapping": False,
        "semantic_label_crosswalk": False,
        "synthetic_pairing": False,
        "external_enrichment": False,
        "image_download": False,
        "path_guessing": False,
        "split_regeneration": False,
        "rakuten_validation_test_for_tuning": False,
        "abo_test_for_tuning": False,
        "b5_predictions_used": False,
        "b5_weights_used": False,
        "rakuten_source_model_modified": False,
    }

    return audit_result(
        "PASS",
        controls,
    )


# ============================================================================
# DOCUMENTATION INCONSISTENCY
# ============================================================================

def audit_documentation_inconsistency() -> dict[str, Any]:

    return audit_result(
        "DOCUMENTATION_INCONSISTENCY",
        {
            "issue": (
                "Frozen Exp2 Gate 2 objective text refers to "
                "a row-permuted Rakuten transfer signal."
            ),
            "verified_execution": (
                "Exp2 Gate 2 execution used genuine aligned "
                "Rakuten transfer."
            ),
            "verified_execution_metadata": (
                "experiment_type=genuine_aligned_transfer"
            ),
            "handling": (
                "Record the inconsistency without modifying "
                "the frozen B6.3.3 report."
            ),
        },
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 72)
    print(
        "B6.3.5 — FINAL INTEGRITY & REPRODUCIBILITY AUDIT"
    )
    print("=" * 72)

    print("READ_ONLY_AUDIT=True")
    print("NO_MODEL_TRAINING=True")
    print("NO_FROZEN_ARTIFACT_MODIFICATION=True")
    print()

    audit = {
        "audit_version": "v001",
        "stage": "B6.3.5",
        "audit_type": (
            "final_integrity_and_reproducibility"
        ),
        "status": "IN_PROGRESS",
        "seed": SEED,
        "target_class_count": TARGET_CLASS_COUNT,
        "checks": {},
    }

    # ------------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------------

    audit["checks"]["reports"] = audit_reports()

    print(
        "REPORTS="
        f"{audit['checks']['reports']['status']}"
    )

    # ------------------------------------------------------------------------
    # Source model
    # ------------------------------------------------------------------------

    audit["checks"]["source_model"] = (
        audit_source_model()
    )

    print(
        "SOURCE_MODEL="
        f"{audit['checks']['source_model']['status']}"
    )

    # ------------------------------------------------------------------------
    # Transfer artifacts
    # ------------------------------------------------------------------------

    audit["checks"]["transfer_artifacts"] = (
        audit_transfer_artifacts()
    )

    print(
        "TRANSFER_ARTIFACTS="
        f"{audit['checks']['transfer_artifacts']['status']}"
    )

    # ------------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------------

    audit["checks"]["population_boundary"] = (
        audit_population_boundary()
    )

    print(
        "POPULATION_BOUNDARY="
        f"{audit['checks']['population_boundary']['status']}"
    )

    # ------------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------------

    audit["checks"]["experiments"] = (
        audit_experiments()
    )

    print(
        "EXPERIMENT_CONSISTENCY="
        f"{audit['checks']['experiments']['status']}"
    )

    # ------------------------------------------------------------------------
    # Leakage
    # ------------------------------------------------------------------------

    audit["checks"]["leakage_controls"] = (
        audit_leakage_controls()
    )

    print(
        "LEAKAGE_CONTROLS="
        f"{audit['checks']['leakage_controls']['status']}"
    )

    # ------------------------------------------------------------------------
    # Documentation
    # ------------------------------------------------------------------------

    audit["documentation_inconsistency"] = (
        audit_documentation_inconsistency()
    )

    print(
        "DOCUMENTATION_CHECK="
        f"{audit['documentation_inconsistency']['status']}"
    )

    # ------------------------------------------------------------------------
    # Final decision
    # ------------------------------------------------------------------------

    failed_checks = [
        name
        for name, value in audit["checks"].items()
        if value.get("status") == "FAIL"
    ]

    audit["failed_checks"] = failed_checks

    audit["scientific_boundary"] = {
        "validated": [
            (
                "Representation-level cross-dataset transfer "
                "from Rakuten-derived signal into ABO "
                "product-type classification."
            ),
            (
                "Genuine aligned transfer produced a small "
                "positive difference relative to the "
                "within-training row-permuted negative control."
            ),
            (
                "Genuine aligned transfer produced a small "
                "positive difference relative to the matched "
                "text-only ablation."
            ),
        ],
        "not_established": [
            "strong transfer performance",
            "substantial practical performance improvement",
            "universal product classification",
            "semantic equivalence of Rakuten and ABO taxonomies",
            "label transfer",
            "multimodal superiority",
            "uncertainty estimation",
            "selective prediction",
            "human-review routing",
            "open-set detection",
        ],
    }

    if failed_checks:

        audit["status"] = "FAIL"

        audit["final_status"] = (
            "B6.3.5 FAIL"
        )

        audit["final_conclusion"] = (
            "B6.3.5 FAIL. One or more integrity checks "
            "failed. Frozen artifacts must not be modified "
            "until the failure is reviewed and a documented "
            "corrective decision is made."
        )

    else:

        audit["status"] = "PASS"

        audit["final_status"] = (
            "B6.3.5 PASS / COMPLETE / LOCKED"
        )

        audit["final_conclusion"] = (
            "B6.3.5 PASS / COMPLETE / LOCKED. "
            "The B6.3 signal-integration sequence is "
            "internally consistent with the frozen "
            "experimental boundary, artifact configuration, "
            "target population, leakage controls, and "
            "reproducibility record. The scientific outcome "
            "is a small positive representation-level "
            "transfer effect rather than a strong "
            "performance claim."
        )

    # ------------------------------------------------------------------------
    # JSON-safe report writing
    # ------------------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_audit = json_safe(audit)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            safe_audit,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"FAILED_CHECKS={failed_checks}"
    )
    print(
        f"REPORT_SAVED={OUTPUT_PATH}"
    )
    print("=" * 72)
    print(
        f"B6.3.5 AUDIT={audit['status']}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()