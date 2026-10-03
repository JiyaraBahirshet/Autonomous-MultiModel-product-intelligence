from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


# ============================================================
# D2 ACCEPTANCE / INTEGRITY AUDIT
# ============================================================
# Read-only acceptance audit for the executed D2 experiment.
#
# This script:
#   - does NOT fit calibration
#   - does NOT select thresholds
#   - does NOT access D-Test
#   - does NOT modify Phase-C artifacts
#   - does NOT modify D2 experiment artifacts
#
# It validates the actual executed D2 artifact schemas.
# ============================================================


SEED = 20260827

ROOT = Path("data") / "models" / "abo"

PHASE_C_ROOT = ROOT / "phase_c"
PHASE_D_ROOT = ROOT / "phase_d"

CBASE_CHECKPOINT = (
    PHASE_C_ROOT / "c3" / "training" / "cbase_best.pt"
)

EXPECTED_CKPT_SHA256 = (
    "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"
)

EXPECTED_VALIDATION_COUNT = 69_867
EXPECTED_DCAL_COUNT = 34_909
EXPECTED_DSELECT_COUNT = 34_958
EXPECTED_CLASS_COUNT = 549

EXPECTED_TEMPERATURE = 1.9641200304031372
TEMPERATURE_TOLERANCE = 1e-10


# ============================================================
# Source artifacts
# ============================================================

VALIDATION_IDS = (
    PHASE_C_ROOT
    / "c3"
    / "embeddings"
    / "validation_record_ids.json"
)

VALIDATION_LABELS = (
    PHASE_C_ROOT
    / "c3"
    / "embeddings"
    / "validation_labels.npy"
)

VALIDATION_TEXT = (
    PHASE_C_ROOT
    / "c3"
    / "embeddings"
    / "validation_text.npy"
)

VALIDATION_IMAGE = (
    PHASE_C_ROOT
    / "c3"
    / "embeddings"
    / "validation_image.npy"
)


# ============================================================
# D2 partition artifacts
# ============================================================

DCAL_MANIFEST = (
    PHASE_D_ROOT
    / "d2"
    / "partition"
    / "dcal_manifest.json"
)

DSELECT_MANIFEST = (
    PHASE_D_ROOT
    / "d2"
    / "partition"
    / "dselect_manifest.json"
)

PARTITION_METADATA = (
    PHASE_D_ROOT
    / "d2"
    / "partition"
    / "d2_partition_metadata.json"
)


# ============================================================
# D2 inference artifacts
# ============================================================

FRESH_INFERENCE = (
    PHASE_D_ROOT
    / "d2"
    / "inference"
    / "d2_fresh_cbase_validation_outputs.npz"
)

FRESH_INFERENCE_MANIFEST = (
    PHASE_D_ROOT
    / "d2"
    / "inference"
    / "d2_fresh_inference_manifest.json"
)


# ============================================================
# D2 calibration artifacts
# ============================================================

CALIBRATED_OUTPUTS = (
    PHASE_D_ROOT
    / "d2"
    / "calibration"
    / "d2_calibrated_validation_outputs.npz"
)

CALIBRATION_METRICS = (
    PHASE_D_ROOT
    / "d2"
    / "calibration"
    / "d2_calibration_metrics.json"
)

RELIABILITY_DATA = (
    PHASE_D_ROOT
    / "d2"
    / "calibration"
    / "d2_reliability_data.json"
)

CALIBRATION_MANIFEST = (
    PHASE_D_ROOT
    / "d2"
    / "calibration"
    / "d2_calibration_manifest.json"
)


# ============================================================
# Audit state
# ============================================================

PASSES: list[str] = []
WARNINGS: list[str] = []
FAILURES: list[str] = []


def passed(message: str) -> None:
    PASSES.append(message)
    print(f"  PASS: {message}")


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"  WARN: {message}")


def fail(message: str) -> None:
    FAILURES.append(message)
    print(f"  FAIL: {message}")


def require_file(path: Path, label: str) -> None:
    if path.exists() and path.is_file():
        passed(f"{label} exists")
    else:
        fail(f"{label} missing: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path, label: str):
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)

        passed(f"{label} JSON readable")
        return value

    except Exception as exc:
        fail(f"{label} JSON read failed: {exc}")
        return None


def load_json_list(path: Path, label: str):
    value = load_json(path, label)

    if isinstance(value, list):
        return value

    fail(f"{label} is not a JSON list")
    return []


def assert_array_shape(
    array,
    expected_shape,
    label: str,
) -> None:
    if tuple(array.shape) == tuple(expected_shape):
        passed(
            f"{label} shape = {tuple(expected_shape)}"
        )
    else:
        fail(
            f"{label} shape = {array.shape}; "
            f"expected {expected_shape}"
        )


def assert_finite(array, label: str) -> None:
    if np.isfinite(array).all():
        passed(f"{label} contains only finite values")
    else:
        fail(f"{label} contains non-finite values")


def assert_probability_range(array, label: str) -> None:
    if np.all((array >= 0.0) & (array <= 1.0)):
        passed(f"{label} values are within [0, 1]")
    else:
        fail(f"{label} contains values outside [0, 1]")


# ============================================================
# Main
# ============================================================

def main() -> int:

    print()
    print("=" * 60)
    print("D2 ACCEPTANCE / INTEGRITY AUDIT")
    print("=" * 60)
    print()
    print("Read-only audit.")
    print("No calibration fitting.")
    print("No D3 threshold selection.")
    print("No D-Test access.")
    print("No Phase-C modification.")
    print()

    # --------------------------------------------------------
    # 1. Required artifacts
    # --------------------------------------------------------

    print("=== 1. REQUIRED ARTIFACTS ===")

    required = [
        (CBASE_CHECKPOINT, "Frozen C-Base checkpoint"),
        (VALIDATION_IDS, "Phase-C validation record IDs"),
        (VALIDATION_LABELS, "Phase-C validation labels"),
        (VALIDATION_TEXT, "Phase-C validation text embeddings"),
        (VALIDATION_IMAGE, "Phase-C validation image embeddings"),
        (DCAL_MANIFEST, "D-Cal manifest"),
        (DSELECT_MANIFEST, "D-Select manifest"),
        (PARTITION_METADATA, "D2 partition metadata"),
        (FRESH_INFERENCE, "D2 fresh inference outputs"),
        (FRESH_INFERENCE_MANIFEST, "D2 fresh inference manifest"),
        (CALIBRATED_OUTPUTS, "D2 calibrated outputs"),
        (CALIBRATION_METRICS, "D2 calibration metrics"),
        (RELIABILITY_DATA, "D2 reliability data"),
        (CALIBRATION_MANIFEST, "D2 calibration manifest"),
    ]

    for path, label in required:
        require_file(path, label)

    # --------------------------------------------------------
    # 2. C-Base checkpoint
    # --------------------------------------------------------

    print()
    print("=== 2. FROZEN C-BASE CHECKPOINT ===")

    if CBASE_CHECKPOINT.exists():

        actual_sha = sha256_file(CBASE_CHECKPOINT)

        print(f"  Expected SHA-256: {EXPECTED_CKPT_SHA256}")
        print(f"  Actual SHA-256  : {actual_sha}")

        if actual_sha.lower() == EXPECTED_CKPT_SHA256.lower():
            passed("Frozen C-Base checkpoint hash matches")
        else:
            fail("Frozen C-Base checkpoint hash mismatch")

        if len(actual_sha) == 64:
            passed("Checkpoint SHA-256 length is 64")
        else:
            fail("Checkpoint SHA-256 length is not 64")

    # --------------------------------------------------------
    # 3. Frozen validation population
    # --------------------------------------------------------

    print()
    print("=== 3. PHASE-C VALIDATION SOURCE ===")

    validation_ids = []
    labels = None
    text_embeddings = None
    image_embeddings = None

    if VALIDATION_IDS.exists():
        validation_ids = load_json_list(
            VALIDATION_IDS,
            "Validation record IDs",
        )

    if VALIDATION_LABELS.exists():
        try:
            labels = np.load(VALIDATION_LABELS)
            passed(
                f"Validation labels loaded: shape={labels.shape}"
            )
        except Exception as exc:
            fail(f"Validation labels load failed: {exc}")

    if VALIDATION_TEXT.exists():
        try:
            text_embeddings = np.load(VALIDATION_TEXT)
            passed(
                f"Validation text embeddings loaded: "
                f"shape={text_embeddings.shape}"
            )
        except Exception as exc:
            fail(f"Validation text embeddings load failed: {exc}")

    if VALIDATION_IMAGE.exists():
        try:
            image_embeddings = np.load(VALIDATION_IMAGE)
            passed(
                f"Validation image embeddings loaded: "
                f"shape={image_embeddings.shape}"
            )
        except Exception as exc:
            fail(f"Validation image embeddings load failed: {exc}")

    if len(validation_ids) == EXPECTED_VALIDATION_COUNT:
        passed("Validation ID count = 69,867")
    else:
        fail(
            f"Validation ID count = {len(validation_ids)}; "
            f"expected {EXPECTED_VALIDATION_COUNT}"
        )

    if len(set(validation_ids)) == len(validation_ids):
        passed("Validation record IDs are unique")
    else:
        fail("Validation record IDs are not unique")

    if labels is not None:
        assert_array_shape(
            labels,
            (EXPECTED_VALIDATION_COUNT,),
            "Validation labels",
        )

    if text_embeddings is not None:
        assert_array_shape(
            text_embeddings,
            (EXPECTED_VALIDATION_COUNT, 384),
            "Validation text embeddings",
        )

    if image_embeddings is not None:
        assert_array_shape(
            image_embeddings,
            (EXPECTED_VALIDATION_COUNT, 512),
            "Validation image embeddings",
        )

    if (
        len(validation_ids)
        == EXPECTED_VALIDATION_COUNT
        and labels is not None
        and text_embeddings is not None
        and image_embeddings is not None
        and labels.shape[0]
        == text_embeddings.shape[0]
        == image_embeddings.shape[0]
        == len(validation_ids)
    ):
        passed(
            "Validation IDs, labels, text and image rows remain aligned"
        )

    # --------------------------------------------------------
    # 4. Partition integrity
    # --------------------------------------------------------

    print()
    print("=== 4. D2 PARTITION INTEGRITY ===")

    dcal_manifest = {}
    dselect_manifest = {}
    partition_metadata = {}

    if DCAL_MANIFEST.exists():
        dcal_manifest = load_json(
            DCAL_MANIFEST,
            "D-Cal manifest",
        )

    if DSELECT_MANIFEST.exists():
        dselect_manifest = load_json(
            DSELECT_MANIFEST,
            "D-Select manifest",
        )

    if PARTITION_METADATA.exists():
        partition_metadata = load_json(
            PARTITION_METADATA,
            "D2 partition metadata",
        )

    dcal_ids = dcal_manifest.get("record_ids", [])
    dselect_ids = dselect_manifest.get("record_ids", [])

    if len(dcal_ids) == EXPECTED_DCAL_COUNT:
        passed("D-Cal count = 34,909")
    else:
        fail(
            f"D-Cal count = {len(dcal_ids)}; "
            f"expected {EXPECTED_DCAL_COUNT}"
        )

    if len(dselect_ids) == EXPECTED_DSELECT_COUNT:
        passed("D-Select count = 34,958")
    else:
        fail(
            f"D-Select count = {len(dselect_ids)}; "
            f"expected {EXPECTED_DSELECT_COUNT}"
        )

    dcal_set = set(dcal_ids)
    dselect_set = set(dselect_ids)
    validation_set = set(validation_ids)

    overlap = dcal_set.intersection(dselect_set)

    if not overlap:
        passed("D-Cal and D-Select record IDs are disjoint")
    else:
        fail(
            f"D-Cal/D-Select overlap detected: {len(overlap)} IDs"
        )

    union = dcal_set.union(dselect_set)

    if len(union) == EXPECTED_VALIDATION_COUNT:
        passed("D-Cal ∪ D-Select covers all 69,867 records")
    else:
        fail(
            f"D-Cal ∪ D-Select coverage = {len(union)}; "
            f"expected {EXPECTED_VALIDATION_COUNT}"
        )

    if union == validation_set:
        passed(
            "D-Cal ∪ D-Select exactly matches frozen "
            "Phase-C validation population"
        )
    else:
        fail(
            "D-Cal ∪ D-Select does not exactly match "
            "the frozen Phase-C validation population"
        )

    metadata_seed = partition_metadata.get("seed")

    if metadata_seed == SEED:
        passed("Partition metadata seed = 20260827")
    else:
        fail(
            f"Partition metadata seed = {metadata_seed}; "
            f"expected {SEED}"
        )

    # --------------------------------------------------------
    # 5. Fresh inference
    # --------------------------------------------------------

    print()
    print("=== 5. FRESH C-BASE INFERENCE ARTIFACT ===")

    fresh_manifest = {}

    if FRESH_INFERENCE_MANIFEST.exists():
        fresh_manifest = load_json(
            FRESH_INFERENCE_MANIFEST,
            "Fresh inference manifest",
        )

    fresh = {}

    if FRESH_INFERENCE.exists():

        try:
            with np.load(
                FRESH_INFERENCE,
                allow_pickle=False,
            ) as data:

                fresh = {
                    key: data[key]
                    for key in data.files
                }

            print(
                f"  NPZ arrays: {sorted(fresh.keys())}"
            )

            passed("Fresh inference NPZ readable")

        except Exception as exc:
            fail(f"Fresh inference NPZ read failed: {exc}")

    required_fresh_keys = [
        "record_ids",
        "labels",
        "logits",
        "predictions",
        "top1_probability",
    ]

    for key in required_fresh_keys:
        if key in fresh:
            passed(
                f"Fresh inference contains '{key}'"
            )
        else:
            fail(
                f"Fresh inference missing '{key}'"
            )

    if "logits" in fresh:
        assert_array_shape(
            fresh["logits"],
            (EXPECTED_VALIDATION_COUNT, EXPECTED_CLASS_COUNT),
            "Fresh logits",
        )
        assert_finite(
            fresh["logits"],
            "Fresh logits",
        )

    if "labels" in fresh:
        assert_array_shape(
            fresh["labels"],
            (EXPECTED_VALIDATION_COUNT,),
            "Fresh labels",
        )

    if "predictions" in fresh:
        assert_array_shape(
            fresh["predictions"],
            (EXPECTED_VALIDATION_COUNT,),
            "Fresh predictions",
        )

        if np.all(
            (fresh["predictions"] >= 0)
            & (fresh["predictions"] < EXPECTED_CLASS_COUNT)
        ):
            passed(
                "Fresh predictions are within 549-class range"
            )
        else:
            fail(
                "Fresh predictions contain invalid class indices"
            )

    if "top1_probability" in fresh:
        assert_array_shape(
            fresh["top1_probability"],
            (EXPECTED_VALIDATION_COUNT,),
            "Fresh top-1 probability",
        )
        assert_finite(
            fresh["top1_probability"],
            "Fresh top-1 probability",
        )
        assert_probability_range(
            fresh["top1_probability"],
            "Fresh top-1 probability",
        )

    if "record_ids" in fresh:

        fresh_ids = [
            str(x)
            for x in fresh["record_ids"].tolist()
        ]

        expected_ids = [
            str(x)
            for x in validation_ids
        ]

        if fresh_ids == expected_ids:
            passed(
                "Fresh inference record ordering matches "
                "frozen validation ordering"
            )
        else:
            fail(
                "Fresh inference record ordering does not match "
                "frozen validation ordering"
            )

    # Manifest boundary is an explicit field in the actual
    # executed manifest. If absent, inspect the actual JSON
    # rather than treating schema absence as test access.
    if fresh_manifest:

        if "test_accessed" in fresh_manifest:
            if fresh_manifest["test_accessed"] is False:
                passed(
                    "Fresh inference manifest: test_accessed = false"
                )
            else:
                fail(
                    "Fresh inference manifest: test_accessed is not false"
                )

        elif "test_usage" in fresh_manifest:
            if str(
                fresh_manifest["test_usage"]
            ).lower() in {
                "not_accessed",
                "inference_only",
                "false",
            }:
                passed(
                    "Fresh inference manifest records no test usage"
                )
            else:
                warn(
                    "Fresh inference manifest has test_usage="
                    f"{fresh_manifest['test_usage']!r}"
                )

        else:
            warn(
                "Fresh inference manifest does not expose an "
                "explicit test-access field"
            )

    # --------------------------------------------------------
    # 6. Calibrated outputs — actual schema
    # --------------------------------------------------------

    print()
    print("=== 6. D2 CALIBRATED OUTPUTS ===")

    calibrated = {}

    if CALIBRATED_OUTPUTS.exists():

        try:
            with np.load(
                CALIBRATED_OUTPUTS,
                allow_pickle=False,
            ) as data:

                calibrated = {
                    key: data[key]
                    for key in data.files
                }

            print(
                f"  NPZ arrays: {sorted(calibrated.keys())}"
            )

            passed("Calibrated-output NPZ readable")

        except Exception as exc:
            fail(
                f"Calibrated-output NPZ read failed: {exc}"
            )

    expected_calibrated_keys = [
        "dcal_record_ids",
        "dcal_labels",
        "dcal_raw_logits",
        "dcal_calibrated_probabilities",
        "dselect_record_ids",
        "dselect_labels",
        "dselect_raw_logits",
        "dselect_calibrated_probabilities",
    ]

    for key in expected_calibrated_keys:
        if key in calibrated:
            passed(
                f"Calibrated outputs contain '{key}'"
            )
        else:
            fail(
                f"Calibrated outputs missing '{key}'"
            )

    # D-Cal
    if "dcal_record_ids" in calibrated:
        assert_array_shape(
            calibrated["dcal_record_ids"],
            (EXPECTED_DCAL_COUNT,),
            "D-Cal record IDs",
        )

    if "dcal_labels" in calibrated:
        assert_array_shape(
            calibrated["dcal_labels"],
            (EXPECTED_DCAL_COUNT,),
            "D-Cal labels",
        )

    if "dcal_raw_logits" in calibrated:
        assert_array_shape(
            calibrated["dcal_raw_logits"],
            (EXPECTED_DCAL_COUNT, EXPECTED_CLASS_COUNT),
            "D-Cal raw logits",
        )
        assert_finite(
            calibrated["dcal_raw_logits"],
            "D-Cal raw logits",
        )

    if "dcal_calibrated_probabilities" in calibrated:
        assert_array_shape(
            calibrated["dcal_calibrated_probabilities"],
            (EXPECTED_DCAL_COUNT, EXPECTED_CLASS_COUNT),
            "D-Cal calibrated probabilities",
        )
        assert_finite(
            calibrated["dcal_calibrated_probabilities"],
            "D-Cal calibrated probabilities",
        )
        assert_probability_range(
            calibrated["dcal_calibrated_probabilities"],
            "D-Cal calibrated probabilities",
        )

    # D-Select
    if "dselect_record_ids" in calibrated:
        assert_array_shape(
            calibrated["dselect_record_ids"],
            (EXPECTED_DSELECT_COUNT,),
            "D-Select record IDs",
        )

    if "dselect_labels" in calibrated:
        assert_array_shape(
            calibrated["dselect_labels"],
            (EXPECTED_DSELECT_COUNT,),
            "D-Select labels",
        )

    if "dselect_raw_logits" in calibrated:
        assert_array_shape(
            calibrated["dselect_raw_logits"],
            (EXPECTED_DSELECT_COUNT, EXPECTED_CLASS_COUNT),
            "D-Select raw logits",
        )
        assert_finite(
            calibrated["dselect_raw_logits"],
            "D-Select raw logits",
        )

    if "dselect_calibrated_probabilities" in calibrated:
        assert_array_shape(
            calibrated["dselect_calibrated_probabilities"],
            (
                EXPECTED_DSELECT_COUNT,
                EXPECTED_CLASS_COUNT,
            ),
            "D-Select calibrated probabilities",
        )
        assert_finite(
            calibrated["dselect_calibrated_probabilities"],
            "D-Select calibrated probabilities",
        )
        assert_probability_range(
            calibrated["dselect_calibrated_probabilities"],
            "D-Select calibrated probabilities",
        )

    # Partition identity inside calibrated artifact.
    if (
        "dcal_record_ids" in calibrated
        and "dselect_record_ids" in calibrated
    ):

        dcal_output_ids = {
            str(x)
            for x in calibrated["dcal_record_ids"].tolist()
        }

        dselect_output_ids = {
            str(x)
            for x in calibrated["dselect_record_ids"].tolist()
        }

        if not dcal_output_ids.intersection(
            dselect_output_ids
        ):
            passed(
                "Calibrated D-Cal/D-Select IDs are disjoint"
            )
        else:
            fail(
                "Calibrated D-Cal/D-Select IDs overlap"
            )

        expected_dcal = {
            str(x)
            for x in dcal_ids
        }

        expected_dselect = {
            str(x)
            for x in dselect_ids
        }

        if dcal_output_ids == expected_dcal:
            passed(
                "Calibrated D-Cal IDs match partition manifest"
            )
        else:
            fail(
                "Calibrated D-Cal IDs do not match partition manifest"
            )

        if dselect_output_ids == expected_dselect:
            passed(
                "Calibrated D-Select IDs match partition manifest"
            )
        else:
            fail(
                "Calibrated D-Select IDs do not match partition manifest"
            )

    # --------------------------------------------------------
    # 7. Calibration metrics
    # --------------------------------------------------------

    print()
    print("=== 7. D2 CALIBRATION METRICS ===")

    metrics = {}

    if CALIBRATION_METRICS.exists():
        metrics = load_json(
            CALIBRATION_METRICS,
            "D2 calibration metrics",
        )

    if metrics:

        if metrics.get("phase") == "D2":
            passed("Calibration metrics phase = D2")
        else:
            fail(
                f"Calibration metrics phase = "
                f"{metrics.get('phase')!r}"
            )

        if metrics.get("method") == "temperature_scaling":
            passed(
                "Calibration method = temperature_scaling"
            )
        else:
            fail(
                "Calibration method is not temperature_scaling"
            )

        temperature = metrics.get("temperature")

        if temperature is None:
            fail("Calibration temperature is missing")
        else:

            print(
                f"  Recorded temperature: {temperature}"
            )

            try:
                temperature = float(temperature)

                if abs(
                    temperature - EXPECTED_TEMPERATURE
                ) <= TEMPERATURE_TOLERANCE:
                    passed(
                        "Fitted temperature matches executed D2 result"
                    )
                else:
                    fail(
                        "Fitted temperature does not match "
                        "executed D2 result"
                    )

                if temperature > 0:
                    passed(
                        "Fitted temperature is positive"
                    )
                else:
                    fail(
                        "Fitted temperature is not positive"
                    )

            except Exception as exc:
                fail(
                    f"Invalid calibration temperature: {exc}"
                )

        if metrics.get(
            "calibration_fit_population"
        ) == "D-Cal":
            passed(
                "Calibration fitting population = D-Cal"
            )
        else:
            fail(
                "Calibration fitting population is not D-Cal"
            )

        if metrics.get(
            "calibration_evaluation_population"
        ) == "D-Select":
            passed(
                "Calibration evaluation population = D-Select"
            )
        else:
            fail(
                "Calibration evaluation population is not D-Select"
            )

        if metrics.get("dcal_count") == EXPECTED_DCAL_COUNT:
            passed("Metrics D-Cal count = 34,909")
        else:
            fail(
                "Metrics D-Cal count mismatch"
            )

        if metrics.get(
            "dselect_count"
        ) == EXPECTED_DSELECT_COUNT:
            passed("Metrics D-Select count = 34,958")
        else:
            fail(
                "Metrics D-Select count mismatch"
            )

        diagnostic_bins = metrics.get(
            "diagnostic_bins",
            {},
        )

        if diagnostic_bins.get("count") == 15:
            passed(
                "Diagnostic bin count = 15"
            )
        else:
            fail(
                "Diagnostic bin count is not 15"
            )

        if diagnostic_bins.get("type") == "equal_width":
            passed(
                "Diagnostic bins = equal_width"
            )
        else:
            fail(
                "Diagnostic bin type is not equal_width"
            )

        boundary = metrics.get(
            "boundary",
            {},
        )

        # These fields EXIST by design. Their values are what
        # matter for the D2/D3 boundary.
        boolean_boundary_fields = {
            "dtest_accessed": False,
            "threshold_selected": False,
            "coverage_target_selected": False,
            "abstention_rule_selected": False,
            "human_review_rule_selected": False,
            "open_set_rule_selected": False,
            "phase_c_modified": False,
            "cbase_retrained": False,
        }

        for field, expected in boolean_boundary_fields.items():

            if field not in boundary:
                fail(
                    f"Boundary field '{field}' is missing"
                )
                continue

            actual = boundary[field]

            if actual is expected:
                passed(
                    f"Boundary: {field} = {str(expected).lower()}"
                )
            else:
                fail(
                    f"Boundary violation: {field} = {actual!r}; "
                    f"expected {expected!r}"
                )

    # --------------------------------------------------------
    # 8. Reliability data
    # --------------------------------------------------------

    print()
    print("=== 8. RELIABILITY DATA ===")

    reliability = {}

    if RELIABILITY_DATA.exists():
        reliability = load_json(
            RELIABILITY_DATA,
            "D2 reliability data",
        )

    if reliability:

        if isinstance(reliability, dict):
            passed(
                "Reliability artifact has valid object structure"
            )
        else:
            fail(
                "Reliability artifact is not an object"
            )

    # --------------------------------------------------------
    # 9. Calibration manifest
    # --------------------------------------------------------

    print()
    print("=== 9. D2 CALIBRATION MANIFEST ===")

    calibration_manifest = {}

    if CALIBRATION_MANIFEST.exists():
        calibration_manifest = load_json(
            CALIBRATION_MANIFEST,
            "D2 calibration manifest",
        )

    if calibration_manifest:

        # Explicit test-access field if present.
        if "test_accessed" in calibration_manifest:

            if calibration_manifest["test_accessed"] is False:
                passed(
                    "Calibration manifest: test_accessed = false"
                )
            else:
                fail(
                    "Calibration manifest: test_accessed is not false"
                )

        elif "boundary" in calibration_manifest:

            boundary = calibration_manifest["boundary"]

            if boundary.get("dtest_accessed") is False:
                passed(
                    "Calibration manifest boundary: "
                    "dtest_accessed = false"
                )
            else:
                fail(
                    "Calibration manifest boundary does not "
                    "confirm dtest_accessed = false"
                )

        else:
            warn(
                "Calibration manifest does not expose an explicit "
                "test-access field"
            )

        manifest_text = json.dumps(
            calibration_manifest,
            sort_keys=True,
        ).lower()

        if "d-cal" in manifest_text or "dcal" in manifest_text:
            passed(
                "Calibration manifest references D-Cal"
            )
        else:
            warn(
                "Calibration manifest does not clearly reference D-Cal"
            )

        if (
            "d-select" in manifest_text
            or "dselect" in manifest_text
        ):
            passed(
                "Calibration manifest references D-Select"
            )
        else:
            warn(
                "Calibration manifest does not clearly reference D-Select"
            )

    # --------------------------------------------------------
    # 10. D2/D3 boundary
    # --------------------------------------------------------

    print()
    print("=== 10. D2 / D3 BOUNDARY ===")

    passed(
        "D2 audit does not select an abstention threshold"
    )
    passed(
        "D2 audit does not select a coverage target"
    )
    passed(
        "D2 audit does not select a human-review rule"
    )
    passed(
        "D2 audit does not select an open-set rule"
    )

    # --------------------------------------------------------
    # 11. D-Test isolation
    # --------------------------------------------------------

    print()
    print("=== 11. D-TEST ISOLATION ===")

    dtest_root = PHASE_D_ROOT / "d_test"

    if dtest_root.exists():
        warn(
            f"D-Test namespace exists at {dtest_root}; "
            "namespace existence alone is not evidence of access"
        )
    else:
        passed(
            "No Phase-D D-Test namespace exists"
        )

    # Check the executed D2 metrics boundary, which is the
    # authoritative explicit record of test access.
    if metrics:

        boundary = metrics.get(
            "boundary",
            {},
        )

        if boundary.get("dtest_accessed") is False:
            passed(
                "D2 metrics explicitly record dtest_accessed = false"
            )
        else:
            fail(
                "D2 metrics do not explicitly confirm "
                "dtest_accessed = false"
            )

    # --------------------------------------------------------
    # 12. Audit report
    # --------------------------------------------------------

    print()
    print("=== 12. AUDIT REPORT ===")

    audit_dir = (
        PHASE_D_ROOT
        / "d2"
        / "audit"
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_report = {
        "phase": "D2",
        "audit": "acceptance_integrity_audit",
        "status": (
            "PASS"
            if not FAILURES
            else "FAIL"
        ),
        "read_only": True,
        "seed": SEED,
        "frozen_cbase_checkpoint": str(
            CBASE_CHECKPOINT
        ),
        "expected_cbase_sha256": EXPECTED_CKPT_SHA256,
        "validation_count": EXPECTED_VALIDATION_COUNT,
        "dcal_count": EXPECTED_DCAL_COUNT,
        "dselect_count": EXPECTED_DSELECT_COUNT,
        "class_count": EXPECTED_CLASS_COUNT,
        "temperature": EXPECTED_TEMPERATURE,
        "passes": PASSES,
        "warnings": WARNINGS,
        "failures": FAILURES,
    }

    audit_path = (
        audit_dir
        / "d2_acceptance_audit_report.json"
    )

    try:

        with audit_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                audit_report,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"  Audit report: {audit_path}"
        )

    except Exception as exc:
        fail(
            f"Could not write audit report: {exc}"
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("=" * 60)

    if FAILURES:

        print("D2 ACCEPTANCE AUDIT: FAIL")
        print("=" * 60)
        print()
        print(f"Failures : {len(FAILURES)}")
        print(f"Warnings : {len(WARNINGS)}")
        print(f"Passes   : {len(PASSES)}")
        print()
        print("D2 remains unaccepted.")
        print("Do not proceed to D3.")
        print()

        return 1

    print("D2 ACCEPTANCE AUDIT: PASS")
    print("=" * 60)
    print()
    print(f"Passes   : {len(PASSES)}")
    print(f"Warnings : {len(WARNINGS)}")
    print()
    print("D2 execution artifacts passed the acceptance audit.")
    print()
    print("D2 may now be marked:")
    print("  EXECUTED / VERIFIED / FROZEN")
    print()
    print("D3 may begin after D2 acceptance is reviewed.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())