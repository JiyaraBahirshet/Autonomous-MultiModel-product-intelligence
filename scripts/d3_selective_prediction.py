"""
Phase D3 — Selective Prediction / Abstention Characterization

D3 v002 — Approved contract.

Purpose
-------
Characterize the complete D-Select coverage–risk relationship using the
frozen D2 calibrated top-1 probabilities.

This script DOES:
- consume frozen D2 calibrated outputs;
- use D-Select only;
- use a deterministic, pre-specified threshold grid;
- compute coverage, selective risk, selective accuracy, accepted count,
  abstained count, abstention rate, and accepted errors;
- save reproducible D3 characterization artifacts.

This script DOES NOT:
- refit D2 temperature scaling;
- regenerate D2 outputs;
- modify D2 artifacts;
- select a deployment/operating threshold;
- select R_max or C_min;
- access D-Test;
- create or use a D-Test population;
- perform human-review routing;
- perform open-set detection;
- retrain or modify C-Base.

D3 v002 approved contract:
- D-Cal = 34,909 (frozen authoritative D2 output)
- D-Select = 34,958 (frozen authoritative D2 output)
- temperature = 1.9641200304 (frozen authoritative D2 output)
- D3 development population = D-Select only
- threshold grid = 0.00, 0.01, ..., 0.99, 1.00
- initial D3 objective = characterization only
- no universally optimal threshold is claimed
- D-Test is a separate subsequent final-evaluation stage and is NOT
  accessed by this script.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ============================================================
# 1. PROJECT / PHASE CONSTANTS
# ============================================================

SEED = 20260827

EXPECTED_DCAL_COUNT = 34_909
EXPECTED_DSELECT_COUNT = 34_958
EXPECTED_TOTAL_VALIDATION_COUNT = 69_867

EXPECTED_TEMPERATURE = 1.9641200304031372

N_CLASSES = 549

THRESHOLD_START = 0.00
THRESHOLD_END = 1.00
THRESHOLD_STEP = 0.01

ROOT = Path(__file__).resolve().parents[1]

D2_CALIBRATION_DIR = (
    ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_d"
    / "d2"
    / "calibration"
)

D3_ROOT = (
    ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_d"
    / "d3"
)

D3_DEVELOPMENT_DIR = D3_ROOT / "development"
D3_CONFIG_DIR = ROOT / "configs" / "phase_d"

D2_CALIBRATED_OUTPUTS = (
    D2_CALIBRATION_DIR
    / "d2_calibrated_validation_outputs.npz"
)

D2_CALIBRATION_METRICS = (
    D2_CALIBRATION_DIR
    / "d2_calibration_metrics.json"
)

D2_CALIBRATION_MANIFEST = (
    D2_CALIBRATION_DIR
    / "d2_calibration_manifest.json"
)

D3_RESULTS_JSON = (
    D3_DEVELOPMENT_DIR
    / "d3_selective_prediction_results.json"
)

D3_RESULTS_CSV = (
    D3_DEVELOPMENT_DIR
    / "d3_coverage_risk_curve.csv"
)

D3_CONFIG_JSON = (
    D3_CONFIG_DIR
    / "d3_v002_characterization_config.json"
)

D3_MANIFEST_JSON = (
    D3_DEVELOPMENT_DIR
    / "d3_selective_prediction_manifest.json"
)


# ============================================================
# 2. HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    """Return SHA-256 hash of a file."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def utc_now() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    """Write deterministic human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the threshold characterization table without pandas."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("Cannot write empty CSV.")

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def assert_close(
    actual: float,
    expected: float,
    tolerance: float = 1e-10,
    name: str = "value",
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise AssertionError(
            f"{name} mismatch: "
            f"actual={actual}, expected={expected}"
        )


# ============================================================
# 3. DETERMINISTIC THRESHOLD GRID
# ============================================================

def build_threshold_grid() -> np.ndarray:
    """
    Build the contractually frozen threshold grid.

    IMPORTANT:
    This grid is defined independently of observed D-Select results.

    It is NOT derived from:
    - observed probabilities;
    - observed coverage;
    - observed risk;
    - desired coverage;
    - desired error rate.
    """

    count = int(
        round(
            (THRESHOLD_END - THRESHOLD_START)
            / THRESHOLD_STEP
        )
    ) + 1

    thresholds = (
        THRESHOLD_START
        + np.arange(count, dtype=np.float64)
        * THRESHOLD_STEP
    )

    # Remove floating-point representation artifacts such as
    # 0.30000000000000004.
    thresholds = np.round(
        thresholds,
        decimals=2,
    )

    expected = np.arange(
        0.00,
        1.0000001,
        0.01,
        dtype=np.float64,
    )

    expected = np.round(
        expected,
        decimals=2,
    )

    if not np.array_equal(
        thresholds,
        expected,
    ):
        raise AssertionError(
            "Deterministic threshold grid does not match "
            "the frozen D3 v002 grid."
        )

    if len(thresholds) != 101:
        raise AssertionError(
            f"Expected 101 thresholds, got {len(thresholds)}."
        )

    if thresholds[0] != 0.00:
        raise AssertionError("Threshold grid must start at 0.00.")

    if thresholds[-1] != 1.00:
        raise AssertionError("Threshold grid must end at 1.00.")

    return thresholds


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def main() -> int:

    print("=" * 60)
    print("D3 SELECTIVE PREDICTION / ABSTENTION CHARACTERIZATION")
    print("=" * 60)
    print()
    print("D3 v002 — approved characterization experiment.")
    print()
    print("D3 only:")
    print("  - frozen D2 calibrated outputs")
    print("  - D-Select only")
    print("  - deterministic threshold grid")
    print("  - coverage/risk characterization")
    print()
    print("D3 does NOT:")
    print("  - fit calibration")
    print("  - select R_max/C_min")
    print("  - select a deployment threshold")
    print("  - access D-Test")
    print("  - modify D2")
    print("  - modify Phase C")
    print()

    # --------------------------------------------------------
    # 4.1 Create output directories
    # --------------------------------------------------------

    D3_DEVELOPMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    D3_CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 4.2 Explicit D-Test boundary
    # --------------------------------------------------------

    phase_d_root = ROOT / "data" / "models" / "abo" / "phase_d"

    dtest_dir = phase_d_root / "dtest"

    if dtest_dir.exists():
        raise RuntimeError(
            "D3 execution blocked: a Phase-D D-Test namespace exists at "
            f"{dtest_dir}. D3 must not access D-Test."
        )

    print("D-TEST BOUNDARY: PASS")
    print("  No Phase-D D-Test namespace detected.")
    print()

    # --------------------------------------------------------
    # 4.3 Required D2 artifact checks
    # --------------------------------------------------------

    required_d2_files = [
        D2_CALIBRATED_OUTPUTS,
        D2_CALIBRATION_METRICS,
        D2_CALIBRATION_MANIFEST,
    ]

    for path in required_d2_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required frozen D2 artifact missing: {path}"
            )

    print("REQUIRED D2 ARTIFACTS: PASS")

    d2_outputs_hash = sha256_file(
        D2_CALIBRATED_OUTPUTS
    )

    d2_metrics_hash = sha256_file(
        D2_CALIBRATION_METRICS
    )

    d2_manifest_hash = sha256_file(
        D2_CALIBRATION_MANIFEST
    )

    print(
        f"  D2 calibrated outputs SHA-256: "
        f"{d2_outputs_hash}"
    )

    print(
        f"  D2 metrics SHA-256: "
        f"{d2_metrics_hash}"
    )

    print(
        f"  D2 manifest SHA-256: "
        f"{d2_manifest_hash}"
    )

    print()

    # --------------------------------------------------------
    # 4.4 Verify D2 metrics are the frozen authoritative result
    # --------------------------------------------------------

    d2_metrics = json.loads(
        D2_CALIBRATION_METRICS.read_text(
            encoding="utf-8"
        )
    )

    if d2_metrics.get("phase") != "D2":
        raise AssertionError(
            "D2 metrics phase is not D2."
        )

    if d2_metrics.get("method") != "temperature_scaling":
        raise AssertionError(
            "D2 calibration method is not temperature_scaling."
        )

    dcal_count = int(
        d2_metrics["dcal_count"]
    )

    dselect_count = int(
        d2_metrics["dselect_count"]
    )

    temperature = float(
        d2_metrics["temperature"]
    )

    if dcal_count != EXPECTED_DCAL_COUNT:
        raise AssertionError(
            f"D-Cal count mismatch: "
            f"{dcal_count} != {EXPECTED_DCAL_COUNT}"
        )

    if dselect_count != EXPECTED_DSELECT_COUNT:
        raise AssertionError(
            f"D-Select count mismatch: "
            f"{dselect_count} != {EXPECTED_DSELECT_COUNT}"
        )

    assert_close(
        temperature,
        EXPECTED_TEMPERATURE,
        tolerance=1e-9,
        name="D2 temperature",
    )

    boundary = d2_metrics.get(
        "boundary",
        {},
    )

    if boundary.get("dtest_accessed") is not False:
        raise AssertionError(
            "D2 metrics do not explicitly confirm "
            "dtest_accessed=false."
        )

    if boundary.get("threshold_selected") is not False:
        raise AssertionError(
            "D2 metrics indicate a threshold was selected."
        )

    if boundary.get("coverage_target_selected") is not False:
        raise AssertionError(
            "D2 metrics indicate a coverage target was selected."
        )

    print("FROZEN D2 AUTHORITATIVE OUTPUTS: PASS")
    print(f"  D-Cal       : {dcal_count:,}")
    print(f"  D-Select    : {dselect_count:,}")
    print(f"  Temperature : {temperature:.10f}")
    print("  D-Test      : not accessed")
    print()

    # --------------------------------------------------------
    # 4.5 Freeze the D3 threshold grid BEFORE loading results
    # --------------------------------------------------------

    thresholds = build_threshold_grid()

    threshold_grid = [
        float(x)
        for x in thresholds
    ]

    print("D3 THRESHOLD GRID: FROZEN")
    print("  Generation rule : 0.00 to 1.00 inclusive")
    print("  Step            : 0.01")
    print(f"  Count           : {len(threshold_grid)}")
    print("  Result-adaptive : NO")
    print()

    # --------------------------------------------------------
    # 4.6 Save D3 configuration before threshold evaluation
    # --------------------------------------------------------

    d3_config = {
        "phase": "D3",
        "version": "v002",
        "experiment": "selective_prediction_characterization",
        "status": "EXECUTING",
        "seed": SEED,
        "objective": (
            "Characterize the complete D-Select "
            "coverage-risk relationship."
        ),
        "operating_objective": {
            "type": "characterization_only",
            "r_max": None,
            "c_min": None,
            "universal_optimum_claim": False,
            "explanation": (
                "No scientifically justified numerical "
                "R_max/C_min has been established. "
                "No numerical target is invented for D3."
            ),
        },
        "development_population": {
            "name": "D-Select",
            "count": EXPECTED_DSELECT_COUNT,
        },
        "excluded_population": {
            "name": "D-Test",
            "access_allowed": False,
            "role": (
                "separate subsequent final-evaluation stage "
                "after D3 decision-rule and integrity freeze"
            ),
        },
        "upstream_frozen_d2": {
            "dcal_count": EXPECTED_DCAL_COUNT,
            "dselect_count": EXPECTED_DSELECT_COUNT,
            "temperature": EXPECTED_TEMPERATURE,
            "d2_calibration_method": "temperature_scaling",
            "d2_outputs_modified": False,
            "d2_refit": False,
        },
        "decision_signal": {
            "type": "calibrated_top1_probability",
            "prediction": "argmax calibrated probability",
            "confidence": "max calibrated probability",
        },
        "threshold_grid": {
            "generation_rule": (
                "np.arange(0.00, 1.0000001, 0.01), "
                "rounded to two decimal places"
            ),
            "start": THRESHOLD_START,
            "end": THRESHOLD_END,
            "step": THRESHOLD_STEP,
            "count": len(threshold_grid),
            "thresholds": threshold_grid,
            "frozen_before_result_inspection": True,
            "adaptive_from_observed_probabilities": False,
            "adaptive_from_observed_performance": False,
            "adaptive_from_desired_coverage": False,
        },
        "decision_rule": {
            "accept_if": "calibrated_top1_probability >= threshold",
            "otherwise": "abstain",
            "threshold_selected": False,
        },
        "metrics": {
            "primary": [
                "coverage",
                "selective_risk",
                "selective_accuracy",
            ],
            "supporting": [
                "threshold",
                "accepted_count",
                "abstained_count",
                "abstention_rate",
                "accepted_error_count",
            ],
        },
        "human_review": {
            "evaluated": False,
            "routing_rule": None,
        },
        "open_set": {
            "evaluated": False,
            "rule": None,
        },
        "dtest_accessed": False,
        "created_at_utc": utc_now(),
    }

    write_json(
        D3_CONFIG_JSON,
        d3_config,
    )

    d3_config_hash = sha256_file(
        D3_CONFIG_JSON
    )

    print("D3 CONFIGURATION: WRITTEN")
    print(f"  Path: {D3_CONFIG_JSON}")
    print(f"  SHA-256: {d3_config_hash}")
    print()

    # --------------------------------------------------------
    # 4.7 Load frozen D2 calibrated outputs
    # --------------------------------------------------------

    with np.load(
        D2_CALIBRATED_OUTPUTS,
        allow_pickle=False,
    ) as data:

        keys = sorted(
            data.files
        )

        required_keys = {
            "dcal_record_ids",
            "dcal_labels",
            "dcal_raw_logits",
            "dcal_calibrated_probabilities",
            "dselect_record_ids",
            "dselect_labels",
            "dselect_raw_logits",
            "dselect_calibrated_probabilities",
        }

        if not required_keys.issubset(
            set(keys)
        ):
            missing = sorted(
                required_keys - set(keys)
            )

            raise AssertionError(
                "D2 calibrated output is missing "
                f"required fields: {missing}"
            )

        dselect_record_ids = np.asarray(
            data["dselect_record_ids"]
        )

        dselect_labels = np.asarray(
            data["dselect_labels"]
        )

        dselect_probabilities = np.asarray(
            data["dselect_calibrated_probabilities"],
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # 4.8 Validate D-Select source artifact
    # --------------------------------------------------------

    if dselect_record_ids.ndim != 1:
        raise AssertionError(
            "D-Select record IDs must be 1-D."
        )

    if dselect_labels.ndim != 1:
        raise AssertionError(
            "D-Select labels must be 1-D."
        )

    if dselect_probabilities.ndim != 2:
        raise AssertionError(
            "D-Select calibrated probabilities must be 2-D."
        )

    if dselect_record_ids.shape[0] != EXPECTED_DSELECT_COUNT:
        raise AssertionError(
            "D-Select record count mismatch: "
            f"{dselect_record_ids.shape[0]} != "
            f"{EXPECTED_DSELECT_COUNT}"
        )

    if dselect_labels.shape[0] != EXPECTED_DSELECT_COUNT:
        raise AssertionError(
            "D-Select label count mismatch."
        )

    if dselect_probabilities.shape != (
        EXPECTED_DSELECT_COUNT,
        N_CLASSES,
    ):
        raise AssertionError(
            "D-Select calibrated probability shape mismatch: "
            f"{dselect_probabilities.shape}"
        )

    if not np.all(
        np.isfinite(dselect_probabilities)
    ):
        raise AssertionError(
            "D-Select calibrated probabilities contain "
            "non-finite values."
        )

    if np.any(
        dselect_probabilities < 0.0
    ) or np.any(
        dselect_probabilities > 1.0
    ):
        raise AssertionError(
            "D-Select calibrated probabilities contain "
            "values outside [0,1]."
        )

    # Ensure each row is a valid probability distribution.
    probability_sums = np.sum(
        dselect_probabilities,
        axis=1,
    )

    if not np.allclose(
        probability_sums,
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise AssertionError(
            "D-Select calibrated probability rows do not "
            "sum to 1 within tolerance."
        )

    if len(
        np.unique(dselect_record_ids)
    ) != EXPECTED_DSELECT_COUNT:
        raise AssertionError(
            "D-Select record IDs are not unique."
        )

    print("D2 CALIBRATED D-SELECT OUTPUT: PASS")
    print(
        f"  Records       : {len(dselect_record_ids):,}"
    )
    print(
        f"  Probabilities : {dselect_probabilities.shape}"
    )
    print(
        "  Probability rows sum to 1: PASS"
    )
    print()

    # --------------------------------------------------------
    # 4.9 Compute frozen top-1 prediction and probability
    # --------------------------------------------------------

    predictions = np.argmax(
        dselect_probabilities,
        axis=1,
    ).astype(
        np.int64,
        copy=False,
    )

    top1_probability = np.max(
        dselect_probabilities,
        axis=1,
    )

    if not np.all(
        np.isfinite(top1_probability)
    ):
        raise AssertionError(
            "D-Select top-1 probabilities contain "
            "non-finite values."
        )

    if np.any(
        top1_probability < 0.0
    ) or np.any(
        top1_probability > 1.0
    ):
        raise AssertionError(
            "D-Select top-1 probabilities fall outside [0,1]."
        )

    correct = (
        predictions == dselect_labels
    )

    total_count = len(
        dselect_labels
    )

    if total_count != EXPECTED_DSELECT_COUNT:
        raise AssertionError(
            "Unexpected D-Select population size."
        )

    overall_accuracy = float(
        np.mean(correct)
    )

    print("D3 BASELINE REFERENCE")
    print(
        f"  D-Select accuracy: {overall_accuracy:.6f}"
    )
    print(
        "  This is descriptive reference only."
    )
    print()

    # --------------------------------------------------------
    # 4.10 Threshold characterization
    # --------------------------------------------------------

    rows: list[dict] = []

    for threshold in thresholds:

        accepted = (
            top1_probability >= threshold
        )

        accepted_count = int(
            np.sum(accepted)
        )

        abstained_count = (
            total_count
            - accepted_count
        )

        if accepted_count > 0:

            accepted_correct = int(
                np.sum(
                    correct & accepted
                )
            )

            accepted_errors = (
                accepted_count
                - accepted_correct
            )

            selective_accuracy = (
                accepted_correct
                / accepted_count
            )

            selective_risk = (
                accepted_errors
                / accepted_count
            )

        else:

            accepted_correct = 0
            accepted_errors = 0

            selective_accuracy = None
            selective_risk = None

        coverage = (
            accepted_count
            / total_count
        )

        abstention_rate = (
            abstained_count
            / total_count
        )

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                "accepted_count": accepted_count,
                "abstained_count": abstained_count,
                "coverage": float(
                    coverage
                ),
                "abstention_rate": float(
                    abstention_rate
                ),
                "accepted_correct_count": accepted_correct,
                "accepted_error_count": accepted_errors,
                "selective_accuracy": (
                    None
                    if selective_accuracy is None
                    else float(selective_accuracy)
                ),
                "selective_risk": (
                    None
                    if selective_risk is None
                    else float(selective_risk)
                ),
            }
        )

    # --------------------------------------------------------
    # 4.11 Validate threshold-characterization invariants
    # --------------------------------------------------------

    if len(rows) != 101:
        raise AssertionError(
            "Expected exactly 101 threshold rows."
        )

    first = rows[0]
    last = rows[-1]

    # At threshold 0.00, all valid records must be accepted.
    if first["accepted_count"] != total_count:
        raise AssertionError(
            "Threshold 0.00 did not accept all D-Select records."
        )

    if not math.isclose(
        first["coverage"],
        1.0,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            "Threshold 0.00 coverage is not 1.0."
        )

    # At threshold 1.00, only exact probability-1 predictions
    # are accepted. The exact count is data-derived and therefore
    # is not assumed in advance.
    if not (
        0 <= last["accepted_count"] <= total_count
    ):
        raise AssertionError(
            "Invalid threshold-1.00 accepted count."
        )

    previous_coverage = None

    for row in rows:

        coverage = row["coverage"]

        if previous_coverage is not None:

            if coverage > previous_coverage + 1e-12:
                raise AssertionError(
                    "Coverage increased as threshold increased."
                )

        previous_coverage = coverage

    print(
        "THRESHOLD CHARACTERIZATION INVARIANTS: PASS"
    )
    print(
        "  Threshold count : 101"
    )
    print(
        "  Coverage monotonicity: PASS"
    )
    print(
        "  τ=0.00 coverage: "
        f"{first['coverage']:.6f}"
    )
    print(
        "  τ=1.00 accepted: "
        f"{last['accepted_count']:,}"
    )
    print()

    # --------------------------------------------------------
    # 4.12 Write characterization CSV
    # --------------------------------------------------------

    write_csv(
        D3_RESULTS_CSV,
        rows,
    )

    # --------------------------------------------------------
    # 4.13 Write characterization JSON
    # --------------------------------------------------------

    results_payload = {
        "phase": "D3",
        "version": "v002",
        "experiment": "selective_prediction_characterization",
        "status": "EXECUTED",
        "objective": (
            "Characterize the complete D-Select "
            "coverage-risk relationship."
        ),
        "scientific_scope": {
            "characterization_only": True,
            "universal_optimal_threshold_claim": False,
            "r_max": None,
            "c_min": None,
        },
        "frozen_d2_inputs": {
            "dcal_count": EXPECTED_DCAL_COUNT,
            "dselect_count": EXPECTED_DSELECT_COUNT,
            "temperature": EXPECTED_TEMPERATURE,
            "calibration_method": "temperature_scaling",
            "d2_calibrated_outputs": str(
                D2_CALIBRATED_OUTPUTS.relative_to(ROOT)
            ),
            "d2_calibrated_outputs_sha256": d2_outputs_hash,
            "d2_metrics_sha256": d2_metrics_hash,
            "d2_manifest_sha256": d2_manifest_hash,
        },
        "development_population": {
            "name": "D-Select",
            "count": total_count,
            "dtest_accessed": False,
        },
        "decision_signal": {
            "type": "calibrated_top1_probability",
            "prediction": "argmax calibrated probability",
            "confidence": "max calibrated probability",
        },
        "threshold_grid": {
            "start": THRESHOLD_START,
            "end": THRESHOLD_END,
            "step": THRESHOLD_STEP,
            "count": len(rows),
            "generation_rule": (
                "0.00 through 1.00 inclusive at 0.01 increments"
            ),
            "frozen_before_result_inspection": True,
            "result_adaptive": False,
        },
        "baseline": {
            "dselect_accuracy": overall_accuracy,
        },
        "metrics": {
            "primary": [
                "coverage",
                "selective_risk",
                "selective_accuracy",
            ],
            "supporting": [
                "threshold",
                "accepted_count",
                "abstained_count",
                "abstention_rate",
                "accepted_error_count",
            ],
        },
        "results": rows,
        "decision_rule": {
            "threshold_selected": False,
            "frozen_threshold": None,
            "reason": (
                "D3 v002 is characterization-only. "
                "No universal operating threshold is claimed."
            ),
        },
        "human_review": {
            "evaluated": False,
            "routing_rule": None,
        },
        "open_set": {
            "evaluated": False,
            "rule": None,
        },
        "boundaries": {
            "dtest_accessed": False,
            "d2_modified": False,
            "d2_refit": False,
            "cbase_retrained": False,
            "phase_c_modified": False,
            "threshold_selected": False,
            "coverage_target_selected": False,
            "human_review_rule_selected": False,
            "open_set_rule_selected": False,
        },
        "created_at_utc": utc_now(),
    }

    write_json(
        D3_RESULTS_JSON,
        results_payload,
    )

    # --------------------------------------------------------
    # 4.14 Write execution manifest
    # --------------------------------------------------------

    manifest = {
        "phase": "D3",
        "version": "v002",
        "experiment": "selective_prediction_characterization",
        "status": "EXECUTED",
        "seed": SEED,
        "d2_source": str(
            D2_CALIBRATED_OUTPUTS.relative_to(ROOT)
        ),
        "d2_source_sha256": d2_outputs_hash,
        "d3_config": str(
            D3_CONFIG_JSON.relative_to(ROOT)
        ),
        "d3_config_sha256": d3_config_hash,
        "d3_results_json": str(
            D3_RESULTS_JSON.relative_to(ROOT)
        ),
        "d3_results_csv": str(
            D3_RESULTS_CSV.relative_to(ROOT)
        ),
        "dcal_count": EXPECTED_DCAL_COUNT,
        "dselect_count": EXPECTED_DSELECT_COUNT,
        "temperature": EXPECTED_TEMPERATURE,
        "threshold_grid": {
            "start": THRESHOLD_START,
            "end": THRESHOLD_END,
            "step": THRESHOLD_STEP,
            "count": len(threshold_grid),
            "frozen_before_result_inspection": True,
            "result_adaptive": False,
        },
        "objective": "characterization_only",
        "r_max": None,
        "c_min": None,
        "threshold_selected": False,
        "dtest_accessed": False,
        "d2_modified": False,
        "phase_c_modified": False,
        "created_at_utc": utc_now(),
    }

    write_json(
        D3_MANIFEST_JSON,
        manifest,
    )

    # --------------------------------------------------------
    # 4.15 Final execution summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("D3 RESULTS")
    print("=" * 60)

    print(
        f"D-Select records       : {total_count:,}"
    )

    print(
        f"Overall accuracy       : {overall_accuracy:.6f}"
    )

    print(
        f"Threshold evaluations  : {len(rows)}"
    )

    print(
        "Threshold range        : "
        f"{THRESHOLD_START:.2f}–{THRESHOLD_END:.2f}"
    )

    print(
        "Threshold step         : "
        f"{THRESHOLD_STEP:.2f}"
    )

    print(
        "Threshold selected     : NO"
    )

    print(
        "R_max selected         : NO"
    )

    print(
        "C_min selected         : NO"
    )

    print(
        "D-Test accessed        : NO"
    )

    print(
        "D2 modified            : NO"
    )

    print(
        "Phase-C modified       : NO"
    )

    print()
    print("Artifacts:")
    print(
        f"  {D3_RESULTS_JSON}"
    )
    print(
        f"  {D3_RESULTS_CSV}"
    )
    print(
        f"  {D3_CONFIG_JSON}"
    )
    print(
        f"  {D3_MANIFEST_JSON}"
    )

    print()
    print("=" * 60)
    print("D3 SELECTIVE CHARACTERIZATION COMPLETE")
    print("=" * 60)
    print()
    print(
        "The complete D-Select coverage-risk relationship "
        "was characterized."
    )
    print(
        "No threshold or operating point was selected."
    )
    print(
        "No R_max/C_min was invented."
    )
    print(
        "D-Test was not accessed."
    )
    print(
        "D2 artifacts were not modified."
    )
    print(
        "Phase-C artifacts were not modified."
    )
    print()
    print(
        "Next step: inspect the D3 characterization outputs "
        "and perform the D3 integrity audit."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )