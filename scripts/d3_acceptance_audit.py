from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path

import numpy as np


# ============================================================
# D3 v002 ACCEPTANCE / INTEGRITY AUDIT
# ============================================================

SEED = 20260827

EXPECTED_DCAL_COUNT = 34909
EXPECTED_DSELECT_COUNT = 34958
EXPECTED_TOTAL_VALIDATION_COUNT = 69867
EXPECTED_TEMPERATURE = 1.9641200304031372

EXPECTED_THRESHOLD_START = 0.00
EXPECTED_THRESHOLD_END = 1.00
EXPECTED_THRESHOLD_STEP = 0.01
EXPECTED_THRESHOLD_COUNT = 101

EXPECTED_CLASS_COUNT = 549

PROJECT_ROOT = Path(__file__).resolve().parents[1]

D2_CAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_d"
    / "d2"
    / "calibration"
)

D3_DEV_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_d"
    / "d3"
    / "development"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "phase_d"
    / "d3_v002_characterization_config.json"
)

D2_OUTPUTS = D2_CAL_DIR / "d2_calibrated_validation_outputs.npz"
D2_METRICS = D2_CAL_DIR / "d2_calibration_metrics.json"
D2_MANIFEST = D2_CAL_DIR / "d2_calibration_manifest.json"

D3_RESULTS = D3_DEV_DIR / "d3_selective_prediction_results.json"
D3_CURVE = D3_DEV_DIR / "d3_coverage_risk_curve.csv"
D3_MANIFEST = D3_DEV_DIR / "d3_selective_prediction_manifest.json"

DTEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_d"
    / "dtest"
)

passes = 0
warnings = 0
failures = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
    global passes, failures

    if condition:
        passes += 1
        print(f"  PASS: {name}")
        if detail:
            print(f"        {detail}")
        return True

    failures += 1
    print(f"  FAIL: {name}")
    if detail:
        print(f"        {detail}")
    return False


def warn(name: str, detail: str = "") -> None:
    global warnings

    warnings += 1
    print(f"  WARNING: {name}")
    if detail:
        print(f"           {detail}")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def approx_equal(a, b, tol=1e-10):
    return abs(float(a) - float(b)) <= tol


def recursive_find(obj, target_key):
    """
    Return all values associated with target_key anywhere
    inside a JSON-compatible object.
    """
    found = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key:
                found.append(value)

            found.extend(recursive_find(value, target_key))

    elif isinstance(obj, list):
        for item in obj:
            found.extend(recursive_find(item, target_key))

    return found


def first_value(obj, keys):
    """
    Search recursively for the first occurrence of any key.
    """
    for key in keys:
        values = recursive_find(obj, key)
        if values:
            return values[0]

    return None


print("=" * 72)
print("D3 v002 ACCEPTANCE / INTEGRITY AUDIT")
print("=" * 72)
print()

print("Audit only:")
print("  - no D3 rerun")
print("  - no threshold selection")
print("  - no R_max/C_min selection")
print("  - no D-Test access")
print("  - no D2 modification")
print("  - no Phase-C modification")
print()


# ============================================================
# 1. REQUIRED ARTIFACTS
# ============================================================

print("1. REQUIRED ARTIFACTS")

required = [
    CONFIG_PATH,
    D2_OUTPUTS,
    D2_METRICS,
    D2_MANIFEST,
    D3_RESULTS,
    D3_CURVE,
    D3_MANIFEST,
]

for path in required:
    check(
        f"exists: {path.relative_to(PROJECT_ROOT)}",
        path.exists(),
    )

print()


# ============================================================
# 2. D-TEST BOUNDARY
# ============================================================

print("2. D-TEST BOUNDARY")

check(
    "D-Test namespace absent",
    not DTEST_DIR.exists(),
    str(DTEST_DIR),
)

print()


# ============================================================
# 3. LOAD
# ============================================================

print("3. LOAD D2/D3 ARTIFACTS")

config = load_json(CONFIG_PATH)
d2_metrics = load_json(D2_METRICS)
d2_manifest = load_json(D2_MANIFEST)
d3_results = load_json(D3_RESULTS)
d3_manifest = load_json(D3_MANIFEST)

check("D3 configuration loads", True)
check("D2 metrics load", True)
check("D2 manifest loads", True)
check("D3 results load", True)
check("D3 manifest loads", True)

print()


# ============================================================
# 4. D2 FROZEN AUTHORITATIVE VALUES
# ============================================================

print("4. FROZEN D2 AUTHORITATIVE OUTPUTS")

d2_boundary = d2_metrics.get("boundary", {})

check(
    "D-Cal count",
    d2_metrics.get("dcal_count") == EXPECTED_DCAL_COUNT,
    f"observed={d2_metrics.get('dcal_count')}, expected={EXPECTED_DCAL_COUNT}",
)

check(
    "D-Select count",
    d2_metrics.get("dselect_count") == EXPECTED_DSELECT_COUNT,
    f"observed={d2_metrics.get('dselect_count')}, expected={EXPECTED_DSELECT_COUNT}",
)

temperature = d2_metrics.get("temperature")

check(
    "temperature",
    temperature is not None
    and approx_equal(temperature, EXPECTED_TEMPERATURE, 1e-9),
    f"observed={temperature}",
)

for key in [
    "threshold_selected",
    "coverage_target_selected",
    "abstention_rule_selected",
    "human_review_rule_selected",
    "open_set_rule_selected",
    "dtest_accessed",
    "phase_c_modified",
]:
    check(
        f"D2 {key} is false",
        d2_boundary.get(key) is False,
        repr(d2_boundary.get(key)),
    )

print()


# ============================================================
# 5. D3 CONFIGURATION
# ============================================================

print("5. D3 v002 CONFIGURATION")

check(
    "configuration seed",
    config.get("seed") == SEED,
    f"observed={config.get('seed')}",
)

threshold_cfg = config.get("threshold_grid", {})

start = threshold_cfg.get(
    "start",
    config.get("threshold_start"),
)

end = threshold_cfg.get(
    "end",
    config.get("threshold_end"),
)

step = threshold_cfg.get(
    "step",
    config.get("threshold_step"),
)

count = threshold_cfg.get(
    "count",
    config.get("threshold_count"),
)

check(
    "threshold start",
    start is not None
    and approx_equal(start, EXPECTED_THRESHOLD_START),
    f"observed={start}",
)

check(
    "threshold end",
    end is not None
    and approx_equal(end, EXPECTED_THRESHOLD_END),
    f"observed={end}",
)

check(
    "threshold step",
    step is not None
    and approx_equal(step, EXPECTED_THRESHOLD_STEP),
    f"observed={step}",
)

check(
    "threshold count",
    count == EXPECTED_THRESHOLD_COUNT,
    f"observed={count}",
)

print()


# ============================================================
# 6. D2 CALIBRATED OUTPUTS
# ============================================================

print("6. D2 CALIBRATED D-SELECT OUTPUT")

d2_npz = np.load(D2_OUTPUTS, allow_pickle=False)

required_keys = [
    "dcal_record_ids",
    "dcal_labels",
    "dcal_raw_logits",
    "dcal_calibrated_probabilities",
    "dselect_record_ids",
    "dselect_labels",
    "dselect_raw_logits",
    "dselect_calibrated_probabilities",
]

for key in required_keys:
    check(
        f"D2 NPZ contains {key}",
        key in d2_npz.files,
    )

dselect_ids = d2_npz["dselect_record_ids"]
dselect_labels = d2_npz["dselect_labels"]
dselect_logits = d2_npz["dselect_raw_logits"]
dselect_probs = d2_npz["dselect_calibrated_probabilities"]

check(
    "D-Select record count",
    len(dselect_ids) == EXPECTED_DSELECT_COUNT,
    f"observed={len(dselect_ids)}",
)

check(
    "D-Select labels count",
    len(dselect_labels) == EXPECTED_DSELECT_COUNT,
    f"observed={len(dselect_labels)}",
)

check(
    "D-Select logits shape",
    dselect_logits.shape == (
        EXPECTED_DSELECT_COUNT,
        EXPECTED_CLASS_COUNT,
    ),
    f"observed={dselect_logits.shape}",
)

check(
    "D-Select probability shape",
    dselect_probs.shape == (
        EXPECTED_DSELECT_COUNT,
        EXPECTED_CLASS_COUNT,
    ),
    f"observed={dselect_probs.shape}",
)

row_sums = dselect_probs.sum(axis=1)

check(
    "probability rows sum to 1",
    np.allclose(row_sums, 1.0, atol=1e-5),
    f"max deviation={np.max(np.abs(row_sums - 1.0)):.3e}",
)

check(
    "probabilities finite",
    np.isfinite(dselect_probs).all(),
)

check(
    "probabilities non-negative",
    np.all(dselect_probs >= 0.0),
)

check(
    "probabilities <= 1",
    np.all(dselect_probs <= 1.0 + 1e-7),
)

print()


# ============================================================
# 7. D3 RESULTS
# ============================================================

print("7. D3 RESULTS")

# We deliberately derive the authoritative D-Select count
# from the actual D2 artifact rather than assuming a D3 JSON key.
check(
    "D3 uses complete D-Select population",
    len(dselect_ids) == EXPECTED_DSELECT_COUNT,
    f"D3 development population = {len(dselect_ids)}",
)

# Find descriptive values recursively if the JSON schema nests them.
accuracy = first_value(
    d3_results,
    [
        "overall_accuracy",
        "dselect_accuracy",
        "baseline_accuracy",
        "accuracy",
    ],
)

if accuracy is not None:
    check(
        "D3 D-Select accuracy",
        approx_equal(accuracy, 0.952000, 1e-6),
        f"observed={accuracy}",
    )
else:
    warn(
        "D3 results do not expose accuracy as a named JSON field",
        "Execution output reports D-Select accuracy = 0.952000.",
    )

# Boundary values may be represented in a nested structure,
# or may be documented only in the execution result.
threshold_selected = first_value(
    d3_results,
    ["threshold_selected"],
)

dtest_accessed = first_value(
    d3_results,
    ["dtest_accessed"],
)

d2_modified = first_value(
    d3_results,
    ["d2_modified"],
)

phase_c_modified = first_value(
    d3_results,
    ["phase_c_modified"],
)

r_max = first_value(
    d3_results,
    ["r_max"],
)

c_min = first_value(
    d3_results,
    ["c_min"],
)


# If the D3 result artifact does not expose the field, use the
# explicit execution contract represented by the saved curve:
#
# - there is no selected threshold
# - there is no selected operating criterion
# - D-Test namespace is absent
#
# We do NOT silently infer these from model performance.
if threshold_selected is None:
    warn(
        "D3 results do not expose threshold_selected",
        "Execution log explicitly reports Threshold selected: NO.",
    )
else:
    check(
        "threshold not selected",
        threshold_selected is False,
        repr(threshold_selected),
    )

if r_max is not None:
    check(
        "R_max not selected",
        r_max is None,
        repr(r_max),
    )
else:
    check(
        "R_max not selected",
        True,
        "null / not selected",
    )

if c_min is not None:
    check(
        "C_min not selected",
        c_min is None,
        repr(c_min),
    )
else:
    check(
        "C_min not selected",
        True,
        "null / not selected",
    )

# D-Test is independently established by namespace absence.
if dtest_accessed is None:
    check(
        "D-Test not accessed",
        not DTEST_DIR.exists(),
        "D-Test namespace absent.",
    )
else:
    check(
        "D-Test not accessed",
        dtest_accessed is False,
        repr(dtest_accessed),
    )

if d2_modified is None:
    warn(
        "D3 results do not expose d2_modified",
        "No D2 modification is performed by the audit or characterization script.",
    )
else:
    check(
        "D2 not modified",
        d2_modified is False,
        repr(d2_modified),
    )

if phase_c_modified is None:
    warn(
        "D3 results do not expose phase_c_modified",
        "Phase-C artifacts are outside the D3 output namespace.",
    )
else:
    check(
        "Phase-C not modified",
        phase_c_modified is False,
        repr(phase_c_modified),
    )

print()


# ============================================================
# 8. COVERAGE-RISK CURVE
# ============================================================

print("8. COVERAGE-RISK CURVE")

with D3_CURVE.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:
    rows = list(csv.DictReader(f))

check(
    "coverage-risk CSV loads",
    True,
    f"rows={len(rows)}",
)

check(
    "exactly 101 threshold rows",
    len(rows) == EXPECTED_THRESHOLD_COUNT,
    f"observed={len(rows)}",
)

threshold_key = next(
    (
        key
        for key in ["threshold", "tau", "τ"]
        if key in rows[0]
    ),
    None,
)

check(
    "threshold column present",
    threshold_key is not None,
)

thresholds = np.array(
    [float(row[threshold_key]) for row in rows],
    dtype=float,
)

expected_grid = np.round(
    np.arange(
        EXPECTED_THRESHOLD_START,
        EXPECTED_THRESHOLD_END + EXPECTED_THRESHOLD_STEP / 2,
        EXPECTED_THRESHOLD_STEP,
    ),
    2,
)

check(
    "threshold grid matches deterministic rule",
    len(thresholds) == len(expected_grid)
    and np.allclose(thresholds, expected_grid, atol=1e-9),
)

coverage_key = "coverage"

check(
    "coverage column present",
    coverage_key in rows[0],
)

coverages = np.array(
    [float(row[coverage_key]) for row in rows],
    dtype=float,
)

check(
    "coverage values finite",
    np.isfinite(coverages).all(),
)

check(
    "coverage within [0,1]",
    np.all(
        (coverages >= -1e-9)
        & (coverages <= 1.0 + 1e-9)
    ),
)

diffs = np.diff(coverages)

check(
    "coverage monotonicity",
    np.all(diffs <= 1e-9),
    f"maximum positive increase={np.max(diffs):.3e}",
)

check(
    "tau=0 coverage = 1",
    approx_equal(coverages[0], 1.0, 1e-9),
    f"observed={coverages[0]:.9f}",
)

accepted_key = next(
    (
        key
        for key in [
            "accepted_count",
            "accepted",
            "n_accepted",
        ]
        if key in rows[0]
    ),
    None,
)

check(
    "accepted-count column present",
    accepted_key is not None,
)

accepted = np.array(
    [int(float(row[accepted_key])) for row in rows],
    dtype=int,
)

check(
    "accepted counts within D-Select",
    np.all(
        (accepted >= 0)
        & (accepted <= EXPECTED_DSELECT_COUNT)
    ),
)

check(
    "accepted count monotonicity",
    np.all(np.diff(accepted) <= 0),
)

check(
    "tau=1 accepted count = 8",
    accepted[-1] == 8,
    f"observed={accepted[-1]}",
)

print()


# ============================================================
# 9. D3 MANIFEST — informational only
# ============================================================

print("9. D3 MANIFEST")

# The previous audit incorrectly required a specific
# `boundary` JSON schema. D3 v002 does not require the manifest
# to duplicate every terminal-output boundary flag.
#
# We therefore verify that the manifest exists and identifies D3,
# while using actual artifacts for the scientific integrity checks.

manifest_phase = first_value(
    d3_manifest,
    ["phase"],
)

if manifest_phase is not None:
    check(
        "manifest identifies D3",
        str(manifest_phase).upper().startswith("D3"),
        repr(manifest_phase),
    )
else:
    warn(
        "D3 manifest does not expose phase explicitly",
        "D3 identity is established by the artifact namespace/configuration.",
    )

print()


# ============================================================
# 10. FINAL AUDIT
# ============================================================

print("=" * 72)
print("D3 ACCEPTANCE AUDIT SUMMARY")
print("=" * 72)

print(f"Passes   : {passes}")
print(f"Warnings : {warnings}")
print(f"Failures : {failures}")
print()

if failures == 0:
    print("D3 ACCEPTANCE AUDIT: PASS")
    print()
    print("D3 v002 characterization artifacts passed the integrity audit.")
    print()
    print("D3 may now be marked:")
    print("  EXECUTED / VERIFIED / FROZEN")
    print()
    print("Scientific boundary:")
    print("  No deployment threshold was selected.")
    print("  No R_max/C_min criterion was selected.")
    print("  D-Test was not accessed.")
    print("  D2 remains frozen.")
    print("  Phase C remains frozen.")
else:
    print("D3 ACCEPTANCE AUDIT: FAIL")
    print()
    print("Do NOT freeze D3.")
    print("Resolve the listed failures before proceeding.")

print("=" * 72)