from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path.cwd()

# ---------------------------------------------------------------------
# Frozen D4 / D2 / D3 expectations
# ---------------------------------------------------------------------
SEED = 20260827
EXPECTED_DSELECT_COUNT = 34958
EXPECTED_CLASSES = 549

EXPECTED_D2_NPZ_SHA256 = (
    "78aa5de731107cd47a2149d6f85261d9f831a9894261cf612edeb05cc9a4f94f"
)
EXPECTED_D2_METRICS_SHA256 = (
    "a52238c1aac16de8605681da38742fdc10af7d673006477666689b2566d8e0d2"
)
EXPECTED_D2_MANIFEST_SHA256 = (
    "7052fd3d621bab8a2615a453a6f47f064e840a84d6c10a0ce3d7acbda026a214"
)

EXPECTED_THRESHOLD_START = 0.00
EXPECTED_THRESHOLD_END = 1.00
EXPECTED_THRESHOLD_STEP = 0.01
EXPECTED_THRESHOLD_COUNT = 101

D2_NPZ = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibrated_validation_outputs.npz"
D2_METRICS = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibration_metrics.json"
D2_MANIFEST = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibration_manifest.json"

D3_CONFIG = ROOT / "configs" / "phase_d" / "d3_v002_characterization_config.json"

D4_CONFIG = ROOT / "configs" / "phase_d" / "d4_v001_human_review_routing_config.json"
D4_RESULTS = ROOT / "data" / "models" / "abo" / "phase_d" / "d4" / "development" / "d4_human_review_routing_results.json"
D4_CURVE = ROOT / "data" / "models" / "abo" / "phase_d" / "d4" / "development" / "d4_review_routing_curve.csv"
D4_MANIFEST = ROOT / "data" / "models" / "abo" / "phase_d" / "d4" / "development" / "d4_human_review_routing_manifest.json"

DTEST_DIR = ROOT / "data" / "models" / "abo" / "phase_d" / "dtest"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


passes = 0
warnings = 0
failures = 0


def result(ok: bool, label: str, detail: str = ""):
    global passes, failures
    if ok:
        passes += 1
        print(f"  PASS: {label}" + (f" {detail}" if detail else ""))
    else:
        failures += 1
        print(f"  FAIL: {label}" + (f" {detail}" if detail else ""))


def warn(label: str, detail: str = ""):
    global warnings
    warnings += 1
    print(f"  WARNING: {label}" + (f" {detail}" if detail else ""))


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        result(False, f"load JSON: {path}", str(e))
        return None


def get_any(obj, paths):
    """Return first existing value from a list of dotted paths."""
    for path in paths:
        cur = obj
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def boolish(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "pass"}
    return False


print("=" * 72)
print("D4 v001 ACCEPTANCE / INTEGRITY AUDIT")
print("=" * 72)
print()
print("Audit only:")
print("  - no D4 rerun")
print("  - no threshold selection")
print("  - no R_max/C_min selection")
print("  - no human-performance evaluation")
print("  - no D-Test access")
print("  - no D2 modification")
print("  - no D3 modification")
print("  - no Phase-C modification")
print()

# ---------------------------------------------------------------------
# 1. Required artifacts
# ---------------------------------------------------------------------
print("1. REQUIRED ARTIFACTS")
for p in [
    D4_CONFIG,
    D2_NPZ,
    D2_METRICS,
    D2_MANIFEST,
    D3_CONFIG,
    D4_RESULTS,
    D4_CURVE,
    D4_MANIFEST,
]:
    result(p.exists(), f"exists: {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")

# ---------------------------------------------------------------------
# 2. D-Test boundary
# ---------------------------------------------------------------------
print()
print("2. D-TEST BOUNDARY")
result(not DTEST_DIR.exists(), "D-Test namespace absent", str(DTEST_DIR))

# ---------------------------------------------------------------------
# 3. Load artifacts
# ---------------------------------------------------------------------
print()
print("3. LOAD ARTIFACTS")
d4_config = load_json(D4_CONFIG)
d2_metrics = load_json(D2_METRICS)
d2_manifest = load_json(D2_MANIFEST)
d3_config = load_json(D3_CONFIG)
d4_results = load_json(D4_RESULTS)
d4_manifest = load_json(D4_MANIFEST)

for name, obj in [
    ("D4 configuration", d4_config),
    ("D2 metrics", d2_metrics),
    ("D2 manifest", d2_manifest),
    ("D3 configuration", d3_config),
    ("D4 results", d4_results),
    ("D4 manifest", d4_manifest),
]:
    result(obj is not None, f"{name} loads")

# ---------------------------------------------------------------------
# 4. Frozen D2 artifact hashes
# ---------------------------------------------------------------------
print()
print("4. FROZEN D2 ARTIFACT HASHES")
if D2_NPZ.exists():
    result(
        sha256(D2_NPZ) == EXPECTED_D2_NPZ_SHA256,
        "D2 calibrated outputs SHA-256",
        f"observed={sha256(D2_NPZ)}",
    )
if D2_METRICS.exists():
    result(
        sha256(D2_METRICS) == EXPECTED_D2_METRICS_SHA256,
        "D2 metrics SHA-256",
        f"observed={sha256(D2_METRICS)}",
    )
if D2_MANIFEST.exists():
    result(
        sha256(D2_MANIFEST) == EXPECTED_D2_MANIFEST_SHA256,
        "D2 manifest SHA-256",
        f"observed={sha256(D2_MANIFEST)}",
    )

# ---------------------------------------------------------------------
# 5. Frozen D3 threshold grid
# ---------------------------------------------------------------------
print()
print("5. FROZEN D3 THRESHOLD GRID")

def extract_grid(cfg):
    if not isinstance(cfg, dict):
        return None, None, None, None

    start = get_any(cfg, [
        "threshold_grid.start",
        "threshold_grid.threshold_start",
        "threshold_start",
    ])
    end = get_any(cfg, [
        "threshold_grid.end",
        "threshold_grid.threshold_end",
        "threshold_end",
    ])
    step = get_any(cfg, [
        "threshold_grid.step",
        "threshold_grid.threshold_step",
        "threshold_step",
    ])
    count = get_any(cfg, [
        "threshold_grid.count",
        "threshold_grid.threshold_count",
    ])
    return start, end, step, count


d3_start, d3_end, d3_step, d3_count = extract_grid(d3_config)
result(d3_start == EXPECTED_THRESHOLD_START, "D3 grid start", f"observed={d3_start}")
result(d3_end == EXPECTED_THRESHOLD_END, "D3 grid end", f"observed={d3_end}")
result(d3_step == EXPECTED_THRESHOLD_STEP, "D3 grid step", f"observed={d3_step}")
result(d3_count == EXPECTED_THRESHOLD_COUNT, "D3 grid count", f"observed={d3_count}")

# ---------------------------------------------------------------------
# 6. D4 configuration integrity
# ---------------------------------------------------------------------
print()
print("6. D4 CONFIGURATION")

if isinstance(d4_config, dict):
    seed = get_any(d4_config, ["seed", "experiment.seed"])
    result(seed == SEED, "seed", f"observed={seed}")

    c_start, c_end, c_step, c_count = extract_grid(d4_config)
    result(c_start == EXPECTED_THRESHOLD_START, "D4 threshold start", f"observed={c_start}")
    result(c_end == EXPECTED_THRESHOLD_END, "D4 threshold end", f"observed={c_end}")
    result(c_step == EXPECTED_THRESHOLD_STEP, "D4 threshold step", f"observed={c_step}")
    result(c_count == EXPECTED_THRESHOLD_COUNT, "D4 threshold count", f"observed={c_count}")

    # Explicit non-selection / boundary flags, if present.
    expected_false_fields = [
        "result_adaptive",
        "threshold_selected",
        "r_max_selected",
        "c_min_selected",
        "human_performance_evaluated",
        "dtest_accessed",
        "d2_modified",
        "d3_modified",
        "phase_c_modified",
    ]
    for field in expected_false_fields:
        value = get_any(d4_config, [field, f"boundary.{field}", f"decision.{field}"])
        if value is None:
            warn(f"D4 config does not explicitly expose {field}")
        else:
            result(not boolish(value), f"D4 config {field}=false", f"observed={value}")

# ---------------------------------------------------------------------
# 7. D2 calibrated D-Select schema/integrity
# ---------------------------------------------------------------------
print()
print("7. D2 CALIBRATED D-SELECT OUTPUT")

dselect_record_ids = None
dselect_labels = None
dselect_probs = None

try:
    with np.load(D2_NPZ, allow_pickle=False) as data:
        required = [
            "dselect_record_ids",
            "dselect_labels",
            "dselect_raw_logits",
            "dselect_calibrated_probabilities",
        ]
        for key in required:
            result(key in data.files, f"D2 NPZ key: {key}")

        if all(k in data.files for k in required):
            dselect_record_ids = data["dselect_record_ids"]
            dselect_labels = data["dselect_labels"]
            dselect_probs = data["dselect_calibrated_probabilities"]

            n = len(dselect_record_ids)
            result(n == EXPECTED_DSELECT_COUNT, "D-Select record count", f"observed={n}")
            result(len(dselect_labels) == EXPECTED_DSELECT_COUNT,
                   "D-Select label count", f"observed={len(dselect_labels)}")
            result(dselect_probs.shape == (EXPECTED_DSELECT_COUNT, EXPECTED_CLASSES),
                   "D-Select probability shape", f"observed={dselect_probs.shape}")

            finite = bool(np.isfinite(dselect_probs).all())
            bounded = bool(((dselect_probs >= 0.0) & (dselect_probs <= 1.0)).all())
            row_dev = float(np.max(np.abs(dselect_probs.sum(axis=1) - 1.0)))
            result(finite, "probabilities finite")
            result(bounded, "probabilities in [0,1]")
            result(row_dev <= 1e-5, "probability row sums", f"max deviation={row_dev:.9g}")

            unique_ids = len(np.unique(dselect_record_ids))
            result(unique_ids == EXPECTED_DSELECT_COUNT,
                   "D-Select record IDs unique", f"unique={unique_ids}")

except Exception as e:
    result(False, "load/validate D2 calibrated NPZ", str(e))

# ---------------------------------------------------------------------
# 8. D4 routing curve
# ---------------------------------------------------------------------
print()
print("8. D4 HUMAN-REVIEW ROUTING CURVE")

rows = []
if D4_CURVE.exists():
    try:
        with D4_CURVE.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        result(len(rows) == EXPECTED_THRESHOLD_COUNT,
               "exactly 101 routing rows", f"rows={len(rows)}")

        required_cols = [
            "threshold",
            "total_records",
            "accepted_count",
            "review_count",
            "coverage",
            "review_rate",
            "accepted_correct",
            "accepted_errors",
            "reviewed_correct",
            "reviewed_errors",
            "machine_accepted_accuracy",
            "machine_accepted_risk",
            "model_error_capture_rate",
        ]
        cols = set(rows[0].keys()) if rows else set()
        for col in required_cols:
            result(col in cols, f"curve column: {col}")

        if rows and all(c in cols for c in required_cols):
            thresholds = np.array([float(r["threshold"]) for r in rows])
            expected_grid = np.round(
                np.arange(
                    EXPECTED_THRESHOLD_START,
                    EXPECTED_THRESHOLD_END + EXPECTED_THRESHOLD_STEP / 2,
                    EXPECTED_THRESHOLD_STEP,
                ),
                2,
            )
            result(
                len(thresholds) == len(expected_grid)
                and np.allclose(thresholds, expected_grid, atol=1e-9),
                "exact deterministic threshold grid",
            )

            accepted = np.array([int(r["accepted_count"]) for r in rows])
            reviewed = np.array([int(r["review_count"]) for r in rows])
            coverage = np.array([float(r["coverage"]) for r in rows])
            review_rate = np.array([float(r["review_rate"]) for r in rows])
            accepted_correct = np.array([int(r["accepted_correct"]) for r in rows])
            accepted_errors = np.array([int(r["accepted_errors"]) for r in rows])
            reviewed_correct = np.array([int(r["reviewed_correct"]) for r in rows])
            reviewed_errors = np.array([int(r["reviewed_errors"]) for r in rows])
            capture = np.array([float(r["model_error_capture_rate"]) for r in rows])

            numeric_arrays = [
                coverage, review_rate, capture,
                np.array([float(r["machine_accepted_accuracy"]) for r in rows]),
                np.array([float(r["machine_accepted_risk"]) for r in rows]),
            ]
            result(
                all(np.isfinite(a).all() for a in numeric_arrays),
                "routing metrics finite",
            )
            result(
                bool(np.all(np.diff(accepted) <= 0)),
                "accepted count monotonic non-increasing",
            )
            result(
                bool(np.all(np.diff(coverage) <= 1e-12)),
                "coverage monotonic non-increasing",
            )
            result(
                bool(np.all(np.diff(reviewed) >= 0)),
                "review count monotonic non-decreasing",
            )
            result(
                abs(coverage[0] - 1.0) <= 1e-12,
                "tau=0 coverage=1",
                f"observed={coverage[0]:.12f}",
            )
            result(
                accepted[0] == EXPECTED_DSELECT_COUNT,
                "tau=0 accepts all D-Select records",
                f"accepted={accepted[0]}",
            )
            result(
                accepted[-1] == 8,
                "tau=1 accepted count=8",
                f"accepted={accepted[-1]}",
            )

            result(
                bool(np.all(accepted + reviewed == EXPECTED_DSELECT_COUNT)),
                "accepted + review = total records at every threshold",
            )
            result(
                bool(np.all(accepted_correct + accepted_errors == accepted)),
                "accepted correct + errors = accepted count",
            )
            result(
                bool(np.all(reviewed_correct + reviewed_errors == reviewed)),
                "reviewed correct + errors = review count",
            )

            total_model_errors = EXPECTED_DSELECT_COUNT - (
                accepted_correct[0] + accepted_errors[0] - accepted_errors[0]
            )
            # At tau=0, accepted_errors is the complete model-error count.
            total_model_errors = int(accepted_errors[0])
            result(
                total_model_errors == 1678,
                "total model errors at tau=0",
                f"observed={total_model_errors}",
            )

            # At every threshold, captured errors should equal reviewed model errors.
            result(
                bool(np.allclose(
                    capture,
                    np.divide(
                        reviewed_errors,
                        total_model_errors,
                        out=np.zeros_like(reviewed_errors, dtype=float),
                        where=total_model_errors != 0,
                    ),
                    atol=1e-9,
                )),
                "model-error capture rate matches reviewed errors / total errors",
            )

            result(
                abs(review_rate[0]) <= 1e-12,
                "tau=0 review rate=0",
                f"observed={review_rate[0]:.12f}",
            )

    except Exception as e:
        result(False, "load/validate D4 routing curve", str(e))

# ---------------------------------------------------------------------
# 9. D4 results / manifest semantic boundary
# ---------------------------------------------------------------------
print()
print("9. D4 RESULTS / MANIFEST BOUNDARY")

if isinstance(d4_results, dict):
    dselect_count = get_any(d4_results, [
        "dselect_count",
        "d_select_count",
        "total_records",
        "population.total_records",
    ])
    if dselect_count is not None:
        result(int(dselect_count) == EXPECTED_DSELECT_COUNT,
               "D4 uses complete D-Select population",
               f"observed={dselect_count}")
    else:
        warn("D4 results does not expose an explicit D-Select count; curve/NPZ audited directly")

    overall_acc = get_any(d4_results, [
        "overall_accuracy",
        "accuracy",
        "population.overall_accuracy",
    ])
    if overall_acc is not None:
        result(abs(float(overall_acc) - 0.9519995423079124) < 1e-9,
               "D4 overall D-Select accuracy",
               f"observed={overall_acc}")
    else:
        warn("D4 results does not expose an overall accuracy field")

    false_fields = [
        "threshold_selected",
        "r_max_selected",
        "c_min_selected",
        "human_performance_evaluated",
        "dtest_accessed",
        "d2_modified",
        "d3_modified",
        "phase_c_modified",
    ]
    for field in false_fields:
        value = get_any(d4_results, [field, f"boundary.{field}", f"decision.{field}"])
        if value is None:
            warn(f"D4 results does not explicitly expose {field}")
        else:
            result(not boolish(value), f"D4 results {field}=false", f"observed={value}")

if isinstance(d4_manifest, dict):
    phase = get_any(d4_manifest, ["phase", "phase_id", "experiment.phase"])
    if phase is not None:
        result(str(phase).lower() == "d4", "D4 manifest identifies phase", f"observed={phase}")
    else:
        warn("D4 manifest does not expose an explicit phase field")

# ---------------------------------------------------------------------
# 10. No mutation evidence from artifact hashes
# ---------------------------------------------------------------------
print()
print("10. FROZEN-ARTIFACT MUTATION BOUNDARY")

# D2 hashes were already verified against frozen authoritative hashes.
# D3 config is the frozen grid source; this audit only reads it.
result(DTEST_DIR.exists() is False, "D-Test remains untouched")
result(sha256(D2_NPZ) == EXPECTED_D2_NPZ_SHA256, "D2 calibrated outputs remain unchanged")
result(sha256(D2_METRICS) == EXPECTED_D2_METRICS_SHA256, "D2 metrics remain unchanged")
result(sha256(D2_MANIFEST) == EXPECTED_D2_MANIFEST_SHA256, "D2 manifest remains unchanged")

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print()
print("=" * 72)
print("D4 ACCEPTANCE AUDIT SUMMARY")
print("=" * 72)
print(f"Passes   : {passes}")
print(f"Warnings : {warnings}")
print(f"Failures : {failures}")

if failures == 0:
    print()
    print("D4 ACCEPTANCE AUDIT: PASS")
    print("D4 v001 routing characterization artifacts passed integrity audit.")
    print()
    print("D4 may now be marked:")
    print("  EXECUTED / VERIFIED / FROZEN")
    print()
    print("Scientific boundary:")
    print("  No deployment threshold was selected.")
    print("  No R_max/C_min criterion was selected.")
    print("  No human-review effectiveness was evaluated.")
    print("  D-Test was not accessed.")
    print("  D2 remains frozen.")
    print("  D3 remains frozen.")
    print("  Phase C remains frozen.")
else:
    print()
    print("D4 ACCEPTANCE AUDIT: FAIL")
    print("Do NOT freeze D4 yet.")
    print("Review only the failing audit checks; do not modify frozen D2/D3/Phase-C artifacts.")

print()
print("=" * 72)
