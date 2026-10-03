from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path.cwd()

SEED = 20260827
EXPECTED_DSELECT_COUNT = 34958
EXPECTED_CLASSES = 549
EXPECTED_CORRECT = 33280
EXPECTED_ERRORS = 1678
EXPECTED_ACCURACY = 0.9519995423079124

EXPECTED_D2_NPZ_SHA256 = (
    "78aa5de731107cd47a2149d6f85261d9f831a9894261cf612edeb05cc9a4f94f"
)
EXPECTED_D2_METRICS_SHA256 = (
    "a52238c1aac16de8605681da38742fdc10af7d673006477666689b2566d8e0d2"
)
EXPECTED_D2_MANIFEST_SHA256 = (
    "7052fd3d621bab8a2615a453a6f47f064e840a84d6c10a0ce3d7acbda026a214"
)

D2_NPZ = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibrated_validation_outputs.npz"
D2_METRICS = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibration_metrics.json"
D2_MANIFEST = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibration_manifest.json"

D3_CONFIG = ROOT / "configs" / "phase_d" / "d3_v002_characterization_config.json"
D3_CURVE = ROOT / "data" / "models" / "abo" / "phase_d" / "d3" / "development" / "d3_coverage_risk_curve.csv"

D4_CURVE = ROOT / "data" / "models" / "abo" / "phase_d" / "d4" / "development" / "d4_review_routing_curve.csv"

D5_CONFIG = ROOT / "configs" / "phase_d" / "d5_v001_reliability_error_analysis_config.json"
D5_RESULTS = ROOT / "data" / "models" / "abo" / "phase_d" / "d5" / "development" / "d5_reliability_error_analysis_results.json"
D5_BINS = ROOT / "data" / "models" / "abo" / "phase_d" / "d5" / "development" / "d5_confidence_error_bins.csv"
D5_CLASSES = ROOT / "data" / "models" / "abo" / "phase_d" / "d5" / "development" / "d5_class_error_analysis.csv"
D5_ROUTING = ROOT / "data" / "models" / "abo" / "phase_d" / "d5" / "development" / "d5_routing_error_analysis.csv"
D5_MANIFEST = ROOT / "data" / "models" / "abo" / "phase_d" / "d5" / "development" / "d5_reliability_error_analysis_manifest.json"

DTEST_DIR = ROOT / "data" / "models" / "abo" / "phase_d" / "dtest"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"load JSON {path}", str(e))
        return None


passes = 0
warnings = 0
failures = 0


def ok(label, detail=""):
    global passes
    passes += 1
    print(f"  PASS: {label}" + (f" {detail}" if detail else ""))


def fail(label, detail=""):
    global failures
    failures += 1
    print(f"  FAIL: {label}" + (f" {detail}" if detail else ""))


def warn(label, detail=""):
    global warnings
    warnings += 1
    print(f"  WARNING: {label}" + (f" {detail}" if detail else ""))


def require(path, label):
    if path.exists():
        ok(f"exists: {path.relative_to(ROOT)}")
        return True
    fail(f"exists: {path.relative_to(ROOT)}", label)
    return False


def get(obj, paths):
    if not isinstance(obj, dict):
        return None
    for path in paths:
        cur = obj
        found = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                found = False
                break
        if found:
            return cur
    return None


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "pass"}
    return False


print("=" * 72)
print("D5 v001 ACCEPTANCE / INTEGRITY AUDIT")
print("=" * 72)
print()
print("Audit only:")
print("  - no D5 rerun")
print("  - no threshold selection")
print("  - no R_max/C_min selection")
print("  - no human-performance evaluation")
print("  - no D-Test access")
print("  - no D2/D3/D4/Phase-C modification")
print()

# 1. Required artifacts
print("1. REQUIRED ARTIFACTS")
required_paths = [
    D5_CONFIG, D5_RESULTS, D5_BINS, D5_CLASSES, D5_ROUTING, D5_MANIFEST,
    D2_NPZ, D2_METRICS, D2_MANIFEST, D3_CONFIG, D3_CURVE, D4_CURVE,
]
for p in required_paths:
    require(p, "required artifact missing")

# 2. D-Test boundary
print()
print("2. D-TEST BOUNDARY")
if not DTEST_DIR.exists():
    ok("D-Test namespace absent", str(DTEST_DIR))
else:
    fail("D-Test namespace absent", str(DTEST_DIR))

# 3. Load JSON artifacts
print()
print("3. LOAD ARTIFACTS")
d5_config = load_json(D5_CONFIG)
d5_results = load_json(D5_RESULTS)
d5_manifest = load_json(D5_MANIFEST)
d2_metrics = load_json(D2_METRICS)
d2_manifest = load_json(D2_MANIFEST)
d3_config = load_json(D3_CONFIG)

for name, obj in [
    ("D5 configuration", d5_config),
    ("D5 results", d5_results),
    ("D5 manifest", d5_manifest),
    ("D2 metrics", d2_metrics),
    ("D2 manifest", d2_manifest),
    ("D3 configuration", d3_config),
]:
    if obj is not None:
        ok(f"{name} loads")
    else:
        fail(f"{name} loads")

# 4. Frozen upstream hashes
print()
print("4. FROZEN UPSTREAM ARTIFACT HASHES")
for p, expected, label in [
    (D2_NPZ, EXPECTED_D2_NPZ_SHA256, "D2 calibrated outputs"),
    (D2_METRICS, EXPECTED_D2_METRICS_SHA256, "D2 metrics"),
    (D2_MANIFEST, EXPECTED_D2_MANIFEST_SHA256, "D2 manifest"),
]:
    if p.exists():
        observed = sha256(p)
        if observed == expected:
            ok(f"{label} SHA-256", f"observed={observed}")
        else:
            fail(f"{label} SHA-256", f"observed={observed}")

# 5. D5 configuration
print()
print("5. D5 CONFIGURATION")
if isinstance(d5_config, dict):
    phase = get(d5_config, ["phase", "experiment.phase"])
    if str(phase).upper() == "D5":
        ok("phase identity", f"observed={phase}")
    else:
        fail("phase identity", f"observed={phase}")

    version = get(d5_config, ["version", "experiment.version"])
    if version == "v001":
        ok("version identity", f"observed={version}")
    else:
        fail("version identity", f"observed={version}")

    seed = get(d5_config, ["seed"])
    if seed == SEED:
        ok("seed", f"observed={seed}")
    else:
        fail("seed", f"observed={seed}")

    method = get(d5_config, ["confidence_bins.method"])
    if method == "equal_width":
        ok("confidence-bin method", f"observed={method}")
    else:
        fail("confidence-bin method", f"observed={method}")

    count = get(d5_config, ["confidence_bins.count"])
    if count == 15:
        ok("confidence-bin count", f"observed={count}")
    else:
        fail("confidence-bin count", f"observed={count}")

    start = get(d5_config, ["confidence_bins.start"])
    end = get(d5_config, ["confidence_bins.end"])
    width = get(d5_config, ["confidence_bins.width"])
    adaptive = get(d5_config, ["confidence_bins.result_adaptive"])

    ok("confidence-bin start", f"observed={start}") if start == 0.0 else fail("confidence-bin start", f"observed={start}")
    ok("confidence-bin end", f"observed={end}") if end == 1.0 else fail("confidence-bin end", f"observed={end}")
    if width is not None and abs(float(width) - (1.0 / 15.0)) < 1e-12:
        ok("confidence-bin width", f"observed={width}")
    else:
        fail("confidence-bin width", f"observed={width}")
    if adaptive is False:
        ok("result-adaptive=false")
    else:
        fail("result-adaptive=false", f"observed={adaptive}")

    boundary = get(d5_config, ["boundary"])
    for field in [
        "threshold_selected", "r_max_selected", "c_min_selected",
        "human_performance_evaluated", "dtest_accessed",
        "d2_modified", "d3_modified", "d4_modified", "phase_c_modified",
    ]:
        value = get(boundary, [field])
        if value is False:
            ok(f"boundary {field}=false")
        else:
            fail(f"boundary {field}=false", f"observed={value}")

# 6. D2 D-Select output integrity
print()
print("6. D2 CALIBRATED D-SELECT INPUT")
record_ids = labels = probs = None
try:
    with np.load(D2_NPZ, allow_pickle=False) as data:
        required_keys = [
            "dselect_record_ids", "dselect_labels",
            "dselect_raw_logits", "dselect_calibrated_probabilities",
        ]
        for key in required_keys:
            if key in data.files:
                ok(f"D2 NPZ key: {key}")
            else:
                fail(f"D2 NPZ key: {key}")

        record_ids = data["dselect_record_ids"]
        labels = data["dselect_labels"].astype(np.int64)
        probs = data["dselect_calibrated_probabilities"].astype(np.float64)

        ok("D-Select count", f"observed={len(record_ids)}") if len(record_ids) == EXPECTED_DSELECT_COUNT else fail("D-Select count", f"observed={len(record_ids)}")
        ok("label count", f"observed={len(labels)}") if len(labels) == EXPECTED_DSELECT_COUNT else fail("label count", f"observed={len(labels)}")
        ok("probability shape", f"observed={probs.shape}") if probs.shape == (EXPECTED_DSELECT_COUNT, EXPECTED_CLASSES) else fail("probability shape", f"observed={probs.shape}")
        ok("probabilities finite") if np.isfinite(probs).all() else fail("probabilities finite")
        ok("probabilities in [0,1]") if ((probs >= 0) & (probs <= 1)).all() else fail("probabilities in [0,1]")
        dev = float(np.max(np.abs(probs.sum(axis=1) - 1.0)))
        ok("probability row sums", f"max deviation={dev:.9g}") if dev <= 1e-5 else fail("probability row sums", f"max deviation={dev:.9g}")
        unique = len(np.unique(record_ids))
        ok("record IDs unique", f"unique={unique}") if unique == EXPECTED_DSELECT_COUNT else fail("record IDs unique", f"unique={unique}")
except Exception as e:
    fail("load D2 calibrated NPZ", str(e))

# 7. Reproduce primary D5 population metrics from frozen D2 input
print()
print("7. PRIMARY D5 METRIC REPRODUCTION")
if probs is not None and labels is not None:
    pred = np.argmax(probs, axis=1)
    top1 = probs[np.arange(len(probs)), pred]
    correct = pred == labels
    errors = ~correct

    correct_n = int(correct.sum())
    error_n = int(errors.sum())
    acc = float(correct.mean())

    ok("correct count", f"observed={correct_n}") if correct_n == EXPECTED_CORRECT else fail("correct count", f"observed={correct_n}")
    ok("error count", f"observed={error_n}") if error_n == EXPECTED_ERRORS else fail("error count", f"observed={error_n}")
    ok("accuracy", f"observed={acc:.12f}") if abs(acc - EXPECTED_ACCURACY) < 1e-12 else fail("accuracy", f"observed={acc:.12f}")

    # D5-specific diagnostics from execution output
    hc = int(((top1 >= 0.90) & errors).sum())
    lc = int(((top1 < 0.60) & correct).sum())
    ok("high-confidence errors >=0.90", f"observed={hc}") if hc == 124 else fail("high-confidence errors >=0.90", f"observed={hc}")
    ok("low-confidence correct <0.60", f"observed={lc}") if lc == 766 else fail("low-confidence correct <0.60", f"observed={lc}")

# 8. Confidence-bin CSV
print()
print("8. CONFIDENCE-BIN ANALYSIS")
bin_rows = []
try:
    with D5_BINS.open("r", newline="", encoding="utf-8") as f:
        bin_rows = list(csv.DictReader(f))

    ok("15 confidence-bin rows", f"rows={len(bin_rows)}") if len(bin_rows) == 15 else fail("15 confidence-bin rows", f"rows={len(bin_rows)}")

    required_cols = [
        "bin_index", "lower_bound", "upper_bound", "interval", "count",
        "correct", "errors", "coverage_fraction",
        "mean_calibrated_confidence", "empirical_accuracy",
        "empirical_error_rate", "confidence_correctness_gap",
    ]
    cols = set(bin_rows[0].keys()) if bin_rows else set()
    for col in required_cols:
        if col in cols:
            ok(f"bin column: {col}")
        else:
            fail(f"bin column: {col}")

    if len(bin_rows) == 15 and all(c in cols for c in required_cols):
        counts = np.array([int(r["count"]) for r in bin_rows])
        corrects = np.array([int(r["correct"]) for r in bin_rows])
        errors = np.array([int(r["errors"]) for r in bin_rows])
        lows = np.array([float(r["lower_bound"]) for r in bin_rows])
        highs = np.array([float(r["upper_bound"]) for r in bin_rows])
        means = np.array([float(r["mean_calibrated_confidence"]) for r in bin_rows])
        accs = np.array([float(r["empirical_accuracy"]) for r in bin_rows])
        ers = np.array([float(r["empirical_error_rate"]) for r in bin_rows])
        gaps = np.array([float(r["confidence_correctness_gap"]) for r in bin_rows])

        ok("bin indices 0..14", "deterministic")
        if np.array_equal(np.arange(15), np.array([int(r["bin_index"]) for r in bin_rows])):
            ok("bin indices exact")
        else:
            fail("bin indices exact")

        expected_lows = np.linspace(0, 14 / 15, 15)
        expected_highs = np.linspace(1 / 15, 1, 15)
        ok("bin lower bounds deterministic") if np.allclose(lows, expected_lows, atol=1e-12) else fail("bin lower bounds deterministic")
        ok("bin upper bounds deterministic") if np.allclose(highs, expected_highs, atol=1e-12) else fail("bin upper bounds deterministic")

        ok("bin counts sum to D-Select", f"sum={counts.sum()}") if counts.sum() == EXPECTED_DSELECT_COUNT else fail("bin counts sum to D-Select", f"sum={counts.sum()}")
        ok("bin correct+errors accounting") if np.array_equal(corrects + errors, counts) else fail("bin correct+errors accounting")
        ok("bin correct sum", f"sum={corrects.sum()}") if corrects.sum() == EXPECTED_CORRECT else fail("bin correct sum", f"sum={corrects.sum()}")
        ok("bin error sum", f"sum={errors.sum()}") if errors.sum() == EXPECTED_ERRORS else fail("bin error sum", f"sum={errors.sum()}")
        ok("bin metrics finite") if np.isfinite(np.r_[means, accs, ers, gaps]).all() else fail("bin metrics finite")
        ok("bin empirical accuracy in [0,1]") if ((accs >= 0) & (accs <= 1)).all() else fail("bin empirical accuracy in [0,1]")
        ok("bin empirical error rate in [0,1]") if ((ers >= 0) & (ers <= 1)).all() else fail("bin empirical error rate in [0,1]")

except Exception as e:
    fail("confidence-bin CSV processing", str(e))

# 9. Class-level CSV
print()
print("9. CLASS-LEVEL ERROR ANALYSIS")
class_rows = []
try:
    with D5_CLASSES.open("r", newline="", encoding="utf-8") as f:
        class_rows = list(csv.DictReader(f))
    ok("class CSV loads", f"rows={len(class_rows)}")

    required_cols = [
        "class_id", "support", "correct", "errors",
        "accuracy", "error_rate", "mean_confidence",
        "high_confidence_error_count_ge_0_90",
        "high_confidence_error_rate_ge_0_90",
    ]
    cols = set(class_rows[0].keys()) if class_rows else set()
    for col in required_cols:
        if col in cols:
            ok(f"class column: {col}")
        else:
            fail(f"class column: {col}")

    if class_rows and all(c in cols for c in required_cols):
        support = np.array([int(r["support"]) for r in class_rows])
        corrects = np.array([int(r["correct"]) for r in class_rows])
        errors = np.array([int(r["errors"]) for r in class_rows])
        accs = np.array([float(r["accuracy"]) for r in class_rows])
        ers = np.array([float(r["error_rate"]) for r in class_rows])
        conf = np.array([float(r["mean_confidence"]) for r in class_rows])

        ok("class accounting correct+errors=support") if np.array_equal(corrects + errors, support) else fail("class accounting correct+errors=support")
        ok("class support sum=D-Select") if support.sum() == EXPECTED_DSELECT_COUNT else fail("class support sum=D-Select", f"sum={support.sum()}")
        ok("class correct sum=D-Select correct") if corrects.sum() == EXPECTED_CORRECT else fail("class correct sum=D-Select correct", f"sum={corrects.sum()}")
        ok("class error sum=D-Select errors") if errors.sum() == EXPECTED_ERRORS else fail("class error sum=D-Select errors", f"sum={errors.sum()}")
        ok("class metrics finite") if np.isfinite(np.r_[accs, ers, conf]).all() else fail("class metrics finite")
        ok("class accuracy/error rates bounded") if ((accs >= 0) & (accs <= 1)).all() and ((ers >= 0) & (ers <= 1)).all() else fail("class accuracy/error rates bounded")
        ok("classes represented", f"count={len(class_rows)}") if len(class_rows) == 290 else fail("classes represented", f"count={len(class_rows)}")

except Exception as e:
    fail("class-level CSV processing", str(e))

# 10. Routing analysis
print()
print("10. FROZEN D3/D4 ROUTING ANALYSIS")
routing_rows = []
try:
    with D5_ROUTING.open("r", newline="", encoding="utf-8") as f:
        routing_rows = list(csv.DictReader(f))
    ok("D5 routing CSV loads", f"rows={len(routing_rows)}") if len(routing_rows) == 101 else fail("D5 routing CSV loads", f"rows={len(routing_rows)}")

    required_cols = [
        "threshold", "total_records", "accepted_count", "review_count",
        "coverage", "review_rate", "accepted_errors", "reviewed_errors",
        "model_error_capture_rate", "selective_risk",
    ]
    cols = set(routing_rows[0].keys()) if routing_rows else set()
    for col in required_cols:
        if col in cols:
            ok(f"routing column: {col}")
        else:
            fail(f"routing column: {col}")

    if len(routing_rows) == 101 and all(c in cols for c in required_cols):
        thresholds = np.array([float(r["threshold"]) for r in routing_rows])
        expected_grid = np.round(np.arange(0, 1.0001, 0.01), 2)
        accepted = np.array([int(r["accepted_count"]) for r in routing_rows])
        review = np.array([int(r["review_count"]) for r in routing_rows])
        coverage = np.array([float(r["coverage"]) for r in routing_rows])
        review_rate = np.array([float(r["review_rate"]) for r in routing_rows])
        accepted_errors = np.array([int(r["accepted_errors"]) for r in routing_rows])
        reviewed_errors = np.array([int(r["reviewed_errors"]) for r in routing_rows])
        capture = np.array([float(r["model_error_capture_rate"]) for r in routing_rows])
        selective_risk = np.array([float(r["selective_risk"]) for r in routing_rows])

        ok("routing threshold grid exact") if np.allclose(thresholds, expected_grid, atol=1e-9) else fail("routing threshold grid exact")
        ok("routing total records constant") if np.all(np.array([int(r["total_records"]) for r in routing_rows]) == EXPECTED_DSELECT_COUNT) else fail("routing total records constant")
        ok("accepted+review=total") if np.all(accepted + review == EXPECTED_DSELECT_COUNT) else fail("accepted+review=total")
        ok("accepted monotonic") if np.all(np.diff(accepted) <= 0) else fail("accepted monotonic")
        ok("review monotonic") if np.all(np.diff(review) >= 0) else fail("review monotonic")
        ok("coverage monotonic") if np.all(np.diff(coverage) <= 1e-12) else fail("coverage monotonic")
        ok("tau=0 coverage=1") if abs(coverage[0] - 1.0) <= 1e-12 else fail("tau=0 coverage=1")
        ok("tau=1 accepted=8") if accepted[-1] == 8 else fail("tau=1 accepted=8", f"observed={accepted[-1]}")
        ok("routing metrics finite") if np.isfinite(np.r_[coverage, review_rate, capture, selective_risk]).all() else fail("routing metrics finite")
        ok("routing reviewed errors bounded") if np.all((reviewed_errors >= 0) & (reviewed_errors <= EXPECTED_ERRORS)) else fail("routing reviewed errors bounded")
        ok("capture matches reviewed errors / total errors") if np.allclose(capture, reviewed_errors / EXPECTED_ERRORS, atol=1e-9) else fail("capture matches reviewed errors / total errors")

except Exception as e:
    fail("routing CSV processing", str(e))
# 11. Results/manifest semantic boundary
print()
print("11. D5 RESULTS / MANIFEST BOUNDARY")
if isinstance(d5_results, dict):
    pop = get(d5_results, ["population"])
    if isinstance(pop, dict):
        n = pop.get("count")
        c = pop.get("correct")
        e = pop.get("errors")
        a = pop.get("accuracy")
        ok("results population count", f"observed={n}") if n == EXPECTED_DSELECT_COUNT else fail("results population count", f"observed={n}")
        ok("results correct", f"observed={c}") if c == EXPECTED_CORRECT else fail("results correct", f"observed={c}")
        ok("results errors", f"observed={e}") if e == EXPECTED_ERRORS else fail("results errors", f"observed={e}")
        ok("results accuracy", f"observed={a}") if a is not None and abs(float(a) - EXPECTED_ACCURACY) < 1e-12 else fail("results accuracy", f"observed={a}")
    else:
        fail("results population block present")

    boundary = get(d5_results, ["boundary"])
    if isinstance(boundary, dict):
        for field in [
            "threshold_selected", "r_max_selected", "c_min_selected",
            "human_performance_evaluated", "dtest_accessed",
            "d2_modified", "d3_modified", "d4_modified", "phase_c_modified",
        ]:
            v = boundary.get(field)
            ok(f"results boundary {field}=false") if v is False else fail(f"results boundary {field}=false", f"observed={v}")
    else:
        fail("results boundary block present")

if isinstance(d5_manifest, dict):
    phase = get(d5_manifest, ["phase"])
    version = get(d5_manifest, ["version"])
    ok("manifest phase D5", f"observed={phase}") if str(phase).upper() == "D5" else fail("manifest phase D5", f"observed={phase}")
    ok("manifest version v001", f"observed={version}") if version == "v001" else fail("manifest version v001", f"observed={version}")

# 12. Frozen upstream artifacts remain unchanged
print()
print("12. FROZEN-UPSTREAM IMMUTABILITY")
for p, expected, label in [
    (D2_NPZ, EXPECTED_D2_NPZ_SHA256, "D2 calibrated outputs"),
    (D2_METRICS, EXPECTED_D2_METRICS_SHA256, "D2 metrics"),
    (D2_MANIFEST, EXPECTED_D2_MANIFEST_SHA256, "D2 manifest"),
]:
    observed = sha256(p)
    ok(f"{label} unchanged") if observed == expected else fail(f"{label} unchanged", f"observed={observed}")

# ---------------------------------------------------------------------
print()
print("=" * 72)
print("D5 ACCEPTANCE AUDIT SUMMARY")
print("=" * 72)
print(f"Passes   : {passes}")
print(f"Warnings : {warnings}")
print(f"Failures : {failures}")

if failures == 0:
    print()
    print("D5 ACCEPTANCE AUDIT: PASS")
    print("D5 v001 reliability error-analysis artifacts passed integrity audit.")
    print()
    print("D5 may now be marked:")
    print("  EXECUTED / VERIFIED / FROZEN")
    print()
    print("Scientific boundary:")
    print("  No deployment threshold was selected.")
    print("  No R_max/C_min criterion was selected.")
    print("  No human-review effectiveness was evaluated.")
    print("  No open-set detection was performed.")
    print("  D-Test was not accessed.")
    print("  D2 remains frozen.")
    print("  D3 remains frozen.")
    print("  D4 remains frozen.")
    print("  Phase C remains frozen.")
else:
    print()
    print("D5 ACCEPTANCE AUDIT: FAIL")
    print("Do NOT freeze D5 yet.")
    print("Review failing checks only; do not modify frozen upstream artifacts.")
