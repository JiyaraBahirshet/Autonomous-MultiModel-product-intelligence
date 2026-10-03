"""
D4 v001 — Human-Review Routing Characterization

Frozen contract:
- D-Select only: 34,958 records
- frozen D2 calibrated top-1 probabilities
- frozen D3 threshold grid: 0.00..1.00, step 0.01
- characterization only; no threshold selection
- no D-Test
- no D2/D3/Phase-C modification
- no claim about actual human reviewer performance
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SEED = 20260827
EXPECTED_DSELECT_COUNT = 34958
N_CLASSES = 549
THRESHOLD_START = 0.00
THRESHOLD_END = 1.00
THRESHOLD_STEP = 0.01

D2_NPZ = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibrated_validation_outputs.npz"
D2_METRICS = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibration_metrics.json"
D2_MANIFEST = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibration_manifest.json"

D3_CONFIG = ROOT / "configs" / "phase_d" / "d3_v002_characterization_config.json"

D4_DIR = ROOT / "data" / "models" / "abo" / "phase_d" / "d4" / "development"
D4_DIR.mkdir(parents=True, exist_ok=True)
D4_RESULTS = D4_DIR / "d4_human_review_routing_results.json"
D4_CURVE = D4_DIR / "d4_review_routing_curve.csv"
D4_MANIFEST = D4_DIR / "d4_human_review_routing_manifest.json"

D4_CONFIG = ROOT / "configs" / "phase_d" / "d4_v001_human_review_routing_config.json"
D4_CONFIG.parent.mkdir(parents=True, exist_ok=True)

DTEST_DIR = ROOT / "data" / "models" / "abo" / "phase_d" / "dtest"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str):
    raise RuntimeError("D4 INTEGRITY FAILURE: " + msg)


print("=" * 72)
print("D4 v001 HUMAN-REVIEW ROUTING CHARACTERIZATION")
print("=" * 72)
print()
print("D4 only:")
print("  - frozen D2 calibrated D-Select outputs")
print("  - frozen D3 threshold grid")
print("  - routing characterization")
print()
print("D4 does NOT:")
print("  - select a deployment threshold")
print("  - select R_max/C_min")
print("  - measure human reviewer performance")
print("  - access D-Test")
print("  - modify D2")
print("  - modify D3")
print("  - modify Phase C")
print()

# ---------------------------------------------------------------------
# 1. Boundary checks
# ---------------------------------------------------------------------
if DTEST_DIR.exists():
    fail(f"D-Test namespace exists: {DTEST_DIR}")
print("D-TEST BOUNDARY: PASS")
print("  No Phase-D D-Test namespace detected.")
print()

# ---------------------------------------------------------------------
# 2. Required frozen inputs
# ---------------------------------------------------------------------
for p in (D2_NPZ, D2_METRICS, D2_MANIFEST, D3_CONFIG):
    if not p.exists():
        fail(f"required input missing: {p}")

with D2_METRICS.open("r", encoding="utf-8") as f:
    d2_metrics = json.load(f)
with D2_MANIFEST.open("r", encoding="utf-8") as f:
    d2_manifest = json.load(f)
with D3_CONFIG.open("r", encoding="utf-8") as f:
    d3_config = json.load(f)

print("FROZEN INPUTS: PASS")
print(f"  D2 NPZ SHA-256     : {sha256(D2_NPZ)}")
print(f"  D2 metrics SHA-256 : {sha256(D2_METRICS)}")
print(f"  D2 manifest SHA-256: {sha256(D2_MANIFEST)}")
print()

# ---------------------------------------------------------------------
# 3. Verify D3's frozen deterministic threshold grid
# ---------------------------------------------------------------------
def get_nested(obj, *keys):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

# Support the exact config schema used by D3, while refusing to invent
# a new grid. Values must resolve to the frozen D3 rule.
start = get_nested(d3_config, "threshold_grid", "start")
end = get_nested(d3_config, "threshold_grid", "end")
step = get_nested(d3_config, "threshold_grid", "step")
count = get_nested(d3_config, "threshold_grid", "count")

if start is None:
    start = get_nested(d3_config, "threshold_start")
if end is None:
    end = get_nested(d3_config, "threshold_end")
if step is None:
    step = get_nested(d3_config, "threshold_step")
if count is None:
    count = get_nested(d3_config, "threshold_count")

if not all(v is not None for v in (start, end, step, count)):
    fail("could not resolve frozen D3 threshold-grid fields")

if not (
    math.isclose(float(start), THRESHOLD_START, abs_tol=1e-12)
    and math.isclose(float(end), THRESHOLD_END, abs_tol=1e-12)
    and math.isclose(float(step), THRESHOLD_STEP, abs_tol=1e-12)
    and int(count) == 101
):
    fail("D3 threshold grid does not match frozen 0.00..1.00 step 0.01 rule")

thresholds = np.round(
    np.arange(THRESHOLD_START, THRESHOLD_END + THRESHOLD_STEP / 2, THRESHOLD_STEP),
    2,
)

if len(thresholds) != 101:
    fail("internal deterministic threshold grid has unexpected length")

print("FROZEN D3 THRESHOLD GRID: PASS")
print("  0.00 to 1.00 inclusive, step 0.01, count 101")
print()

# ---------------------------------------------------------------------
# 4. Freeze D4 configuration BEFORE observing D-Select results
# ---------------------------------------------------------------------
config = {
    "phase": "D4",
    "version": "v001",
    "experiment": "human_review_routing_characterization",
    "seed": SEED,
    "population": {
        "source": "frozen D2 D-Select",
        "record_count": EXPECTED_DSELECT_COUNT,
    },
    "signal": {
        "name": "frozen calibrated top-1 probability",
        "source_artifact": str(D2_NPZ.relative_to(ROOT)),
    },
    "routing_rule": {
        "machine_accept": "p >= tau",
        "human_review": "p < tau",
    },
    "threshold_grid": {
        "source": "frozen D3 v002 characterization grid",
        "start": THRESHOLD_START,
        "end": THRESHOLD_END,
        "step": THRESHOLD_STEP,
        "count": 101,
        "result_adaptive": False,
    },
    "objective": "characterize routing workload, machine coverage, residual machine risk, and model-error capture",
    "threshold_selected": False,
    "r_max_selected": False,
    "c_min_selected": False,
    "human_performance_evaluated": False,
    "dtest_accessed": False,
    "d2_modified": False,
    "d3_modified": False,
    "phase_c_modified": False,
}

with D4_CONFIG.open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("D4 CONFIGURATION: WRITTEN")
print(f"  Path: {D4_CONFIG}")
print(f"  SHA-256: {sha256(D4_CONFIG)}")
print()

# ---------------------------------------------------------------------
# 5. Load frozen D2 calibrated D-Select output
# ---------------------------------------------------------------------
with np.load(D2_NPZ, allow_pickle=False) as data:
    required = {
        "dselect_record_ids",
        "dselect_labels",
        "dselect_calibrated_probabilities",
    }
    missing = sorted(required - set(data.files))
    if missing:
        fail("D2 NPZ missing required keys: " + ", ".join(missing))

    record_ids = data["dselect_record_ids"]
    labels = data["dselect_labels"]
    probs = data["dselect_calibrated_probabilities"]

if len(record_ids) != EXPECTED_DSELECT_COUNT:
    fail(f"D-Select count {len(record_ids)} != {EXPECTED_DSELECT_COUNT}")
if labels.shape != (EXPECTED_DSELECT_COUNT,):
    fail(f"D-Select labels shape unexpected: {labels.shape}")
if probs.shape != (EXPECTED_DSELECT_COUNT, N_CLASSES):
    fail(f"D-Select probability shape unexpected: {probs.shape}")
if not np.all(np.isfinite(probs)):
    fail("D-Select probabilities contain non-finite values")
if np.any(probs < 0) or np.any(probs > 1):
    fail("D-Select probabilities outside [0,1]")

row_sums = probs.sum(axis=1)
max_dev = float(np.max(np.abs(row_sums - 1.0)))
if max_dev > 1e-5:
    fail(f"probability rows do not sum to 1; max deviation={max_dev}")

predictions = np.argmax(probs, axis=1)
top1 = probs[np.arange(len(probs)), predictions]
correct = predictions == labels

overall_accuracy = float(np.mean(correct))

print("D2 CALIBRATED D-SELECT OUTPUT: PASS")
print(f"  records       : {len(record_ids)}")
print(f"  probabilities : {probs.shape}")
print(f"  row-sum max deviation: {max_dev:.3e}")
print(f"  overall accuracy: {overall_accuracy:.12f}")
print()

# ---------------------------------------------------------------------
# 6. Characterize routing at every frozen threshold
# ---------------------------------------------------------------------
rows = []

total = len(top1)
total_errors = int(np.sum(~correct))

for tau in thresholds:
    accepted = top1 >= tau
    reviewed = ~accepted

    accepted_count = int(np.sum(accepted))
    review_count = int(np.sum(reviewed))

    accepted_correct = int(np.sum(accepted & correct))
    accepted_errors = int(np.sum(accepted & ~correct))

    reviewed_correct = int(np.sum(reviewed & correct))
    reviewed_errors = int(np.sum(reviewed & ~correct))

    coverage = accepted_count / total
    review_rate = review_count / total

    machine_accuracy = (
        accepted_correct / accepted_count if accepted_count > 0 else None
    )
    selective_risk = (
        accepted_errors / accepted_count if accepted_count > 0 else None
    )

    error_capture = (
        reviewed_errors / total_errors if total_errors > 0 else None
    )

    rows.append({
        "threshold": float(tau),
        "total_records": total,
        "accepted_count": accepted_count,
        "review_count": review_count,
        "coverage": float(coverage),
        "review_rate": float(review_rate),
        "accepted_correct": accepted_correct,
        "accepted_errors": accepted_errors,
        "reviewed_correct": reviewed_correct,
        "reviewed_errors": reviewed_errors,
        "machine_accepted_accuracy": machine_accuracy,
        "machine_accepted_risk": selective_risk,
        "model_error_capture_rate": error_capture,
    })

# Invariants
coverages = np.array([r["coverage"] for r in rows], dtype=float)
accepted_counts = np.array([r["accepted_count"] for r in rows], dtype=int)

if not np.all(np.diff(coverages) <= 1e-12):
    fail("coverage is not monotonically non-increasing")
if not np.all(np.diff(accepted_counts) <= 0):
    fail("accepted count is not monotonically non-increasing")
if not math.isclose(rows[0]["coverage"], 1.0, abs_tol=1e-12):
    fail("tau=0 coverage is not 1.0")
if rows[-1]["accepted_count"] != int(np.sum(top1 >= 1.0)):
    fail("tau=1 accepted count invariant failed")

print("D4 ROUTING INVARIANTS: PASS")
print(f"  threshold evaluations: {len(rows)}")
print(f"  coverage at tau=0    : {rows[0]['coverage']:.9f}")
print(f"  accepted at tau=1    : {rows[-1]['accepted_count']}")
print(f"  total model errors   : {total_errors}")
print()

# ---------------------------------------------------------------------
# 7. Save curve
# ---------------------------------------------------------------------
fieldnames = list(rows[0].keys())
with D4_CURVE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# ---------------------------------------------------------------------
# 8. Save results
# ---------------------------------------------------------------------
results = {
    "phase": "D4",
    "version": "v001",
    "status": "EXECUTED",
    "experiment": "human_review_routing_characterization",
    "population": {
        "name": "D-Select",
        "count": total,
        "overall_accuracy": overall_accuracy,
        "total_model_errors": total_errors,
    },
    "routing_signal": "frozen D2 calibrated top-1 probability",
    "threshold_grid": {
        "start": THRESHOLD_START,
        "end": THRESHOLD_END,
        "step": THRESHOLD_STEP,
        "count": 101,
        "source": "frozen D3 v002",
    },
    "threshold_selected": None,
    "r_max_selected": None,
    "c_min_selected": None,
    "human_performance_evaluated": False,
    "dtest_accessed": False,
    "d2_modified": False,
    "d3_modified": False,
    "phase_c_modified": False,
    "scientific_boundary": [
        "Routing workload and machine-side behavior are characterized.",
        "Model-error capture by the review queue is characterized.",
        "No actual human reviewer performance was evaluated.",
        "No deployment threshold was selected.",
        "No R_max/C_min criterion was selected.",
        "D-Test was not accessed.",
    ],
    "threshold_results": rows,
}

with D4_RESULTS.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------------
# 9. Save manifest
# ---------------------------------------------------------------------
manifest = {
    "phase": "D4",
    "version": "v001",
    "method": "human_review_routing_characterization",
    "seed": SEED,
    "inputs": {
        "d2_npz": str(D2_NPZ.relative_to(ROOT)),
        "d2_npz_sha256": sha256(D2_NPZ),
        "d2_metrics": str(D2_METRICS.relative_to(ROOT)),
        "d2_metrics_sha256": sha256(D2_METRICS),
        "d2_manifest": str(D2_MANIFEST.relative_to(ROOT)),
        "d2_manifest_sha256": sha256(D2_MANIFEST),
        "d3_config": str(D3_CONFIG.relative_to(ROOT)),
        "d3_config_sha256": sha256(D3_CONFIG),
    },
    "population": {
        "dselect_count": total,
    },
    "threshold_grid": {
        "start": THRESHOLD_START,
        "end": THRESHOLD_END,
        "step": THRESHOLD_STEP,
        "count": 101,
        "result_adaptive": False,
    },
    "boundary": {
        "threshold_selected": False,
        "r_max_selected": False,
        "c_min_selected": False,
        "human_performance_evaluated": False,
        "dtest_accessed": False,
        "d2_modified": False,
        "d3_modified": False,
        "phase_c_modified": False,
    },
    "outputs": {
        "results": str(D4_RESULTS.relative_to(ROOT)),
        "curve": str(D4_CURVE.relative_to(ROOT)),
        "config": str(D4_CONFIG.relative_to(ROOT)),
    },
    "status": "EXECUTED",
}

with D4_MANIFEST.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("=" * 72)
print("D4 RESULTS")
print("=" * 72)
print(f"D-Select records       : {total}")
print(f"Overall accuracy       : {overall_accuracy:.12f}")
print(f"Threshold evaluations  : 101")
print("Threshold range        : 0.00–1.00")
print("Threshold selected     : NO")
print("R_max selected         : NO")
print("C_min selected         : NO")
print("Human performance      : NOT EVALUATED")
print("D-Test accessed        : NO")
print("D2 modified            : NO")
print("D3 modified            : NO")
print("Phase-C modified       : NO")
print()
print("Artifacts:")
print(f"  {D4_RESULTS}")
print(f"  {D4_CURVE}")
print(f"  {D4_CONFIG}")
print(f"  {D4_MANIFEST}")
print()
print("=" * 72)
print("D4 HUMAN-REVIEW ROUTING CHARACTERIZATION COMPLETE")
print("=" * 72)
print()
print("The frozen D3 threshold grid was used without adaptation.")
print("No deployment threshold was selected.")
print("No R_max/C_min criterion was invented.")
print("No human-review effectiveness claim was made.")
print("D-Test was not accessed.")
print("D2/D3/Phase-C artifacts were not modified.")
