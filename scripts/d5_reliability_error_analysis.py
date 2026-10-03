from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path.cwd()

SEED = 20260827
EXPECTED_DSELECT_COUNT = 34958
N_CLASSES = 549

EXPECTED_D2_NPZ_SHA256 = (
    "78aa5de731107cd47a2149d6f85261d9f831a9894261cf612edeb05cc9a4f94f"
)
EXPECTED_D2_METRICS_SHA256 = (
    "a52238c1aac16de8605681da38742fdc10af7d673006477666689b2566d8e0d2"
)
EXPECTED_D2_MANIFEST_SHA256 = (
    "7052fd3d621bab8a2615a453a6f47f064e840a84d6c10a0ce3d7acbda026a214"
)

EXPECTED_D3_CONFIG = ROOT / "configs" / "phase_d" / "d3_v002_characterization_config.json"

D2_NPZ = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibrated_validation_outputs.npz"
D2_METRICS = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibration_metrics.json"
D2_MANIFEST = ROOT / "data" / "models" / "abo" / "phase_d" / "d2" / "calibration" / "d2_calibration_manifest.json"

D3_RESULTS = ROOT / "data" / "models" / "abo" / "phase_d" / "d3" / "development" / "d3_selective_prediction_results.json"
D3_CURVE = ROOT / "data" / "models" / "abo" / "phase_d" / "d3" / "development" / "d3_coverage_risk_curve.csv"
D3_CONFIG = EXPECTED_D3_CONFIG

D4_RESULTS = ROOT / "data" / "models" / "abo" / "phase_d" / "d4" / "development" / "d4_human_review_routing_results.json"
D4_CURVE = ROOT / "data" / "models" / "abo" / "phase_d" / "d4" / "development" / "d4_review_routing_curve.csv"

DTEST_DIR = ROOT / "data" / "models" / "abo" / "phase_d" / "dtest"

CONFIG = ROOT / "configs" / "phase_d" / "d5_v001_reliability_error_analysis_config.json"
OUT_DIR = ROOT / "data" / "models" / "abo" / "phase_d" / "d5" / "development"
RESULTS = OUT_DIR / "d5_reliability_error_analysis_results.json"
CONF_BINS = OUT_DIR / "d5_confidence_error_bins.csv"
CLASS_ERRORS = OUT_DIR / "d5_class_error_analysis.csv"
ROUTING_ERRORS = OUT_DIR / "d5_routing_error_analysis.csv"
MANIFEST = OUT_DIR / "d5_reliability_error_analysis_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(x):
    return bool(np.isfinite(x).all())


def safe_div(a, b):
    return float(a / b) if b else 0.0


def find_threshold_row(rows, tau):
    for row in rows:
        if abs(float(row["threshold"]) - tau) <= 1e-9:
            return row
    return None


print("=" * 72)
print("D5 v001 RELIABILITY ERROR ANALYSIS")
print("=" * 72)
print()
print("D5 only:")
print("  - frozen D2 calibrated D-Select outputs")
print("  - frozen D3 selective characterization")
print("  - frozen D4 routing characterization")
print("  - deterministic 15-bin confidence analysis")
print("  - diagnostic error analysis")
print()
print("D5 does NOT:")
print("  - retrain or modify C-Base")
print("  - refit D2 calibration")
print("  - modify D3/D4")
print("  - select a deployment threshold")
print("  - select R_max/C_min")
print("  - evaluate human performance")
print("  - access D-Test")
print()

# ---------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------
if DTEST_DIR.exists():
    raise RuntimeError(
        f"D-Test namespace exists; D5 execution is blocked: {DTEST_DIR}"
    )
print("D-TEST BOUNDARY: PASS")
print("  No Phase-D D-Test namespace detected.")
print()

# ---------------------------------------------------------------------
# Required inputs
# ---------------------------------------------------------------------
for p, label in [
    (D2_NPZ, "D2 calibrated outputs"),
    (D2_METRICS, "D2 metrics"),
    (D2_MANIFEST, "D2 manifest"),
    (D3_CONFIG, "D3 configuration"),
    (D3_CURVE, "D3 coverage-risk curve"),
    (D4_CURVE, "D4 routing curve"),
]:
    require(p, label)

if sha256(D2_NPZ) != EXPECTED_D2_NPZ_SHA256:
    raise RuntimeError("D2 calibrated outputs SHA-256 mismatch.")
if sha256(D2_METRICS) != EXPECTED_D2_METRICS_SHA256:
    raise RuntimeError("D2 metrics SHA-256 mismatch.")
if sha256(D2_MANIFEST) != EXPECTED_D2_MANIFEST_SHA256:
    raise RuntimeError("D2 manifest SHA-256 mismatch.")

print("FROZEN INPUTS: PASS")
print(f"  D2 NPZ SHA-256     : {sha256(D2_NPZ)}")
print(f"  D2 metrics SHA-256 : {sha256(D2_METRICS)}")
print(f"  D2 manifest SHA-256: {sha256(D2_MANIFEST)}")
print()

# ---------------------------------------------------------------------
# Freeze D5 configuration BEFORE inspecting D5 results.
# ---------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG.parent.mkdir(parents=True, exist_ok=True)

config = {
    "phase": "D5",
    "version": "v001",
    "experiment": "reliability_error_analysis",
    "seed": SEED,
    "population": {
        "source": "frozen D2 calibrated D-Select outputs",
        "count": EXPECTED_DSELECT_COUNT,
        "test_access": False,
    },
    "confidence_bins": {
        "method": "equal_width",
        "start": 0.0,
        "end": 1.0,
        "count": 15,
        "width": 1.0 / 15.0,
        "result_adaptive": False,
    },
    "analyses": [
        "confidence_error_bins",
        "high_confidence_error_characterization",
        "low_confidence_correct_characterization",
        "class_level_error_analysis",
        "selective_error_characterization_from_frozen_D3_curve",
        "human_review_routing_error_characterization_from_frozen_D4_curve",
    ],
    "allowed_conclusions": [
        "descriptive confidence-error relationships",
        "descriptive residual-error patterns",
        "descriptive class-level error concentration",
        "descriptive routing workload and model-error capture",
    ],
    "prohibited_decisions": [
        "deployment_threshold_selection",
        "R_max_selection",
        "C_min_selection",
        "human_performance_evaluation",
        "open_set_detection",
        "autonomy_claim",
    ],
    "boundary": {
        "threshold_selected": False,
        "r_max_selected": False,
        "c_min_selected": False,
        "human_performance_evaluated": False,
        "dtest_accessed": False,
        "d2_modified": False,
        "d3_modified": False,
        "d4_modified": False,
        "phase_c_modified": False,
    },
}

CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")
print("D5 CONFIGURATION: WRITTEN")
print(f"  Path: {CONFIG}")
print(f"  SHA-256: {sha256(CONFIG)}")
print("  Confidence bins: 15 equal-width bins over [0,1]")
print("  Result-adaptive: NO")
print()

# ---------------------------------------------------------------------
# Load D2 calibrated outputs
# ---------------------------------------------------------------------
with np.load(D2_NPZ, allow_pickle=False) as data:
    required = [
        "dselect_record_ids",
        "dselect_labels",
        "dselect_raw_logits",
        "dselect_calibrated_probabilities",
    ]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"Missing D2 NPZ keys: {missing}")

    record_ids = data["dselect_record_ids"]
    labels = data["dselect_labels"].astype(np.int64)
    probs = data["dselect_calibrated_probabilities"].astype(np.float64)

if len(record_ids) != EXPECTED_DSELECT_COUNT:
    raise RuntimeError("D-Select count mismatch.")
if probs.shape != (EXPECTED_DSELECT_COUNT, N_CLASSES):
    raise RuntimeError(f"Probability shape mismatch: {probs.shape}")
if labels.shape != (EXPECTED_DSELECT_COUNT,):
    raise RuntimeError(f"Label shape mismatch: {labels.shape}")
if not finite(probs):
    raise RuntimeError("Non-finite calibrated probabilities detected.")
if np.any(probs < 0) or np.any(probs > 1):
    raise RuntimeError("Calibrated probabilities outside [0,1].")

row_dev = float(np.max(np.abs(probs.sum(axis=1) - 1.0)))
if row_dev > 1e-5:
    raise RuntimeError(f"Probability row-sum deviation too large: {row_dev}")

pred = np.argmax(probs, axis=1).astype(np.int64)
top1 = probs[np.arange(len(probs)), pred]
correct = pred == labels
errors = ~correct

print("D2 CALIBRATED D-SELECT INPUT: PASS")
print(f"  records              : {len(record_ids)}")
print(f"  probabilities        : {probs.shape}")
print(f"  row-sum max deviation: {row_dev:.9g}")
print(f"  correct              : {int(correct.sum())}")
print(f"  errors               : {int(errors.sum())}")
print()

# ---------------------------------------------------------------------
# 15 fixed equal-width confidence bins.
# Bin convention:
# [0,b1), ..., [b13,b14), [b14,1.0]
# ---------------------------------------------------------------------
edges = np.linspace(0.0, 1.0, 16)
bin_ids = np.searchsorted(edges, top1, side="right") - 1
bin_ids = np.clip(bin_ids, 0, 14)

bin_rows = []
for i in range(15):
    lo = float(edges[i])
    hi = float(edges[i + 1])
    mask = bin_ids == i
    n = int(mask.sum())
    correct_n = int(correct[mask].sum())
    error_n = n - correct_n

    # Descriptive high/low confidence within the frozen bins.
    mean_conf = float(top1[mask].mean()) if n else 0.0
    empirical_acc = safe_div(correct_n, n)
    empirical_err = safe_div(error_n, n)

    bin_rows.append({
        "bin_index": i,
        "lower_bound": lo,
        "upper_bound": hi,
        "interval": f"[{lo:.6f},{hi:.6f})" if i < 14 else f"[{lo:.6f},1.000000]",
        "count": n,
        "correct": correct_n,
        "errors": error_n,
        "coverage_fraction": safe_div(n, EXPECTED_DSELECT_COUNT),
        "mean_calibrated_confidence": mean_conf,
        "empirical_accuracy": empirical_acc,
        "empirical_error_rate": empirical_err,
        "confidence_correctness_gap": mean_conf - empirical_acc,
    })

with CONF_BINS.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
    writer.writeheader()
    writer.writerows(bin_rows)

bin_total = sum(r["count"] for r in bin_rows)
if bin_total != EXPECTED_DSELECT_COUNT:
    raise RuntimeError("Confidence-bin accounting does not cover D-Select.")

# ---------------------------------------------------------------------
# Class-level error analysis
# ---------------------------------------------------------------------
classes_present = np.unique(labels)
class_rows = []

for cls in classes_present:
    mask = labels == cls
    support = int(mask.sum())
    correct_n = int(correct[mask].sum())
    error_n = support - correct_n
    conf = top1[mask]

    class_rows.append({
        "class_id": int(cls),
        "support": support,
        "correct": correct_n,
        "errors": error_n,
        "accuracy": safe_div(correct_n, support),
        "error_rate": safe_div(error_n, support),
        "mean_confidence": float(conf.mean()),
        "high_confidence_error_count_ge_0_90": int((~correct[mask] & (conf >= 0.90)).sum()),
        "high_confidence_error_rate_ge_0_90": safe_div(
            int((~correct[mask] & (conf >= 0.90)).sum()),
            error_n,
        ),
    })

class_rows.sort(key=lambda r: (-r["errors"], -r["support"], r["class_id"]))

with CLASS_ERRORS.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
    writer.writeheader()
    writer.writerows(class_rows)

# ---------------------------------------------------------------------
# Frozen D3 selective characterization import.
# ---------------------------------------------------------------------
with D3_CURVE.open("r", newline="", encoding="utf-8") as f:
    d3_rows = list(csv.DictReader(f))

if len(d3_rows) != 101:
    raise RuntimeError(f"D3 curve expected 101 rows, got {len(d3_rows)}")

# ---------------------------------------------------------------------
# Frozen D4 routing characterization import.
# ---------------------------------------------------------------------
with D4_CURVE.open("r", newline="", encoding="utf-8") as f:
    d4_rows = list(csv.DictReader(f))

if len(d4_rows) != 101:
    raise RuntimeError(f"D4 curve expected 101 rows, got {len(d4_rows)}")

routing_rows = []
for d3, d4 in zip(d3_rows, d4_rows):
    tau3 = float(d3["threshold"])
    tau4 = float(d4["threshold"])
    if abs(tau3 - tau4) > 1e-9:
        raise RuntimeError("D3/D4 threshold grids do not align.")

    routing_rows.append({
        "threshold": tau4,
        "total_records": int(d4["total_records"]),
        "accepted_count": int(d4["accepted_count"]),
        "review_count": int(d4["review_count"]),
        "coverage": float(d4["coverage"]),
        "review_rate": float(d4["review_rate"]),
        "accepted_errors": int(d4["accepted_errors"]),
        "reviewed_errors": int(d4["reviewed_errors"]),
        "model_error_capture_rate": float(d4["model_error_capture_rate"]),
        "selective_risk": (
            float(d3["risk"])
            if "risk" in d3
            else (
                float(d3["selective_risk"])
                if "selective_risk" in d3
                else math.nan
            )
        ),
    })

with ROUTING_ERRORS.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(routing_rows[0].keys()))
    writer.writeheader()
    writer.writerows(routing_rows)

# ---------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------
high_conf_error_bins = [
    r for r in bin_rows
    if r["errors"] > 0 and r["upper_bound"] >= 0.90
]
low_conf_correct_bins = [
    r for r in bin_rows
    if r["correct"] > 0 and r["upper_bound"] <= 0.60
]

high_conf_errors = int(
    ((top1 >= 0.90) & errors).sum()
)
low_conf_correct = int(
    ((top1 < 0.60) & correct).sum()
)

# Largest error-contributing classes are descriptive only.
top_error_classes = class_rows[:20]

results = {
    "phase": "D5",
    "version": "v001",
    "status": "EXECUTED",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "population": {
        "name": "D-Select",
        "count": EXPECTED_DSELECT_COUNT,
        "classes": N_CLASSES,
        "correct": int(correct.sum()),
        "errors": int(errors.sum()),
        "accuracy": float(correct.mean()),
    },
    "confidence_bin_analysis": {
        "method": "15 equal-width bins over [0,1]",
        "bins": bin_rows,
        "total_records_accounted": bin_total,
    },
    "diagnostic_high_confidence_errors": {
        "definition": "calibrated top-1 probability >= 0.90",
        "error_count": high_conf_errors,
        "error_fraction_of_all_model_errors": safe_div(
            high_conf_errors, int(errors.sum())
        ),
        "note": "diagnostic region only; not a deployment threshold",
    },
    "diagnostic_low_confidence_correct": {
        "definition": "calibrated top-1 probability < 0.60",
        "correct_count": low_conf_correct,
        "fraction_of_all_correct_predictions": safe_div(
            low_conf_correct, int(correct.sum())
        ),
        "note": "diagnostic region only; not a deployment threshold",
    },
    "class_error_analysis": {
        "classes_present": int(len(classes_present)),
        "top_20_by_error_count": top_error_classes,
        "note": "class-level observations are descriptive and support-aware",
    },
    "routing_error_analysis": {
        "threshold_count": len(routing_rows),
        "source": "frozen D4 routing characterization",
        "rows": routing_rows,
        "human_effectiveness_evaluated": False,
    },
    "boundary": config["boundary"],
    "upstream_artifacts": {
        "d2_npz_sha256": sha256(D2_NPZ),
        "d2_metrics_sha256": sha256(D2_METRICS),
        "d2_manifest_sha256": sha256(D2_MANIFEST),
        "d3_config_sha256": sha256(D3_CONFIG),
        "d3_curve_sha256": sha256(D3_CURVE),
        "d4_curve_sha256": sha256(D4_CURVE),
    },
    "scientific_boundary": [
        "No deployment threshold selected.",
        "No R_max/C_min selected.",
        "No human-review effectiveness evaluated.",
        "No open-set detection performed.",
        "No autonomy claim.",
        "D-Test not accessed.",
        "D2/D3/D4/Phase-C artifacts not modified.",
    ],
}

RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")

manifest = {
    "phase": "D5",
    "version": "v001",
    "method": "reliability_error_analysis",
    "status": "EXECUTED",
    "seed": SEED,
    "population": {
        "source": str(D2_NPZ.relative_to(ROOT)),
        "count": EXPECTED_DSELECT_COUNT,
    },
    "confidence_bins": {
        "count": 15,
        "equal_width": True,
        "start": 0.0,
        "end": 1.0,
        "result_adaptive": False,
    },
    "inputs": {
        "d2_npz_sha256": sha256(D2_NPZ),
        "d2_metrics_sha256": sha256(D2_METRICS),
        "d2_manifest_sha256": sha256(D2_MANIFEST),
        "d3_config_sha256": sha256(D3_CONFIG),
        "d3_curve_sha256": sha256(D3_CURVE),
        "d4_curve_sha256": sha256(D4_CURVE),
    },
    "outputs": {
        "config": str(CONFIG.relative_to(ROOT)),
        "results": str(RESULTS.relative_to(ROOT)),
        "confidence_bins": str(CONF_BINS.relative_to(ROOT)),
        "class_errors": str(CLASS_ERRORS.relative_to(ROOT)),
        "routing_errors": str(ROUTING_ERRORS.relative_to(ROOT)),
    },
    "boundary": config["boundary"],
}

MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("D5 ANALYSIS: COMPLETE")
print(f"  D-Select records           : {EXPECTED_DSELECT_COUNT}")
print(f"  Correct                    : {int(correct.sum())}")
print(f"  Errors                     : {int(errors.sum())}")
print(f"  Accuracy                   : {correct.mean():.12f}")
print(f"  Confidence bins            : 15")
print(f"  High-confidence errors >=.90 : {high_conf_errors}")
print(f"  Low-confidence correct <.60  : {low_conf_correct}")
print(f"  Classes represented        : {len(classes_present)}")
print(f"  D3 curve rows consumed     : {len(d3_rows)}")
print(f"  D4 curve rows consumed     : {len(d4_rows)}")
print()
print("D5 BOUNDARY: PASS")
print("  Threshold selected         : NO")
print("  R_max selected             : NO")
print("  C_min selected             : NO")
print("  Human performance          : NOT EVALUATED")
print("  D-Test accessed            : NO")
print("  D2 modified                : NO")
print("  D3 modified                : NO")
print("  D4 modified                : NO")
print("  Phase-C modified           : NO")
print()
print("Artifacts:")
print(f"  {RESULTS}")
print(f"  {CONF_BINS}")
print(f"  {CLASS_ERRORS}")
print(f"  {ROUTING_ERRORS}")
print(f"  {MANIFEST}")
print()
print("=" * 72)
print("D5 RELIABILITY ERROR ANALYSIS COMPLETE")
print("=" * 72)
print()
print("D5 is executed but NOT YET FROZEN.")
print("Run the independent D5 acceptance audit before freezing.")
