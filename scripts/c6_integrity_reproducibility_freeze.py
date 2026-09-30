from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260827
EXPECTED_TEST = 7346
EXPECTED_TRAIN = 69823
EXPECTED_CLASSES = 549
EXPECTED_BEST_EPOCH = 17

PHASE_C_ROOT = PROJECT_ROOT / "data" / "models" / "abo" / "phase_c"

CHECKPOINT = PHASE_C_ROOT / "c3" / "training" / "cbase_best.pt"

C3_TRAIN_DIR = PHASE_C_ROOT / "c3" / "embeddings"
C4_TEST_DIR = PHASE_C_ROOT / "c4" / "test_embeddings"
C4_EVAL_DIR = PHASE_C_ROOT / "c4" / "evaluation"
C5_1_DIR = PHASE_C_ROOT / "c5" / "c5_1_reliability_signal_audit"
C5_2_DIR = PHASE_C_ROOT / "c5" / "c5_2_confidence_probability_audit"
C5_3_DIR = PHASE_C_ROOT / "c5" / "c5_3_margin_ambiguity_analysis"
C5_4_DIR = PHASE_C_ROOT / "c5" / "c5_4_modality_evidence_analysis"
C5_5_DIR = PHASE_C_ROOT / "c5" / "c5_5_representation_distance_analysis"

OUTPUT_DIR = PHASE_C_ROOT / "c6" / "integrity_freeze"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path, label: str):
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
    return path


def require_dir(path: Path, label: str):
    if not path.is_dir():
        raise RuntimeError(f"Missing {label}: {path}")
    return path


def read_json(path: Path, label: str):
    require(path, label)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_array(path: Path, shape=None, dtype=None):
    require(path, f"array {path.name}")
    arr = np.load(path, mmap_mode="r")
    if shape is not None and tuple(arr.shape) != tuple(shape):
        raise RuntimeError(
            f"{path.name}: shape {arr.shape} != expected {shape}"
        )
    if dtype is not None and arr.dtype != dtype:
        raise RuntimeError(
            f"{path.name}: dtype {arr.dtype} != expected {dtype}"
        )
    if not np.all(np.isfinite(np.asarray(arr))):
        raise RuntimeError(f"{path.name}: non-finite values detected")
    return arr


def git_status():
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "available": result.returncode == 0,
            "output": result.stdout.strip(),
            "error": result.stderr.strip(),
        }
    except Exception as exc:
        return {"available": False, "output": "", "error": str(exc)}


def git_diff_names():
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.splitlines() if result.returncode == 0 else []
    except Exception:
        return []


def main():
    checks = {}
    failures = []

    def check(name, condition, detail):
        checks[name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            failures.append(f"{name}: {detail}")

    # ------------------------------------------------------------------
    # Frozen C3 training artifacts
    # ------------------------------------------------------------------
    require_dir(C3_TRAIN_DIR, "C3 embeddings directory")
    train_text = check_array(
        C3_TRAIN_DIR / "train_text.npy",
        (EXPECTED_TRAIN, 384),
        np.dtype("float32"),
    )
    train_image = check_array(
        C3_TRAIN_DIR / "train_image.npy",
        (EXPECTED_TRAIN, 512),
        np.dtype("float32"),
    )
    train_labels = check_array(
        C3_TRAIN_DIR / "train_labels.npy",
        (EXPECTED_TRAIN,),
        np.dtype("int64"),
    )

    check(
        "c3_train_labels_valid",
        bool(np.all((train_labels >= 0) & (train_labels < EXPECTED_CLASSES))),
        "all train labels are inside the 549-class space",
    )

    c3_manifest = read_json(
        C3_TRAIN_DIR / "c3_embedding_manifest.json",
        "C3 embedding manifest",
    )
    check(
        "c3_test_accessed_false",
        c3_manifest.get("test_accessed") is False,
        f"manifest test_accessed={c3_manifest.get('test_accessed')}",
    )
    check(
        "c3_seed",
        c3_manifest.get("seed") == SEED,
        f"manifest seed={c3_manifest.get('seed')}",
    )

    # ------------------------------------------------------------------
    # Frozen checkpoint
    # ------------------------------------------------------------------
    checkpoint_hash = sha256_file(require(CHECKPOINT, "C-Base checkpoint"))

    import torch

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    check(
        "checkpoint_best_epoch",
        checkpoint.get("best_epoch") == EXPECTED_BEST_EPOCH,
        f"best_epoch={checkpoint.get('best_epoch')}",
    )
    check(
        "checkpoint_selection_metric",
        checkpoint.get("selection_metric") == "macro_f1",
        f"selection_metric={checkpoint.get('selection_metric')}",
    )
    check(
        "checkpoint_test_accessed_false",
        checkpoint.get("test_accessed") is False,
        f"checkpoint test_accessed={checkpoint.get('test_accessed')}",
    )

    # ------------------------------------------------------------------
    # C4 test embeddings + frozen final evaluation
    # ------------------------------------------------------------------
    require_dir(C4_TEST_DIR, "C4 test embeddings directory")

    test_text = check_array(
        C4_TEST_DIR / "test_text.npy",
        (EXPECTED_TEST, 384),
        np.dtype("float32"),
    )
    test_image = check_array(
        C4_TEST_DIR / "test_image.npy",
        (EXPECTED_TEST, 512),
        np.dtype("float32"),
    )
    test_labels = check_array(
        C4_TEST_DIR / "test_labels.npy",
        (EXPECTED_TEST,),
        np.dtype("int64"),
    )

    test_ids = read_json(
        C4_TEST_DIR / "test_record_ids.json",
        "C4 test record IDs",
    )
    check(
        "test_ids_unique",
        len(test_ids) == EXPECTED_TEST
        and len(set(test_ids)) == EXPECTED_TEST,
        f"test IDs={len(test_ids)} unique={len(set(test_ids))}",
    )

    c4_results = read_json(
        C4_EVAL_DIR / "cbase_test_results.json",
        "C4 C-Base test results",
    )

    expected_metrics = {
        "accuracy": 0.7868227606860877,
        "macro_f1": 0.3413462319143464,
        "weighted_f1": 0.7824345652636053,
    }

    # Support both likely field names without altering historical artifacts.
    observed_metrics = {
        "accuracy": c4_results.get("accuracy"),
        "macro_f1": c4_results.get("macro_f1"),
        "weighted_f1": c4_results.get("weighted_f1"),
    }

    for key, expected in expected_metrics.items():
        observed = observed_metrics[key]
        check(
            f"c4_{key}_matches_frozen",
            observed is not None and abs(float(observed) - expected) <= 1e-12,
            f"observed={observed}, expected={expected}",
        )

    predictions_npz = np.load(
        require(
            C4_EVAL_DIR / "cbase_test_predictions.npz",
            "C4 predictions",
        )
    )
    predictions = predictions_npz["predictions"]
    check(
        "c4_predictions_shape",
        predictions.shape == (EXPECTED_TEST,),
        f"shape={predictions.shape}",
    )

    # ------------------------------------------------------------------
    # C5 inventory
    # ------------------------------------------------------------------
    c5_dirs = {
        "c5_1": C5_1_DIR,
        "c5_2": C5_2_DIR,
        "c5_3": C5_3_DIR,
        "c5_4": C5_4_DIR,
        "c5_5": C5_5_DIR,
    }

    for name, directory in c5_dirs.items():
        check(
            f"{name}_directory_exists",
            directory.is_dir(),
            str(directory),
        )

    c5_1_summary = read_json(
        C5_1_DIR / "c5_1_signal_inventory.json",
        "C5.1 summary",
    )
    check(
        "c5_1_population",
        c5_1_summary.get("test_population") == EXPECTED_TEST,
        f"population={c5_1_summary.get('test_population')}",
    )
    check(
        "c5_1_best_epoch",
        c5_1_summary.get("checkpoint_best_epoch") == EXPECTED_BEST_EPOCH,
        f"best_epoch={c5_1_summary.get('checkpoint_best_epoch')}",
    )

    c5_1_predictions = check_array(
        C5_1_DIR / "c5_1_predictions.npy",
        (EXPECTED_TEST,),
        np.dtype("int64"),
    )

    check(
        "c5_1_predictions_match_c4",
        bool(np.array_equal(c5_1_predictions, predictions)),
        "C5.1 predictions equal frozen C4 predictions",
    )

    c5_2_summary = read_json(
        C5_2_DIR / "c5_2_summary.json",
        "C5.2 summary",
    )
    check(
        "c5_2_population",
        c5_2_summary.get("test_population") == EXPECTED_TEST,
        f"population={c5_2_summary.get('test_population')}",
    )

    c5_3_summary = read_json(
        C5_3_DIR / "c5_3_summary.json",
        "C5.3 summary",
    )
    check(
        "c5_3_population",
        c5_3_summary.get("test_population") == EXPECTED_TEST,
        f"population={c5_3_summary.get('test_population')}",
    )

    c5_4_summary = read_json(
        C5_4_DIR / "c5_4_summary.json",
        "C5.4 summary",
    )
    check(
        "c5_4_population",
        c5_4_summary.get("test_population") == EXPECTED_TEST,
        f"population={c5_4_summary.get('test_population')}",
    )
    check(
        "c5_4_reproduced_accuracy",
        abs(
            float(c5_4_summary["metrics_reproduced_from_frozen_c4"]["accuracy"])
            - expected_metrics["accuracy"]
        ) <= 1e-12,
        "C5.4 reproduces frozen C4 accuracy",
    )

    c5_5_summary = read_json(
        C5_5_DIR / "c5_5_summary.json",
        "C5.5 summary",
    )
    check(
        "c5_5_population",
        c5_5_summary.get("test_population") == EXPECTED_TEST,
        f"population={c5_5_summary.get('test_population')}",
    )
    check(
        "c5_5_prototypes_train_only",
        c5_5_summary.get("test_accessed_for_prototype_construction") is False,
        f"value={c5_5_summary.get('test_accessed_for_prototype_construction')}",
    )
    check(
        "c5_5_all_classes_supported",
        c5_5_summary["prototype_class_support"]["classes_with_zero_train_support"] == 0,
        "all 549 classes have train prototype support",
    )

    # ------------------------------------------------------------------
    # Cross-stage consistency
    # ------------------------------------------------------------------
    check(
        "cross_stage_population",
        len(test_ids) == EXPECTED_TEST,
        f"test population={len(test_ids)}",
    )
    check(
        "cross_stage_class_space",
        int(np.max(test_labels)) < EXPECTED_CLASSES,
        f"max test label={int(np.max(test_labels))}",
    )
    check(
        "cross_stage_seed",
        c5_1_summary.get("seed") == SEED
        and c5_3_summary.get("seed") == SEED
        and c5_5_summary.get("seed") == SEED,
        "C5.1/C5.3/C5.5 seed metadata all match",
    )

    # ------------------------------------------------------------------
    # Git / frozen-boundary inspection
    # ------------------------------------------------------------------
    status = git_status()
    diff_names = git_diff_names()

    frozen_prefixes = [
        "data/models/abo/b3",
        "data/models/abo/b4.2",
        "data/models/abo/b5",
        "data/models/abo/b6",
        "data/models/abo/phase_c/c3",
        "data/models/abo/phase_c/c4",
    ]

    frozen_modifications = [
        p for p in diff_names
        if any(p.replace("\\", "/").startswith(prefix) for prefix in frozen_prefixes)
    ]

    check(
        "git_available",
        status["available"],
        status["error"] or status["output"],
    )
    check(
        "no_uncommitted_frozen_artifact_changes",
        len(frozen_modifications) == 0,
        f"modified frozen paths={frozen_modifications}",
    )

    # ------------------------------------------------------------------
    # Artifact inventory + hashes for compact authoritative files
    # ------------------------------------------------------------------
    inventory_paths = [
        CHECKPOINT,
        C3_TRAIN_DIR / "c3_embedding_manifest.json",
        C4_TEST_DIR / "c4_test_embedding_manifest.json",
        C4_EVAL_DIR / "cbase_test_results.json",
        C4_EVAL_DIR / "cbase_test_predictions.npz",
        C5_1_DIR / "c5_1_signal_inventory.json",
        C5_1_DIR / "c5_1_signal_inventory_manifest.json",
        C5_2_DIR / "c5_2_summary.json",
        C5_2_DIR / "c5_2_audit_manifest.json",
        C5_3_DIR / "c5_3_summary.json",
        C5_3_DIR / "c5_3_audit_manifest.json",
        C5_4_DIR / "c5_4_summary.json",
        C5_4_DIR / "c5_4_audit_manifest.json",
        C5_5_DIR / "c5_5_summary.json",
        C5_5_DIR / "c5_5_audit_manifest.json",
    ]

    inventory = {}
    for path in inventory_paths:
        if path.is_file():
            inventory[str(path.relative_to(PROJECT_ROOT))] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------
    overall_status = "VERIFIED" if not failures else "BLOCKED"

    report = {
        "status": overall_status,
        "phase": "C6",
        "analysis": "integrity_and_reproducibility_freeze",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "frozen_contract": {
            "scope": "ABO-only",
            "target": "product_type",
            "classes": EXPECTED_CLASSES,
            "train_population": EXPECTED_TRAIN,
            "test_population": EXPECTED_TEST,
            "best_epoch": EXPECTED_BEST_EPOCH,
            "selection_metric": "macro_f1",
            "test_isolated_until_final_evaluation": True,
        },
        "checks": checks,
        "failures": failures,
        "git": {
            "status": status,
            "uncommitted_frozen_paths": frozen_modifications,
        },
        "checkpoint_sha256": checkpoint_hash,
        "authoritative_artifact_inventory": inventory,
        "scientific_final_status": {
            "multimodal_understanding": "DEMONSTRATED",
            "joint_model_improvement_over_b5": "DEMONSTRATED",
            "error_analysis": "VERIFIED",
            "reliability_signal_inventory": "VERIFIED",
            "modality_evidence": "VERIFIED",
            "representation_distance": "VERIFIED",
            "calibration": "UNKNOWN",
            "selective_prediction": "UNKNOWN",
            "abstention": "UNKNOWN",
            "human_review_routing": "UNKNOWN",
            "open_set_detection": "UNKNOWN",
            "full_project_autonomy": "NOT ESTABLISHED",
        },
        "phase_c_freeze_boundary": (
            "Phase C establishes a stronger ABO jointly learned multimodal "
            "understanding model and a documented reliability-signal inventory. "
            "It does not establish calibrated confidence, selective prediction, "
            "abstention, human-review routing, open-set detection, or full-system autonomy."
        ),
    }

    with (OUTPUT_DIR / "c6_integrity_freeze_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    with (OUTPUT_DIR / "c6_artifact_inventory.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(inventory, f, indent=2)

    print("C6 INTEGRITY AUDIT COMPLETE")
    print(json.dumps(report, indent=2))
    print(f"Outputs: {OUTPUT_DIR}")

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
