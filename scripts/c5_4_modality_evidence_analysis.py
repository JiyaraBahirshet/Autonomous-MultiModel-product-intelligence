"""
C5.4 — Modality Evidence / Sensitivity Analysis
Autonomous Multimodal Product Intelligence
Phase C — ABO-only

Purpose
-------
Inference-only analysis of the frozen Phase C C-Base model to inspect:
    1. separate text/image projected representation strength;
    2. descriptive relative representation strength;
    3. prediction sensitivity to text/image branch ablation;
    4. probability-distribution change under branch ablation;
    5. relationship between these signals and correctness.

This script does NOT:
    - train or fine-tune;
    - select a checkpoint;
    - tune thresholds;
    - fit calibration;
    - modify B3/B4/B5/C3/C4 artifacts;
    - use test results to change the model;
    - claim calibrated confidence, selective prediction, abstention,
      human-review routing, or open-set detection.

Interpretation boundary
-----------------------
Representation magnitude is descriptive evidence, not causal contribution.
Ablation sensitivity is a model-behavior diagnostic, not a validated routing
policy. All thresholds in this script are descriptive diagnostics only.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

SEED = 20260827
NUM_CLASSES = 549
TEXT_DIM = 384
IMAGE_DIM = 512
PROJECTED_DIM = 512
EXPECTED_TEST = 7_346

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_ROOT = PROJECT_ROOT / "data" / "models" / "abo" / "phase_c" / "c4" / "test_embeddings"
TEXT_PATH = EMBEDDING_ROOT / "test_text.npy"
IMAGE_PATH = EMBEDDING_ROOT / "test_image.npy"
LABEL_PATH = EMBEDDING_ROOT / "test_labels.npy"
ID_PATH = EMBEDDING_ROOT / "test_record_ids.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "models" / "abo" / "phase_c" / "c3" / "training" / "cbase_best.pt"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "models" / "abo" / "phase_c" / "c5" / "c5_4_modality_evidence_analysis"
SUMMARY_PATH = OUTPUT_ROOT / "c5_4_summary.json"
STRENGTH_PATH = OUTPUT_ROOT / "c5_4_modality_strength.json"
ABLATION_PATH = OUTPUT_ROOT / "c5_4_ablation_sensitivity.json"
CORRECTNESS_PATH = OUTPUT_ROOT / "c5_4_correct_incorrect_summary.json"
RELATIONSHIP_PATH = OUTPUT_ROOT / "c5_4_modality_relationships.json"
MANIFEST_PATH = OUTPUT_ROOT / "c5_4_audit_manifest.json"

# Per-record signals are saved so the analysis is auditable without rerunning
# model inference.
SIGNAL_ARRAYS = {
    "text_projected_l2": OUTPUT_ROOT / "c5_4_text_projected_l2.npy",
    "image_projected_l2": OUTPUT_ROOT / "c5_4_image_projected_l2.npy",
    "text_strength_share": OUTPUT_ROOT / "c5_4_text_strength_share.npy",
    "image_strength_share": OUTPUT_ROOT / "c5_4_image_strength_share.npy",
    "full_predictions": OUTPUT_ROOT / "c5_4_full_predictions.npy",
    "text_only_predictions": OUTPUT_ROOT / "c5_4_text_only_predictions.npy",
    "image_only_predictions": OUTPUT_ROOT / "c5_4_image_only_predictions.npy",
    "full_top1_probability": OUTPUT_ROOT / "c5_4_full_top1_probability.npy",
    "text_only_top1_probability": OUTPUT_ROOT / "c5_4_text_only_top1_probability.npy",
    "image_only_top1_probability": OUTPUT_ROOT / "c5_4_image_only_top1_probability.npy",
    "text_ablation_kl": OUTPUT_ROOT / "c5_4_text_ablation_kl.npy",
    "image_ablation_kl": OUTPUT_ROOT / "c5_4_image_ablation_kl.npy",
    "text_ablation_prediction_changed": OUTPUT_ROOT / "c5_4_text_ablation_prediction_changed.npy",
    "image_ablation_prediction_changed": OUTPUT_ROOT / "c5_4_image_ablation_prediction_changed.npy",
    "correct": OUTPUT_ROOT / "c5_4_correct.npy",
}

DEVICE = torch.device("cpu")


class CBase(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, num_classes: int) -> None:
        super().__init__()
        self.text_projection = nn.Sequential(nn.Linear(text_dim, 512), nn.ReLU())
        self.image_projection = nn.Sequential(nn.Linear(image_dim, 512), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, num_classes))

    def branches(self, text: torch.Tensor, image: torch.Tensor):
        return self.text_projection(text), self.image_projection(image)

    def logits_from_projected(self, text_projected: torch.Tensor, image_projected: torch.Tensor):
        combined = torch.cat([text_projected, image_projected], dim=1)
        joint = self.fusion[1](self.fusion[0](combined))
        logits = self.fusion[2](joint)
        return logits, joint


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def finite_summary(x: np.ndarray) -> dict[str, Any]:
    return {
        "finite": bool(np.isfinite(x).all()),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def quantiles(x: np.ndarray) -> dict[str, float]:
    return {f"p{p}": float(np.quantile(x, p / 100.0)) for p in [1, 5, 25, 50, 75, 95, 99]}


def binned_accuracy(signal: np.ndarray, correct: np.ndarray, edges: list[float]) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (signal >= lo) & (signal <= hi)
        else:
            mask = (signal >= lo) & (signal < hi)
        n = int(mask.sum())
        rows.append({
            "lower": lo,
            "upper": hi,
            "count": n,
            "accuracy": float(correct[mask].mean()) if n else None,
        })
    return rows


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    required = [TEXT_PATH, IMAGE_PATH, LABEL_PATH, ID_PATH, CHECKPOINT_PATH]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required frozen artifact not found: {path}")

    text = np.load(TEXT_PATH, mmap_mode="r")
    image = np.load(IMAGE_PATH, mmap_mode="r")
    labels = np.load(LABEL_PATH, mmap_mode="r")
    with ID_PATH.open("r", encoding="utf-8") as f:
        record_ids = json.load(f)

    if text.shape != (EXPECTED_TEST, TEXT_DIM):
        raise RuntimeError(f"Unexpected text shape: {text.shape}")
    if image.shape != (EXPECTED_TEST, IMAGE_DIM):
        raise RuntimeError(f"Unexpected image shape: {image.shape}")
    if labels.shape != (EXPECTED_TEST,):
        raise RuntimeError(f"Unexpected label shape: {labels.shape}")
    if len(record_ids) != EXPECTED_TEST or len(set(record_ids)) != EXPECTED_TEST:
        raise RuntimeError("Test record IDs are not exactly 7,346 unique IDs")
    if not np.isfinite(text).all() or not np.isfinite(image).all():
        raise RuntimeError("Non-finite test embeddings detected")
    if np.any(labels < 0) or np.any(labels >= NUM_CLASSES):
        raise RuntimeError("Invalid test labels detected")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if checkpoint.get("test_accessed") is not False:
        raise RuntimeError(f"Checkpoint test_accessed is not explicitly false: {checkpoint.get('test_accessed')}")

    model = CBase(TEXT_DIM, IMAGE_DIM, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Chunking is purely for memory control; no optimization or selection occurs.
    batch_size = 256
    all_arrays = {k: [] for k in SIGNAL_ARRAYS}

    with torch.no_grad():
        for start in range(0, EXPECTED_TEST, batch_size):
            end = min(start + batch_size, EXPECTED_TEST)
            t = torch.from_numpy(np.asarray(text[start:end], dtype=np.float32)).to(DEVICE)
            im = torch.from_numpy(np.asarray(image[start:end], dtype=np.float32)).to(DEVICE)
            y = np.asarray(labels[start:end], dtype=np.int64)

            tp, ip = model.branches(t, im)
            zero_t = torch.zeros_like(tp)
            zero_i = torch.zeros_like(ip)

            full_logits, _ = model.logits_from_projected(tp, ip)
            text_only_logits, _ = model.logits_from_projected(tp, zero_i)
            image_only_logits, _ = model.logits_from_projected(zero_t, ip)

            full_p = torch.softmax(full_logits, dim=1)
            text_p = torch.softmax(text_only_logits, dim=1)
            image_p = torch.softmax(image_only_logits, dim=1)

            full_pred = full_p.argmax(dim=1)
            text_pred = text_p.argmax(dim=1)
            image_pred = image_p.argmax(dim=1)

            # KL(full || ablated). This is a descriptive distribution-change signal.
            eps = 1e-12
            full_safe = full_p.clamp_min(eps)
            text_safe = text_p.clamp_min(eps)
            image_safe = image_p.clamp_min(eps)
            text_kl = (full_safe * (full_safe.log() - text_safe.log())).sum(dim=1)
            image_kl = (full_safe * (full_safe.log() - image_safe.log())).sum(dim=1)

            t_norm = torch.linalg.vector_norm(tp, dim=1)
            i_norm = torch.linalg.vector_norm(ip, dim=1)
            denom = (t_norm + i_norm).clamp_min(eps)
            t_share = t_norm / denom
            i_share = i_norm / denom

            all_arrays["text_projected_l2"].append(t_norm.cpu().numpy())
            all_arrays["image_projected_l2"].append(i_norm.cpu().numpy())
            all_arrays["text_strength_share"].append(t_share.cpu().numpy())
            all_arrays["image_strength_share"].append(i_share.cpu().numpy())
            all_arrays["full_predictions"].append(full_pred.cpu().numpy())
            all_arrays["text_only_predictions"].append(text_pred.cpu().numpy())
            all_arrays["image_only_predictions"].append(image_pred.cpu().numpy())
            all_arrays["full_top1_probability"].append(full_p.max(dim=1).values.cpu().numpy())
            all_arrays["text_only_top1_probability"].append(text_p.max(dim=1).values.cpu().numpy())
            all_arrays["image_only_top1_probability"].append(image_p.max(dim=1).values.cpu().numpy())
            all_arrays["text_ablation_kl"].append(text_kl.cpu().numpy())
            all_arrays["image_ablation_kl"].append(image_kl.cpu().numpy())
            all_arrays["text_ablation_prediction_changed"].append((full_pred != text_pred).cpu().numpy())
            all_arrays["image_ablation_prediction_changed"].append((full_pred != image_pred).cpu().numpy())
            all_arrays["correct"].append((full_pred.cpu().numpy() == y))

    arrays = {}
    for key, chunks in all_arrays.items():
        arr = np.concatenate(chunks)
        arrays[key] = arr
        np.save(SIGNAL_ARRAYS[key], arr)

    correct = arrays["correct"]
    full_pred = arrays["full_predictions"]
    y = np.asarray(labels, dtype=np.int64)

    # Reproduce frozen C4 metrics exactly: fixed 549-class macro-F1.
    metrics = {
        "accuracy": float(accuracy_score(y, full_pred)),
        "macro_f1_fixed_549": float(f1_score(y, full_pred, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, full_pred, average="weighted", zero_division=0)),
    }

    text_change = arrays["text_ablation_prediction_changed"]
    image_change = arrays["image_ablation_prediction_changed"]
    text_kl = arrays["text_ablation_kl"]
    image_kl = arrays["image_ablation_kl"]
    t_norm = arrays["text_projected_l2"]
    i_norm = arrays["image_projected_l2"]
    t_share = arrays["text_strength_share"]
    i_share = arrays["image_strength_share"]

    def group_stats(signal: np.ndarray) -> dict[str, Any]:
        return {
            "all": finite_summary(signal),
            "correct": finite_summary(signal[correct]),
            "incorrect": finite_summary(signal[~correct]),
        }

    summary = {
        "status": "VERIFIED",
        "phase": "C5.4",
        "analysis": "modality_evidence_and_sensitivity",
        "test_population": EXPECTED_TEST,
        "num_classes": NUM_CLASSES,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "seed": SEED,
        "metrics_reproduced_from_frozen_c4": metrics,
        "correct_count": int(correct.sum()),
        "incorrect_count": int((~correct).sum()),
        "representation_strength": {
            "text_projected_l2": group_stats(t_norm),
            "image_projected_l2": group_stats(i_norm),
            "text_strength_share": group_stats(t_share),
            "image_strength_share": group_stats(i_share),
        },
        "ablation_sensitivity": {
            "text_ablation_kl_full_vs_text_only": group_stats(text_kl),
            "image_ablation_kl_full_vs_image_only": group_stats(image_kl),
            "text_ablation_prediction_change_rate": float(text_change.mean()),
            "image_ablation_prediction_change_rate": float(image_change.mean()),
            "text_ablation_change_rate_correct": float(text_change[correct].mean()),
            "text_ablation_change_rate_incorrect": float(text_change[~correct].mean()),
            "image_ablation_change_rate_correct": float(image_change[correct].mean()),
            "image_ablation_change_rate_incorrect": float(image_change[~correct].mean()),
        },
        "scientific_status": {
            "modality_representation_strength": "VERIFIED descriptive signal",
            "modality_ablation_sensitivity": "VERIFIED descriptive signal",
            "modality_sensitivity_correctness_association": "DEMONSTRATED descriptively",
            "calibration": "UNKNOWN",
            "selective_prediction": "UNKNOWN",
            "abstention": "UNKNOWN",
            "human_review_routing": "UNKNOWN",
            "open_set_detection": "UNKNOWN",
        },
    }

    strength = {
        "text_projected_l2": {**finite_summary(t_norm), **quantiles(t_norm)},
        "image_projected_l2": {**finite_summary(i_norm), **quantiles(i_norm)},
        "text_strength_share": {**finite_summary(t_share), **quantiles(t_share)},
        "image_strength_share": {**finite_summary(i_share), **quantiles(i_share)},
        "identity_check_share_sum_max_error": float(np.max(np.abs((t_share + i_share) - 1.0))),
        "descriptive_bins": {
            "text_strength_share": binned_accuracy(t_share, correct, [0, .2, .4, .5, .6, .8, 1.000001]),
            "image_strength_share": binned_accuracy(i_share, correct, [0, .2, .4, .5, .6, .8, 1.000001]),
        },
    }

    ablation = {
        "text_ablation_kl": {**finite_summary(text_kl), **quantiles(text_kl)},
        "image_ablation_kl": {**finite_summary(image_kl), **quantiles(image_kl)},
        "text_ablation_prediction_changed": {
            "count": int(text_change.sum()), "rate": float(text_change.mean())
        },
        "image_ablation_prediction_changed": {
            "count": int(image_change.sum()), "rate": float(image_change.mean())
        },
        "both_modalities_change": {
            "count": int((text_change & image_change).sum()),
            "rate": float((text_change & image_change).mean()),
        },
        "neither_modality_change": {
            "count": int((~text_change & ~image_change).sum()),
            "rate": float((~text_change & ~image_change).mean()),
        },
    }

    correctness = {
        "correct_count": int(correct.sum()),
        "incorrect_count": int((~correct).sum()),
        "text_strength_share": group_stats(t_share),
        "image_strength_share": group_stats(i_share),
        "text_ablation_kl": group_stats(text_kl),
        "image_ablation_kl": group_stats(image_kl),
        "text_prediction_change": {
            "correct": float(text_change[correct].mean()),
            "incorrect": float(text_change[~correct].mean()),
        },
        "image_prediction_change": {
            "correct": float(image_change[correct].mean()),
            "incorrect": float(image_change[~correct].mean()),
        },
    }

    relationships = {
        "text_vs_image_strength_share_mean": {
            "text": float(t_share.mean()), "image": float(i_share.mean())
        },
        "text_vs_image_ablation_kl_mean": {
            "text": float(text_kl.mean()), "image": float(image_kl.mean())
        },
        "interpretation": (
            "Ablation sensitivity measures how the frozen model's output distribution "
            "changes when one projected modality branch is zeroed. It is not a causal "
            "importance estimate and is not a calibrated confidence or routing score."
        ),
    }

    save_json(SUMMARY_PATH, summary)
    save_json(STRENGTH_PATH, strength)
    save_json(ABLATION_PATH, ablation)
    save_json(CORRECTNESS_PATH, correctness)
    save_json(RELATIONSHIP_PATH, relationships)

    manifest = {
        "status": "PASS",
        "phase": "C5.4",
        "test_accessed": True,
        "test_used_only_for_final_descriptive_analysis": True,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "threshold_optimization_performed": False,
        "calibration_fitting_performed": False,
        "upstream_artifacts_modified": False,
        "seed": SEED,
        "inputs": {
            "test_text": {"path": str(TEXT_PATH), "sha256": sha256_file(TEXT_PATH)},
            "test_image": {"path": str(IMAGE_PATH), "sha256": sha256_file(IMAGE_PATH)},
            "test_labels": {"path": str(LABEL_PATH), "sha256": sha256_file(LABEL_PATH)},
            "test_record_ids": {"path": str(ID_PATH), "sha256": sha256_file(ID_PATH)},
            "checkpoint": {"path": str(CHECKPOINT_PATH), "sha256": sha256_file(CHECKPOINT_PATH)},
        },
        "outputs": {name: str(path) for name, path in SIGNAL_ARRAYS.items()},
        "scientific_boundary": summary["scientific_status"],
    }
    save_json(MANIFEST_PATH, manifest)

    print("C5.4 COMPLETE")
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
