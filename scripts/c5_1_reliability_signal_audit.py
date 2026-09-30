"""
C5.1 — Reliability Signal Inventory & Audit
Autonomous Multimodal Product Intelligence
Phase C — ABO-only

Purpose
-------
Run inference-only reliability-signal extraction from the frozen Phase C
C-Base best checkpoint (validation-selected epoch 17) on the exact frozen
7,346-record Phase C test population.

Signals extracted:
    - logits
    - softmax probabilities
    - top-1 predicted class
    - top-1 probability
    - top-2 probability
    - top-1/top-2 margin
    - predictive entropy
    - correctness
    - joint representation (post-fusion hidden representation)

This script does NOT:
    - train
    - tune hyperparameters
    - select a checkpoint
    - modify B3/B4/B5/C3 artifacts
    - regenerate splits
    - introduce missing modalities
    - use test results to modify the model

Important
---------
This is a reliability-evidence inventory, NOT calibration, selective
prediction, abstention, human-review routing, or open-set detection.
Those claims remain unestablished until separately evaluated.
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 20260827
NUM_CLASSES = 549
TEXT_DIM = 384
IMAGE_DIM = 512
FUSION_DIM = 512
EXPECTED_TEST = 7_346

EMBEDDING_ROOT = Path(
    r"data\models\abo\phase_c\c4\test_embeddings"
)

TEXT_PATH = EMBEDDING_ROOT / "test_text.npy"
IMAGE_PATH = EMBEDDING_ROOT / "test_image.npy"
LABEL_PATH = EMBEDDING_ROOT / "test_labels.npy"
ID_PATH = EMBEDDING_ROOT / "test_record_ids.json"

C4_MANIFEST = EMBEDDING_ROOT / "c4_test_embedding_manifest.json"

CHECKPOINT_PATH = Path(
    r"data\models\abo\phase_c\c3\training\cbase_best.pt"
)

OUTPUT_ROOT = Path(
    r"data\models\abo\phase_c\c5\c5_1_reliability_signal_audit"
)

LOGITS_PATH = OUTPUT_ROOT / "c5_1_logits.npy"
PROBABILITIES_PATH = OUTPUT_ROOT / "c5_1_probabilities.npy"
PREDICTIONS_PATH = OUTPUT_ROOT / "c5_1_predictions.npy"
TOP1_PROB_PATH = OUTPUT_ROOT / "c5_1_top1_probability.npy"
TOP2_PROB_PATH = OUTPUT_ROOT / "c5_1_top2_probability.npy"
MARGIN_PATH = OUTPUT_ROOT / "c5_1_top1_top2_margin.npy"
ENTROPY_PATH = OUTPUT_ROOT / "c5_1_predictive_entropy.npy"
CORRECT_PATH = OUTPUT_ROOT / "c5_1_correct.npy"
JOINT_REP_PATH = OUTPUT_ROOT / "c5_1_joint_representation.npy"

SUMMARY_PATH = OUTPUT_ROOT / "c5_1_signal_inventory.json"
MANIFEST_PATH = OUTPUT_ROOT / "c5_1_signal_inventory_manifest.json"

DEVICE = torch.device("cpu")


# ============================================================================
# HELPERS
# ============================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def sha256_jsonable(obj) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    tmp.replace(path)


def require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )


# ============================================================================
# EXACT C-BASE ARCHITECTURE
# ============================================================================

class CBase(nn.Module):
    """
    Exact Phase C C-Base architecture verified from cbase_best.pt:

        text_projection: 384 -> 512 + ReLU
        image_projection: 512 -> 512 + ReLU
        fusion:
            concat -> 1024
            1024 -> 512 + ReLU
            512 -> 549

    The 512-D output of fusion[0] + fusion[1] is the joint multimodal
    representation used for C5 representation inspection.
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(
                text_dim,
                512,
            ),
            nn.ReLU(),
        )

        self.image_projection = nn.Sequential(
            nn.Linear(
                image_dim,
                512,
            ),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                1024,
                512,
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                num_classes,
            ),
        )

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        return_joint: bool = False,
    ):
        text_projected = self.text_projection(text)

        image_projected = self.image_projection(image)

        combined = torch.cat(
            [
                text_projected,
                image_projected,
            ],
            dim=1,
        )

        joint = self.fusion[1](
            self.fusion[0](combined)
        )

        logits = self.fusion[2](joint)

        if return_joint:
            return logits, joint

        return logits


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    section(
        "C5.1 — RELIABILITY SIGNAL INVENTORY & AUDIT"
    )

    set_seed(SEED)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Required artifacts
    # ------------------------------------------------------------------------

    section("ARTIFACT AVAILABILITY")

    required = [
        TEXT_PATH,
        IMAGE_PATH,
        LABEL_PATH,
        ID_PATH,
        C4_MANIFEST,
        CHECKPOINT_PATH,
    ]

    for path in required:
        require_exists(path)
        print(f"PASS | {path}")

    # ------------------------------------------------------------------------
    # Load C4.1 frozen test artifacts
    # ------------------------------------------------------------------------

    section("LOAD FROZEN C4.1 TEST ARTIFACTS")

    text = np.load(
        TEXT_PATH,
        mmap_mode="r",
    )

    image = np.load(
        IMAGE_PATH,
        mmap_mode="r",
    )

    labels = np.load(
        LABEL_PATH,
        mmap_mode="r",
    )

    record_ids = load_json(
        ID_PATH
    )

    print(
        f"Text shape   : {text.shape}"
    )
    print(
        f"Image shape  : {image.shape}"
    )
    print(
        f"Labels shape : {labels.shape}"
    )
    print(
        f"Record IDs   : {len(record_ids)}"
    )

    if text.shape != (
        EXPECTED_TEST,
        TEXT_DIM,
    ):
        raise RuntimeError(
            f"Unexpected test text shape: {text.shape}"
        )

    if image.shape != (
        EXPECTED_TEST,
        IMAGE_DIM,
    ):
        raise RuntimeError(
            f"Unexpected test image shape: {image.shape}"
        )

    if labels.shape != (
        EXPECTED_TEST,
    ):
        raise RuntimeError(
            f"Unexpected test label shape: {labels.shape}"
        )

    if len(record_ids) != EXPECTED_TEST:
        raise RuntimeError(
            f"Unexpected test ID count: {len(record_ids)}"
        )

    if len(set(record_ids)) != EXPECTED_TEST:
        raise RuntimeError(
            "Test record IDs are not unique."
        )

    if not np.isfinite(text).all():
        raise RuntimeError(
            "Non-finite values found in test text embeddings."
        )

    if not np.isfinite(image).all():
        raise RuntimeError(
            "Non-finite values found in test image embeddings."
        )

    if not np.isfinite(labels).all():
        raise RuntimeError(
            "Non-finite labels found."
        )

    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise RuntimeError(
            "Test labels outside frozen 549-class space."
        )

    # ------------------------------------------------------------------------
    # Load frozen C3 checkpoint
    # ------------------------------------------------------------------------

    section("LOAD FROZEN C-BASE CHECKPOINT")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Frozen C-Base checkpoint is not a dictionary."
        )

    print(
        f"Checkpoint version : "
        f"{checkpoint.get('checkpoint_version')}"
    )

    print(
        f"Phase               : "
        f"{checkpoint.get('phase')}"
    )

    print(
        f"Model name          : "
        f"{checkpoint.get('model_name')}"
    )

    print(
        f"Best epoch          : "
        f"{checkpoint.get('best_epoch')}"
    )

    print(
        f"Selection metric    : "
        f"{checkpoint.get('selection_metric')}"
    )

    print(
        f"Test accessed       : "
        f"{checkpoint.get('test_accessed')}"
    )

    # ------------------------------------------------------------------------
    # Frozen checkpoint integrity checks
    # ------------------------------------------------------------------------

    if checkpoint.get("phase") != "C3":
        raise RuntimeError(
            f"Unexpected checkpoint phase: "
            f"{checkpoint.get('phase')}"
        )

    checkpoint_best_epoch = checkpoint.get(
        "best_epoch"
    )

    if checkpoint_best_epoch != 17:
        raise RuntimeError(
            f"Expected frozen best epoch 17, "
            f"found {checkpoint_best_epoch}"
        )

    if checkpoint.get("selection_metric") != "macro_f1":
        raise RuntimeError(
            "Unexpected checkpoint selection metric: "
            f"{checkpoint.get('selection_metric')}"
        )

    if checkpoint.get("test_accessed") is not False:
        raise RuntimeError(
            "Checkpoint reports test_accessed != False."
        )

    architecture = checkpoint.get(
        "architecture"
    )

    if not isinstance(
        architecture,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint does not contain valid architecture metadata."
        )

    if architecture.get("name") != "C-Base":
        raise RuntimeError(
            "Unexpected checkpoint architecture name: "
            f"{architecture.get('name')}"
        )

    if architecture.get("num_classes") != NUM_CLASSES:
        raise RuntimeError(
            "Checkpoint class count mismatch: "
            f"{architecture.get('num_classes')}"
        )

    if architecture.get("text_input_dim") != TEXT_DIM:
        raise RuntimeError(
            "Checkpoint text dimension mismatch: "
            f"{architecture.get('text_input_dim')}"
        )

    if architecture.get("image_input_dim") != IMAGE_DIM:
        raise RuntimeError(
            "Checkpoint image dimension mismatch: "
            f"{architecture.get('image_input_dim')}"
        )

    if architecture.get(
        "trainable_parameter_count"
    ) != 1_266_213:
        raise RuntimeError(
            "Checkpoint parameter-count metadata mismatch: "
            f"{architecture.get('trainable_parameter_count')}"
        )

    # ------------------------------------------------------------------------
    # Reconstruct exact frozen C-Base
    # ------------------------------------------------------------------------

    model = CBase(
        text_dim=TEXT_DIM,
        image_dim=IMAGE_DIM,
        num_classes=NUM_CLASSES,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print(
        "PASS | Frozen C-Base state_dict loaded."
    )

    print(
        f"PASS | Best validation epoch: "
        f"{checkpoint_best_epoch}"
    )

    print(
        f"PASS | Best validation Macro-F1: "
        f"{checkpoint.get('best_metric'):.12f}"
    )

    # ------------------------------------------------------------------------
    # Inference-only signal extraction
    # ------------------------------------------------------------------------

    section(
        "INFERENCE-ONLY RELIABILITY SIGNAL EXTRACTION"
    )

    # Copy read-only memmaps into ordinary writable NumPy arrays before
    # conversion to PyTorch tensors.
    #
    # This does NOT modify the underlying C4.1 artifacts.

    text_tensor = torch.from_numpy(
        np.asarray(
            text,
            dtype=np.float32,
        ).copy()
    ).to(DEVICE)

    image_tensor = torch.from_numpy(
        np.asarray(
            image,
            dtype=np.float32,
        ).copy()
    ).to(DEVICE)

    with torch.no_grad():

        logits_tensor, joint_tensor = model(
            text_tensor,
            image_tensor,
            return_joint=True,
        )

        probabilities_tensor = torch.softmax(
            logits_tensor,
            dim=1,
        )

    logits = (
        logits_tensor
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    probabilities = (
        probabilities_tensor
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    joint_representation = (
        joint_tensor
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

    # ------------------------------------------------------------------------
    # Top-1 / Top-2 probabilities
    # ------------------------------------------------------------------------

    top_two = np.partition(
        probabilities,
        kth=NUM_CLASSES - 2,
        axis=1,
    )[:, -2:]

    top_two.sort(
        axis=1
    )

    top2_probability = (
        top_two[:, 0]
        .astype(np.float32)
    )

    top1_probability = (
        top_two[:, 1]
        .astype(np.float32)
    )

    margin = (
        top1_probability
        - top2_probability
    ).astype(np.float32)

    # ------------------------------------------------------------------------
    # Predictive entropy
    # ------------------------------------------------------------------------

    # Natural-log units (nats).
    safe_probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    entropy = (
        -np.sum(
            safe_probabilities
            * np.log(safe_probabilities),
            axis=1,
        )
    ).astype(np.float32)

    true_labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    correct = (
        predictions == true_labels
    )

    # ------------------------------------------------------------------------
    # Mathematical / structural integrity
    # ------------------------------------------------------------------------

    section("SIGNAL INTEGRITY")

    if logits.shape != (
        EXPECTED_TEST,
        NUM_CLASSES,
    ):
        raise RuntimeError(
            f"Logit shape mismatch: {logits.shape}"
        )

    if probabilities.shape != (
        EXPECTED_TEST,
        NUM_CLASSES,
    ):
        raise RuntimeError(
            f"Probability shape mismatch: "
            f"{probabilities.shape}"
        )

    if predictions.shape != (
        EXPECTED_TEST,
    ):
        raise RuntimeError(
            f"Prediction shape mismatch: "
            f"{predictions.shape}"
        )

    if top1_probability.shape != (
        EXPECTED_TEST,
    ):
        raise RuntimeError(
            f"Top-1 probability shape mismatch: "
            f"{top1_probability.shape}"
        )

    if top2_probability.shape != (
        EXPECTED_TEST,
    ):
        raise RuntimeError(
            f"Top-2 probability shape mismatch: "
            f"{top2_probability.shape}"
        )

    if margin.shape != (
        EXPECTED_TEST,
    ):
        raise RuntimeError(
            f"Margin shape mismatch: "
            f"{margin.shape}"
        )

    if entropy.shape != (
        EXPECTED_TEST,
    ):
        raise RuntimeError(
            f"Entropy shape mismatch: "
            f"{entropy.shape}"
        )

    if correct.shape != (
        EXPECTED_TEST,
    ):
        raise RuntimeError(
            f"Correctness shape mismatch: "
            f"{correct.shape}"
        )

    if joint_representation.shape != (
        EXPECTED_TEST,
        FUSION_DIM,
    ):
        raise RuntimeError(
            "Joint representation shape mismatch: "
            f"{joint_representation.shape}"
        )

    if not np.isfinite(logits).all():
        raise RuntimeError(
            "Non-finite logits detected."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Non-finite probabilities detected."
        )

    if not np.isfinite(
        joint_representation
    ).all():
        raise RuntimeError(
            "Non-finite joint representations detected."
        )

    if not np.isfinite(
        top1_probability
    ).all():
        raise RuntimeError(
            "Non-finite top-1 probabilities detected."
        )

    if not np.isfinite(
        top2_probability
    ).all():
        raise RuntimeError(
            "Non-finite top-2 probabilities detected."
        )

    if not np.isfinite(
        margin
    ).all():
        raise RuntimeError(
            "Non-finite margins detected."
        )

    if not np.isfinite(
        entropy
    ).all():
        raise RuntimeError(
            "Non-finite entropy values detected."
        )

    probability_sums = probabilities.sum(
        axis=1
    )

    max_probability_sum_error = float(
        np.max(
            np.abs(
                probability_sums - 1.0
            )
        )
    )

    if max_probability_sum_error > 1e-5:
        raise RuntimeError(
            "Softmax probability rows do not "
            "sum to 1 within tolerance."
        )

    if np.any(
        top1_probability < 0
    ) or np.any(
        top1_probability > 1
    ):
        raise RuntimeError(
            "Top-1 probabilities outside [0,1]."
        )

    if np.any(
        top2_probability < 0
    ) or np.any(
        top2_probability > 1
    ):
        raise RuntimeError(
            "Top-2 probabilities outside [0,1]."
        )

    if np.any(
        margin < -1e-7
    ):
        raise RuntimeError(
            "Top-1/top-2 margins contain negative values."
        )

    if np.any(
        entropy < -1e-6
    ):
        raise RuntimeError(
            "Predictive entropy contains negative values."
        )

    print(
        "PASS | Logits finite."
    )

    print(
        "PASS | Probabilities finite."
    )

    print(
        "PASS | Joint representations finite."
    )

    print(
        "PASS | Softmax rows sum to 1."
    )

    print(
        f"PASS | Maximum probability-sum error: "
        f"{max_probability_sum_error:.12e}"
    )

    # ------------------------------------------------------------------------
    # Reproduce C4.2 metrics exactly
    # ------------------------------------------------------------------------

    section("C4.2 METRIC REPRODUCTION")

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        true_labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        average="weighted",
        zero_division=0,
    )

    expected_accuracy = 0.786823
    expected_macro_f1 = 0.341346
    expected_weighted_f1 = 0.782435

    print(
        f"Accuracy    : {accuracy:.6f}"
    )

    print(
        f"Macro-F1    : {macro_f1:.6f}"
    )

    print(
        f"Weighted-F1 : {weighted_f1:.6f}"
    )

    if abs(
        accuracy - expected_accuracy
    ) > 5e-5:
        raise RuntimeError(
            "C4.2 accuracy reproduction failed."
        )

    if abs(
        macro_f1 - expected_macro_f1
    ) > 5e-5:
        raise RuntimeError(
            "C4.2 Macro-F1 reproduction failed."
        )

    if abs(
        weighted_f1 - expected_weighted_f1
    ) > 5e-5:
        raise RuntimeError(
            "C4.2 Weighted-F1 reproduction failed."
        )

    print(
        "PASS | C4.2 metrics reproduced from "
        "frozen epoch-17 checkpoint."
    )

    # ------------------------------------------------------------------------
    # Save arrays
    # ------------------------------------------------------------------------

    section("SAVE C5.1 SIGNAL ARTIFACTS")

    np.save(
        LOGITS_PATH,
        logits,
    )

    np.save(
        PROBABILITIES_PATH,
        probabilities,
    )

    np.save(
        PREDICTIONS_PATH,
        predictions,
    )

    np.save(
        TOP1_PROB_PATH,
        top1_probability,
    )

    np.save(
        TOP2_PROB_PATH,
        top2_probability,
    )

    np.save(
        MARGIN_PATH,
        margin,
    )

    np.save(
        ENTROPY_PATH,
        entropy,
    )

    np.save(
        CORRECT_PATH,
        correct.astype(np.bool_),
    )

    np.save(
        JOINT_REP_PATH,
        joint_representation,
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    summary = {
        "phase": "C5",
        "stage": "C5.1",
        "task": "Reliability Signal Inventory and Audit",
        "dataset": "ABO",
        "target_field": "product_type",
        "num_classes": NUM_CLASSES,
        "test_population": EXPECTED_TEST,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_best_epoch": checkpoint_best_epoch,
        "checkpoint_selection_metric": checkpoint.get(
            "selection_metric"
        ),
        "best_validation_macro_f1": checkpoint.get(
            "best_metric"
        ),
        "signals": {
            "logits": {
                "available": True,
                "shape": list(
                    logits.shape
                ),
                "dtype": str(
                    logits.dtype
                ),
            },
            "softmax_probabilities": {
                "available": True,
                "shape": list(
                    probabilities.shape
                ),
                "dtype": str(
                    probabilities.dtype
                ),
            },
            "top1_prediction": {
                "available": True,
                "shape": list(
                    predictions.shape
                ),
            },
            "top1_probability": {
                "available": True,
                "shape": list(
                    top1_probability.shape
                ),
            },
            "top2_probability": {
                "available": True,
                "shape": list(
                    top2_probability.shape
                ),
            },
            "top1_top2_margin": {
                "available": True,
                "shape": list(
                    margin.shape
                ),
            },
            "predictive_entropy": {
                "available": True,
                "shape": list(
                    entropy.shape
                ),
                "units": "nats",
            },
            "correctness": {
                "available": True,
                "shape": list(
                    correct.shape
                ),
            },
            "joint_representation": {
                "available": True,
                "shape": list(
                    joint_representation.shape
                ),
                "dimension": FUSION_DIM,
                "definition": (
                    "C-Base fusion hidden representation after "
                    "1024->512 linear layer and ReLU, before "
                    "512->549 classifier."
                ),
            },
        },
        "metric_reproduction": {
            "accuracy": float(
                accuracy
            ),
            "macro_f1": float(
                macro_f1
            ),
            "weighted_f1": float(
                weighted_f1
            ),
            "expected_c4_2_accuracy": expected_accuracy,
            "expected_c4_2_macro_f1": expected_macro_f1,
            "expected_c4_2_weighted_f1": expected_weighted_f1,
        },
        "signal_integrity": {
            "max_probability_row_sum_error": (
                max_probability_sum_error
            ),
            "all_logits_finite": bool(
                np.isfinite(
                    logits
                ).all()
            ),
            "all_probabilities_finite": bool(
                np.isfinite(
                    probabilities
                ).all()
            ),
            "all_joint_representation_finite": bool(
                np.isfinite(
                    joint_representation
                ).all()
            ),
        },
        "test_usage": {
            "test_access": True,
            "usage": (
                "final inference-only reliability "
                "signal extraction"
            ),
            "training": False,
            "model_selection": False,
            "hyperparameter_tuning": False,
        },
        "scientific_status": {
            "calibration": "UNKNOWN",
            "selective_prediction": "UNKNOWN",
            "abstention": "UNKNOWN",
            "human_review_routing": "UNKNOWN",
            "open_set_detection": "UNKNOWN",
            "confidence_interpretation": (
                "Raw predictive probability/top-1 score "
                "is available as a model output, but "
                "calibration or reliability is not established."
            ),
        },
        "provenance": {
            "seed": SEED,
            "b3_modified": False,
            "b4_modified": False,
            "b5_modified": False,
            "c3_modified": False,
        },
    }

    save_json_atomic(
        SUMMARY_PATH,
        summary,
    )

    # ------------------------------------------------------------------------
    # Manifest with hashes
    # ------------------------------------------------------------------------

    manifest = {
        "phase": "C5",
        "stage": "C5.1",
        "artifact_type": (
            "reliability_signal_inventory"
        ),
        "source_artifacts": {
            "checkpoint": {
                "path": str(
                    CHECKPOINT_PATH
                ),
                "sha256": sha256_file(
                    CHECKPOINT_PATH
                ),
            },
            "c4_manifest": {
                "path": str(
                    C4_MANIFEST
                ),
                "sha256": sha256_file(
                    C4_MANIFEST
                ),
            },
            "test_text": {
                "path": str(
                    TEXT_PATH
                ),
                "sha256": sha256_file(
                    TEXT_PATH
                ),
            },
            "test_image": {
                "path": str(
                    IMAGE_PATH
                ),
                "sha256": sha256_file(
                    IMAGE_PATH
                ),
            },
            "test_labels": {
                "path": str(
                    LABEL_PATH
                ),
                "sha256": sha256_file(
                    LABEL_PATH
                ),
            },
            "test_record_ids": {
                "path": str(
                    ID_PATH
                ),
                "sha256": sha256_file(
                    ID_PATH
                ),
            },
        },
        "checkpoint_metadata": {
            "checkpoint_version": checkpoint.get(
                "checkpoint_version"
            ),
            "phase": checkpoint.get(
                "phase"
            ),
            "model_name": checkpoint.get(
                "model_name"
            ),
            "best_epoch": checkpoint.get(
                "best_epoch"
            ),
            "best_metric": checkpoint.get(
                "best_metric"
            ),
            "selection_metric": checkpoint.get(
                "selection_metric"
            ),
            "test_accessed": checkpoint.get(
                "test_accessed"
            ),
        },
        "outputs": {
            "logits": str(
                LOGITS_PATH
            ),
            "probabilities": str(
                PROBABILITIES_PATH
            ),
            "predictions": str(
                PREDICTIONS_PATH
            ),
            "top1_probability": str(
                TOP1_PROB_PATH
            ),
            "top2_probability": str(
                TOP2_PROB_PATH
            ),
            "margin": str(
                MARGIN_PATH
            ),
            "entropy": str(
                ENTROPY_PATH
            ),
            "correct": str(
                CORRECT_PATH
            ),
            "joint_representation": str(
                JOINT_REP_PATH
            ),
            "summary": str(
                SUMMARY_PATH
            ),
        },
        "output_shapes": {
            "logits": list(
                logits.shape
            ),
            "probabilities": list(
                probabilities.shape
            ),
            "predictions": list(
                predictions.shape
            ),
            "top1_probability": list(
                top1_probability.shape
            ),
            "top2_probability": list(
                top2_probability.shape
            ),
            "margin": list(
                margin.shape
            ),
            "entropy": list(
                entropy.shape
            ),
            "correct": list(
                correct.shape
            ),
            "joint_representation": list(
                joint_representation.shape
            ),
        },
        "integrity": {
            "exact_test_population": EXPECTED_TEST,
            "unique_test_record_ids": True,
            "test_labels_in_549_class_space": True,
            "c4_2_metrics_reproduced": True,
            "training": False,
            "model_selection": False,
            "hyperparameter_tuning": False,
            "b3_modified": False,
            "b4_modified": False,
            "b5_modified": False,
            "c3_modified": False,
        },
        "scientific_boundary": (
            "C5.1 inventories raw reliability-related "
            "model signals. It does not establish "
            "calibration, selective prediction, "
            "abstention, human-review routing, "
            "or open-set detection."
        ),
    }

    manifest["manifest_sha256"] = (
        sha256_jsonable(
            {
                k: v
                for k, v in manifest.items()
                if k != "manifest_sha256"
            }
        )
    )

    save_json_atomic(
        MANIFEST_PATH,
        manifest,
    )

    # ------------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------------

    section("C5.1 FINAL STATUS")

    print(
        "C5.1 COMPLETE"
    )

    print()
    print(
        f"Test records        : {EXPECTED_TEST}"
    )
    print(
        f"Logits              : {logits.shape}"
    )
    print(
        f"Probabilities       : {probabilities.shape}"
    )
    print(
        f"Joint representation: "
        f"{joint_representation.shape}"
    )

    print(
        f"Accuracy            : "
        f"{accuracy:.6f}"
    )

    print(
        f"Macro-F1            : "
        f"{macro_f1:.6f}"
    )

    print(
        f"Weighted-F1         : "
        f"{weighted_f1:.6f}"
    )

    print()
    print(
        "Scientific status:"
    )

    print(
        "  Calibration             : UNKNOWN"
    )

    print(
        "  Selective prediction    : UNKNOWN"
    )

    print(
        "  Abstention              : UNKNOWN"
    )

    print(
        "  Human-review routing    : UNKNOWN"
    )

    print(
        "  Open-set detection      : UNKNOWN"
    )

    print()
    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    print()
    print(
        "Integrity:"
    )

    print(
        "  Training               : NO"
    )

    print(
        "  Model selection        : NO"
    )

    print(
        "  Hyperparameter tuning  : NO"
    )

    print(
        "  B3 modified            : NO"
    )

    print(
        "  B4 modified            : NO"
    )

    print(
        "  B5 modified            : NO"
    )

    print(
        "  C3 modified            : NO"
    )

    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------

    del text_tensor
    del image_tensor
    del logits_tensor
    del probabilities_tensor
    del joint_tensor

    gc.collect()


if __name__ == "__main__":
    main()