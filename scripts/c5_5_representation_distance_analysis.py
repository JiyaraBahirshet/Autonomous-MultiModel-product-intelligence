from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


# ============================================================================
# Phase C5.5 — Representation Distance Analysis
# ============================================================================
# Purpose:
#   Analyze frozen C-Base joint-representation geometry on the isolated
#   7,346-record test population using class prototypes computed ONLY from
#   the frozen Phase C training population.
#
# Scientific boundary:
#   - inference / descriptive analysis only
#   - no training
#   - no checkpoint selection
#   - no threshold optimization
#   - no calibration fitting
#   - no test-driven prototype construction
#   - no selective-prediction / abstention / human-review claim
#   - no open-set claim
#
# The class prototype is the mean C-Base joint representation for each class
# over the frozen Phase C training population.
#
# Distances reported:
#   1. Euclidean distance to the true-class prototype
#   2. Euclidean distance to the nearest competing class prototype
#   3. Prototype margin = nearest competitor distance - true-class distance
#   4. Cosine distance to the true-class prototype
#   5. Cosine distance to the nearest competing class prototype
#   6. Cosine prototype margin
#
# IMPORTANT:
#   These are representation-geometry diagnostics, not confidence scores.
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NUM_CLASSES = 549
TEXT_DIM = 384
IMAGE_DIM = 512
JOINT_DIM = 512
EXPECTED_TRAIN = 69823
EXPECTED_TEST = 7346
SEED = 20260827

TRAIN_TEXT_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c3"
    / "embeddings"
    / "train_text.npy"
)
TRAIN_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c3"
    / "embeddings"
    / "train_image.npy"
)
TRAIN_LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c3"
    / "embeddings"
    / "train_labels.npy"
)
TRAIN_IDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c3"
    / "embeddings"
    / "train_record_ids.json"
)

TEST_TEXT_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c4"
    / "test_embeddings"
    / "test_text.npy"
)
TEST_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c4"
    / "test_embeddings"
    / "test_image.npy"
)
TEST_LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c4"
    / "test_embeddings"
    / "test_labels.npy"
)
TEST_IDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c4"
    / "test_embeddings"
    / "test_record_ids.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c3"
    / "training"
    / "cbase_best.pt"
)

C5_1_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c5"
    / "c5_1_reliability_signal_audit"
)
C5_1_PREDICTIONS_PATH = C5_1_DIR / "c5_1_predictions.npy"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c5"
    / "c5_5_representation_distance_analysis"
)

DEVICE = torch.device("cpu")
BATCH_SIZE = 256


class CBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(TEXT_DIM, 512),
            torch.nn.ReLU(),
        )

        self.image_projection = torch.nn.Sequential(
            torch.nn.Linear(IMAGE_DIM, 512),
            torch.nn.ReLU(),
        )

        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, text, image, return_joint=False):
        text_projected = self.text_projection(text)
        image_projected = self.image_projection(image)
        joint = self.fusion[0](
            torch.cat([text_projected, image_projected], dim=1)
        )
        joint = self.fusion[1](joint)
        logits = self.fusion[2](joint)

        if return_joint:
            return logits, joint

        return logits


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_required(path: Path, description: str):
    if not path.is_file():
        fail(f"Missing {description}: {path}")
    return path


def load_json(path: Path, description: str):
    load_required(path, description)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_ids(ids, expected_count, name):
    if not isinstance(ids, list):
        fail(f"{name} is not a JSON list.")
    if len(ids) != expected_count:
        fail(f"{name} count {len(ids)} != expected {expected_count}.")
    if len(set(ids)) != len(ids):
        fail(f"{name} contains duplicates.")


def validate_embedding_array(array, expected_rows, expected_dim, name):
    if array.shape != (expected_rows, expected_dim):
        fail(
            f"{name} shape {array.shape} != "
            f"({expected_rows}, {expected_dim})."
        )
    if not np.all(np.isfinite(array)):
        fail(f"{name} contains non-finite values.")


def load_checkpoint():
    load_required(CHECKPOINT_PATH, "C-Base checkpoint")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("test_accessed") is not False:
        fail(
            "Checkpoint metadata does not explicitly report "
            "test_accessed=false."
        )

    if checkpoint.get("best_epoch") != 17:
        fail(
            f"Expected frozen best_epoch=17, got "
            f"{checkpoint.get('best_epoch')}."
        )

    if checkpoint.get("selection_metric") != "macro_f1":
        fail(
            "Checkpoint selection metric is not the frozen macro_f1."
        )

    model = CBase().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint, model


@torch.inference_mode()
def compute_joint_representation(model, text, image):
    n = text.shape[0]
    output = np.empty((n, JOINT_DIM), dtype=np.float32)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)

        # Explicit writable copies avoid the non-writable memmap warning.
        t = torch.from_numpy(
            np.array(text[start:end], dtype=np.float32, copy=True)
        )
        im = torch.from_numpy(
            np.array(image[start:end], dtype=np.float32, copy=True)
        )

        _, joint = model(t, im, return_joint=True)
        output[start:end] = joint.cpu().numpy()

    return output


def build_prototypes(train_joint, train_labels):
    prototypes = np.empty(
        (NUM_CLASSES, JOINT_DIM),
        dtype=np.float32,
    )

    counts = np.bincount(
        train_labels,
        minlength=NUM_CLASSES,
    )

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        fail(
            "Training prototype construction has classes with zero support: "
            f"{missing[:20]}"
        )

    for cls in range(NUM_CLASSES):
        members = train_joint[train_labels == cls]
        prototypes[cls] = members.mean(axis=0)

    return prototypes, counts


def euclidean_distance_matrix(test_joint, prototypes):
    # Squared-distance identity avoids a huge (N, C, D) temporary.
    test_sq = np.sum(test_joint.astype(np.float64) ** 2, axis=1)
    proto_sq = np.sum(prototypes.astype(np.float64) ** 2, axis=1)

    distances_sq = (
        test_sq[:, None]
        + proto_sq[None, :]
        - 2.0
        * (
            test_joint.astype(np.float64)
            @ prototypes.astype(np.float64).T
        )
    )

    distances_sq = np.maximum(distances_sq, 0.0)
    return np.sqrt(distances_sq).astype(np.float32)


def cosine_distance_matrix(test_joint, prototypes):
    test = test_joint.astype(np.float64)
    proto = prototypes.astype(np.float64)

    test_norm = np.linalg.norm(test, axis=1, keepdims=True)
    proto_norm = np.linalg.norm(proto, axis=1, keepdims=True)

    if np.any(test_norm <= 0) or np.any(proto_norm <= 0):
        fail("Zero-norm representation encountered.")

    normalized_test = test / test_norm
    normalized_proto = proto / proto_norm

    cosine_similarity = normalized_test @ normalized_proto.T
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

    return (1.0 - cosine_similarity).astype(np.float32)


def summarize(values, name):
    values = np.asarray(values)

    return {
        "name": name,
        "finite": bool(np.all(np.isfinite(values))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if np.std(a) == 0 or np.std(b) == 0:
        return None

    return float(np.corrcoef(a, b)[0, 1])


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Load and validate frozen inputs
    # ----------------------------------------------------------------------
    train_text = np.load(
        load_required(TRAIN_TEXT_PATH, "train text embeddings"),
        mmap_mode="r",
    )
    train_image = np.load(
        load_required(TRAIN_IMAGE_PATH, "train image embeddings"),
        mmap_mode="r",
    )
    train_labels = np.load(
        load_required(TRAIN_LABELS_PATH, "train labels"),
        mmap_mode="r",
    )
    train_ids = load_json(TRAIN_IDS_PATH, "train record IDs")

    test_text = np.load(
        load_required(TEST_TEXT_PATH, "test text embeddings"),
        mmap_mode="r",
    )
    test_image = np.load(
        load_required(TEST_IMAGE_PATH, "test image embeddings"),
        mmap_mode="r",
    )
    test_labels = np.load(
        load_required(TEST_LABELS_PATH, "test labels"),
        mmap_mode="r",
    )
    test_ids = load_json(TEST_IDS_PATH, "test record IDs")

    validate_embedding_array(
        train_text,
        EXPECTED_TRAIN,
        TEXT_DIM,
        "train_text",
    )
    validate_embedding_array(
        train_image,
        EXPECTED_TRAIN,
        IMAGE_DIM,
        "train_image",
    )
    validate_embedding_array(
        test_text,
        EXPECTED_TEST,
        TEXT_DIM,
        "test_text",
    )
    validate_embedding_array(
        test_image,
        EXPECTED_TEST,
        IMAGE_DIM,
        "test_image",
    )

    if train_labels.shape != (EXPECTED_TRAIN,):
        fail(f"train_labels shape {train_labels.shape} is invalid.")
    if test_labels.shape != (EXPECTED_TEST,):
        fail(f"test_labels shape {test_labels.shape} is invalid.")

    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    if np.any(train_labels < 0) or np.any(train_labels >= NUM_CLASSES):
        fail("Training labels outside 549-class space.")
    if np.any(test_labels < 0) or np.any(test_labels >= NUM_CLASSES):
        fail("Test labels outside 549-class space.")

    validate_ids(train_ids, EXPECTED_TRAIN, "train_record_ids")
    validate_ids(test_ids, EXPECTED_TEST, "test_record_ids")

    if set(train_ids) & set(test_ids):
        fail("Train/test record-ID overlap detected.")

    # ----------------------------------------------------------------------
    # Frozen C5.1 prediction consistency
    # ----------------------------------------------------------------------
    c5_1_predictions = np.load(
        load_required(
            C5_1_PREDICTIONS_PATH,
            "frozen C5.1 predictions",
        )
    )

    if c5_1_predictions.shape != (EXPECTED_TEST,):
        fail(
            f"C5.1 predictions shape {c5_1_predictions.shape} is invalid."
        )

    c5_1_predictions = np.asarray(
        c5_1_predictions,
        dtype=np.int64,
    )

    # ----------------------------------------------------------------------
    # Load frozen checkpoint and compute train/test joint representations.
    # ----------------------------------------------------------------------
    checkpoint, model = load_checkpoint()

    train_joint = compute_joint_representation(
        model,
        train_text,
        train_image,
    )
    test_joint = compute_joint_representation(
        model,
        test_text,
        test_image,
    )

    if not np.all(np.isfinite(train_joint)):
        fail("Train joint representations contain non-finite values.")
    if not np.all(np.isfinite(test_joint)):
        fail("Test joint representations contain non-finite values.")

    # ----------------------------------------------------------------------
    # Build class prototypes from TRAIN ONLY.
    # ----------------------------------------------------------------------
    prototypes, class_counts = build_prototypes(
        train_joint,
        train_labels,
    )

    # ----------------------------------------------------------------------
    # Compute representation distances on TEST.
    # ----------------------------------------------------------------------
    euclidean = euclidean_distance_matrix(
        test_joint,
        prototypes,
    )
    cosine = cosine_distance_matrix(
        test_joint,
        prototypes,
    )

    if not np.all(np.isfinite(euclidean)):
        fail("Euclidean distances contain non-finite values.")
    if not np.all(np.isfinite(cosine)):
        fail("Cosine distances contain non-finite values.")

    rows = np.arange(EXPECTED_TEST)

    true_euclidean = euclidean[rows, test_labels]
    true_cosine = cosine[rows, test_labels]

    euclidean_masked = euclidean.copy()
    euclidean_masked[rows, test_labels] = np.inf

    cosine_masked = cosine.copy()
    cosine_masked[rows, test_labels] = np.inf

    nearest_competitor_euclidean = np.min(
        euclidean_masked,
        axis=1,
    )
    nearest_competitor_euclidean_class = np.argmin(
        euclidean_masked,
        axis=1,
    )

    nearest_competitor_cosine = np.min(
        cosine_masked,
        axis=1,
    )
    nearest_competitor_cosine_class = np.argmin(
        cosine_masked,
        axis=1,
    )

    euclidean_margin = (
        nearest_competitor_euclidean - true_euclidean
    )
    cosine_margin = (
        nearest_competitor_cosine - true_cosine
    )

    euclidean_nearest_class = np.argmin(euclidean, axis=1)
    cosine_nearest_class = np.argmin(cosine, axis=1)

    euclidean_proto_accuracy = accuracy_score(
        test_labels,
        euclidean_nearest_class,
    )
    cosine_proto_accuracy = accuracy_score(
        test_labels,
        cosine_nearest_class,
    )

    model_correct = c5_1_predictions == test_labels

    # ----------------------------------------------------------------------
    # Relationship summaries
    # ----------------------------------------------------------------------
    summary = {
        "status": "VERIFIED",
        "phase": "C5.5",
        "analysis": "representation_distance_geometry",
        "test_population": EXPECTED_TEST,
        "train_population": EXPECTED_TRAIN,
        "num_classes": NUM_CLASSES,
        "joint_dimension": JOINT_DIM,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "seed": SEED,
        "prototype_definition": (
            "Per-class mean of the frozen C-Base joint representation "
            "computed over the frozen Phase C training population only."
        ),
        "test_accessed_for_prototype_construction": False,
        "distance_metrics": {
            "euclidean": "L2 distance",
            "cosine": "1 - cosine similarity",
        },
        "prototype_class_support": {
            "min": int(np.min(class_counts)),
            "max": int(np.max(class_counts)),
            "mean": float(np.mean(class_counts)),
            "median": float(np.median(class_counts)),
            "classes_with_zero_train_support": int(
                np.sum(class_counts == 0)
            ),
        },
        "nearest_prototype_diagnostic_accuracy": {
            "euclidean": float(euclidean_proto_accuracy),
            "cosine": float(cosine_proto_accuracy),
        },
        "all_test": {
            "true_class_euclidean": summarize(
                true_euclidean,
                "true_class_euclidean",
            ),
            "nearest_competitor_euclidean": summarize(
                nearest_competitor_euclidean,
                "nearest_competitor_euclidean",
            ),
            "euclidean_margin": summarize(
                euclidean_margin,
                "euclidean_margin",
            ),
            "true_class_cosine_distance": summarize(
                true_cosine,
                "true_class_cosine_distance",
            ),
            "nearest_competitor_cosine_distance": summarize(
                nearest_competitor_cosine,
                "nearest_competitor_cosine_distance",
            ),
            "cosine_margin": summarize(
                cosine_margin,
                "cosine_margin",
            ),
        },
        "correct_vs_incorrect": {
            "true_class_euclidean": {
                "correct": summarize(
                    true_euclidean[model_correct],
                    "correct_true_class_euclidean",
                ),
                "incorrect": summarize(
                    true_euclidean[~model_correct],
                    "incorrect_true_class_euclidean",
                ),
            },
            "euclidean_margin": {
                "correct": summarize(
                    euclidean_margin[model_correct],
                    "correct_euclidean_margin",
                ),
                "incorrect": summarize(
                    euclidean_margin[~model_correct],
                    "incorrect_euclidean_margin",
                ),
            },
            "true_class_cosine_distance": {
                "correct": summarize(
                    true_cosine[model_correct],
                    "correct_true_class_cosine_distance",
                ),
                "incorrect": summarize(
                    true_cosine[~model_correct],
                    "incorrect_true_class_cosine_distance",
                ),
            },
            "cosine_margin": {
                "correct": summarize(
                    cosine_margin[model_correct],
                    "correct_cosine_margin",
                ),
                "incorrect": summarize(
                    cosine_margin[~model_correct],
                    "incorrect_cosine_margin",
                ),
            },
        },
        "signal_relationships": {
            "euclidean_margin_vs_model_correct": safe_corr(
                euclidean_margin.astype(np.float64),
                model_correct.astype(np.float64),
            ),
            "cosine_margin_vs_model_correct": safe_corr(
                cosine_margin.astype(np.float64),
                model_correct.astype(np.float64),
            ),
            "true_euclidean_vs_model_correct": safe_corr(
                true_euclidean.astype(np.float64),
                model_correct.astype(np.float64),
            ),
            "true_cosine_distance_vs_model_correct": safe_corr(
                true_cosine.astype(np.float64),
                model_correct.astype(np.float64),
            ),
        },
        "scientific_status": {
            "representation_distance_behavior": (
                "VERIFIED descriptive signal"
            ),
            "prototype_geometry_correctness_association": (
                "DEMONSTRATED descriptively"
            ),
            "representation_distance_as_confidence": "UNKNOWN",
            "calibration": "UNKNOWN",
            "selective_prediction": "UNKNOWN",
            "abstention": "UNKNOWN",
            "human_review_routing": "UNKNOWN",
            "open_set_detection": "UNKNOWN",
        },
    }

    # ----------------------------------------------------------------------
    # Save compact per-record arrays.
    # ----------------------------------------------------------------------
    np.save(
        OUTPUT_DIR / "c5_5_true_class_euclidean.npy",
        true_euclidean,
    )
    np.save(
        OUTPUT_DIR / "c5_5_nearest_competitor_euclidean.npy",
        nearest_competitor_euclidean,
    )
    np.save(
        OUTPUT_DIR / "c5_5_euclidean_margin.npy",
        euclidean_margin,
    )
    np.save(
        OUTPUT_DIR / "c5_5_true_class_cosine_distance.npy",
        true_cosine,
    )
    np.save(
        OUTPUT_DIR / "c5_5_nearest_competitor_cosine_distance.npy",
        nearest_competitor_cosine,
    )
    np.save(
        OUTPUT_DIR / "c5_5_cosine_margin.npy",
        cosine_margin,
    )
    np.save(
        OUTPUT_DIR / "c5_5_euclidean_nearest_class.npy",
        euclidean_nearest_class,
    )
    np.save(
        OUTPUT_DIR / "c5_5_cosine_nearest_class.npy",
        cosine_nearest_class,
    )
    np.save(
        OUTPUT_DIR / "c5_5_nearest_competitor_euclidean_class.npy",
        nearest_competitor_euclidean_class,
    )
    np.save(
        OUTPUT_DIR / "c5_5_nearest_competitor_cosine_class.npy",
        nearest_competitor_cosine_class,
    )

    np.save(
        OUTPUT_DIR / "c5_5_class_prototypes.npy",
        prototypes,
    )
    np.save(
        OUTPUT_DIR / "c5_5_class_train_counts.npy",
        class_counts,
    )

    with (OUTPUT_DIR / "c5_5_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    manifest = {
        "status": "VERIFIED",
        "phase": "C5.5",
        "analysis": "representation_distance_geometry",
        "seed": SEED,
        "test_accessed_for_prototype_construction": False,
        "train_population": EXPECTED_TRAIN,
        "test_population": EXPECTED_TEST,
        "num_classes": NUM_CLASSES,
        "artifacts": {
            "checkpoint": str(CHECKPOINT_PATH),
            "train_text": str(TRAIN_TEXT_PATH),
            "train_image": str(TRAIN_IMAGE_PATH),
            "train_labels": str(TRAIN_LABELS_PATH),
            "test_text": str(TEST_TEXT_PATH),
            "test_image": str(TEST_IMAGE_PATH),
            "test_labels": str(TEST_LABELS_PATH),
            "c5_1_predictions": str(C5_1_PREDICTIONS_PATH),
        },
        "scientific_controls": [
            "No training.",
            "No checkpoint selection.",
            "No hyperparameter tuning.",
            "No threshold optimization.",
            "No calibration fitting.",
            "Class prototypes computed from training population only.",
            "Test labels used only for descriptive correctness comparisons.",
            "Frozen C5.1 predictions used only as the model-correctness reference.",
            "No selective prediction, abstention, human-review, or open-set claim.",
        ],
        "file_hashes": {
            "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        },
    }

    with (OUTPUT_DIR / "c5_5_audit_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    print("C5.5 COMPLETE")
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
