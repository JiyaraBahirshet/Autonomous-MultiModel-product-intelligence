"""
C3 controlled training — C-Base
Autonomous Multimodal Product Intelligence

Primary Phase C model:
    frozen MiniLM text embedding  : 384-D
    frozen ResNet-18 image        : 512-D
    text projection               : 384 -> 512
    image projection              : 512 -> 512
    concatenation                 : 1024
    fusion hidden                 : 1024 -> 512
    classifier                    : 512 -> 549

Important:
- Uses ONLY precomputed Phase C3 embeddings.
- Does NOT access B3/raw data or the test split.
- Does NOT modify B3/B4/B5 artifacts.
- Checkpoints every completed epoch.
- Supports safe resume after interruption.
- Best checkpoint is selected on VALIDATION MACRO-F1 only.
  This is an initial C3 training-selection choice, not a Phase C success threshold.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c3"
    / "embeddings"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "abo"
    / "phase_c"
    / "c3"
    / "training"
)

MANIFEST_PATH = EMBEDDING_DIR / "c3_embedding_manifest.json"

TRAIN_TEXT_PATH = EMBEDDING_DIR / "train_text.npy"
TRAIN_IMAGE_PATH = EMBEDDING_DIR / "train_image.npy"
TRAIN_LABELS_PATH = EMBEDDING_DIR / "train_labels.npy"
TRAIN_IDS_PATH = EMBEDDING_DIR / "train_record_ids.json"

VAL_TEXT_PATH = EMBEDDING_DIR / "validation_text.npy"
VAL_IMAGE_PATH = EMBEDDING_DIR / "validation_image.npy"
VAL_LABELS_PATH = EMBEDDING_DIR / "validation_labels.npy"
VAL_IDS_PATH = EMBEDDING_DIR / "validation_record_ids.json"


# ============================================================================
# LOCKED PHASE C CONSTANTS
# ============================================================================

SEED = 20260827

NUM_CLASSES = 549
TEXT_DIM = 384
IMAGE_DIM = 512

TRAIN_COUNT = 69_823
VALIDATION_COUNT = 69_867

# Initial controlled-training configuration.
# These are C3 training choices, not frozen Phase C benchmark facts.
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Checkpoint after every completed epoch.
CHECKPOINT_EVERY_EPOCH = 1

# Primary validation selection criterion for this controlled run.
SELECTION_METRIC = "macro_f1"

# No test path is defined or loaded anywhere in this script.


# ============================================================================
# OUTPUT FILES
# ============================================================================

LATEST_CHECKPOINT_PATH = OUTPUT_DIR / "cbase_latest.pt"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "cbase_best.pt"
PROGRESS_PATH = OUTPUT_DIR / "cbase_training_progress.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "cbase_training_manifest.json"
HISTORY_PATH = OUTPUT_DIR / "cbase_training_history.json"


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CPU-only controlled run.
    # These settings make the intended deterministic behavior explicit.
    torch.use_deterministic_algorithms(True)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


# ============================================================================
# ATOMIC JSON
# ============================================================================

def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    os.replace(tmp_path, path)


# ============================================================================
# SHA-256
# ============================================================================

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def sha256_json_file(path: Path) -> str:
    return sha256_file(path)


# ============================================================================
# DATASET
# ============================================================================

class EmbeddingDataset(Dataset):
    def __init__(
        self,
        text_path: Path,
        image_path: Path,
        labels_path: Path,
        expected_count: int,
    ) -> None:
        self.text = np.load(text_path, mmap_mode="r")
        self.image = np.load(image_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

        if self.text.shape != (expected_count, TEXT_DIM):
            raise RuntimeError(
                f"Unexpected text shape: {self.text.shape}; "
                f"expected {(expected_count, TEXT_DIM)}"
            )

        if self.image.shape != (expected_count, IMAGE_DIM):
            raise RuntimeError(
                f"Unexpected image shape: {self.image.shape}; "
                f"expected {(expected_count, IMAGE_DIM)}"
            )

        if self.labels.shape != (expected_count,):
            raise RuntimeError(
                f"Unexpected labels shape: {self.labels.shape}; "
                f"expected {(expected_count,)}"
            )

        if self.text.dtype != np.float32:
            raise RuntimeError(
                f"Unexpected text dtype: {self.text.dtype}; expected float32"
            )

        if self.image.dtype != np.float32:
            raise RuntimeError(
                f"Unexpected image dtype: {self.image.dtype}; expected float32"
            )

        if self.labels.dtype != np.int64:
            raise RuntimeError(
                f"Unexpected labels dtype: {self.labels.dtype}; expected int64"
            )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int):
        text = torch.from_numpy(np.asarray(self.text[index])).float()
        image = torch.from_numpy(np.asarray(self.image[index])).float()
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return text, image, label


# ============================================================================
# MODEL
# ============================================================================

class CBase(nn.Module):
    def __init__(
        self,
        text_dim: int = TEXT_DIM,
        image_dim: int = IMAGE_DIM,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
        )

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> torch.Tensor:
        text_features = self.text_projection(text_embedding)
        image_features = self.image_projection(image_embedding)

        fused = torch.cat([text_features, image_features], dim=1)

        return self.fusion(fused)


# ============================================================================
# MODEL IDENTITY
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def architecture_config() -> dict[str, Any]:
    return {
        "name": "C-Base",
        "type": "feature_level_learned_fusion",
        "text_input_dim": TEXT_DIM,
        "image_input_dim": IMAGE_DIM,
        "text_projection": "384->512 + ReLU",
        "image_projection": "512->512 + ReLU",
        "fusion": "concat(512,512)->1024->512 + ReLU->549",
        "num_classes": NUM_CLASSES,
        "trainable_parameter_count": 1_266_213,
    }


# ============================================================================
# MANIFEST / ARTIFACT VALIDATION
# ============================================================================

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_required_files() -> None:
    required = [
        MANIFEST_PATH,
        TRAIN_TEXT_PATH,
        TRAIN_IMAGE_PATH,
        TRAIN_LABELS_PATH,
        TRAIN_IDS_PATH,
        VAL_TEXT_PATH,
        VAL_IMAGE_PATH,
        VAL_LABELS_PATH,
        VAL_IDS_PATH,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Required C3 embedding artifacts are missing:\n"
            + "\n".join(missing)
        )


def validate_manifest(manifest: dict[str, Any]) -> None:
    # The manifest is the source of the prepared C3 population facts.
    if manifest.get("dataset") != "ABO":
        raise RuntimeError(
            f"Manifest dataset is {manifest.get('dataset')!r}, expected 'ABO'."
        )

    if manifest.get("phase") != "C3":
        raise RuntimeError(
            f"Manifest phase is {manifest.get('phase')!r}, expected 'C3'."
        )

    if manifest.get("b3_modified") is True:
        raise RuntimeError("Manifest reports B3 was modified. Refusing to train.")

    if manifest.get("test_accessed") is True:
        raise RuntimeError(
            "Manifest reports test access. Refusing to use this artifact set."
        )

    if manifest.get("seed") != SEED:
        raise RuntimeError(
            f"Manifest seed {manifest.get('seed')!r} != locked seed {SEED}."
        )


def validate_arrays_and_ids() -> dict[str, str]:
    print("=" * 72)
    print("VALIDATING C3 EMBEDDING ARTIFACTS")
    print("=" * 72)

    checks = {}

    for split, expected in [
        ("train", TRAIN_COUNT),
        ("validation", VALIDATION_COUNT),
    ]:
        if split == "train":
            text_path = TRAIN_TEXT_PATH
            image_path = TRAIN_IMAGE_PATH
            labels_path = TRAIN_LABELS_PATH
            ids_path = TRAIN_IDS_PATH
        else:
            text_path = VAL_TEXT_PATH
            image_path = VAL_IMAGE_PATH
            labels_path = VAL_LABELS_PATH
            ids_path = VAL_IDS_PATH

        text = np.load(text_path, mmap_mode="r")
        image = np.load(image_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        ids = load_json(ids_path)

        print(f"\n{split.upper()}")
        print("text   :", text.shape, text.dtype)
        print("image  :", image.shape, image.dtype)
        print("labels :", labels.shape, labels.dtype)
        print("ids    :", len(ids))

        if text.shape != (expected, TEXT_DIM):
            raise RuntimeError(
                f"{split} text shape mismatch: {text.shape}"
            )

        if image.shape != (expected, IMAGE_DIM):
            raise RuntimeError(
                f"{split} image shape mismatch: {image.shape}"
            )

        if labels.shape != (expected,):
            raise RuntimeError(
                f"{split} labels shape mismatch: {labels.shape}"
            )

        if len(ids) != expected:
            raise RuntimeError(
                f"{split} ID count mismatch: {len(ids)}"
            )

        if text.dtype != np.float32 or image.dtype != np.float32:
            raise RuntimeError(
                f"{split} embedding dtype must be float32."
            )

        if labels.dtype != np.int64:
            raise RuntimeError(
                f"{split} label dtype must be int64."
            )

        if not np.isfinite(text).all():
            raise RuntimeError(f"{split} text contains non-finite values.")

        if not np.isfinite(image).all():
            raise RuntimeError(f"{split} image contains non-finite values.")

        if int(labels.min()) < 0 or int(labels.max()) >= NUM_CLASSES:
            raise RuntimeError(
                f"{split} labels outside [0,{NUM_CLASSES - 1}]."
            )

        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{split} contains duplicate record IDs.")

        print("finite text :", bool(np.isfinite(text).all()))
        print("finite image:", bool(np.isfinite(image).all()))
        print("label range:", int(labels.min()), int(labels.max()))
        print("ids unique :", len(ids) == len(set(ids)))
        print("PASS")

        # Exact artifact hashes provide strong resume identity protection.
        checks[f"{split}_text_sha256"] = sha256_file(text_path)
        checks[f"{split}_image_sha256"] = sha256_file(image_path)
        checks[f"{split}_labels_sha256"] = sha256_file(labels_path)
        checks[f"{split}_ids_sha256"] = sha256_json_file(ids_path)

    train_ids = load_json(TRAIN_IDS_PATH)
    val_ids = load_json(VAL_IDS_PATH)

    overlap = set(train_ids).intersection(val_ids)

    if overlap:
        raise RuntimeError(
            f"Train/validation record-ID overlap detected: {len(overlap)}"
        )

    print("\nTrain/validation record-ID overlap:", len(overlap))
    print("C3 artifact validation: PASS")

    return checks


# ============================================================================
# TRAINING METRICS
# ============================================================================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    total_loss = 0.0
    total_count = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for text, image, labels in loader:
            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            logits = model(text, image)
            loss = criterion(logits, labels)

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(
                predictions.cpu().numpy()
            )
            all_labels.append(
                labels.cpu().numpy()
            )

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_labels)

    return {
        "loss": total_loss / total_count,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================================
# CHECKPOINTS
# ============================================================================

@dataclass
class TrainingConfig:
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    selection_metric: str
    checkpoint_every_epoch: int
    num_classes: int
    text_dim: int
    image_dim: int


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    history: list[dict[str, Any]],
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "checkpoint_version": "c3_cbase_v001",
        "phase": "C3",
        "model_name": "C-Base",
        "epoch_completed": epoch,
        "total_epochs": EPOCHS,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "selection_metric": SELECTION_METRIC,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "history": history,
        "seed": SEED,
        "architecture": architecture_config(),
        "training_config": asdict(
            TrainingConfig(
                seed=SEED,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                selection_metric=SELECTION_METRIC,
                checkpoint_every_epoch=CHECKPOINT_EVERY_EPOCH,
                num_classes=NUM_CLASSES,
                text_dim=TEXT_DIM,
                image_dim=IMAGE_DIM,
            )
        ),
        "artifact_hashes": artifact_hashes,
        "test_accessed": False,
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(path.name + ".tmp")

    torch.save(payload, tmp_path)

    os.replace(tmp_path, path)


def validate_checkpoint_identity(
    checkpoint: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> None:
    if checkpoint.get("checkpoint_version") != "c3_cbase_v001":
        raise RuntimeError("Unsupported checkpoint version.")

    if checkpoint.get("phase") != "C3":
        raise RuntimeError("Checkpoint is not a Phase C3 checkpoint.")

    if checkpoint.get("model_name") != "C-Base":
        raise RuntimeError("Checkpoint model is not C-Base.")

    if checkpoint.get("seed") != SEED:
        raise RuntimeError(
            f"Checkpoint seed {checkpoint.get('seed')} != {SEED}."
        )

    if checkpoint.get("architecture") != architecture_config():
        raise RuntimeError(
            "Checkpoint architecture does not match current C-Base architecture."
        )

    if checkpoint.get("training_config") != asdict(
        TrainingConfig(
            seed=SEED,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            selection_metric=SELECTION_METRIC,
            checkpoint_every_epoch=CHECKPOINT_EVERY_EPOCH,
            num_classes=NUM_CLASSES,
            text_dim=TEXT_DIM,
            image_dim=IMAGE_DIM,
        )
    ):
        raise RuntimeError(
            "Checkpoint training configuration does not match current configuration."
        )

    saved_hashes = checkpoint.get("artifact_hashes")

    if saved_hashes != artifact_hashes:
        raise RuntimeError(
            "Checkpoint input-artifact hashes do not match current embeddings. "
            "Refusing to resume."
        )

    if checkpoint.get("test_accessed") is True:
        raise RuntimeError(
            "Checkpoint reports test access. Refusing to resume."
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("C3 CONTROLLED TRAINING — C-BASE")
    print("=" * 72)
    print("Project root :", PROJECT_ROOT)
    print("Embedding dir:", EMBEDDING_DIR)
    print("Output dir   :", OUTPUT_DIR)
    print("Seed         :", SEED)
    print("Device       : cpu")
    print("Classes      :", NUM_CLASSES)
    print("Train        :", TRAIN_COUNT)
    print("Validation   :", VALIDATION_COUNT)
    print("Batch size   :", BATCH_SIZE)
    print("Epochs       :", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay :", WEIGHT_DECAY)
    print("Selection    :", SELECTION_METRIC)
    print()

    seed_everything(SEED)

    validate_required_files()

    manifest = load_json(MANIFEST_PATH)
    validate_manifest(manifest)

    artifact_hashes = validate_arrays_and_ids()

    train_dataset = EmbeddingDataset(
        TRAIN_TEXT_PATH,
        TRAIN_IMAGE_PATH,
        TRAIN_LABELS_PATH,
        TRAIN_COUNT,
    )

    val_dataset = EmbeddingDataset(
        VAL_TEXT_PATH,
        VAL_IMAGE_PATH,
        VAL_LABELS_PATH,
        VALIDATION_COUNT,
    )

    # One DataLoader generator is checkpointed through torch RNG.
    # shuffle=True is appropriate because the dataset order itself is frozen
    # and the training RNG state is saved at each checkpoint.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    device = torch.device("cpu")

    model = CBase().to(device)
    parameter_count = count_parameters(model)

    if parameter_count != 1_266_213:
        raise RuntimeError(
            f"Unexpected C-Base parameter count: {parameter_count}; "
            "expected 1266213."
        )

    print("Trainable parameters:", parameter_count)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_metric = float("-inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------------
    # RESUME
    # ------------------------------------------------------------------------

    if LATEST_CHECKPOINT_PATH.exists():
        print()
        print("=" * 72)
        print("RESUME CHECKPOINT FOUND")
        print("=" * 72)
        print("Checkpoint:", LATEST_CHECKPOINT_PATH)

        checkpoint = torch.load(
            LATEST_CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        validate_checkpoint_identity(
            checkpoint,
            artifact_hashes,
        )

        completed_epoch = int(checkpoint["epoch_completed"])

        if completed_epoch >= EPOCHS:
            print(
                f"Checkpoint already completed {completed_epoch}/{EPOCHS} epochs."
            )
            print("Nothing more to train.")
            print("Use cbase_best.pt for the selected validation checkpoint.")
            return

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        restore_rng_state(checkpoint["rng_state"])

        start_epoch = completed_epoch + 1
        best_metric = float(checkpoint["best_metric"])
        best_epoch = int(checkpoint["best_epoch"])
        history = list(checkpoint["history"])

        print("Completed epoch :", completed_epoch)
        print("Resuming epoch  :", start_epoch)
        print("Best epoch      :", best_epoch)
        print("Best Macro-F1   :", best_metric)
        print("Resume identity : PASS")

    else:
        print()
        print("=" * 72)
        print("STARTING NEW C3 C-BASE RUN")
        print("=" * 72)

    # ------------------------------------------------------------------------
    # RUN MANIFEST
    # ------------------------------------------------------------------------

    run_manifest = {
        "manifest_version": "c3_cbase_training_v001",
        "phase": "C3",
        "dataset": "ABO",
        "model": architecture_config(),
        "training_config": asdict(
            TrainingConfig(
                seed=SEED,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                selection_metric=SELECTION_METRIC,
                checkpoint_every_epoch=CHECKPOINT_EVERY_EPOCH,
                num_classes=NUM_CLASSES,
                text_dim=TEXT_DIM,
                image_dim=IMAGE_DIM,
            )
        ),
        "primary_population": {
            "train": TRAIN_COUNT,
            "validation": VALIDATION_COUNT,
            "test_accessed": False,
        },
        "comparator": {
            "name": "B5 frozen late-fusion multimodal baseline",
            "accuracy": 0.538660,
            "macro_f1": 0.122360,
            "weighted_f1": 0.507353,
        },
        "inputs": {
            "text_embedding": "all-MiniLM-L6-v2",
            "text_dimension": TEXT_DIM,
            "image_embedding": "ResNet-18",
            "image_dimension": IMAGE_DIM,
            "image_weights": "ResNet18_Weights.IMAGENET1K_V1",
            "embeddings_frozen": True,
            "embedding_manifest": str(MANIFEST_PATH),
            "artifact_hashes": artifact_hashes,
        },
        "integrity": {
            "b3_modified": False,
            "test_accessed": False,
            "raw_data_accessed": False,
            "target_classes": NUM_CLASSES,
        },
        "software": {
            "python": f"{__import__('sys').version_info.major}."
                       f"{__import__('sys').version_info.minor}."
                       f"{__import__('sys').version_info.micro}",
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
    }

    atomic_write_json(RUN_MANIFEST_PATH, run_manifest)

    # ------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        sample_count = 0

        for batch_index, (text, image, labels) in enumerate(train_loader, start=1):
            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(text, image)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        train_loss = running_loss / sample_count

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        epoch_seconds = time.time() - epoch_start

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": val_metrics,
            "epoch_seconds": epoch_seconds,
            "timestamp_unix": time.time(),
        }

        history.append(epoch_result)

        current_metric = float(val_metrics[SELECTION_METRIC])

        improved = current_metric > best_metric

        if improved:
            best_metric = current_metric
            best_epoch = epoch

            best_payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                history=history,
                artifact_hashes=artifact_hashes,
            )

            atomic_torch_save(
                best_payload,
                BEST_CHECKPOINT_PATH,
            )

        latest_payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            best_epoch=best_epoch,
            history=history,
            artifact_hashes=artifact_hashes,
        )

        if epoch % CHECKPOINT_EVERY_EPOCH == 0:
            atomic_torch_save(
                latest_payload,
                LATEST_CHECKPOINT_PATH,
            )

        atomic_write_json(
            PROGRESS_PATH,
            {
                "phase": "C3",
                "model": "C-Base",
                "epoch_completed": epoch,
                "total_epochs": EPOCHS,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_metric,
                "latest_checkpoint": str(LATEST_CHECKPOINT_PATH),
                "best_checkpoint": str(BEST_CHECKPOINT_PATH),
                "test_accessed": False,
                "artifact_hashes": artifact_hashes,
            },
        )

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_acc={val_metrics['accuracy']:.6f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.6f} | "
            f"val_weighted_f1={val_metrics['weighted_f1']:.6f} | "
            f"time={epoch_seconds:.1f}s"
        )

        if improved:
            print(
                f"  BEST CHECKPOINT UPDATED | "
                f"epoch={best_epoch} | "
                f"val_macro_f1={best_metric:.6f}"
            )
        else:
            print(
                f"  checkpoint saved | "
                f"best_epoch={best_epoch} | "
                f"best_val_macro_f1={best_metric:.6f}"
            )

        gc.collect()

    # ------------------------------------------------------------------------
    # FINAL TRAINING SUMMARY
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("C3 C-BASE CONTROLLED TRAINING COMPLETE")
    print("=" * 72)
    print("Latest checkpoint :", LATEST_CHECKPOINT_PATH)
    print("Best checkpoint   :", BEST_CHECKPOINT_PATH)
    print("Progress file     :", PROGRESS_PATH)
    print("Training manifest :", RUN_MANIFEST_PATH)
    print("History           :", HISTORY_PATH)
    print("Completed epochs  :", EPOCHS)
    print("Best epoch        :", best_epoch)
    print("Best validation")
    print("  Macro-F1        :", f"{best_metric:.6f}")
    print()
    print("TEST SET: NOT ACCESSED")
    print("B3: NOT MODIFIED")
    print("B4/B5: NOT MODIFIED")
    print("=" * 72)

    atomic_write_json(
        HISTORY_PATH,
        {
            "phase": "C3",
            "model": "C-Base",
            "selection_metric": SELECTION_METRIC,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_metric,
            "history": history,
            "test_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
