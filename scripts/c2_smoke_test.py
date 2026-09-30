"""
C3 — Prepare Frozen Multimodal Representations
WITH RESUMABLE, IDENTITY-LOCKED CHECKPOINTS

Purpose
-------
Generate frozen Phase C text and image embeddings from the authoritative
ABO B3 v002 representations for the locked Phase C train/validation
populations.

Locked Phase C populations
--------------------------
Train      : 69,823
Validation : 69,867
Test       : 7,346  <-- NEVER READ BY THIS SCRIPT

Frozen target space
-------------------
549 product_type classes, using the frozen B4.2 image LabelEncoder ordering.

Text
----
Model      : sentence-transformers/all-MiniLM-L6-v2
Loaded via : Hugging Face AutoModel
Embedding  : 384-D
Sequence   : 256
Content    : 254 tokens/chunk
Chunking   : exact token-preserving B3 combined_tokens -> tokenizer IDs
Pooling    : mean pool within each chunk, then mean across chunks

Image
-----
Model      : ResNet-18
Weights    : ResNet18_Weights.IMAGENET1K_V1
Embedding  : 512-D
Transform  : official torchvision pretrained-weight transform

Checkpoint / Resume
-------------------
Each split and modality has its own NumPy .npy memory-mapped checkpoint
and atomic JSON progress file:

    train_text
    train_image
    validation_text
    validation_image

The checkpoint stores an ordered population fingerprint derived from
record IDs, labels, targets, and image paths. Resume is refused if the
current B3-derived population differs from the population used to create
the checkpoint.

If execution is interrupted, rerun this same script. Work resumes from
the last atomically committed checkpoint boundary. At most
CHECKPOINT_EVERY-1 records are recomputed.

Important
---------
- B3 is read-only.
- No test records are accessed.
- No missing modality is fabricated.
- No paths are guessed or downloaded.
- Encoders remain frozen.
- This script prepares representations only; it does not train C-Base.
- Existing incompatible checkpoints are never silently discarded.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision.models import ResNet18_Weights, resnet18


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 20260827

B3_TRAIN = Path(r"data\representations\abo\b3\train.jsonl")
B3_VALIDATION = Path(r"data\representations\abo\b3\validation.jsonl")

IMAGE_ROOT = Path(r"ABO_Audit\images\small")

B4_TEXT_MODEL = Path(r"data\models\abo\b4.2\text_model.joblib")
B4_IMAGE_MODEL = Path(r"data\models\abo\b4.2\image_model.joblib")

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "ResNet-18"
IMAGE_WEIGHT_ENUM = "ResNet18_Weights.IMAGENET1K_V1"

OUTPUT_ROOT = Path(r"data\models\abo\phase_c\c3\embeddings")

EXPECTED_CLASSES = 549
EXPECTED_TRAIN_POPULATION = 69_823
EXPECTED_VALIDATION_POPULATION = 69_867
EXPECTED_TRAIN_MISSING_MODALITY = 461
EXPECTED_VALIDATION_MISSING_MODALITY = 70
EXPECTED_VALIDATION_OUTSIDE_TARGET = 59

TEXT_DIM = 384
IMAGE_DIM = 512

MAX_SEQ_LENGTH = 256
CONTENT_PER_CHUNK = 254

# CPU-safe starting batch sizes.
TEXT_CHUNK_BATCH_SIZE = 16
IMAGE_BATCH_SIZE = 32

# Atomic progress is committed every N completed records.
CHECKPOINT_EVERY = 250

DEVICE = torch.device("cpu")


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ============================================================================
# HASH / ATOMIC FILE HELPERS
# ============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, path)


def population_fingerprint(records: list[dict]) -> str:
    """
    Stable identity fingerprint for the exact ordered Phase C population.

    It intentionally includes the fields that determine row alignment:
    record_id, target, numeric label, and exact image path.
    """
    h = hashlib.sha256()

    for record in records:
        row = (
            str(record["record_id"]),
            str(record["target"]),
            int(record["label"]),
            str(record["image_path"]),
        )
        h.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")

    return h.hexdigest()


def record_ids_fingerprint(records: list[dict]) -> str:
    h = hashlib.sha256()
    for record in records:
        h.update(str(record["record_id"]).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def labels_fingerprint(records: list[dict]) -> str:
    h = hashlib.sha256()
    for record in records:
        h.update(np.int64(record["label"]).tobytes())
    return h.hexdigest()


def memmap_is_finite(arr: np.ndarray, block_rows: int = 4096) -> bool:
    for start in range(0, arr.shape[0], block_rows):
        end = min(start + block_rows, arr.shape[0])
        if not np.isfinite(arr[start:end]).all():
            return False
    return True


# ============================================================================
# TARGET EXTRACTION
# ============================================================================

def extract_product_type(record: dict) -> str | None:
    value = record.get("product_type")

    if not isinstance(value, list):
        return None

    for item in value:
        if not isinstance(item, dict):
            continue
        candidate = item.get("value")
        if candidate is None:
            continue
        candidate = str(candidate).strip()
        if candidate:
            return candidate

    return None


# ============================================================================
# LOAD B3 RECORDS
# ============================================================================

def load_jsonl_records(
    path: Path,
    expected_population: int,
    class_to_index: dict[str, int],
    split_name: str,
) -> tuple[list[dict], int, int, list[str]]:
    """
    Construct the locked Phase C primary multimodal population.

    Missing physical/usable images are expected missing-modality exclusions,
    not malformed records. A B3 record that claims a physical usable image
    but whose documented local file is absent is malformed/ineligible.
    """

    records: list[dict] = []
    excluded_target = 0
    excluded_missing_modality = 0
    bad_ids: list[str] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            record_id = record.get("record_id")

            if not record_id:
                bad_ids.append(f"{split_name}:line_{line_number}:missing_record_id")
                continue

            if record_id in seen_ids:
                raise RuntimeError(f"Duplicate record_id in {split_name}: {record_id}")
            seen_ids.add(record_id)

            b3 = record.get("b3_representation")
            if not isinstance(b3, dict):
                bad_ids.append(f"{record_id}:missing_b3_representation")
                continue

            if b3.get("representation_version") != "v002":
                bad_ids.append(f"{record_id}:wrong_b3_version")
                continue

            text_block = b3.get("text")
            if not isinstance(text_block, dict):
                bad_ids.append(f"{record_id}:missing_b3_text")
                continue

            combined_tokens = text_block.get("combined_tokens")
            if not isinstance(combined_tokens, list):
                bad_ids.append(f"{record_id}:combined_tokens_not_list")
                continue

            if len(combined_tokens) == 0:
                bad_ids.append(f"{record_id}:empty_combined_tokens")
                continue

            target = extract_product_type(record)
            if target is None:
                bad_ids.append(f"{record_id}:missing_product_type")
                continue

            if target not in class_to_index:
                excluded_target += 1
                continue

            image_block = b3.get("image")
            if not isinstance(image_block, dict):
                bad_ids.append(f"{record_id}:missing_b3_image")
                continue

            main_image = image_block.get("main")
            if not isinstance(main_image, dict):
                bad_ids.append(f"{record_id}:missing_main_image")
                continue

            if not main_image.get("physical_file_available", False):
                excluded_missing_modality += 1
                continue

            if not main_image.get("usable_for_local_image_model", False):
                excluded_missing_modality += 1
                continue

            relative_path = main_image.get("relative_path")
            if not relative_path:
                bad_ids.append(f"{record_id}:missing_image_relative_path")
                continue

            image_path = IMAGE_ROOT / relative_path
            if not image_path.is_file():
                bad_ids.append(f"{record_id}:image_file_missing:{relative_path}")
                continue

            records.append(
                {
                    "record_id": record_id,
                    "target": target,
                    "label": class_to_index[target],
                    "combined_tokens": combined_tokens,
                    "image_path": str(image_path),
                }
            )

    if len(records) != expected_population:
        raise RuntimeError(
            f"{split_name} population mismatch.\n"
            f"Expected primary population : {expected_population}\n"
            f"Observed primary population : {len(records)}\n"
            f"Missing-modality excluded   : {excluded_missing_modality}\n"
            f"Target-space excluded       : {excluded_target}\n"
            f"Malformed records           : {len(bad_ids)}"
        )

    return records, excluded_target, excluded_missing_modality, bad_ids


# ============================================================================
# FROZEN TARGET SPACE
# ============================================================================

def load_frozen_class_space():
    print("\n" + "=" * 72)
    print("LOADING FROZEN 549-CLASS TARGET SPACE")
    print("=" * 72)

    text_bundle = joblib.load(B4_TEXT_MODEL)
    image_bundle = joblib.load(B4_IMAGE_MODEL)

    text_encoder = text_bundle["label_encoder"]
    image_encoder = image_bundle["label_encoder"]

    text_classes = list(text_encoder.classes_)
    image_classes = list(image_encoder.classes_)

    print(f"Text classes  : {len(text_classes)}")
    print(f"Image classes : {len(image_classes)}")

    if len(text_classes) != 553:
        raise RuntimeError(f"Expected 553 B4.2 text classes, found {len(text_classes)}")

    if len(image_classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} B4.2 image classes, found {len(image_classes)}"
        )

    text_set = set(text_classes)
    image_set = set(image_classes)

    missing_from_text = image_set - text_set
    text_only = text_set - image_set

    if missing_from_text:
        raise RuntimeError(
            "Image classes missing from frozen text class space: "
            f"{sorted(missing_from_text)}"
        )

    if len(text_only) != 4:
        raise RuntimeError(f"Expected 4 text-only classes, found {len(text_only)}")

    frozen_classes = image_classes
    class_to_index = {label: index for index, label in enumerate(frozen_classes)}

    print(f"Intersection    : {len(frozen_classes)}")
    print(f"Text-only       : {len(text_only)}")
    print(f"First 10        : {frozen_classes[:10]}")
    print(f"Last 10         : {frozen_classes[-10:]}")

    return frozen_classes, class_to_index


# ============================================================================
# TEXT ENCODER
# ============================================================================

def load_text_encoder():
    print("\n" + "=" * 72)
    print("LOADING TEXT ENCODER")
    print("=" * 72)

    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, use_fast=True)
    tokenizer.model_max_length = 1_000_000

    text_encoder, loading_info = AutoModel.from_pretrained(
        TEXT_MODEL_NAME,
        output_loading_info=True,
    )

    missing_keys = set(loading_info["missing_keys"])
    unexpected_keys = set(loading_info["unexpected_keys"])
    allowed_unexpected = {"embeddings.position_ids"}

    if missing_keys:
        raise RuntimeError(
            f"Unexpected missing text encoder keys: {sorted(missing_keys)}"
        )

    unexpected_not_allowed = unexpected_keys - allowed_unexpected
    if unexpected_not_allowed:
        raise RuntimeError(
            "Unexpected text encoder checkpoint keys: "
            f"{sorted(unexpected_not_allowed)}"
        )

    if unexpected_keys:
        print(
            "INFO: Known Sentence-Transformers checkpoint buffer key ignored "
            f"by AutoModel loader: {sorted(unexpected_keys)}"
        )

    text_encoder.to(DEVICE)
    text_encoder.eval()
    for parameter in text_encoder.parameters():
        parameter.requires_grad = False

    hidden_size = int(text_encoder.config.hidden_size)
    if hidden_size != TEXT_DIM:
        raise RuntimeError(f"Expected text dimension {TEXT_DIM}, found {hidden_size}")

    print(f"Tokenizer class      : {tokenizer.__class__.__name__}")
    print(f"Text embedding dim   : {hidden_size}")
    print(f"CLS token ID         : {tokenizer.cls_token_id}")
    print(f"SEP token ID         : {tokenizer.sep_token_id}")
    print(f"Phase C max length   : {MAX_SEQ_LENGTH}")
    print(f"Content/chunk        : {CONTENT_PER_CHUNK}")
    print(f"Load time            : {time.time() - start:.2f} sec")

    return tokenizer, text_encoder


# ============================================================================
# TEXT TOKENIZATION / POOLING
# ============================================================================

def tokenize_full(tokenizer, combined_tokens: list) -> list[int]:
    text = " ".join(str(token) for token in combined_tokens)
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    return list(token_ids)


def build_chunk_ids(tokenizer, content_token_ids: list[int]) -> list[list[int]]:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    if cls_id is None or sep_id is None:
        raise RuntimeError("Tokenizer does not provide CLS/SEP token IDs.")

    chunks = []
    for start in range(0, len(content_token_ids), CONTENT_PER_CHUNK):
        content = content_token_ids[start:start + CONTENT_PER_CHUNK]
        input_ids = [cls_id, *content, sep_id]

        if len(input_ids) > MAX_SEQ_LENGTH:
            raise RuntimeError(
                "Constructed text chunk exceeds Phase C maximum sequence length: "
                f"{len(input_ids)}"
            )

        chunks.append(input_ids)

    if not chunks:
        raise RuntimeError("No chunks produced for non-empty text.")

    return chunks


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def checkpoint_array_path(split_name: str, modality: str) -> Path:
    return OUTPUT_ROOT / f"{split_name}_{modality}_checkpoint.npy"


def checkpoint_progress_path(split_name: str, modality: str) -> Path:
    return OUTPUT_ROOT / f"{split_name}_{modality}_checkpoint_progress.json"


def final_array_path(split_name: str, modality: str) -> Path:
    return OUTPUT_ROOT / f"{split_name}_{modality}.npy"


def checkpoint_meta(
    split_name: str,
    modality: str,
    completed: int,
    total: int,
    dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
) -> dict:
    return {
        "phase": "C3",
        "type": "resumable_embedding_checkpoint",
        "split": split_name,
        "modality": modality,
        "completed_records": completed,
        "total_records": total,
        "embedding_dim": dim,
        "dtype": "float32",
        "seed": SEED,
        "device": str(DEVICE),
        "test_accessed": False,
        "b3_modified": False,
        "checkpoint_every": CHECKPOINT_EVERY,
        "population_fingerprint": population_fp,
        "record_ids_fingerprint": ids_fp,
        "labels_fingerprint": labels_fp,
        "text_model": TEXT_MODEL_NAME,
        "image_model": IMAGE_MODEL_NAME,
        "image_weight_enum": IMAGE_WEIGHT_ENUM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "content_tokens_per_chunk": CONTENT_PER_CHUNK,
    }


def validate_checkpoint_progress(
    progress: dict,
    split_name: str,
    modality: str,
    total_records: int,
    embedding_dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
) -> int:
    expected = {
        "split": split_name,
        "modality": modality,
        "total_records": total_records,
        "embedding_dim": embedding_dim,
        "dtype": "float32",
        "seed": SEED,
        "population_fingerprint": population_fp,
        "record_ids_fingerprint": ids_fp,
        "labels_fingerprint": labels_fp,
        "text_model": TEXT_MODEL_NAME,
        "image_model": IMAGE_MODEL_NAME,
        "image_weight_enum": IMAGE_WEIGHT_ENUM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "content_tokens_per_chunk": CONTENT_PER_CHUNK,
    }

    for key, value in expected.items():
        if progress.get(key) != value:
            raise RuntimeError(
                f"Checkpoint compatibility mismatch for {split_name}/{modality}: "
                f"{key}: checkpoint={progress.get(key)!r}, current={value!r}"
            )

    completed = int(progress.get("completed_records", -1))
    if completed < 0 or completed > total_records:
        raise RuntimeError(
            f"Invalid checkpoint progress: {completed}/{total_records}"
        )

    return completed


def open_or_create_checkpoint(
    split_name: str,
    modality: str,
    total_records: int,
    embedding_dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_array_path(split_name, modality)
    progress_path = checkpoint_progress_path(split_name, modality)

    if checkpoint_path.exists() or progress_path.exists():
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Progress file exists without checkpoint array: {progress_path}"
            )
        if not progress_path.exists():
            raise RuntimeError(
                f"Checkpoint array exists without progress file: {checkpoint_path}"
            )

        with progress_path.open("r", encoding="utf-8") as f:
            progress = json.load(f)

        completed = validate_checkpoint_progress(
            progress,
            split_name,
            modality,
            total_records,
            embedding_dim,
            population_fp,
            ids_fp,
            labels_fp,
        )

        mmap = np.lib.format.open_memmap(checkpoint_path, mode="r+")
        expected_shape = (total_records, embedding_dim)

        if mmap.shape != expected_shape:
            raise RuntimeError(
                f"Checkpoint shape mismatch: {mmap.shape} != {expected_shape}"
            )
        if mmap.dtype != np.float32:
            raise RuntimeError(
                f"Checkpoint dtype mismatch: {mmap.dtype} != float32"
            )

        print("\nRESUME CHECKPOINT")
        print(f"  Split              : {split_name}")
        print(f"  Modality           : {modality}")
        print(f"  Completed          : {completed:,}/{total_records:,}")
        print(f"  Remaining          : {total_records - completed:,}")
        print(f"  Population FP      : {population_fp}")
        print(f"  Checkpoint         : {checkpoint_path}")

        return mmap, completed

    print("\nCREATING NEW CHECKPOINT")
    print(f"  Split              : {split_name}")
    print(f"  Modality           : {modality}")
    print(f"  Shape              : ({total_records:,}, {embedding_dim})")
    print(f"  Population FP      : {population_fp}")
    print(f"  Path               : {checkpoint_path}")

    mmap = np.lib.format.open_memmap(
        checkpoint_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_records, embedding_dim),
    )
    mmap.flush()

    progress = checkpoint_meta(
        split_name,
        modality,
        0,
        total_records,
        embedding_dim,
        population_fp,
        ids_fp,
        labels_fp,
    )
    atomic_write_json(progress_path, progress)

    return mmap, 0


def save_checkpoint_progress(
    split_name: str,
    modality: str,
    completed: int,
    total_records: int,
    embedding_dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
    mmap,
) -> None:
    mmap.flush()

    progress = checkpoint_meta(
        split_name,
        modality,
        completed,
        total_records,
        embedding_dim,
        population_fp,
        ids_fp,
        labels_fp,
    )
    atomic_write_json(
        checkpoint_progress_path(split_name, modality),
        progress,
    )

    print(
        f"CHECKPOINT SAVED | {split_name}/{modality} | "
        f"{completed:,}/{total_records:,}"
    )


def mark_checkpoint_complete(
    split_name: str,
    modality: str,
    total_records: int,
    embedding_dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
    mmap,
) -> Path:
    mmap.flush()

    final_path = final_array_path(split_name, modality)
    checkpoint_path = checkpoint_array_path(split_name, modality)
    progress_path = checkpoint_progress_path(split_name, modality)

    # Ensure the data is no longer actively mapped before Windows rename.
    del mmap

    if final_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing final output: {final_path}"
        )

    os.replace(checkpoint_path, final_path)

    completion_progress = checkpoint_meta(
        split_name,
        modality,
        total_records,
        total_records,
        embedding_dim,
        population_fp,
        ids_fp,
        labels_fp,
    )
    completion_progress["completed"] = True
    completion_progress["final_output"] = str(final_path)
    completion_progress["final_output_sha256"] = sha256_file(final_path)

    atomic_write_json(progress_path, completion_progress)

    print(f"COMPLETE | {split_name}/{modality}")
    print(f"  Final output : {final_path}")

    return final_path


def final_output_is_complete(
    split_name: str,
    modality: str,
    total_records: int,
    embedding_dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
) -> bool:
    final_path = final_array_path(split_name, modality)
    progress_path = checkpoint_progress_path(split_name, modality)

    if not final_path.exists() or not progress_path.exists():
        return False

    try:
        with progress_path.open("r", encoding="utf-8") as f:
            progress = json.load(f)

        if not progress.get("completed", False):
            return False

        completed = validate_checkpoint_progress(
            progress,
            split_name,
            modality,
            total_records,
            embedding_dim,
            population_fp,
            ids_fp,
            labels_fp,
        )
        if completed != total_records:
            return False

        arr = np.load(final_path, mmap_mode="r")
        if arr.shape != (total_records, embedding_dim):
            return False
        if arr.dtype != np.float32:
            return False

        stored_hash = progress.get("final_output_sha256")
        if stored_hash and sha256_file(final_path) != stored_hash:
            raise RuntimeError(
                f"Final output hash mismatch for {split_name}/{modality}: {final_path}"
            )

        return True

    except RuntimeError:
        raise
    except Exception:
        return False


# ============================================================================
# TEXT EMBEDDING GENERATION — RESUMABLE
# ============================================================================

def encode_text_records_resumable(
    split_name: str,
    records: list[dict],
    tokenizer,
    text_encoder,
) -> Path:
    print("\n" + "=" * 72)
    print(f"GENERATING TEXT EMBEDDINGS — {split_name.upper()}")
    print("=" * 72)

    total_records = len(records)
    population_fp = population_fingerprint(records)
    ids_fp = record_ids_fingerprint(records)
    labels_fp = labels_fingerprint(records)

    if final_output_is_complete(
        split_name,
        "text",
        total_records,
        TEXT_DIM,
        population_fp,
        ids_fp,
        labels_fp,
    ):
        final_path = final_array_path(split_name, "text")
        print(f"SKIP: completed text embeddings already exist: {final_path}")
        return final_path

    embeddings, completed = open_or_create_checkpoint(
        split_name,
        "text",
        total_records,
        TEXT_DIM,
        population_fp,
        ids_fp,
        labels_fp,
    )

    start_time = time.time()
    run_start_completed = completed
    total_chunks_processed = 0
    max_chunks = 0

    for record_index in range(completed, total_records):
        record = records[record_index]

        content_ids = tokenize_full(tokenizer, record["combined_tokens"])
        chunks = build_chunk_ids(tokenizer, content_ids)

        total_chunks_processed += len(chunks)
        max_chunks = max(max_chunks, len(chunks))

        # Accumulate directly rather than retaining all chunk vectors.
        sum_embedding = np.zeros(TEXT_DIM, dtype=np.float64)
        chunk_count = 0

        for start in range(0, len(chunks), TEXT_CHUNK_BATCH_SIZE):
            chunk_batch = chunks[start:start + TEXT_CHUNK_BATCH_SIZE]
            max_length = max(len(ids) for ids in chunk_batch)

            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = 0

            input_ids = []
            attention_masks = []

            for ids in chunk_batch:
                padding = max_length - len(ids)
                input_ids.append(ids + [pad_id] * padding)
                attention_masks.append([1] * len(ids) + [0] * padding)

            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)
            attention_mask_tensor = torch.tensor(
                attention_masks,
                dtype=torch.long,
                device=DEVICE,
            )

            with torch.inference_mode():
                outputs = text_encoder(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                )
                pooled = mean_pool(outputs.last_hidden_state, attention_mask_tensor)

            pooled_np = (
                pooled.detach().cpu().numpy().astype(np.float32)
            )

            if pooled_np.shape[1] != TEXT_DIM:
                raise RuntimeError(
                    f"Unexpected text chunk embedding shape: {pooled_np.shape}"
                )

            sum_embedding += pooled_np.astype(np.float64).sum(axis=0)
            chunk_count += pooled_np.shape[0]

        record_embedding = (
            sum_embedding / chunk_count
        ).astype(np.float32)

        if record_embedding.shape != (TEXT_DIM,):
            raise RuntimeError(
                f"Unexpected text embedding shape for record {record['record_id']}: "
                f"{record_embedding.shape}"
            )

        if not np.isfinite(record_embedding).all():
            raise RuntimeError(
                f"Non-finite text embedding for record {record['record_id']}"
            )

        embeddings[record_index] = record_embedding
        completed = record_index + 1

        if completed % 100 == 0 or completed == total_records:
            elapsed = time.time() - start_time
            current_run = completed - run_start_completed
            throughput = current_run / max(elapsed, 1e-9)
            remaining = total_records - completed
            eta_seconds = remaining / max(throughput, 1e-9)
            print(
                f"TEXT | {split_name} | {completed:,}/{total_records:,} | "
                f"{throughput:.2f} records/sec | ETA {eta_seconds / 60:.1f} min"
            )

        if completed % CHECKPOINT_EVERY == 0 or completed == total_records:
            save_checkpoint_progress(
                split_name,
                "text",
                completed,
                total_records,
                TEXT_DIM,
                population_fp,
                ids_fp,
                labels_fp,
                embeddings,
            )

    if not memmap_is_finite(embeddings):
        raise RuntimeError(
            f"Non-finite values found in completed {split_name} text checkpoint."
        )

    final_path = mark_checkpoint_complete(
        split_name,
        "text",
        total_records,
        TEXT_DIM,
        population_fp,
        ids_fp,
        labels_fp,
        embeddings,
    )

    elapsed = time.time() - start_time
    print(f"Text generation elapsed: {elapsed:.2f} sec")
    print(f"Text total chunks processed this run: {total_chunks_processed:,}")
    print(f"Text maximum chunks for one record this run: {max_chunks:,}")
    print(f"Text final output: {final_path}")

    return final_path


# ============================================================================
# IMAGE ENCODER
# ============================================================================

def load_image_encoder():
    print("\n" + "=" * 72)
    print("LOADING IMAGE ENCODER")
    print("=" * 72)

    start = time.time()

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    image_encoder = nn.Sequential(*list(model.children())[:-1])
    image_encoder.to(DEVICE)
    image_encoder.eval()

    for parameter in image_encoder.parameters():
        parameter.requires_grad = False

    print(f"Image encoder       : {IMAGE_MODEL_NAME}")
    print(f"Image embedding dim : {IMAGE_DIM}")
    print(f"Weight enum         : {weights}")
    print(f"Load time           : {time.time() - start:.2f} sec")

    return weights.transforms(), image_encoder


# ============================================================================
# IMAGE EMBEDDING GENERATION — RESUMABLE
# ============================================================================

def encode_image_records_resumable(
    split_name: str,
    records: list[dict],
    image_transform,
    image_encoder,
) -> Path:
    print("\n" + "=" * 72)
    print(f"GENERATING IMAGE EMBEDDINGS — {split_name.upper()}")
    print("=" * 72)

    total_records = len(records)
    population_fp = population_fingerprint(records)
    ids_fp = record_ids_fingerprint(records)
    labels_fp = labels_fingerprint(records)

    if final_output_is_complete(
        split_name,
        "image",
        total_records,
        IMAGE_DIM,
        population_fp,
        ids_fp,
        labels_fp,
    ):
        final_path = final_array_path(split_name, "image")
        print(f"SKIP: completed image embeddings already exist: {final_path}")
        return final_path

    embeddings, completed = open_or_create_checkpoint(
        split_name,
        "image",
        total_records,
        IMAGE_DIM,
        population_fp,
        ids_fp,
        labels_fp,
    )

    start_time = time.time()
    run_start_completed = completed

    batch_images = []
    batch_indices = []

    def flush_batch():
        nonlocal batch_images, batch_indices

        if not batch_images:
            return

        image_tensor = torch.stack(batch_images, dim=0).to(DEVICE)

        with torch.inference_mode():
            features = image_encoder(image_tensor)

        features = features.flatten(start_dim=1)
        features_np = features.detach().cpu().numpy().astype(np.float32)

        if features_np.shape[1] != IMAGE_DIM:
            raise RuntimeError(
                f"Unexpected image embedding dimension: {features_np.shape}"
            )

        for row, record_index in enumerate(batch_indices):
            if not np.isfinite(features_np[row]).all():
                raise RuntimeError(
                    f"Non-finite image embedding at record index {record_index}"
                )
            embeddings[record_index] = features_np[row]

        batch_images = []
        batch_indices = []

    for record_index in range(completed, total_records):
        record = records[record_index]
        image_path = Path(record["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = image_transform(image)

        batch_images.append(tensor)
        batch_indices.append(record_index)

        if len(batch_images) >= IMAGE_BATCH_SIZE:
            flush_batch()

        next_completed = record_index + 1

        if next_completed % CHECKPOINT_EVERY == 0 or next_completed == total_records:
            flush_batch()

            if not memmap_is_finite(embeddings[:next_completed]):
                raise RuntimeError(
                    f"Non-finite image embedding detected before checkpoint at "
                    f"{next_completed} records."
                )

            save_checkpoint_progress(
                split_name,
                "image",
                next_completed,
                total_records,
                IMAGE_DIM,
                population_fp,
                ids_fp,
                labels_fp,
                embeddings,
            )

            elapsed = time.time() - start_time
            current_run = next_completed - run_start_completed
            throughput = current_run / max(elapsed, 1e-9)
            remaining = total_records - next_completed
            eta_seconds = remaining / max(throughput, 1e-9)

            print(
                f"IMAGE | {split_name} | {next_completed:,}/{total_records:,} | "
                f"{throughput:.2f} records/sec | ETA {eta_seconds / 60:.1f} min"
            )

    flush_batch()

    if not memmap_is_finite(embeddings):
        raise RuntimeError(
            f"Non-finite values found in completed {split_name} image checkpoint."
        )

    final_path = mark_checkpoint_complete(
        split_name,
        "image",
        total_records,
        IMAGE_DIM,
        population_fp,
        ids_fp,
        labels_fp,
        embeddings,
    )

    elapsed = time.time() - start_time
    print(f"Image generation elapsed: {elapsed:.2f} sec")
    print(f"Image final output: {final_path}")

    return final_path


# ============================================================================
# SAVE LABELS + RECORD IDS
# ============================================================================

def save_metadata_arrays(split_name: str, records: list[dict]):
    labels_path = OUTPUT_ROOT / f"{split_name}_labels.npy"
    ids_path = OUTPUT_ROOT / f"{split_name}_record_ids.json"

    labels = np.asarray(
        [record["label"] for record in records],
        dtype=np.int64,
    )
    record_ids = [record["record_id"] for record in records]

    np.save(labels_path, labels)

    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(record_ids, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"Saved labels      : {labels_path}")
    print(f"Saved record IDs  : {ids_path}")

    return {
        "labels_path": str(labels_path),
        "record_ids_path": str(ids_path),
        "labels_sha256": sha256_file(labels_path),
        "record_ids_sha256": sha256_file(ids_path),
    }


# ============================================================================
# VERIFY FINAL EMBEDDINGS
# ============================================================================

def verify_embeddings(
    split_name: str,
    records: list[dict],
    text_path: Path,
    image_path: Path,
):
    print("\n" + "=" * 72)
    print(f"VERIFYING {split_name.upper()} EMBEDDINGS")
    print("=" * 72)

    expected_count = len(records)
    text_embeddings = np.load(text_path, mmap_mode="r")
    image_embeddings = np.load(image_path, mmap_mode="r")

    expected_text_shape = (expected_count, TEXT_DIM)
    expected_image_shape = (expected_count, IMAGE_DIM)

    if text_embeddings.shape != expected_text_shape:
        raise RuntimeError(
            f"{split_name} text shape mismatch: "
            f"{text_embeddings.shape} != {expected_text_shape}"
        )

    if image_embeddings.shape != expected_image_shape:
        raise RuntimeError(
            f"{split_name} image shape mismatch: "
            f"{image_embeddings.shape} != {expected_image_shape}"
        )

    if text_embeddings.dtype != np.float32 or image_embeddings.dtype != np.float32:
        raise RuntimeError("Final embedding dtype is not float32.")

    if not memmap_is_finite(text_embeddings):
        raise RuntimeError(f"{split_name} text embeddings contain non-finite values.")

    if not memmap_is_finite(image_embeddings):
        raise RuntimeError(f"{split_name} image embeddings contain non-finite values.")

    labels = np.asarray(
        [record["label"] for record in records],
        dtype=np.int64,
    )

    if labels.shape != (expected_count,):
        raise RuntimeError(f"{split_name} label shape mismatch.")

    if labels.min() < 0 or labels.max() >= EXPECTED_CLASSES:
        raise RuntimeError(
            f"{split_name} contains labels outside [0, {EXPECTED_CLASSES - 1}]."
        )

    record_ids = [record["record_id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError(f"{split_name} contains duplicate record IDs.")

    print(f"Records          : {expected_count:,}")
    print(f"Text shape       : {text_embeddings.shape}")
    print(f"Image shape      : {image_embeddings.shape}")
    print(f"Text finite      : {memmap_is_finite(text_embeddings)}")
    print(f"Image finite     : {memmap_is_finite(image_embeddings)}")
    print(f"Labels shape     : {labels.shape}")
    print(f"Labels valid     : {labels.min()}..{labels.max()} within 549 classes")
    print("Record IDs unique: True")
    print("PASS.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("C3 — FROZEN MULTIMODAL EMBEDDING PREPARATION")
    print("IDENTITY-LOCKED RESUMABLE CHECKPOINT VERSION")
    print("=" * 72)

    print("\nEnvironment")
    print("-" * 72)
    print(f"Python               : {sys.version.split()[0]}")
    print(f"PyTorch              : {torch.__version__}")
    print(f"NumPy                : {np.__version__}")
    print(f"Device               : {DEVICE}")
    print(f"Seed                 : {SEED}")
    print(f"Checkpoint every     : {CHECKPOINT_EVERY:,} records")
    print(f"B3 train             : {B3_TRAIN}")
    print(f"B3 validation        : {B3_VALIDATION}")
    print(f"Image root           : {IMAGE_ROOT}")
    print(f"Text encoder         : {TEXT_MODEL_NAME}")
    print(f"Image encoder        : {IMAGE_MODEL_NAME}")
    print(f"Output root          : {OUTPUT_ROOT}")

    # ========================================================================
    # REQUIRED INPUTS
    # ========================================================================
    print("\n" + "=" * 72)
    print("VERIFYING REQUIRED INPUT PATHS")
    print("=" * 72)

    required_paths = [
        B3_TRAIN,
        B3_VALIDATION,
        IMAGE_ROOT,
        B4_TEXT_MODEL,
        B4_IMAGE_MODEL,
    ]

    for path in required_paths:
        print(f"{path} | exists={path.exists()}")
        if not path.exists():
            raise FileNotFoundError(f"Required frozen input missing: {path}")

    print("PASS: all required frozen inputs exist.")

    # ========================================================================
    # FROZEN TARGET SPACE
    # ========================================================================
    frozen_classes, class_to_index = load_frozen_class_space()

    if len(frozen_classes) != EXPECTED_CLASSES:
        raise RuntimeError("Frozen target space size mismatch.")

    # ========================================================================
    # LOAD ENCODERS
    # ========================================================================
    tokenizer, text_encoder = load_text_encoder()
    image_transform, image_encoder = load_image_encoder()

    # ========================================================================
    # TRAIN POPULATION
    # ========================================================================
    print("\n" + "=" * 72)
    print("VERIFYING PRIMARY TRAIN POPULATION")
    print("=" * 72)

    (
        train_records,
        train_excluded,
        train_missing_modality,
        train_bad,
    ) = load_jsonl_records(
        B3_TRAIN,
        EXPECTED_TRAIN_POPULATION,
        class_to_index,
        "train",
    )

    print(f"Eligible train records              : {len(train_records):,}")
    print(f"Missing-modality exclusions        : {train_missing_modality:,}")
    print(f"Excluded outside frozen 549 space  : {train_excluded:,}")
    print(f"Malformed records                  : {len(train_bad):,}")

    if train_bad:
        raise RuntimeError(
            "Train contains genuinely malformed B3 records: "
            f"{train_bad[:10]}"
        )
    if train_missing_modality != EXPECTED_TRAIN_MISSING_MODALITY:
        raise RuntimeError(
            "Expected exactly 461 train missing-modality exclusions, "
            f"found {train_missing_modality}."
        )
    if train_excluded != 0:
        raise RuntimeError(
            "Train population contains records outside the frozen 549-class target space."
        )

    print("PASS: exact Phase C train population verified.")

    # ========================================================================
    # VALIDATION POPULATION
    # ========================================================================
    print("\n" + "=" * 72)
    print("VERIFYING PRIMARY VALIDATION POPULATION")
    print("=" * 72)

    (
        validation_records,
        validation_excluded,
        validation_missing_modality,
        validation_bad,
    ) = load_jsonl_records(
        B3_VALIDATION,
        EXPECTED_VALIDATION_POPULATION,
        class_to_index,
        "validation",
    )

    print(f"Eligible validation records          : {len(validation_records):,}")
    print(f"Missing-modality exclusions          : {validation_missing_modality:,}")
    print(f"Excluded outside frozen 549 space   : {validation_excluded:,}")
    print(f"Malformed records                    : {len(validation_bad):,}")

    if validation_bad:
        raise RuntimeError(
            "Validation contains genuinely malformed B3 records: "
            f"{validation_bad[:10]}"
        )
    if validation_missing_modality != EXPECTED_VALIDATION_MISSING_MODALITY:
        raise RuntimeError(
            "Expected exactly 70 validation missing-modality exclusions, "
            f"found {validation_missing_modality}."
        )
    if validation_excluded != EXPECTED_VALIDATION_OUTSIDE_TARGET:
        raise RuntimeError(
            "Expected exactly 59 valid validation records outside the frozen "
            f"549-class target space, found {validation_excluded}."
        )

    print("PASS: exact Phase C validation population verified.")

    # ========================================================================
    # CROSS-SPLIT RECORD ID CHECK
    # ========================================================================
    train_ids = {record["record_id"] for record in train_records}
    validation_ids = {record["record_id"] for record in validation_records}
    overlap = train_ids & validation_ids

    if overlap:
        raise RuntimeError(
            "Cross-split record-ID overlap detected: "
            f"{sorted(list(overlap))[:10]}"
        )

    print("\nPASS: train/validation record IDs are disjoint.")

    # ========================================================================
    # POPULATION FINGERPRINTS
    # ========================================================================
    train_population_fp = population_fingerprint(train_records)
    validation_population_fp = population_fingerprint(validation_records)

    train_ids_fp = record_ids_fingerprint(train_records)
    validation_ids_fp = record_ids_fingerprint(validation_records)

    train_labels_fp = labels_fingerprint(train_records)
    validation_labels_fp = labels_fingerprint(validation_records)

    print("\n" + "=" * 72)
    print("LOCKED POPULATION FINGERPRINTS")
    print("=" * 72)
    print(f"Train population FP      : {train_population_fp}")
    print(f"Validation population FP : {validation_population_fp}")
    print(f"Train record-ID FP        : {train_ids_fp}")
    print(f"Validation record-ID FP   : {validation_ids_fp}")
    print(f"Train labels FP           : {train_labels_fp}")
    print(f"Validation labels FP      : {validation_labels_fp}")

    # ========================================================================
    # SAVE LABELS + IDS
    # ========================================================================
    print("\n" + "=" * 72)
    print("SAVING LOCKED LABEL / RECORD-ID METADATA")
    print("=" * 72)

    train_metadata = save_metadata_arrays("train", train_records)
    validation_metadata = save_metadata_arrays("validation", validation_records)

    # ========================================================================
    # TRAIN TEXT
    # ========================================================================
    train_text_path = encode_text_records_resumable(
        "train",
        train_records,
        tokenizer,
        text_encoder,
    )

    # ========================================================================
    # TRAIN IMAGE
    # ========================================================================
    train_image_path = encode_image_records_resumable(
        "train",
        train_records,
        image_transform,
        image_encoder,
    )

    verify_embeddings(
        "train",
        train_records,
        train_text_path,
        train_image_path,
    )

    # ========================================================================
    # VALIDATION TEXT
    # ========================================================================
    validation_text_path = encode_text_records_resumable(
        "validation",
        validation_records,
        tokenizer,
        text_encoder,
    )

    # ========================================================================
    # VALIDATION IMAGE
    # ========================================================================
    validation_image_path = encode_image_records_resumable(
        "validation",
        validation_records,
        image_transform,
        image_encoder,
    )

    verify_embeddings(
        "validation",
        validation_records,
        validation_text_path,
        validation_image_path,
    )

    # ========================================================================
    # MANIFEST
    # ========================================================================
    print("\n" + "=" * 72)
    print("WRITING C3 EMBEDDING MANIFEST")
    print("=" * 72)

    manifest = {
        "phase": "C3",
        "task": "frozen_multimodal_embedding_preparation",
        "checkpointing": {
            "enabled": True,
            "checkpoint_every_records": CHECKPOINT_EVERY,
            "resumable": True,
            "checkpoint_unit": "completed_record",
            "identity_locked": True,
        },
        "seed": SEED,
        "device": str(DEVICE),
        "dataset": "ABO",
        "b3_representation": {
            "version": "v002",
            "train_source": str(B3_TRAIN),
            "validation_source": str(B3_VALIDATION),
            "read_only": True,
        },
        "target": {
            "field": "product_type",
            "class_count": EXPECTED_CLASSES,
            "class_order_source": "frozen B4.2 image LabelEncoder",
            "classes": frozen_classes,
        },
        "populations": {
            "train": len(train_records),
            "validation": len(validation_records),
            "train_missing_modality_excluded": train_missing_modality,
            "validation_missing_modality_excluded": validation_missing_modality,
            "train_excluded_outside_549": train_excluded,
            "validation_excluded_outside_549": validation_excluded,
            "train_population_fingerprint": train_population_fp,
            "validation_population_fingerprint": validation_population_fp,
            "train_record_ids_fingerprint": train_ids_fp,
            "validation_record_ids_fingerprint": validation_ids_fp,
            "train_labels_fingerprint": train_labels_fp,
            "validation_labels_fingerprint": validation_labels_fp,
            "test_accessed": False,
        },
        "text": {
            "model": TEXT_MODEL_NAME,
            "embedding_dim": TEXT_DIM,
            "max_sequence_length": MAX_SEQ_LENGTH,
            "content_tokens_per_chunk": CONTENT_PER_CHUNK,
            "chunking": "token_preserving",
            "chunk_pooling": "mean",
            "encoder_frozen": True,
            "tokenizer_class": tokenizer.__class__.__name__,
        },
        "image": {
            "model": IMAGE_MODEL_NAME,
            "weight_enum": IMAGE_WEIGHT_ENUM,
            "embedding_dim": IMAGE_DIM,
            "encoder_frozen": True,
            "image_root": str(IMAGE_ROOT),
        },
        "outputs": {
            "train": {
                "text_path": str(train_text_path),
                "image_path": str(train_image_path),
                "labels_path": str(OUTPUT_ROOT / "train_labels.npy"),
                "record_ids_path": str(OUTPUT_ROOT / "train_record_ids.json"),
            },
            "validation": {
                "text_path": str(validation_text_path),
                "image_path": str(validation_image_path),
                "labels_path": str(OUTPUT_ROOT / "validation_labels.npy"),
                "record_ids_path": str(OUTPUT_ROOT / "validation_record_ids.json"),
            },
        },
        "integrity": {
            "train_bad_records": len(train_bad),
            "validation_bad_records": len(validation_bad),
            "train_missing_modality_excluded": train_missing_modality,
            "validation_missing_modality_excluded": validation_missing_modality,
            "train_excluded_target": train_excluded,
            "validation_excluded_target": validation_excluded,
            "train_validation_record_id_overlap": len(overlap),
            "test_accessed": False,
            "b3_modified": False,
        },
        "software": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "numpy": np.__version__,
        },
        "checkpoint_outputs": {
            "train_text": str(checkpoint_progress_path("train", "text")),
            "train_image": str(checkpoint_progress_path("train", "image")),
            "validation_text": str(checkpoint_progress_path("validation", "text")),
            "validation_image": str(checkpoint_progress_path("validation", "image")),
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = OUTPUT_ROOT / "c3_embedding_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(f"Manifest             : {manifest_path}")

    # ========================================================================
    # FINAL INTEGRITY
    # ========================================================================
    print("\n" + "=" * 72)
    print("FINAL C3 EMBEDDING PREPARATION INTEGRITY")
    print("=" * 72)

    checks = [
        ("B3 v002 read-only source", True),
        ("Exact train population", len(train_records) == EXPECTED_TRAIN_POPULATION),
        ("Exact validation population", len(validation_records) == EXPECTED_VALIDATION_POPULATION),
        ("Train bad records = 0", len(train_bad) == 0),
        ("Validation bad records = 0", len(validation_bad) == 0),
        ("Train outside-target exclusions = 0", train_excluded == 0),
        ("Validation outside-target exclusions = 59", validation_excluded == EXPECTED_VALIDATION_OUTSIDE_TARGET),
        ("Train missing-modality exclusions = 461", train_missing_modality == EXPECTED_TRAIN_MISSING_MODALITY),
        ("Validation missing-modality exclusions = 70", validation_missing_modality == EXPECTED_VALIDATION_MISSING_MODALITY),
        ("549 frozen target classes", len(frozen_classes) == EXPECTED_CLASSES),
        (
            "384-D text embeddings",
            np.load(train_text_path, mmap_mode="r").shape == (EXPECTED_TRAIN_POPULATION, TEXT_DIM)
            and np.load(validation_text_path, mmap_mode="r").shape == (EXPECTED_VALIDATION_POPULATION, TEXT_DIM),
        ),
        (
            "512-D image embeddings",
            np.load(train_image_path, mmap_mode="r").shape == (EXPECTED_TRAIN_POPULATION, IMAGE_DIM)
            and np.load(validation_image_path, mmap_mode="r").shape == (EXPECTED_VALIDATION_POPULATION, IMAGE_DIM),
        ),
        ("No train/validation record-ID overlap", len(overlap) == 0),
        ("No test accessed", True),
        ("B3 unmodified", True),
        (
            "Train population fingerprint stable",
            population_fingerprint(train_records) == train_population_fp,
        ),
        (
            "Validation population fingerprint stable",
            population_fingerprint(validation_records) == validation_population_fp,
        ),
        (
            "All four embedding stages completed",
            all(
                final_output_is_complete(
                    split_name,
                    modality,
                    len(records),
                    dim,
                    population_fingerprint(records),
                    record_ids_fingerprint(records),
                    labels_fingerprint(records),
                )
                for split_name, records in [
                    ("train", train_records),
                    ("validation", validation_records),
                ]
                for modality, dim in [
                    ("text", TEXT_DIM),
                    ("image", IMAGE_DIM),
                ]
            ),
        ),
    ]

    all_pass = True
    for description, passed in checks:
        if passed:
            print(f"PASS: {description}.")
        else:
            print(f"FAIL: {description}.")
            all_pass = False

    if not all_pass:
        raise RuntimeError("C3 embedding preparation integrity failed.")

    print("\n" + "=" * 72)
    print("C3 EMBEDDING PREPARATION COMPLETE")
    print("=" * 72)
    print(f"Output directory: {OUTPUT_ROOT}")
    print("\nAll four embedding stages are complete.")


if __name__ == "__main__":
    main()
