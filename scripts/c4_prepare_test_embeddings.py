"""
C4.1 — Prepare Isolated ABO Test Embeddings

Purpose:
    Generate frozen Phase C text/image embeddings for the locked
    7,346-record ABO common-target test population.

Scientific boundary:
    - TEST INFERENCE PREPARATION ONLY.
    - No training.
    - No hyperparameter selection.
    - No model selection.
    - No test-driven tuning.
    - B3 v002 is read-only.
    - C3 train/validation artifacts are not modified.
    - Frozen 549-class target space is preserved.
    - Exact C3 text/image embedding implementation is reused.

Output:
    data\models\abo\phase_c\c4\test_embeddings\

This script is intentionally separate from C3.
"""

import gc
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
# LOCKED CONFIGURATION
# ============================================================================

SEED = 20260827

B3_TEST = Path(r"data\representations\abo\b3\test.jsonl")
IMAGE_ROOT = Path(r"ABO_Audit\images\small")

B4_TEXT_MODEL = Path(r"data\models\abo\b4.2\text_model.joblib")
B4_IMAGE_MODEL = Path(r"data\models\abo\b4.2\image_model.joblib")

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "ResNet-18"
IMAGE_WEIGHT_ENUM = "ResNet18_Weights.IMAGENET1K_V1"

OUTPUT_ROOT = Path(
    r"data\models\abo\phase_c\c4\test_embeddings"
)

EXPECTED_CLASSES = 549
EXPECTED_TEST_POPULATION = 7_346

TEXT_DIM = 384
IMAGE_DIM = 512

MAX_SEQ_LENGTH = 256
CONTENT_PER_CHUNK = 254

TEXT_CHUNK_BATCH_SIZE = 16
IMAGE_BATCH_SIZE = 32

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

    torch.use_deterministic_algorithms(True)


# ============================================================================
# FILE / HASH HELPERS
# ============================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, path)


def population_fingerprint(records: list[dict]) -> str:
    digest = hashlib.sha256()

    for record in records:
        payload = {
            "record_id": record["record_id"],
            "target": record["target"],
            "label": int(record["label"]),
            "image_path": str(record["image_path"]),
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        digest.update(encoded)
        digest.update(b"\n")

    return digest.hexdigest()


def record_ids_fingerprint(records: list[dict]) -> str:
    digest = hashlib.sha256()

    for record in records:
        digest.update(str(record["record_id"]).encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()


def labels_fingerprint(records: list[dict]) -> str:
    digest = hashlib.sha256()

    for record in records:
        digest.update(str(int(record["label"])).encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()


def memmap_is_finite(array) -> bool:
    chunk_size = 8192

    for start in range(0, array.shape[0], chunk_size):
        stop = min(start + chunk_size, array.shape[0])

        if not np.isfinite(array[start:stop]).all():
            return False

    return True


# ============================================================================
# TARGET EXTRACTION
# ============================================================================

def extract_product_type(record):
    value = record.get("product_type")

    if not isinstance(value, list):
        return None

    for item in value:
        if isinstance(item, dict):
            candidate = item.get("value")

            if candidate is not None:
                candidate = str(candidate).strip()

                if candidate:
                    return candidate

    return None


# ============================================================================
# FROZEN TEST POPULATION
# ============================================================================

def load_test_records(
    path: Path,
    class_to_index: dict[str, int],
) -> tuple[list[dict], int, int, list[str]]:
    """
    Load the B3 v002 test source and construct the frozen 549-class
    common-target test population.

    Missing modality is evaluated BEFORE target-space filtering,
    matching the authoritative C3 population logic.
    """

    records = []

    excluded_missing_modality = 0
    excluded_target = 0
    bad_ids = []

    seen_record_ids = set()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_ids.append(f"line:{line_number}:invalid_json")
                continue

            record_id = record.get("record_id")

            if not record_id:
                bad_ids.append(f"line:{line_number}:missing_record_id")
                continue

            if record_id in seen_record_ids:
                bad_ids.append(str(record_id))
                continue

            seen_record_ids.add(record_id)

            b3 = record.get("b3_representation")

            if not isinstance(b3, dict):
                bad_ids.append(str(record_id))
                continue

            if b3.get("representation_version") != "v002":
                bad_ids.append(str(record_id))
                continue

            text_block = b3.get("text")

            if not isinstance(text_block, dict):
                bad_ids.append(str(record_id))
                continue

            combined_tokens = text_block.get("combined_tokens")

            if not isinstance(combined_tokens, list) or not combined_tokens:
                bad_ids.append(str(record_id))
                continue

            target = extract_product_type(record)

            if target is None:
                bad_ids.append(str(record_id))
                continue

            image_block = b3.get("image")

            if not isinstance(image_block, dict):
                bad_ids.append(str(record_id))
                continue

            main_image = image_block.get("main")

            if not isinstance(main_image, dict):
                bad_ids.append(str(record_id))
                continue

            # IMPORTANT:
            # Missing modality is evaluated before target filtering.
            if not main_image.get("physical_file_available", False):
                excluded_missing_modality += 1
                continue

            if not main_image.get("usable_for_local_image_model", False):
                excluded_missing_modality += 1
                continue

            if target not in class_to_index:
                excluded_target += 1
                continue

            relative_path = main_image.get("relative_path")

            if not relative_path:
                bad_ids.append(str(record_id))
                continue

            image_path = IMAGE_ROOT / relative_path

            if not image_path.exists():
                bad_ids.append(str(record_id))
                continue

            records.append(
                {
                    "record_id": str(record_id),
                    "target": target,
                    "label": int(class_to_index[target]),
                    "combined_tokens": combined_tokens,
                    "image_path": image_path,
                }
            )

    return (
        records,
        excluded_target,
        excluded_missing_modality,
        bad_ids,
    )


# ============================================================================
# FROZEN TARGET SPACE
# ============================================================================

def load_frozen_class_space():
    print("\n" + "=" * 72)
    print("LOADING FROZEN 549-CLASS TARGET SPACE")
    print("=" * 72)

    text_bundle = joblib.load(B4_TEXT_MODEL)
    image_bundle = joblib.load(B4_IMAGE_MODEL)

    text_classes = list(text_bundle["label_encoder"].classes_)
    image_classes = list(image_bundle["label_encoder"].classes_)

    print(f"B4.2 text classes   : {len(text_classes)}")
    print(f"B4.2 image classes  : {len(image_classes)}")

    if len(text_classes) != 553:
        raise RuntimeError(
            f"Expected 553 B4.2 text classes, found {len(text_classes)}."
        )

    if len(image_classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} B4.2 image classes, "
            f"found {len(image_classes)}."
        )

    text_set = set(text_classes)
    image_set = set(image_classes)

    if not image_set.issubset(text_set):
        raise RuntimeError(
            "Frozen B4.2 image classes are not a subset of text classes."
        )

    text_only = sorted(text_set - image_set)

    expected_text_only = sorted(
        [
            "HAIRBAND",
            "PUNCHING_BAG",
            "SALWAR_SUIT_SET",
            "TREADMILL",
        ]
    )

    if text_only != expected_text_only:
        raise RuntimeError(
            "Unexpected text-only classes in frozen B4.2 class alignment:\n"
            f"Observed: {text_only}\n"
            f"Expected: {expected_text_only}"
        )

    frozen_classes = image_classes

    class_to_index = {
        label: index
        for index, label in enumerate(frozen_classes)
    }

    print(f"Frozen Phase C classes: {len(frozen_classes)}")
    print("Class order source      : frozen B4.2 image LabelEncoder")
    print("PASS.")

    return frozen_classes, class_to_index


# ============================================================================
# TEXT ENCODER
# ============================================================================

def load_text_encoder():
    print("\n" + "=" * 72)
    print("LOADING TEXT ENCODER")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL_NAME,
        use_fast=True,
    )

    # Same C3 behavior.
    tokenizer.model_max_length = 1_000_000

    text_encoder, loading_info = AutoModel.from_pretrained(
        TEXT_MODEL_NAME,
        output_loading_info=True,
    )

    missing_keys = loading_info.get("missing_keys", [])
    unexpected_keys = loading_info.get("unexpected_keys", [])

    if missing_keys:
        raise RuntimeError(
            f"Unexpected missing text-encoder keys: {missing_keys}"
        )

    allowed_unexpected = {"embeddings.position_ids"}

    unexpected_set = set(unexpected_keys)

    if unexpected_set != allowed_unexpected:
        raise RuntimeError(
            "Unexpected text-encoder checkpoint loading state:\n"
            f"Observed: {sorted(unexpected_set)}\n"
            f"Allowed: {sorted(allowed_unexpected)}"
        )

    text_encoder.to(DEVICE)
    text_encoder.eval()

    for parameter in text_encoder.parameters():
        parameter.requires_grad = False

    hidden_size = int(text_encoder.config.hidden_size)

    if hidden_size != TEXT_DIM:
        raise RuntimeError(
            f"Expected {TEXT_DIM}-D MiniLM embeddings, found {hidden_size}."
        )

    if tokenizer.cls_token_id is None:
        raise RuntimeError("Tokenizer has no CLS token.")

    if tokenizer.sep_token_id is None:
        raise RuntimeError("Tokenizer has no SEP token.")

    print(f"Text encoder         : {TEXT_MODEL_NAME}")
    print(f"Tokenizer class      : {tokenizer.__class__.__name__}")
    print(f"Embedding dimension  : {hidden_size}")
    print(f"CLS token ID         : {tokenizer.cls_token_id}")
    print(f"SEP token ID         : {tokenizer.sep_token_id}")
    print(f"Maximum sequence     : {MAX_SEQ_LENGTH}")
    print(f"Content/chunk        : {CONTENT_PER_CHUNK}")
    print("Encoder frozen       : True")
    print("PASS.")

    return tokenizer, text_encoder


# ============================================================================
# EXACT C3 TEXT TOKENIZATION / CHUNKING / POOLING
# ============================================================================

def tokenize_full(tokenizer, combined_tokens):
    text = " ".join(
        str(token)
        for token in combined_tokens
    )

    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    return list(tokenized["input_ids"])


def build_chunk_ids(
    tokenizer,
    content_token_ids: list[int],
) -> list[list[int]]:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    if cls_id is None or sep_id is None:
        raise RuntimeError(
            "Tokenizer does not provide CLS/SEP token IDs."
        )

    chunks = []

    for start in range(
        0,
        len(content_token_ids),
        CONTENT_PER_CHUNK,
    ):
        content = content_token_ids[
            start:start + CONTENT_PER_CHUNK
        ]

        input_ids = [
            cls_id,
            *content,
            sep_id,
        ]

        if len(input_ids) > MAX_SEQ_LENGTH:
            raise RuntimeError(
                "Constructed text chunk exceeds Phase C "
                f"maximum sequence length: {len(input_ids)}"
            )

        chunks.append(input_ids)

    if not chunks:
        raise RuntimeError(
            "No chunks produced for non-empty text."
        )

    return chunks


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(
        last_hidden_state.dtype
    )

    summed = (
        last_hidden_state * mask
    ).sum(dim=1)

    counts = mask.sum(dim=1).clamp(
        min=1e-9
    )

    return summed / counts


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def checkpoint_array_path(modality: str) -> Path:
    return OUTPUT_ROOT / f"test_{modality}_checkpoint.npy"


def checkpoint_progress_path(modality: str) -> Path:
    return OUTPUT_ROOT / f"test_{modality}_checkpoint_progress.json"


def final_array_path(modality: str) -> Path:
    return OUTPUT_ROOT / f"test_{modality}.npy"


def checkpoint_meta(
    modality: str,
    completed: int,
    total: int,
    dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
) -> dict:
    return {
        "phase": "C4.1",
        "type": "test_embedding_checkpoint",
        "split": "test",
        "modality": modality,
        "completed_records": completed,
        "total_records": total,
        "embedding_dim": dim,
        "dtype": "float32",
        "seed": SEED,
        "device": str(DEVICE),
        "test_accessed": True,
        "test_usage": "inference_only",
        "model_selection": False,
        "hyperparameter_selection": False,
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
    modality: str,
    total_records: int,
    embedding_dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
) -> int:

    expected = {
        "phase": "C4.1",
        "type": "test_embedding_checkpoint",
        "split": "test",
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
                "C4.1 checkpoint compatibility mismatch:\n"
                f"{key}: checkpoint={progress.get(key)!r}, "
                f"current={value!r}"
            )

    completed = int(
        progress.get("completed_records", -1)
    )

    if completed < 0 or completed > total_records:
        raise RuntimeError(
            f"Invalid checkpoint progress: "
            f"{completed}/{total_records}"
        )

    if progress.get("test_accessed") is not True:
        raise RuntimeError(
            "C4.1 checkpoint does not declare test access."
        )

    if progress.get("test_usage") != "inference_only":
        raise RuntimeError(
            "C4.1 checkpoint does not declare inference-only test usage."
        )

    if progress.get("model_selection") is not False:
        raise RuntimeError(
            "C4.1 checkpoint incorrectly permits model selection."
        )

    if progress.get("hyperparameter_selection") is not False:
        raise RuntimeError(
            "C4.1 checkpoint incorrectly permits hyperparameter selection."
        )

    return completed


def open_or_create_checkpoint(
    modality: str,
    total_records: int,
    embedding_dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
):
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = checkpoint_array_path(modality)
    progress_path = checkpoint_progress_path(modality)

    if (
        checkpoint_path.exists()
        or progress_path.exists()
    ):
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Progress exists without checkpoint: "
                f"{progress_path}"
            )

        if not progress_path.exists():
            raise RuntimeError(
                f"Checkpoint exists without progress: "
                f"{checkpoint_path}"
            )

        with progress_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            progress = json.load(f)

        completed = validate_checkpoint_progress(
            progress,
            modality,
            total_records,
            embedding_dim,
            population_fp,
            ids_fp,
            labels_fp,
        )

        mmap = np.lib.format.open_memmap(
            checkpoint_path,
            mode="r+",
        )

        expected_shape = (
            total_records,
            embedding_dim,
        )

        if mmap.shape != expected_shape:
            raise RuntimeError(
                f"Checkpoint shape mismatch: "
                f"{mmap.shape} != {expected_shape}"
            )

        if mmap.dtype != np.float32:
            raise RuntimeError(
                f"Checkpoint dtype mismatch: "
                f"{mmap.dtype} != float32"
            )

        print("\nRESUME CHECKPOINT")
        print(f"  Modality       : {modality}")
        print(
            f"  Completed      : "
            f"{completed:,}/{total_records:,}"
        )
        print(
            f"  Remaining      : "
            f"{total_records - completed:,}"
        )
        print(
            f"  Population FP  : "
            f"{population_fp}"
        )
        print(
            f"  Checkpoint     : "
            f"{checkpoint_path}"
        )

        return mmap, completed

    print("\nCREATING NEW CHECKPOINT")
    print(f"  Modality       : {modality}")
    print(
        f"  Shape          : "
        f"({total_records:,}, {embedding_dim})"
    )
    print(
        f"  Population FP  : "
        f"{population_fp}"
    )
    print(
        f"  Path           : "
        f"{checkpoint_path}"
    )

    mmap = np.lib.format.open_memmap(
        checkpoint_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            total_records,
            embedding_dim,
        ),
    )

    mmap.flush()

    progress = checkpoint_meta(
        modality,
        0,
        total_records,
        embedding_dim,
        population_fp,
        ids_fp,
        labels_fp,
    )

    atomic_write_json(
        progress_path,
        progress,
    )

    return mmap, 0


def save_checkpoint_progress(
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
        modality,
        completed,
        total_records,
        embedding_dim,
        population_fp,
        ids_fp,
        labels_fp,
    )

    atomic_write_json(
        checkpoint_progress_path(modality),
        progress,
    )

    print(
        f"CHECKPOINT SAVED | test/{modality} | "
        f"{completed:,}/{total_records:,}"
    )


def mark_checkpoint_complete(
    modality: str,
    total_records: int,
    embedding_dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
    mmap,
) -> Path:

    mmap.flush()

    if getattr(mmap, "_mmap", None) is not None:
        mmap._mmap.close()

    final_path = final_array_path(modality)
    checkpoint_path = checkpoint_array_path(modality)
    progress_path = checkpoint_progress_path(modality)

    del mmap
    gc.collect()

    if final_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing final output: "
            f"{final_path}"
        )

    os.replace(
        checkpoint_path,
        final_path,
    )

    completion_progress = checkpoint_meta(
        modality,
        total_records,
        total_records,
        embedding_dim,
        population_fp,
        ids_fp,
        labels_fp,
    )

    completion_progress["completed"] = True
    completion_progress["final_output"] = str(
        final_path
    )
    completion_progress["final_output_sha256"] = (
        sha256_file(final_path)
    )

    atomic_write_json(
        progress_path,
        completion_progress,
    )

    print(
        f"COMPLETE | test/{modality}"
    )
    print(
        f"  Final output : {final_path}"
    )

    return final_path


def final_output_is_complete(
    modality: str,
    total_records: int,
    embedding_dim: int,
    population_fp: str,
    ids_fp: str,
    labels_fp: str,
) -> bool:

    final_path = final_array_path(modality)
    progress_path = checkpoint_progress_path(modality)

    if (
        not final_path.exists()
        or not progress_path.exists()
    ):
        return False

    try:
        with progress_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            progress = json.load(f)

        if not progress.get(
            "completed",
            False,
        ):
            return False

        completed = validate_checkpoint_progress(
            progress,
            modality,
            total_records,
            embedding_dim,
            population_fp,
            ids_fp,
            labels_fp,
        )

        if completed != total_records:
            return False

        arr = np.load(
            final_path,
            mmap_mode="r",
        )

        if arr.shape != (
            total_records,
            embedding_dim,
        ):
            return False

        if arr.dtype != np.float32:
            return False

        stored_hash = progress.get(
            "final_output_sha256"
        )

        if stored_hash:
            actual_hash = sha256_file(
                final_path
            )

            if actual_hash != stored_hash:
                raise RuntimeError(
                    "Final output hash mismatch for "
                    f"test/{modality}: {final_path}"
                )

        return True

    except RuntimeError:
        raise
    except Exception:
        return False


# ============================================================================
# TEXT EMBEDDINGS
# ============================================================================

def encode_text_records_resumable(
    records: list[dict],
    tokenizer,
    text_encoder,
) -> Path:

    print("\n" + "=" * 72)
    print("GENERATING TEST TEXT EMBEDDINGS")
    print("=" * 72)

    total_records = len(records)

    population_fp = population_fingerprint(
        records
    )

    ids_fp = record_ids_fingerprint(
        records
    )

    labels_fp = labels_fingerprint(
        records
    )

    if final_output_is_complete(
        "text",
        total_records,
        TEXT_DIM,
        population_fp,
        ids_fp,
        labels_fp,
    ):
        final_path = final_array_path(
            "text"
        )

        print(
            "SKIP: completed test text embeddings "
            f"already exist: {final_path}"
        )

        return final_path

    embeddings, completed = (
        open_or_create_checkpoint(
            "text",
            total_records,
            TEXT_DIM,
            population_fp,
            ids_fp,
            labels_fp,
        )
    )

    start_time = time.time()
    run_start_completed = completed

    total_chunks_processed = 0
    max_chunks = 0

    for record_index in range(
        completed,
        total_records,
    ):
        record = records[record_index]

        content_ids = tokenize_full(
            tokenizer,
            record["combined_tokens"],
        )

        chunks = build_chunk_ids(
            tokenizer,
            content_ids,
        )

        total_chunks_processed += len(
            chunks
        )

        max_chunks = max(
            max_chunks,
            len(chunks),
        )

        sum_embedding = np.zeros(
            TEXT_DIM,
            dtype=np.float64,
        )

        chunk_count = 0

        for start in range(
            0,
            len(chunks),
            TEXT_CHUNK_BATCH_SIZE,
        ):
            chunk_batch = chunks[
                start:start
                + TEXT_CHUNK_BATCH_SIZE
            ]

            max_length = max(
                len(ids)
                for ids in chunk_batch
            )

            pad_id = tokenizer.pad_token_id

            if pad_id is None:
                pad_id = 0

            input_ids = []
            attention_masks = []

            for ids in chunk_batch:
                padding = (
                    max_length
                    - len(ids)
                )

                input_ids.append(
                    ids
                    + [pad_id] * padding
                )

                attention_masks.append(
                    [1] * len(ids)
                    + [0] * padding
                )

            input_ids_tensor = torch.tensor(
                input_ids,
                dtype=torch.long,
                device=DEVICE,
            )

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

                pooled = mean_pool(
                    outputs.last_hidden_state,
                    attention_mask_tensor,
                )

            pooled_np = (
                pooled
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if pooled_np.shape[1] != TEXT_DIM:
                raise RuntimeError(
                    "Unexpected text chunk embedding "
                    f"shape: {pooled_np.shape}"
                )

            sum_embedding += (
                pooled_np
                .astype(np.float64)
                .sum(axis=0)
            )

            chunk_count += pooled_np.shape[0]

        record_embedding = (
            sum_embedding / chunk_count
        ).astype(np.float32)

        if record_embedding.shape != (
            TEXT_DIM,
        ):
            raise RuntimeError(
                "Unexpected text embedding shape "
                f"for {record['record_id']}: "
                f"{record_embedding.shape}"
            )

        if not np.isfinite(
            record_embedding
        ).all():
            raise RuntimeError(
                "Non-finite text embedding for "
                f"{record['record_id']}"
            )

        embeddings[record_index] = (
            record_embedding
        )

        completed = record_index + 1

        if (
            completed % 100 == 0
            or completed == total_records
        ):
            elapsed = (
                time.time()
                - start_time
            )

            current_run = (
                completed
                - run_start_completed
            )

            throughput = (
                current_run
                / max(elapsed, 1e-9)
            )

            remaining = (
                total_records
                - completed
            )

            eta_seconds = (
                remaining
                / max(throughput, 1e-9)
            )

            print(
                f"TEXT | test | "
                f"{completed:,}/{total_records:,} | "
                f"{throughput:.2f} records/sec | "
                f"ETA {eta_seconds / 60:.1f} min"
            )

        if (
            completed % CHECKPOINT_EVERY == 0
            or completed == total_records
        ):
            save_checkpoint_progress(
                "text",
                completed,
                total_records,
                TEXT_DIM,
                population_fp,
                ids_fp,
                labels_fp,
                embeddings,
            )

    if not memmap_is_finite(
        embeddings
    ):
        raise RuntimeError(
            "Non-finite values found in completed "
            "test text checkpoint."
        )

    final_path = mark_checkpoint_complete(
        "text",
        total_records,
        TEXT_DIM,
        population_fp,
        ids_fp,
        labels_fp,
        embeddings,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Text generation elapsed: "
        f"{elapsed:.2f} sec"
    )

    print(
        "Text total chunks processed "
        f"this run: {total_chunks_processed:,}"
    )

    print(
        "Text maximum chunks for one "
        f"record this run: {max_chunks:,}"
    )

    print(
        f"Text final output: {final_path}"
    )

    return final_path


# ============================================================================
# IMAGE ENCODER
# ============================================================================

def load_image_encoder():
    print("\n" + "=" * 72)
    print("LOADING IMAGE ENCODER")
    print("=" * 72)

    start = time.time()

    weights = (
        ResNet18_Weights.IMAGENET1K_V1
    )

    model = resnet18(
        weights=weights
    )

    image_encoder = nn.Sequential(
        *list(model.children())[:-1]
    )

    image_encoder.to(DEVICE)
    image_encoder.eval()

    for parameter in image_encoder.parameters():
        parameter.requires_grad = False

    print(
        f"Image encoder       : "
        f"{IMAGE_MODEL_NAME}"
    )

    print(
        f"Image embedding dim : "
        f"{IMAGE_DIM}"
    )

    print(
        f"Weight enum         : "
        f"{weights}"
    )

    print(
        f"Load time           : "
        f"{time.time() - start:.2f} sec"
    )

    print(
        "Encoder frozen       : True"
    )

    return (
        weights.transforms(),
        image_encoder,
    )


# ============================================================================
# IMAGE EMBEDDINGS
# ============================================================================

def encode_image_records_resumable(
    records: list[dict],
    image_transform,
    image_encoder,
) -> Path:

    print("\n" + "=" * 72)
    print("GENERATING TEST IMAGE EMBEDDINGS")
    print("=" * 72)

    total_records = len(records)

    population_fp = population_fingerprint(
        records
    )

    ids_fp = record_ids_fingerprint(
        records
    )

    labels_fp = labels_fingerprint(
        records
    )

    if final_output_is_complete(
        "image",
        total_records,
        IMAGE_DIM,
        population_fp,
        ids_fp,
        labels_fp,
    ):
        final_path = final_array_path(
            "image"
        )

        print(
            "SKIP: completed test image embeddings "
            f"already exist: {final_path}"
        )

        return final_path

    embeddings, completed = (
        open_or_create_checkpoint(
            "image",
            total_records,
            IMAGE_DIM,
            population_fp,
            ids_fp,
            labels_fp,
        )
    )

    start_time = time.time()
    run_start_completed = completed

    batch_images = []
    batch_indices = []

    def flush_batch():
        nonlocal batch_images
        nonlocal batch_indices

        if not batch_images:
            return

        image_tensor = torch.stack(
            batch_images,
            dim=0,
        ).to(DEVICE)

        with torch.inference_mode():
            features = image_encoder(
                image_tensor
            )

        features = features.flatten(
            start_dim=1
        )

        features_np = (
            features
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if features_np.shape[1] != IMAGE_DIM:
            raise RuntimeError(
                "Unexpected image embedding "
                f"dimension: {features_np.shape}"
            )

        for row, record_index in enumerate(
            batch_indices
        ):
            if not np.isfinite(
                features_np[row]
            ).all():
                raise RuntimeError(
                    "Non-finite image embedding "
                    f"at record index {record_index}"
                )

            embeddings[record_index] = (
                features_np[row]
            )

        batch_images = []
        batch_indices = []

    for record_index in range(
        completed,
        total_records,
    ):
        record = records[record_index]

        image_path = Path(
            record["image_path"]
        )

        with Image.open(
            image_path
        ) as image:
            image = image.convert("RGB")

            tensor = image_transform(
                image
            )

        batch_images.append(tensor)
        batch_indices.append(
            record_index
        )

        if len(batch_images) >= IMAGE_BATCH_SIZE:
            flush_batch()

        next_completed = (
            record_index + 1
        )

        if (
            next_completed % CHECKPOINT_EVERY == 0
            or next_completed == total_records
        ):
            flush_batch()

            if not memmap_is_finite(
                embeddings[:next_completed]
            ):
                raise RuntimeError(
                    "Non-finite image embedding "
                    "detected before checkpoint at "
                    f"{next_completed} records."
                )

            save_checkpoint_progress(
                "image",
                next_completed,
                total_records,
                IMAGE_DIM,
                population_fp,
                ids_fp,
                labels_fp,
                embeddings,
            )

            elapsed = (
                time.time()
                - start_time
            )

            current_run = (
                next_completed
                - run_start_completed
            )

            throughput = (
                current_run
                / max(elapsed, 1e-9)
            )

            remaining = (
                total_records
                - next_completed
            )

            eta_seconds = (
                remaining
                / max(throughput, 1e-9)
            )

            print(
                f"IMAGE | test | "
                f"{next_completed:,}/{total_records:,} | "
                f"{throughput:.2f} records/sec | "
                f"ETA {eta_seconds / 60:.1f} min"
            )

    flush_batch()

    if not memmap_is_finite(
        embeddings
    ):
        raise RuntimeError(
            "Non-finite values found in completed "
            "test image checkpoint."
        )

    final_path = mark_checkpoint_complete(
        "image",
        total_records,
        IMAGE_DIM,
        population_fp,
        ids_fp,
        labels_fp,
        embeddings,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Image generation elapsed: "
        f"{elapsed:.2f} sec"
    )

    print(
        f"Image final output: {final_path}"
    )

    return final_path


# ============================================================================
# METADATA
# ============================================================================

def save_metadata_arrays(
    records: list[dict],
):
    labels_path = (
        OUTPUT_ROOT
        / "test_labels.npy"
    )

    ids_path = (
        OUTPUT_ROOT
        / "test_record_ids.json"
    )

    labels = np.asarray(
        [
            record["label"]
            for record in records
        ],
        dtype=np.int64,
    )

    record_ids = [
        record["record_id"]
        for record in records
    ]

    np.save(
        labels_path,
        labels,
    )

    with ids_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            record_ids,
            f,
            indent=2,
        )

        f.flush()
        os.fsync(
            f.fileno()
        )

    print(
        f"Saved labels     : {labels_path}"
    )

    print(
        f"Saved record IDs : {ids_path}"
    )

    return {
        "labels_path": str(
            labels_path
        ),
        "record_ids_path": str(
            ids_path
        ),
        "labels_sha256": sha256_file(
            labels_path
        ),
        "record_ids_sha256": sha256_file(
            ids_path
        ),
    }


# ============================================================================
# FINAL VERIFICATION
# ============================================================================

def verify_embeddings(
    records: list[dict],
    text_path: Path,
    image_path: Path,
):
    print("\n" + "=" * 72)
    print("VERIFYING TEST EMBEDDINGS")
    print("=" * 72)

    expected_count = len(records)

    text_embeddings = np.load(
        text_path,
        mmap_mode="r",
    )

    image_embeddings = np.load(
        image_path,
        mmap_mode="r",
    )

    expected_text_shape = (
        expected_count,
        TEXT_DIM,
    )

    expected_image_shape = (
        expected_count,
        IMAGE_DIM,
    )

    if text_embeddings.shape != expected_text_shape:
        raise RuntimeError(
            "Test text shape mismatch: "
            f"{text_embeddings.shape} != "
            f"{expected_text_shape}"
        )

    if image_embeddings.shape != expected_image_shape:
        raise RuntimeError(
            "Test image shape mismatch: "
            f"{image_embeddings.shape} != "
            f"{expected_image_shape}"
        )

    if (
        text_embeddings.dtype != np.float32
        or image_embeddings.dtype != np.float32
    ):
        raise RuntimeError(
            "Final test embedding dtype is not float32."
        )

    if not memmap_is_finite(
        text_embeddings
    ):
        raise RuntimeError(
            "Test text embeddings contain "
            "non-finite values."
        )

    if not memmap_is_finite(
        image_embeddings
    ):
        raise RuntimeError(
            "Test image embeddings contain "
            "non-finite values."
        )

    labels = np.asarray(
        [
            record["label"]
            for record in records
        ],
        dtype=np.int64,
    )

    if labels.shape != (
        expected_count,
    ):
        raise RuntimeError(
            "Test label shape mismatch."
        )

    if (
        labels.min() < 0
        or labels.max() >= EXPECTED_CLASSES
    ):
        raise RuntimeError(
            "Test contains labels outside "
            f"[0, {EXPECTED_CLASSES - 1}]."
        )

    record_ids = [
        record["record_id"]
        for record in records
    ]

    if len(record_ids) != len(
        set(record_ids)
    ):
        raise RuntimeError(
            "Test contains duplicate record IDs."
        )

    print(
        f"Records          : {expected_count:,}"
    )

    print(
        f"Text shape       : "
        f"{text_embeddings.shape}"
    )

    print(
        f"Image shape      : "
        f"{image_embeddings.shape}"
    )

    print(
        "Text finite      : "
        f"{memmap_is_finite(text_embeddings)}"
    )

    print(
        "Image finite     : "
        f"{memmap_is_finite(image_embeddings)}"
    )

    print(
        f"Labels shape     : {labels.shape}"
    )

    print(
        f"Labels valid     : "
        f"{labels.min()}..{labels.max()} "
        "within 549 classes"
    )

    print(
        "Record IDs unique: True"
    )

    print("PASS.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(SEED)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("C4.1 — ISOLATED ABO TEST EMBEDDING PREPARATION")
    print("TEST INFERENCE PREPARATION ONLY")
    print("=" * 72)

    print("\nScientific boundary")
    print("-" * 72)
    print("Test access          : AUTHORIZED")
    print("Test usage           : INFERENCE ONLY")
    print("Model selection      : NONE")
    print("Hyperparameter tuning: NONE")
    print("Training             : NONE")
    print("B3 modification      : NONE")
    print("C3 artifacts         : READ-ONLY / NOT MODIFIED")

    print("\nEnvironment")
    print("-" * 72)
    print(f"Python               : {sys.version.split()[0]}")
    print(f"PyTorch              : {torch.__version__}")
    print(f"NumPy                : {np.__version__}")
    print(f"Device               : {DEVICE}")
    print(f"Seed                 : {SEED}")
    print(f"Checkpoint every     : {CHECKPOINT_EVERY:,}")
    print(f"B3 test              : {B3_TEST}")
    print(f"Image root           : {IMAGE_ROOT}")
    print(f"Output root          : {OUTPUT_ROOT}")

    # ========================================================================
    # REQUIRED INPUTS
    # ========================================================================

    print("\n" + "=" * 72)
    print("VERIFYING REQUIRED FROZEN INPUTS")
    print("=" * 72)

    required_paths = [
        B3_TEST,
        IMAGE_ROOT,
        B4_TEXT_MODEL,
        B4_IMAGE_MODEL,
    ]

    for path in required_paths:
        print(
            f"{path} | exists={path.exists()}"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Required frozen input missing: "
                f"{path}"
            )

    print(
        "PASS: all required frozen inputs exist."
    )

    # ========================================================================
    # TARGET SPACE
    # ========================================================================

    frozen_classes, class_to_index = (
        load_frozen_class_space()
    )

    if len(frozen_classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            "Frozen target space size mismatch."
        )

    # ========================================================================
    # LOAD ENCODERS
    # ========================================================================

    tokenizer, text_encoder = (
        load_text_encoder()
    )

    image_transform, image_encoder = (
        load_image_encoder()
    )

    # ========================================================================
    # TEST POPULATION
    # ========================================================================

    print("\n" + "=" * 72)
    print("VERIFYING LOCKED ABO TEST POPULATION")
    print("=" * 72)

    (
        test_records,
        test_excluded,
        test_missing_modality,
        test_bad,
    ) = load_test_records(
        B3_TEST,
        class_to_index,
    )

    print(
        f"Eligible test records             : "
        f"{len(test_records):,}"
    )

    print(
        f"Missing-modality exclusions       : "
        f"{test_missing_modality:,}"
    )

    print(
        f"Excluded outside frozen 549 space : "
        f"{test_excluded:,}"
    )

    print(
        f"Malformed records                 : "
        f"{len(test_bad):,}"
    )

    if test_bad:
        raise RuntimeError(
            "Test contains genuinely malformed "
            "B3 records: "
            f"{test_bad[:10]}"
        )

    if len(test_records) != EXPECTED_TEST_POPULATION:
        raise RuntimeError(
            "Locked Phase C test population mismatch: "
            f"expected {EXPECTED_TEST_POPULATION:,}, "
            f"found {len(test_records):,}."
        )

    print(
        "PASS: exact 7,346-record Phase C "
        "test population verified."
    )

    # ========================================================================
    # TEST IDENTITY
    # ========================================================================

    test_ids = [
        record["record_id"]
        for record in test_records
    ]

    if len(test_ids) != len(set(test_ids)):
        raise RuntimeError(
            "Duplicate test record IDs detected."
        )

    test_population_fp = (
        population_fingerprint(
            test_records
        )
    )

    test_ids_fp = (
        record_ids_fingerprint(
            test_records
        )
    )

    test_labels_fp = (
        labels_fingerprint(
            test_records
        )
    )

    print("\n" + "=" * 72)
    print("LOCKED TEST POPULATION FINGERPRINTS")
    print("=" * 72)

    print(
        f"Test population FP : "
        f"{test_population_fp}"
    )

    print(
        f"Test record-ID FP  : "
        f"{test_ids_fp}"
    )

    print(
        f"Test labels FP     : "
        f"{test_labels_fp}"
    )

    # ========================================================================
    # SAVE METADATA
    # ========================================================================

    print("\n" + "=" * 72)
    print("SAVING TEST LABEL / RECORD-ID METADATA")
    print("=" * 72)

    metadata = save_metadata_arrays(
        test_records
    )

    # ========================================================================
    # TEXT
    # ========================================================================

    test_text_path = (
        encode_text_records_resumable(
            test_records,
            tokenizer,
            text_encoder,
        )
    )

    # ========================================================================
    # IMAGE
    # ========================================================================

    test_image_path = (
        encode_image_records_resumable(
            test_records,
            image_transform,
            image_encoder,
        )
    )

    # ========================================================================
    # VERIFY
    # ========================================================================

    verify_embeddings(
        test_records,
        test_text_path,
        test_image_path,
    )

    # ========================================================================
    # MANIFEST
    # ========================================================================

    print("\n" + "=" * 72)
    print("WRITING C4.1 TEST EMBEDDING MANIFEST")
    print("=" * 72)

    manifest = {
        "phase": "C4.1",
        "task": "isolated_test_embedding_preparation",
        "scientific_status": "TEST_INFERENCE_PREPARATION",
        "test_accessed": True,
        "test_usage": "inference_only",
        "model_selection": False,
        "hyperparameter_selection": False,
        "training": False,
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
            "test_source": str(B3_TEST),
            "read_only": True,
        },
        "target": {
            "field": "product_type",
            "class_count": EXPECTED_CLASSES,
            "class_order_source": (
                "frozen B4.2 image LabelEncoder"
            ),
            "classes": frozen_classes,
        },
        "population": {
            "test": len(test_records),
            "expected_test": EXPECTED_TEST_POPULATION,
            "missing_modality_excluded": (
                test_missing_modality
            ),
            "excluded_outside_549": (
                test_excluded
            ),
            "malformed_records": len(test_bad),
            "population_fingerprint": (
                test_population_fp
            ),
            "record_ids_fingerprint": (
                test_ids_fp
            ),
            "labels_fingerprint": (
                test_labels_fp
            ),
        },
        "text": {
            "model": TEXT_MODEL_NAME,
            "embedding_dim": TEXT_DIM,
            "max_sequence_length": MAX_SEQ_LENGTH,
            "content_tokens_per_chunk": CONTENT_PER_CHUNK,
            "chunking": "token_preserving",
            "chunk_pooling": "mean",
            "encoder_frozen": True,
            "tokenizer_class": (
                tokenizer.__class__.__name__
            ),
        },
        "image": {
            "model": IMAGE_MODEL_NAME,
            "weight_enum": IMAGE_WEIGHT_ENUM,
            "embedding_dim": IMAGE_DIM,
            "encoder_frozen": True,
            "image_root": str(IMAGE_ROOT),
        },
        "outputs": {
            "text_path": str(test_text_path),
            "image_path": str(test_image_path),
            "labels_path": metadata[
                "labels_path"
            ],
            "record_ids_path": metadata[
                "record_ids_path"
            ],
        },
        "integrity": {
            "bad_records": len(test_bad),
            "missing_modality_excluded": (
                test_missing_modality
            ),
            "excluded_target": test_excluded,
            "test_population_exact": (
                len(test_records)
                == EXPECTED_TEST_POPULATION
            ),
            "b3_modified": False,
            "test_accessed": True,
            "test_usage": "inference_only",
            "model_selection": False,
            "hyperparameter_selection": False,
        },
        "software": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "numpy": np.__version__,
        },
        "checkpoint_outputs": {
            "test_text": str(
                checkpoint_progress_path(
                    "text"
                )
            ),
            "test_image": str(
                checkpoint_progress_path(
                    "image"
                )
            ),
        },
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
    }

    manifest_path = (
        OUTPUT_ROOT
        / "c4_test_embedding_manifest.json"
    )

    atomic_write_json(
        manifest_path,
        manifest,
    )

    print(
        f"Manifest : {manifest_path}"
    )

    # ========================================================================
    # FINAL INTEGRITY
    # ========================================================================

    print("\n" + "=" * 72)
    print("FINAL C4.1 TEST EMBEDDING INTEGRITY")
    print("=" * 72)

    checks = [
        (
            "B3 v002 read-only source",
            True,
        ),
        (
            "Exact test population",
            len(test_records)
            == EXPECTED_TEST_POPULATION,
        ),
        (
            "Malformed records = 0",
            len(test_bad) == 0,
        ),
        (
            "549 frozen target classes",
            len(frozen_classes)
            == EXPECTED_CLASSES,
        ),
        (
            "384-D test text embeddings",
            np.load(
                test_text_path,
                mmap_mode="r",
            ).shape
            == (
                EXPECTED_TEST_POPULATION,
                TEXT_DIM,
            ),
        ),
        (
            "512-D test image embeddings",
            np.load(
                test_image_path,
                mmap_mode="r",
            ).shape
            == (
                EXPECTED_TEST_POPULATION,
                IMAGE_DIM,
            ),
        ),
        (
            "Test record IDs unique",
            len(test_ids)
            == len(set(test_ids)),
        ),
        (
            "Test population fingerprint stable",
            population_fingerprint(
                test_records
            )
            == test_population_fp,
        ),
        (
            "Test embeddings finite",
            memmap_is_finite(
                np.load(
                    test_text_path,
                    mmap_mode="r",
                )
            )
            and memmap_is_finite(
                np.load(
                    test_image_path,
                    mmap_mode="r",
                )
            ),
        ),
        (
            "Test access declared inference-only",
            True,
        ),
        (
            "No model selection from test",
            True,
        ),
        (
            "No hyperparameter selection from test",
            True,
        ),
        (
            "B3 unmodified",
            True,
        ),
        (
            "Test text output complete",
            final_output_is_complete(
                "text",
                len(test_records),
                TEXT_DIM,
                test_population_fp,
                test_ids_fp,
                test_labels_fp,
            ),
        ),
        (
            "Test image output complete",
            final_output_is_complete(
                "image",
                len(test_records),
                IMAGE_DIM,
                test_population_fp,
                test_ids_fp,
                test_labels_fp,
            ),
        ),
    ]

    all_pass = True

    for description, passed in checks:
        if passed:
            print(
                f"PASS: {description}."
            )
        else:
            print(
                f"FAIL: {description}."
            )
            all_pass = False

    if not all_pass:
        raise RuntimeError(
            "C4.1 test embedding preparation "
            "integrity failed."
        )

    print("\n" + "=" * 72)
    print("C4.1 TEST EMBEDDING PREPARATION COMPLETE")
    print("=" * 72)

    print(
        f"Output directory: {OUTPUT_ROOT}"
    )

    print(
        "\nThe isolated 7,346-record test "
        "embedding population is ready for C4.2 "
        "evaluation."
    )


if __name__ == "__main__":
    main()