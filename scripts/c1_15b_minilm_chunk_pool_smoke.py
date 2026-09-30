import json
import time
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

B3_PATH = Path(r"data\representations\abo\b3\train.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_SEQ_LENGTH = 256
CONTENT_PER_CHUNK = MAX_SEQ_LENGTH - 2

# Controlled sample:
# short, medium, long, and very-long examples.
TARGET_B3_TOKEN_LENGTHS = [50, 200, 500, 1000, 2000, 5000]

print("=" * 70)
print("C1.15-B — MiniLM Chunk-and-Pool Embedding Feasibility")
print("=" * 70)

print()
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

print("Loading SentenceTransformer...")
model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

model.max_seq_length = MAX_SEQ_LENGTH

print(f"Model max_seq_length: {model.max_seq_length}")
print()

# ------------------------------------------------------------
# Collect representative records from frozen B3 v002.
# ------------------------------------------------------------

candidates = []

with B3_PATH.open("r", encoding="utf-8") as f:

    for line in f:

        if not line.strip():
            continue

        record = json.loads(line)
        rep = record["b3_representation"]

        if rep.get("representation_version") != "v002":
            continue

        image = rep["image"]["main"]

        if not image.get("physical_file_available", False):
            continue

        if not image.get("usable_for_local_image_model", False):
            continue

        tokens = rep["text"]["combined_tokens"]
        text = " ".join(tokens)

        tokenized = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )

        length = len(tokenized["input_ids"])

        candidates.append(
            {
                "record_id": record["record_id"],
                "text": text,
                "b3_length": len(tokens),
                "token_length": length,
            }
        )

        if len(candidates) >= 69_823:
            break


# ------------------------------------------------------------
# Select nearest records to target lengths.
# ------------------------------------------------------------

selected = []

for target in TARGET_B3_TOKEN_LENGTHS:

    candidate = min(
        candidates,
        key=lambda x: abs(x["token_length"] - target)
    )

    if candidate["record_id"] not in {
        x["record_id"] for x in selected
    }:
        selected.append(candidate)


print("Selected representative records")
print("-" * 70)

for i, item in enumerate(selected, start=1):
    print(
        f"{i}. record_id={item['record_id']} "
        f"| B3 tokens={item['b3_length']:,} "
        f"| MiniLM tokens={item['token_length']:,}"
    )

# ------------------------------------------------------------
# Chunk helper
# ------------------------------------------------------------

def make_chunks(text):
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )["input_ids"]

    chunks = []

    for start in range(0, len(ids), CONTENT_PER_CHUNK):
        chunk_ids = ids[start:start + CONTENT_PER_CHUNK]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        chunks.append(chunk_text)

    return chunks


# ------------------------------------------------------------
# Embedding functions
# ------------------------------------------------------------

def direct_embedding(text):
    start = time.perf_counter()

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )

    elapsed = time.perf_counter() - start

    return embedding.astype(np.float32), elapsed


def chunk_pool_embedding(text):

    start = time.perf_counter()

    chunks = make_chunks(text)

    chunk_embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
        batch_size=16,
    )

    chunk_embeddings = np.asarray(
        chunk_embeddings,
        dtype=np.float32,
    )

    pooled = np.mean(
        chunk_embeddings,
        axis=0,
    ).astype(np.float32)

    elapsed = time.perf_counter() - start

    return pooled, len(chunks), elapsed


# ------------------------------------------------------------
# Run experiment
# ------------------------------------------------------------

print()
print("=" * 70)
print("EMBEDDING RESULTS")
print("=" * 70)

for i, item in enumerate(selected, start=1):

    print()
    print(f"CASE {i}")
    print("-" * 70)
    print(f"Record ID       : {item['record_id']}")
    print(f"B3 token count  : {item['b3_length']:,}")
    print(f"MiniLM tokens   : {item['token_length']:,}")

    # Direct/truncated representation.
    direct_1, direct_time_1 = direct_embedding(item["text"])

    # Chunk-and-pool representation.
    pooled_1, n_chunks_1, pooled_time_1 = chunk_pool_embedding(
        item["text"]
    )

    # Repeat both methods.
    direct_2, direct_time_2 = direct_embedding(item["text"])

    pooled_2, n_chunks_2, pooled_time_2 = chunk_pool_embedding(
        item["text"]
    )

    direct_diff = np.max(
        np.abs(direct_1 - direct_2)
    )

    pooled_diff = np.max(
        np.abs(pooled_1 - pooled_2)
    )

    print()
    print("Direct MiniLM")
    print(f"  Shape              : {direct_1.shape}")
    print(f"  dtype              : {direct_1.dtype}")
    print(f"  finite             : {np.isfinite(direct_1).all()}")
    print(f"  runtime #1         : {direct_time_1:.4f} sec")
    print(f"  runtime #2         : {direct_time_2:.4f} sec")
    print(f"  repeat max abs diff: {direct_diff:.10f}")

    print()
    print("Chunk-and-pool")
    print(f"  Chunks             : {n_chunks_1}")
    print(f"  Shape              : {pooled_1.shape}")
    print(f"  dtype              : {pooled_1.dtype}")
    print(f"  finite             : {np.isfinite(pooled_1).all()}")
    print(f"  runtime #1         : {pooled_time_1:.4f} sec")
    print(f"  runtime #2         : {pooled_time_2:.4f} sec")
    print(f"  repeat max abs diff: {pooled_diff:.10f}")

    print()
    print("Representation difference")
    print(
        f"  Direct vs pooled L2 distance: "
        f"{np.linalg.norm(direct_1 - pooled_1):.6f}"
    )

# ------------------------------------------------------------
# Final integrity
# ------------------------------------------------------------

print()
print("=" * 70)
print("INTEGRITY")
print("=" * 70)

print("PASS: B3 v002 read-only.")
print("PASS: no classifier training.")
print("PASS: no test data used.")
print("PASS: direct and chunk-and-pool embeddings generated.")
print("PASS: repeatability measured.")
print("=" * 70)
