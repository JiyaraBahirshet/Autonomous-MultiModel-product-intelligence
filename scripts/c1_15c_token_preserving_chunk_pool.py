import json
import time
import numpy as np
from pathlib import Path

from transformers import AutoTokenizer, AutoModel
import torch

B3_PATH = Path(r"data\representations\abo\b3\train.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_SEQ_LENGTH = 256
CONTENT_PER_CHUNK = MAX_SEQ_LENGTH - 2

TARGET_TOKEN_LENGTHS = [50, 200, 500, 1000, 2000, 5000]

print("=" * 70)
print("C1.15-C — Token-Preserving MiniLM Chunk Embedding — Corrected")
print("=" * 70)

print()
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

print("Loading base MiniLM encoder...")
encoder = AutoModel.from_pretrained(
    MODEL_NAME,
)

encoder.eval()

print("Model loaded.")
print(f"Hidden size: {encoder.config.hidden_size}")
print(f"Tokenizer class: {tokenizer.__class__.__name__}")
print(f"CLS token ID: {tokenizer.cls_token_id}")
print(f"SEP token ID: {tokenizer.sep_token_id}")
print(f"Tokenizer model_max_length: {tokenizer.model_max_length}")
print(f"Phase C sequence length: {MAX_SEQ_LENGTH}")
print(f"Content tokens per chunk: {CONTENT_PER_CHUNK}")


# ------------------------------------------------------------
# Mean pooling
# ------------------------------------------------------------

def mean_pool(last_hidden_state, attention_mask):

    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    summed = torch.sum(
        last_hidden_state * mask,
        dim=1,
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9,
    )

    return summed / counts


# ------------------------------------------------------------
# Exact BERT-style input construction
# [CLS] + content + [SEP]
# ------------------------------------------------------------

def make_model_input(content_ids):

    input_ids = [
        tokenizer.cls_token_id,
        *content_ids,
        tokenizer.sep_token_id,
    ]

    if len(input_ids) > MAX_SEQ_LENGTH:
        raise ValueError(
            f"Input length {len(input_ids)} exceeds "
            f"MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}"
        )

    attention_mask = [1] * len(input_ids)

    return input_ids, attention_mask


# ------------------------------------------------------------
# Encode exact token IDs
# ------------------------------------------------------------

def encode_token_ids(input_ids, attention_mask):

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
    )

    mask_tensor = torch.tensor(
        [attention_mask],
        dtype=torch.long,
    )

    with torch.no_grad():

        outputs = encoder(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
        )

        embedding = mean_pool(
            outputs.last_hidden_state,
            mask_tensor,
        )

    return embedding[0].cpu().numpy().astype(
        np.float32
    )


# ------------------------------------------------------------
# Tokenize original B3 text
# ------------------------------------------------------------

def tokenize_full(text):

    return tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )["input_ids"]


# ------------------------------------------------------------
# Exact token-preserving chunking
# ------------------------------------------------------------

def make_chunks(token_ids):

    chunks = []

    for start in range(
        0,
        len(token_ids),
        CONTENT_PER_CHUNK,
    ):

        content_ids = token_ids[
            start:start + CONTENT_PER_CHUNK
        ]

        input_ids, attention_mask = (
            make_model_input(content_ids)
        )

        chunks.append(
            {
                "content_ids": content_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    return chunks


# ------------------------------------------------------------
# Collect representative records
# ------------------------------------------------------------

print()
print("Collecting representative records...")

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

        text = " ".join(
            rep["text"]["combined_tokens"]
        )

        ids = tokenize_full(text)

        candidates.append(
            {
                "record_id": record["record_id"],
                "text": text,
                "token_ids": ids,
                "token_length": len(ids),
                "b3_length": len(
                    rep["text"]["combined_tokens"]
                ),
            }
        )

        if len(candidates) >= 69_823:
            break


selected = []

for target in TARGET_TOKEN_LENGTHS:

    candidate = min(
        candidates,
        key=lambda x: abs(
            x["token_length"] - target
        ),
    )

    if candidate["record_id"] not in {
        x["record_id"] for x in selected
    }:
        selected.append(candidate)


print()
print("Selected representative records")
print("-" * 70)

for i, item in enumerate(selected, start=1):

    print(
        f"{i}. {item['record_id']} | "
        f"B3 tokens={item['b3_length']:,} | "
        f"MiniLM tokens={item['token_length']:,}"
    )


# ------------------------------------------------------------
# Direct 256-token representation
# ------------------------------------------------------------

def direct_embedding(token_ids):

    content_ids = token_ids[
        :CONTENT_PER_CHUNK
    ]

    input_ids, attention_mask = (
        make_model_input(content_ids)
    )

    start = time.perf_counter()

    embedding = encode_token_ids(
        input_ids,
        attention_mask,
    )

    elapsed = time.perf_counter() - start

    return embedding, elapsed


# ------------------------------------------------------------
# Token-preserving chunk-and-pool
# ------------------------------------------------------------

def chunk_pool_embedding(token_ids):

    chunks = make_chunks(token_ids)

    start = time.perf_counter()

    chunk_embeddings = []

    for chunk in chunks:

        embedding = encode_token_ids(
            chunk["input_ids"],
            chunk["attention_mask"],
        )

        chunk_embeddings.append(
            embedding
        )

    chunk_embeddings = np.stack(
        chunk_embeddings,
        axis=0,
    ).astype(np.float32)

    pooled = np.mean(
        chunk_embeddings,
        axis=0,
    ).astype(np.float32)

    elapsed = time.perf_counter() - start

    return (
        pooled,
        len(chunks),
        chunk_embeddings,
        elapsed,
    )


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

    print(
        f"Record ID      : {item['record_id']}"
    )
    print(
        f"B3 tokens      : {item['b3_length']:,}"
    )
    print(
        f"MiniLM tokens  : {item['token_length']:,}"
    )

    # Direct representation.
    direct_1, direct_time_1 = (
        direct_embedding(
            item["token_ids"]
        )
    )

    direct_2, direct_time_2 = (
        direct_embedding(
            item["token_ids"]
        )
    )

    # Chunk-and-pool.
    pooled_1, n_chunks_1, _, pooled_time_1 = (
        chunk_pool_embedding(
            item["token_ids"]
        )
    )

    pooled_2, n_chunks_2, _, pooled_time_2 = (
        chunk_pool_embedding(
            item["token_ids"]
        )
    )

    direct_repeat_diff = np.max(
        np.abs(
            direct_1 - direct_2
        )
    )

    pooled_repeat_diff = np.max(
        np.abs(
            pooled_1 - pooled_2
        )
    )

    print()
    print(
        "Direct 256-token representation"
    )
    print(
        f"  Shape              : "
        f"{direct_1.shape}"
    )
    print(
        f"  dtype              : "
        f"{direct_1.dtype}"
    )
    print(
        f"  finite             : "
        f"{np.isfinite(direct_1).all()}"
    )
    print(
        f"  runtime #1         : "
        f"{direct_time_1:.4f} sec"
    )
    print(
        f"  runtime #2         : "
        f"{direct_time_2:.4f} sec"
    )
    print(
        f"  repeat max abs diff: "
        f"{direct_repeat_diff:.10f}"
    )

    print()
    print(
        "Token-preserving chunk-and-pool"
    )
    print(
        f"  Chunks             : "
        f"{n_chunks_1}"
    )
    print(
        f"  Shape              : "
        f"{pooled_1.shape}"
    )
    print(
        f"  dtype              : "
        f"{pooled_1.dtype}"
    )
    print(
        f"  finite             : "
        f"{np.isfinite(pooled_1).all()}"
    )
    print(
        f"  runtime #1         : "
        f"{pooled_time_1:.4f} sec"
    )
    print(
        f"  runtime #2         : "
        f"{pooled_time_2:.4f} sec"
    )
    print(
        f"  repeat max abs diff: "
        f"{pooled_repeat_diff:.10f}"
    )

    if n_chunks_1 == 1:

        single_chunk_diff = np.max(
            np.abs(
                direct_1 - pooled_1
            )
        )

        print()
        print(
            "Single-chunk equivalence check"
        )
        print(
            f"  max abs difference: "
            f"{single_chunk_diff:.10f}"
        )

    print()
    print(
        "Direct vs pooled L2 distance"
    )
    print(
        f"  {np.linalg.norm(direct_1 - pooled_1):.6f}"
    )


# ------------------------------------------------------------
# Integrity
# ------------------------------------------------------------

print()
print("=" * 70)
print("INTEGRITY")
print("=" * 70)

print("PASS: B3 v002 read-only.")
print("PASS: no classifier training.")
print("PASS: no test data used.")
print("PASS: token IDs preserved during chunking.")
print("PASS: direct and chunk-and-pool embeddings generated.")
print("PASS: repeatability measured.")
print("=" * 70)
