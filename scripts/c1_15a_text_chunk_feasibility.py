import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

B3_PATH = Path(r"data\representations\abo\b3\train.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_MAX_LENGTH = 256

# MiniLM/BERT needs special tokens.
# We therefore reserve two positions:
# [CLS] + content + [SEP]
CONTENT_PER_CHUNK = MODEL_MAX_LENGTH - 2

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

print("=" * 70)
print("C1.15-A — Text Representation Strategy Feasibility")
print("MiniLM truncation vs context-preserving chunking")
print("=" * 70)

total_records = 0
eligible_records = 0
bad_records = 0

token_lengths = []
chunk_counts = []

with B3_PATH.open("r", encoding="utf-8") as f:

    for line_number, line in enumerate(f, start=1):

        if not line.strip():
            continue

        total_records += 1

        try:
            record = json.loads(line)
            rep = record["b3_representation"]

            if rep.get("representation_version") != "v002":
                continue

            image = rep["image"]["main"]

            if not image.get("physical_file_available", False):
                continue

            if not image.get("usable_for_local_image_model", False):
                continue

            combined_tokens = rep["text"]["combined_tokens"]

            text = " ".join(combined_tokens)

            # Tokenize WITHOUT special tokens so that we can
            # explicitly calculate chunk sizes.
            ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]

            n_tokens = len(ids)

            # Number of chunks required to preserve all content.
            n_chunks = max(
                1,
                (n_tokens + CONTENT_PER_CHUNK - 1)
                // CONTENT_PER_CHUNK
            )

            token_lengths.append(n_tokens)
            chunk_counts.append(n_chunks)

            eligible_records += 1

        except Exception as e:
            bad_records += 1
            print(
                f"WARNING line {line_number}: "
                f"{type(e).__name__}: {e}"
            )


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    frac = index - lo
    return values[lo] + frac * (values[hi] - values[lo])


print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Total B3 train records read : {total_records:,}")
print(f"Primary eligible records    : {eligible_records:,}")
print(f"Bad records                 : {bad_records:,}")

assert eligible_records == 69_823
assert bad_records == 0

print()
print("PASS: exact locked Phase C train population = 69,823.")

print()
print("Content-token distribution")
print("-" * 50)
print(f"Min    : {min(token_lengths):,}")
print(f"Median : {statistics.median(token_lengths):,.2f}")
print(f"Mean   : {statistics.mean(token_lengths):,.2f}")
print(f"P90    : {percentile(token_lengths, 0.90):,.2f}")
print(f"P95    : {percentile(token_lengths, 0.95):,.2f}")
print(f"P99    : {percentile(token_lengths, 0.99):,.2f}")
print(f"Max    : {max(token_lengths):,}")

print()
print("Chunking requirements")
print("-" * 50)

for k in [1, 2, 3, 4, 5, 8, 10, 16]:

    count = sum(x <= k for x in chunk_counts)

    print(
        f"Records requiring <= {k:2d} chunk(s): "
        f"{count:7,d} "
        f"({count / eligible_records * 100:6.2f}%)"
    )

print()
print("Chunk-count distribution")
print("-" * 50)

max_chunks = max(chunk_counts)

print(f"Maximum chunks required: {max_chunks:,}")

for k in range(1, min(max_chunks, 20) + 1):

    count = sum(x == k for x in chunk_counts)

    if count:
        print(
            f"{k:2d} chunk(s): "
            f"{count:7,d} "
            f"({count / eligible_records * 100:6.2f}%)"
        )

print()
print("Average chunk count")
print("-" * 50)
print(f"Mean chunks/record   : {statistics.mean(chunk_counts):.3f}")
print(f"Median chunks/record : {statistics.median(chunk_counts):.3f}")

print()
print("Comparison")
print("-" * 50)

truncated = sum(x > 254 for x in token_lengths)

print(
    f"Records requiring truncation with 256-token MiniLM: "
    f"{truncated:,} ({truncated / eligible_records * 100:.2f}%)"
)

print(
    "Chunk-and-pool preserves all tokenizer content "
    "by construction, subject to tokenizer/chunk implementation."
)

print()
print("=" * 70)
print("STATUS")
print("=" * 70)
print("C1.15-A is a feasibility analysis only.")
print("No B3 artifact modified.")
print("No classifier trained.")
print("No test data used.")
print("=" * 70)
