import json
import statistics
from pathlib import Path
from transformers import AutoTokenizer

B3_PATH = Path(r"data\representations\abo\b3\train.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MAX_SEQ_LENGTH = 256
BATCH_SIZE = 256

print("=" * 70)
print("C1.14 — MiniLM Tokenizer Context Coverage — Corrected")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

lengths = []
b3_lengths = []

total_records = 0
eligible_records = 0
bad_records = 0

batch_texts = []
batch_b3_lengths = []


def process_batch(texts, b3_lens):
    if not texts:
        return

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    lengths.extend(len(ids) for ids in encoded["input_ids"])
    b3_lengths.extend(b3_lens)


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

            combined_tokens = rep["text"]["combined_tokens"]
            image = rep["image"]["main"]

            if not image.get("physical_file_available", False):
                continue

            if not image.get("usable_for_local_image_model", False):
                continue

            text = " ".join(combined_tokens)

            eligible_records += 1

            batch_texts.append(text)
            batch_b3_lengths.append(len(combined_tokens))

            if len(batch_texts) >= BATCH_SIZE:
                process_batch(batch_texts, batch_b3_lengths)
                batch_texts.clear()
                batch_b3_lengths.clear()

        except Exception as e:
            bad_records += 1
            print(
                f"WARNING line {line_number}: "
                f"{type(e).__name__}: {e}"
            )

process_batch(batch_texts, batch_b3_lengths)


def percentile(sorted_values, p):
    n = len(sorted_values)
    index = (n - 1) * p
    lower = int(index)
    upper = min(lower + 1, n - 1)
    fraction = index - lower
    return (
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def print_summary(title, values):
    s = sorted(values)

    print()
    print(title)
    print("-" * 50)
    print(f"Count  : {len(s):,}")
    print(f"Min    : {s[0]:,.2f}")
    print(f"Max    : {s[-1]:,.2f}")
    print(f"Mean   : {statistics.mean(s):,.2f}")
    print(f"Median : {statistics.median(s):,.2f}")
    print(f"P90    : {percentile(s, 0.90):,.2f}")
    print(f"P95    : {percentile(s, 0.95):,.2f}")
    print(f"P99    : {percentile(s, 0.99):,.2f}")


print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print_summary("Actual MiniLM tokenizer lengths", lengths)

print()
print("MiniLM context coverage")
print("-" * 50)

n = len(lengths)

for limit in [128, 256, 512, 1024]:
    count_over = sum(x > limit for x in lengths)
    count_within = n - count_over

    print(
        f"<= {limit:4d}: {count_within:7,d} "
        f"({count_within / n * 100:6.2f}%)"
    )
    print(
        f">  {limit:4d}: {count_over:7,d} "
        f"({count_over / n * 100:6.2f}%)"
    )

# Number of tokenizer units that would exceed the 256-token window.
tokens_over_256 = sum(
    max(0, x - MODEL_MAX_SEQ_LENGTH)
    for x in lengths
)

records_over_256 = sum(
    x > MODEL_MAX_SEQ_LENGTH
    for x in lengths
)

print()
print("256-token truncation exposure")
print("-" * 50)
print(f"Records >256 tokens       : {records_over_256:,}")
print(f"Fraction >256             : "
      f"{records_over_256 / n * 100:.2f}%")
print(f"Total excess token units  : {tokens_over_256:,}")

print_summary(
    "B3 whitespace-token lengths "
    "(same primary population)",
    b3_lengths,
)

print()
print("=" * 70)
print("INTEGRITY CHECKS")
print("=" * 70)

print(f"Total B3 train records read : {total_records:,}")
print(f"Primary eligible records    : {eligible_records:,}")
print(f"Tokenized records           : {len(lengths):,}")
print(f"Bad records                 : {bad_records:,}")

expected = 69_823

print()
print(f"Expected C1.4 population: {expected:,}")

assert eligible_records == expected, (
    f"Population mismatch: {eligible_records} != {expected}"
)

assert len(lengths) == expected, (
    f"Tokenized count mismatch: {len(lengths)} != {expected}"
)

print("PASS: primary population = 69,823.")
print("PASS: all eligible records tokenized.")
print("PASS: no B3 artifact modified.")
print("PASS: no model trained.")
print("=" * 70)
