import json
import joblib
from pathlib import Path

root = Path(".")

text_bundle = joblib.load(
    root / "data" / "models" / "abo" / "b4.2" / "text_model.joblib"
)

image_bundle = joblib.load(
    root / "data" / "models" / "abo" / "b4.2" / "image_model.joblib"
)

text_classes = {
    str(x).strip()
    for x in text_bundle["label_encoder"].classes_
}

image_classes = {
    str(x).strip()
    for x in image_bundle["label_encoder"].classes_
}

common_classes = text_classes & image_classes

print("=" * 72)
print("B5 LABEL REPRESENTATION DIAGNOSTIC")
print("=" * 72)

print("Text model classes   :", len(text_classes))
print("Image model classes  :", len(image_classes))
print("Common model classes:", len(common_classes))

print()
print("FIRST 20 COMMON MODEL LABELS")
print("-" * 72)

for x in sorted(common_classes)[:20]:
    print(repr(x))

print()
print("FIRST 20 B3 PRODUCT_TYPE VALUES")
print("-" * 72)

path = (
    root
    / "data"
    / "representations"
    / "abo"
    / "b3"
    / "train.jsonl"
)

b3_values = []

with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        record = json.loads(line)
        value = record.get("product_type")

        if value is not None:
            b3_values.append(value)

        if len(b3_values) >= 20:
            break

for value in b3_values:
    print(repr(value))

print()
print("DIRECT STRING MATCH TEST")
print("-" * 72)

matched = 0
total = 0

with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        record = json.loads(line)
        target = record.get("product_type")

        if target is None:
            continue

        total += 1

        if str(target).strip() in common_classes:
            matched += 1

print("B3 non-empty train targets :", total)
print("Direct matches              :", matched)

print()
print("NORMALIZED MATCH TEST")
print("-" * 72)

def normalize(value):
    if value is None:
        return None

    return " ".join(
        str(value).strip().lower().split()
    )

normalized_common = {
    normalize(x)
    for x in common_classes
}

normalized_matched = 0

with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        record = json.loads(line)
        target = record.get("product_type")

        if target is None:
            continue

        if normalize(target) in normalized_common:
            normalized_matched += 1

print(
    "Normalized matches:",
    normalized_matched
)

print()
print("=" * 72)
print("DIAGNOSTIC COMPLETE")
print("=" * 72)
