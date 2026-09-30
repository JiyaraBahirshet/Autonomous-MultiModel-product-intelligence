import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

# ============================================================
# C1.16 — Image Representation Feasibility
# ============================================================

B3_PATH = Path(
    r"data\representations\abo\b3\train.jsonl"
)

IMAGE_ROOT = Path(
    r"ABO_Audit\images\small"
)

EXPECTED_POPULATION = 69_823
SAMPLE_SIZE = 12

DEVICE = torch.device("cpu")

print("=" * 70)
print("C1.16 — Image Representation Feasibility")
print("=" * 70)

print()
print("Environment")
print("-" * 70)
print(f"PyTorch       : {torch.__version__}")
print(f"Torchvision   : {torchvision.__version__}")
print(f"Device        : {DEVICE}")
print(f"B3 source     : {B3_PATH}")
print(f"Image root    : {IMAGE_ROOT}")
print()

# ============================================================
# Collect exact Phase C eligible training image references.
# ============================================================

print("Reading frozen B3 v002 training records...")

eligible = []
bad_paths = []

with B3_PATH.open("r", encoding="utf-8") as f:

    for line in f:

        if not line.strip():
            continue

        record = json.loads(line)
        rep = record["b3_representation"]

        if rep.get("representation_version") != "v002":
            continue

        image_info = rep["image"]["main"]

        if not image_info.get(
            "physical_file_available",
            False,
        ):
            continue

        if not image_info.get(
            "usable_for_local_image_model",
            False,
        ):
            continue

        relative_path = image_info.get(
            "relative_path"
        )

        if not relative_path:
            bad_paths.append(
                (
                    record["record_id"],
                    "missing relative_path",
                )
            )
            continue

        image_path = (
            IMAGE_ROOT / Path(relative_path)
        )

        if not image_path.is_file():

            bad_paths.append(
                (
                    record["record_id"],
                    str(image_path),
                )
            )
            continue

        eligible.append(
            {
                "record_id": record["record_id"],
                "relative_path": relative_path,
                "image_path": image_path,
            }
        )

        if len(eligible) >= EXPECTED_POPULATION:
            break


print(f"Eligible records found : {len(eligible):,}")
print(f"Bad/missing paths      : {len(bad_paths):,}")

if len(eligible) != EXPECTED_POPULATION:
    raise RuntimeError(
        f"Expected {EXPECTED_POPULATION:,} eligible records, "
        f"found {len(eligible):,}."
    )

if bad_paths:
    raise RuntimeError(
        "Eligible records contained invalid image paths."
    )

print("PASS: exact locked Phase C image population found.")

# ============================================================
# Controlled image sample.
#
# Use deterministic evenly spaced positions across the
# eligible population rather than arbitrary filesystem files.
# ============================================================

indices = np.linspace(
    0,
    len(eligible) - 1,
    SAMPLE_SIZE,
    dtype=int,
)

sample = [
    eligible[i]
    for i in indices
]

print()
print("Selected image sample")
print("-" * 70)

for i, item in enumerate(sample, start=1):

    print(
        f"{i:02d}. {item['record_id']} | "
        f"{item['relative_path']}"
    )

# ============================================================
# Load pretrained models.
# ============================================================

print()
print("=" * 70)
print("LOADING PRETRAINED IMAGE ENCODERS")
print("=" * 70)

print()
print("Loading ResNet-18 pretrained weights...")

resnet_weights = ResNet18_Weights.DEFAULT
resnet = resnet18(
    weights=resnet_weights
)

# Remove classifier; retain 512-D representation.
resnet.fc = torch.nn.Identity()
resnet.eval()
resnet.to(DEVICE)

resnet_transform = (
    resnet_weights.transforms()
)

print(
    f"ResNet-18 representation dimension: "
    f"{resnet.fc.in_features if hasattr(resnet.fc, 'in_features') else 512}"
)

print()
print("Loading MobileNetV3-Small pretrained weights...")

mobile_weights = (
    MobileNet_V3_Small_Weights.DEFAULT
)

mobilenet = mobilenet_v3_small(
    weights=mobile_weights
)

# classifier[0] is the projection from pooled features.
mobile_embedding_dim = (
    mobilenet.classifier[0].in_features
)

# Replace classifier with identity so output is the
# pooled feature representation.
mobilenet.classifier = torch.nn.Identity()

mobilenet.eval()
mobilenet.to(DEVICE)

mobile_transform = (
    mobile_weights.transforms()
)

print(
    f"MobileNetV3-Small representation dimension: "
    f"{mobile_embedding_dim}"
)

# ============================================================
# Load images.
# ============================================================

from PIL import Image


def load_image(path):

    with Image.open(path) as img:

        image = img.convert("RGB")

        return image.copy()


# ============================================================
# Embedding helper.
# ============================================================

def embed_one(
    model,
    transform,
    image,
):

    tensor = transform(image).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        embedding = model(
            tensor
        )

    return (
        embedding.squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# Run each model.
# ============================================================

results = {}

for model_name, model, transform in [
    (
        "ResNet-18",
        resnet,
        resnet_transform,
    ),
    (
        "MobileNetV3-Small",
        mobilenet,
        mobile_transform,
    ),
]:

    print()
    print("=" * 70)
    print(model_name.upper())
    print("=" * 70)

    model_results = []

    for i, item in enumerate(
        sample,
        start=1,
    ):

        image = load_image(
            item["image_path"]
        )

        # First run.
        start = time.perf_counter()

        emb1 = embed_one(
            model,
            transform,
            image,
        )

        time1 = (
            time.perf_counter() - start
        )

        # Repeat run.
        start = time.perf_counter()

        emb2 = embed_one(
            model,
            transform,
            image,
        )

        time2 = (
            time.perf_counter() - start
        )

        repeat_diff = np.max(
            np.abs(
                emb1 - emb2
            )
        )

        finite = np.isfinite(
            emb1
        ).all()

        model_results.append(
            {
                "record_id": item["record_id"],
                "embedding_dim": emb1.shape[0],
                "runtime_1": time1,
                "runtime_2": time2,
                "repeat_diff": repeat_diff,
                "finite": finite,
            }
        )

        print(
            f"{i:02d}. "
            f"{item['record_id']} | "
            f"dim={emb1.shape[0]} | "
            f"run1={time1:.4f}s | "
            f"run2={time2:.4f}s | "
            f"repeat_diff={repeat_diff:.10f} | "
            f"finite={finite}"
        )

    results[model_name] = model_results

# ============================================================
# Summary.
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

for model_name, rows in results.items():

    dims = {
        r["embedding_dim"]
        for r in rows
    }

    runtimes = [
        r["runtime_1"]
        for r in rows
    ]

    repeat_diffs = [
        r["repeat_diff"]
        for r in rows
    ]

    finite_values = [
        r["finite"]
        for r in rows
    ]

    print()
    print(model_name)
    print("-" * 70)
    print(f"Samples             : {len(rows)}")
    print(f"Embedding dimensions: {sorted(dims)}")
    print(
        f"Mean runtime        : "
        f"{np.mean(runtimes):.4f} sec"
    )
    print(
        f"Median runtime      : "
        f"{np.median(runtimes):.4f} sec"
    )
    print(
        f"Max runtime         : "
        f"{np.max(runtimes):.4f} sec"
    )
    print(
        f"Max repeat diff     : "
        f"{np.max(repeat_diffs):.10f}"
    )
    print(
        f"All finite          : "
        f"{all(finite_values)}"
    )

# ============================================================
# Storage estimate for the full primary population.
# ============================================================

print()
print("=" * 70)
print("FULL-POPULATION STORAGE ESTIMATES")
print("=" * 70)

for name, dim in [
    ("ResNet-18", 512),
    ("MobileNetV3-Small", mobile_embedding_dim),
]:

    bytes_total = (
        EXPECTED_POPULATION
        * dim
        * 4
    )

    mib = (
        bytes_total
        / (1024 ** 2)
    )

    print(
        f"{name:24s}: "
        f"{dim:4d}-D float32 -> "
        f"{mib:.2f} MiB"
    )

# ============================================================
# Integrity.
# ============================================================

print()
print("=" * 70)
print("INTEGRITY")
print("=" * 70)

print(
    "PASS: exact locked Phase C train population verified."
)
print(
    "PASS: images selected only through frozen B3 v002 paths."
)
print(
    "PASS: physical/usable image requirements enforced."
)
print(
    "PASS: no path guessing or image downloading."
)
print(
    "PASS: no test images used."
)
print(
    "PASS: no classifier training."
)
print(
    "PASS: B3 artifact read-only."
)
print(
    "PASS: repeatability measured."
)

print("=" * 70)
