from pathlib import Path
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn

SEED = 20260827
N_CLASSES = 549
TEXT_DIM = 384
IMAGE_DIM = 512
FUSED_DIM = 512
N_VALIDATION = 69867

EXPECTED_CKPT_SHA256 = (
    "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"
)

ROOT = Path("data/models/abo")
C3_EMB = ROOT / "phase_c/c3/embeddings"
C3_TRAIN = ROOT / "phase_c/c3/training"
D2_ROOT = ROOT / "phase_d/d2"
OUT_ROOT = D2_ROOT / "inference"

TEXT_PATH = C3_EMB / "validation_text.npy"
IMAGE_PATH = C3_EMB / "validation_image.npy"
LABEL_PATH = C3_EMB / "validation_labels.npy"
ID_PATH = C3_EMB / "validation_record_ids.json"

CKPT_PATH = C3_TRAIN / "cbase_best.pt"

PARTITION_ROOT = D2_ROOT / "partition"
DCAL_MANIFEST = PARTITION_ROOT / "dcal_manifest.json"
DSELECT_MANIFEST = PARTITION_ROOT / "dselect_manifest.json"
PARTITION_METADATA = PARTITION_ROOT / "d2_partition_metadata.json"

OUT_NPZ = OUT_ROOT / "d2_fresh_cbase_validation_outputs.npz"
OUT_MANIFEST = OUT_ROOT / "d2_fresh_inference_manifest.json"


class CBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(TEXT_DIM, FUSED_DIM),
            nn.ReLU(),
        )

        self.image_projection = nn.Sequential(
            nn.Linear(IMAGE_DIM, FUSED_DIM),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(FUSED_DIM * 2, FUSED_DIM),
            nn.ReLU(),
            nn.Linear(FUSED_DIM, N_CLASSES),
        )

    def forward(self, text, image):
        text_proj = self.text_projection(text)
        image_proj = self.image_projection(image)
        fused = torch.cat([text_proj, image_proj], dim=1)
        return self.fusion(fused)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    raise RuntimeError(
        "Could not identify model state dictionary in frozen C-Base checkpoint."
    )


def main():
    print("=== D2 FRESH C-BASE INFERENCE ===")
    print("Phase C remains frozen.")
    print("D-Test must remain untouched.")
    print()

    # ------------------------------------------------------------
    # 1. Required artifact existence
    # ------------------------------------------------------------
    required = [
        TEXT_PATH,
        IMAGE_PATH,
        LABEL_PATH,
        ID_PATH,
        CKPT_PATH,
        DCAL_MANIFEST,
        DSELECT_MANIFEST,
        PARTITION_METADATA,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    # ------------------------------------------------------------
    # 2. Verify frozen C-Base checkpoint
    # ------------------------------------------------------------
    actual_sha = sha256_file(CKPT_PATH)

    print("C-Base checkpoint:")
    print(f"  {CKPT_PATH}")
    print(f"  SHA-256: {actual_sha}")

    if actual_sha.lower() != EXPECTED_CKPT_SHA256.lower():
        raise RuntimeError(
            "FATAL: C-Base checkpoint SHA-256 does not match the frozen Phase C checkpoint."
        )

    print("  CHECKPOINT HASH: PASS")
    print()

    # ------------------------------------------------------------
    # 3. Load frozen Phase-C validation embeddings
    # ------------------------------------------------------------
    text = np.load(TEXT_PATH)
    image = np.load(IMAGE_PATH)
    labels = np.load(LABEL_PATH)

    with open(ID_PATH, "r", encoding="utf-8") as f:
        record_ids = json.load(f)

    if len(record_ids) != N_VALIDATION:
        raise RuntimeError(
            f"Unexpected validation ID count: {len(record_ids)}"
        )

    if len(set(record_ids)) != N_VALIDATION:
        raise RuntimeError("Validation record IDs are not unique.")

    if text.shape != (N_VALIDATION, TEXT_DIM):
        raise RuntimeError(f"Unexpected text shape: {text.shape}")

    if image.shape != (N_VALIDATION, IMAGE_DIM):
        raise RuntimeError(f"Unexpected image shape: {image.shape}")

    if labels.shape != (N_VALIDATION,):
        raise RuntimeError(f"Unexpected label shape: {labels.shape}")

    if not np.isfinite(text).all():
        raise RuntimeError("Non-finite values detected in validation text embeddings.")

    if not np.isfinite(image).all():
        raise RuntimeError("Non-finite values detected in validation image embeddings.")

    if not np.isfinite(labels).all():
        raise RuntimeError("Non-finite values detected in validation labels.")

    print("Validation embedding alignment:")
    print(f"  records : {len(record_ids)}")
    print(f"  text    : {text.shape}")
    print(f"  image   : {image.shape}")
    print(f"  labels  : {labels.shape}")
    print("  ALIGNMENT: PASS")
    print()

    # ------------------------------------------------------------
    # 4. Verify D-Cal / D-Select partition artifacts exist
    #    and have expected population counts.
    #
    #    We do NOT alter or regenerate these partitions.
    # ------------------------------------------------------------
    with open(DCAL_MANIFEST, "r", encoding="utf-8") as f:
        dcal = json.load(f)

    with open(DSELECT_MANIFEST, "r", encoding="utf-8") as f:
        dselect = json.load(f)

    with open(PARTITION_METADATA, "r", encoding="utf-8") as f:
        partition_metadata = json.load(f)

    def extract_ids(obj):
        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict):
            for key in (
                "record_ids",
                "ids",
                "validation_record_ids",
                "dcal_record_ids",
                "dselect_record_ids",
            ):
                value = obj.get(key)
                if isinstance(value, list):
                    return value

        return None

    dcal_ids = extract_ids(dcal)
    dselect_ids = extract_ids(dselect)

    if dcal_ids is None or dselect_ids is None:
        raise RuntimeError(
            "Could not identify record IDs in D-Cal/D-Select manifests. "
            "No inference was executed."
        )

    if len(dcal_ids) != 34909:
        raise RuntimeError(
            f"Unexpected D-Cal size: {len(dcal_ids)}"
        )

    if len(dselect_ids) != 34958:
        raise RuntimeError(
            f"Unexpected D-Select size: {len(dselect_ids)}"
        )

    if set(dcal_ids) & set(dselect_ids):
        raise RuntimeError("D-Cal and D-Select overlap.")

    if set(dcal_ids) | set(dselect_ids) != set(record_ids):
        raise RuntimeError(
            "D-Cal/D-Select union does not equal the complete validation population."
        )

    print("Frozen D2 partition verification:")
    print("  D-Cal    : 34,909")
    print("  D-Select : 34,958")
    print("  Union    : 69,867")
    print("  Overlap  : 0")
    print("  PARTITION: PASS")
    print()

    # ------------------------------------------------------------
    # 5. Load frozen C-Base
    # ------------------------------------------------------------
    checkpoint = torch.load(
        CKPT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = CBase()
    state_dict = load_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("C-Base model:")
    print("  architecture: frozen Phase-C C-Base")
    print("  classes     : 549")
    print("  mode        : eval")
    print("  MODEL LOAD  : PASS")
    print()

    # ------------------------------------------------------------
    # 6. Fresh inference over ALL 69,867 non-test validation records
    # ------------------------------------------------------------
    text_tensor = torch.from_numpy(
        text.astype(np.float32, copy=False)
    )

    image_tensor = torch.from_numpy(
        image.astype(np.float32, copy=False)
    )

    batch_size = 512
    logits_chunks = []

    print("Running fresh C-Base inference...")
    print(f"  records    : {N_VALIDATION}")
    print(f"  batch size : {batch_size}")
    print("  test set   : NOT ACCESSED")
    print()

    with torch.no_grad():
        for start in range(0, N_VALIDATION, batch_size):
            end = min(start + batch_size, N_VALIDATION)

            logits = model(
                text_tensor[start:end],
                image_tensor[start:end],
            )

            logits_chunks.append(
                logits.cpu().numpy().astype(np.float32)
            )

            if start == 0 or end == N_VALIDATION or start % (batch_size * 20) == 0:
                print(f"  processed {end}/{N_VALIDATION}")

    logits = np.concatenate(logits_chunks, axis=0)

    if logits.shape != (N_VALIDATION, N_CLASSES):
        raise RuntimeError(
            f"Unexpected logits shape: {logits.shape}"
        )

    if not np.isfinite(logits).all():
        raise RuntimeError("Non-finite logits detected.")

    # ------------------------------------------------------------
    # 7. Derive predictions/probabilities.
    #
    #    This is descriptive inference only.
    #    No calibration or threshold selection occurs here.
    # ------------------------------------------------------------
    logits_tensor = torch.from_numpy(logits)

    probabilities = torch.softmax(logits_tensor, dim=1).numpy().astype(
        np.float32
    )

    predictions = np.argmax(logits, axis=1).astype(np.int64)
    top1_probability = probabilities[
        np.arange(N_VALIDATION),
        predictions,
    ].astype(np.float32)

    correct = (
        predictions == labels.astype(np.int64)
    ).astype(np.uint8)

    # ------------------------------------------------------------
    # 8. Final pre-write integrity checks
    # ------------------------------------------------------------
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite probabilities detected.")

    probability_sums = probabilities.sum(axis=1)

    if not np.allclose(
        probability_sums,
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Softmax probability rows do not sum to 1."
        )

    if predictions.shape != (N_VALIDATION,):
        raise RuntimeError("Prediction shape mismatch.")

    if top1_probability.shape != (N_VALIDATION,):
        raise RuntimeError("Top-1 probability shape mismatch.")

    print()
    print("Inference integrity:")
    print(f"  logits shape           : {logits.shape}")
    print(f"  predictions shape      : {predictions.shape}")
    print(f"  top1 probability shape : {top1_probability.shape}")
    print(f"  correct records        : {int(correct.sum())}")
    print(f"  test accessed          : FALSE")
    print("  INFERENCE: PASS")
    print()

    # ------------------------------------------------------------
    # 9. Write ONLY to Phase-D namespace
    # ------------------------------------------------------------
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUT_NPZ,
        record_ids=np.asarray(record_ids),
        labels=labels.astype(np.int64),
        logits=logits,
        predictions=predictions,
        top1_probability=top1_probability,
        correct=correct,
    )

    output_sha = sha256_file(OUT_NPZ)

    manifest = {
        "status": "GENERATED",
        "phase": "D2",
        "stage": "fresh_cbase_inference",
        "seed": SEED,
        "source": {
            "text_embeddings": str(TEXT_PATH),
            "image_embeddings": str(IMAGE_PATH),
            "labels": str(LABEL_PATH),
            "record_ids": str(ID_PATH),
            "checkpoint": str(CKPT_PATH),
        },
        "checkpoint": {
            "sha256": actual_sha,
            "expected_sha256": EXPECTED_CKPT_SHA256,
            "verified": True,
        },
        "population": {
            "total": N_VALIDATION,
            "dcal": 34909,
            "dselect": 34958,
            "dtest_accessed": False,
        },
        "outputs": {
            "file": str(OUT_NPZ),
            "sha256": output_sha,
            "logits_shape": list(logits.shape),
            "predictions_shape": list(predictions.shape),
        },
        "method_boundary": {
            "fresh_inference_only": True,
            "calibration_fitted": False,
            "temperature_scaling_fitted": False,
            "threshold_selected": False,
            "coverage_target_selected": False,
            "review_rule_selected": False,
            "dtest_used": False,
            "phase_c_modified": False,
        },
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Artifacts written:")
    print(f"  {OUT_NPZ}")
    print(f"  {OUT_MANIFEST}")
    print()
    print("=== D2 FRESH INFERENCE COMPLETE ===")
    print("No calibration was fitted.")
    print("No D3 threshold was selected.")
    print("D-Test was not accessed.")
    print("Phase-C artifacts were not modified.")


if __name__ == "__main__":
    main()
