from pathlib import Path
import json, hashlib, random, os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

SEED = 20260827
N_CLASSES = 549
N_TEST = 7346
TEXT_DIM, IMAGE_DIM, FUSED_DIM = 384, 512, 512

ROOT = Path(r"data\models\abo\phase_c")
TEST_ROOT = ROOT / "c4" / "test_embeddings"
TRAIN_ROOT = ROOT / "c3" / "training"
OUT = ROOT / "c4" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

TEXT = TEST_ROOT / "test_text.npy"
IMAGE = TEST_ROOT / "test_image.npy"
LABELS = TEST_ROOT / "test_labels.npy"
IDS = TEST_ROOT / "test_record_ids.json"
TEST_MANIFEST = TEST_ROOT / "c4_test_embedding_manifest.json"
BEST = TRAIN_ROOT / "cbase_best.pt"
TRAIN_MANIFEST = TRAIN_ROOT / "cbase_training_manifest.json"

RESULTS = OUT / "cbase_test_results.json"
PREDICTIONS = OUT / "cbase_test_predictions.npz"
MANIFEST = OUT / "c4_2_evaluation_manifest.json"

B5 = {"accuracy": 0.538660, "macro_f1": 0.122360, "weighted_f1": 0.507353}

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def write_json(p, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, p)

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

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
        text_features = self.text_projection(text)
        image_features = self.image_projection(image)
        fused = torch.cat([text_features, image_features], dim=1)
        return self.fusion(fused)

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 72)
    print("C4.2 — FROZEN C-BASE TEST EVALUATION")
    print("=" * 72)
    print("Test access          : AUTHORIZED")
    print("Test usage           : FINAL INFERENCE ONLY")
    print("Model selection      : NONE")
    print("Hyperparameter tuning: NONE")
    print("Training             : NONE")
    print("Checkpoint           : cbase_best.pt")
    print("Device               : cpu")
    print(f"Seed                 : {SEED}")

    required = [TEXT, IMAGE, LABELS, IDS, TEST_MANIFEST, BEST, TRAIN_MANIFEST]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    tm = load_json(TEST_MANIFEST)
    if tm.get("test_accessed") is not True:
        raise RuntimeError("C4.1 manifest does not declare test_accessed=true.")
    if tm.get("test_usage") != "inference_only":
        raise RuntimeError("C4.1 test usage is not inference_only.")
    if tm.get("model_selection") not in (False, None):
        raise RuntimeError("Test was declared for model selection.")
    if tm.get("hyperparameter_selection") not in (False, None):
        raise RuntimeError("Test was declared for hyperparameter selection.")

    text = np.load(TEXT, mmap_mode="r")
    image = np.load(IMAGE, mmap_mode="r")
    labels = np.load(LABELS, mmap_mode="r")
    record_ids = load_json(IDS)

    print("\nC4.1 TEST ARTIFACTS")
    print(f"Records     : {len(labels):,}")
    print(f"Text shape  : {text.shape}")
    print(f"Image shape: {image.shape}")

    if text.shape != (N_TEST, TEXT_DIM): raise RuntimeError(f"Bad text shape: {text.shape}")
    if image.shape != (N_TEST, IMAGE_DIM): raise RuntimeError(f"Bad image shape: {image.shape}")
    if labels.shape != (N_TEST,): raise RuntimeError(f"Bad label shape: {labels.shape}")
    if len(record_ids) != N_TEST or len(set(record_ids)) != N_TEST:
        raise RuntimeError("Test record IDs are invalid.")
    if text.dtype != np.float32 or image.dtype != np.float32:
        raise RuntimeError("Embeddings are not float32.")
    if not np.isfinite(text).all() or not np.isfinite(image).all():
        raise RuntimeError("Non-finite test embeddings.")
    if int(labels.min()) < 0 or int(labels.max()) >= N_CLASSES:
        raise RuntimeError("Invalid test label range.")
    print("PASS: isolated test artifacts valid.")

    train_manifest = load_json(TRAIN_MANIFEST)
    if train_manifest.get("test_accessed") is True:
        raise RuntimeError("C3 manifest reports test_accessed=true.")

    ckpt = torch.load(BEST, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt.get("state_dict")))
    if state is None:
        raise RuntimeError("No model state dict found in cbase_best.pt.")

    model = CBase()
    model.load_state_dict(state, strict=True)
    model.eval()

    epoch = ckpt.get("epoch")
    best_epoch = ckpt.get("best_epoch")
    best_metric = ckpt.get("best_metric")
    print("\nFROZEN CHECKPOINT")
    print(f"Best epoch             : {best_epoch}")
    print(f"Best validation Macro-F1: {best_metric}")
    print(f"Checkpoint epoch       : {epoch}")
    print(f"Parameters             : {sum(p.numel() for p in model.parameters()):,}")

    if best_epoch is not None and int(best_epoch) != 17:
        raise RuntimeError(f"Expected frozen best epoch 17, found {best_epoch}.")
    if best_metric is not None and not np.isclose(float(best_metric), 0.4626669030182739, atol=1e-12, rtol=0):
        raise RuntimeError(f"Unexpected frozen validation Macro-F1: {best_metric}")

    predictions = np.empty(N_TEST, dtype=np.int64)
    batch_size = 512

    print("\nFINAL TEST INFERENCE")
    with torch.inference_mode():
        for start in range(0, N_TEST, batch_size):
            end = min(start + batch_size, N_TEST)
            t = torch.from_numpy(np.asarray(text[start:end], dtype=np.float32).copy())
            i = torch.from_numpy(np.asarray(image[start:end], dtype=np.float32).copy())
            predictions[start:end] = torch.argmax(model(t, i), dim=1).numpy()
            print(f"INFERENCE | test | {end:,}/{N_TEST:,}")

    y = np.asarray(labels, dtype=np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, labels=np.arange(N_CLASSES), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predictions, labels=np.arange(N_CLASSES), average="weighted", zero_division=0)),
    }
    delta = {k: metrics[k] - B5[k] for k in metrics}

    print("\n" + "=" * 72)
    print("C4.2 TEST RESULTS")
    print("=" * 72)
    for k in metrics:
        print(f"{k:12s}: {metrics[k]:.6f}")
    print("\nFROZEN B5 COMPARISON")
    for k in B5:
        print(f"B5 {k:10s}: {B5[k]:.6f}")
    print("\nABSOLUTE DELTAS (C4.2 - B5)")
    for k in delta:
        print(f"Delta {k:6s}: {delta[k]:+.6f}")

    np.savez_compressed(PREDICTIONS, predictions=predictions)

    result = {
        "phase": "C4.2",
        "task": "frozen C-Base test evaluation",
        "dataset": "ABO",
        "target_field": "product_type",
        "num_classes": N_CLASSES,
        "test_population": N_TEST,
        "checkpoint": str(BEST),
        "checkpoint_epoch": epoch,
        "best_epoch": best_epoch,
        "selection_metric": "macro_f1",
        "best_validation_macro_f1": best_metric,
        "metrics": metrics,
        "frozen_b5_baseline": B5,
        "absolute_deltas_c4_minus_b5": delta,
        "test_usage": "inference_only",
        "model_selection_from_test": False,
        "hyperparameter_selection_from_test": False,
        "training_on_test": False,
        "b3_modified": False,
        "c3_modified": False,
        "seed": SEED,
        "device": "cpu",
        "test_embedding_hashes": {
            "text": sha256_file(TEXT), "image": sha256_file(IMAGE),
            "labels": sha256_file(LABELS), "record_ids": sha256_file(IDS),
            "manifest": sha256_file(TEST_MANIFEST),
        },
        "checkpoint_sha256": sha256_file(BEST),
        "outputs": {"predictions": str(PREDICTIONS), "results": str(RESULTS), "manifest": str(MANIFEST)},
    }
    write_json(RESULTS, result)

    em = {
        "phase": "C4.2",
        "task": "final frozen C-Base test evaluation",
        "test_accessed": True,
        "test_usage": "inference_only",
        "model_selection_from_test": False,
        "hyperparameter_selection_from_test": False,
        "training_on_test": False,
        "b3_modified": False,
        "c3_modified": False,
        "test_population": N_TEST,
        "target_classes": N_CLASSES,
        "best_epoch": best_epoch,
        "selection_metric": "macro_f1",
        "best_validation_macro_f1": best_metric,
        "metrics": metrics,
        "frozen_b5_baseline": B5,
        "absolute_deltas_c4_minus_b5": delta,
        "checkpoint_sha256": sha256_file(BEST),
        "test_embedding_hashes": result["test_embedding_hashes"],
        "seed": SEED,
        "device": "cpu",
    }
    write_json(MANIFEST, em)

    print("\n" + "=" * 72)
    print("C4.2 FINAL INTEGRITY")
    print("=" * 72)
    print("PASS: exact 7,346-record isolated test population.")
    print("PASS: validation-selected epoch 17 checkpoint.")
    print("PASS: no training/model selection/hyperparameter tuning on test.")
    print("PASS: B3/C3 source artifacts read-only.")
    print(f"Results    : {RESULTS}")
    print(f"Predictions: {PREDICTIONS}")
    print(f"Manifest   : {MANIFEST}")
    print("C4.2 COMPLETE.")

if __name__ == "__main__":
    main()
