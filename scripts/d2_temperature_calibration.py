from pathlib import Path
import json
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

SEED = 20260827
N_CLASSES = 549
N_VALIDATION = 69867
DCAL_COUNT = 34909
DSELECT_COUNT = 34958

EXPECTED_CKPT_SHA256 = (
    "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"
)

D2_ROOT = Path("data/models/abo/phase_d/d2")
PARTITION_ROOT = D2_ROOT / "partition"
INFERENCE_ROOT = D2_ROOT / "inference"
CAL_ROOT = D2_ROOT / "calibration"

INPUT_NPZ = INFERENCE_ROOT / "d2_fresh_cbase_validation_outputs.npz"
INPUT_MANIFEST = INFERENCE_ROOT / "d2_fresh_inference_manifest.json"

DCAL_MANIFEST = PARTITION_ROOT / "dcal_manifest.json"
DSELECT_MANIFEST = PARTITION_ROOT / "dselect_manifest.json"

CHECKPOINT = Path(
    "data/models/abo/phase_c/c3/training/cbase_best.pt"
)

OUTPUT_NPZ = CAL_ROOT / "d2_calibrated_validation_outputs.npz"
OUTPUT_METRICS = CAL_ROOT / "d2_calibration_metrics.json"
OUTPUT_RELIABILITY = CAL_ROOT / "d2_reliability_data.json"
OUTPUT_MANIFEST = CAL_ROOT / "d2_calibration_manifest.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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

    raise RuntimeError(
        "Could not identify record IDs in partition manifest."
    )


def nll_from_logits(logits, labels):
    return float(
        F.cross_entropy(
            torch.from_numpy(logits.astype(np.float32)),
            torch.from_numpy(labels.astype(np.int64)),
        ).item()
    )


def probabilities_from_logits(logits, temperature):
    scaled = torch.from_numpy(
        logits.astype(np.float32)
    ) / float(temperature)

    return torch.softmax(scaled, dim=1).numpy()


def accuracy_and_predictions(logits, labels, temperature):
    probs = probabilities_from_logits(logits, temperature)
    predictions = np.argmax(probs, axis=1)

    return (
        float(np.mean(predictions == labels)),
        predictions,
        probs,
    )


def brier_score_multiclass(probs, labels):
    n = len(labels)

    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(n), labels] = 1.0

    return float(
        np.mean(np.sum((probs.astype(np.float64) - one_hot) ** 2, axis=1))
    )


def calibration_statistics(probs, labels, n_bins=15):
    confidence = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    ece = 0.0
    total = len(labels)

    for b in range(n_bins):
        left = bin_edges[b]
        right = bin_edges[b + 1]

        if b == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        count = int(mask.sum())

        if count == 0:
            rows.append({
                "bin": b + 1,
                "lower": float(left),
                "upper": float(right),
                "count": 0,
                "mean_confidence": None,
                "accuracy": None,
                "gap": None,
            })
            continue

        mean_conf = float(confidence[mask].mean())
        accuracy = float(correctness[mask].mean())
        gap = abs(mean_conf - accuracy)

        ece += (count / total) * gap

        rows.append({
            "bin": b + 1,
            "lower": float(left),
            "upper": float(right),
            "count": count,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
            "gap": gap,
        })

    return float(ece), rows


def confidence_summary(probs, labels):
    confidence = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = predictions == labels

    high_confidence = confidence >= 0.90

    if high_confidence.any():
        high_conf_accuracy = float(correct[high_confidence].mean())
        high_conf_error = float(1.0 - high_conf_accuracy)
        high_conf_count = int(high_confidence.sum())
    else:
        high_conf_accuracy = None
        high_conf_error = None
        high_conf_count = 0

    return {
        "mean": float(confidence.mean()),
        "median": float(np.median(confidence)),
        "min": float(confidence.min()),
        "max": float(confidence.max()),
        "p10": float(np.percentile(confidence, 10)),
        "p25": float(np.percentile(confidence, 25)),
        "p75": float(np.percentile(confidence, 75)),
        "p90": float(np.percentile(confidence, 90)),
        "high_confidence_definition": "top1_probability >= 0.90",
        "high_confidence_count": high_conf_count,
        "high_confidence_accuracy": high_conf_accuracy,
        "high_confidence_error_rate": high_conf_error,
    }


def evaluate(logits, labels, temperature):
    probs = probabilities_from_logits(logits, temperature)

    accuracy, predictions, _ = accuracy_and_predictions(
        logits,
        labels,
        temperature,
    )

    ece, reliability = calibration_statistics(
        probs,
        labels,
        n_bins=15,
    )

    nll = nll_from_logits(
        logits / float(temperature),
        labels,
    )

    brier = brier_score_multiclass(
        probs,
        labels,
    )

    confidence = np.max(probs, axis=1)

    return {
        "temperature": float(temperature),
        "accuracy": accuracy,
        "nll": nll,
        "ece_15_equal_width": ece,
        "brier_score": brier,
        "confidence": confidence_summary(probs, labels),
        "reliability_bins": reliability,
        "predictions": predictions,
        "probabilities": probs,
    }


def fit_temperature(logits, labels):
    """
    Fit one scalar positive temperature using D-Cal only.

    Optimization objective:
        multiclass cross-entropy / NLL

    The temperature is parameterized as exp(log_temperature),
    guaranteeing T > 0.
    """

    logits_t = torch.from_numpy(
        logits.astype(np.float32)
    )

    labels_t = torch.from_numpy(
        labels.astype(np.int64)
    )

    log_temperature = torch.nn.Parameter(
        torch.tensor(0.0, dtype=torch.float32)
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.01,
        max_iter=100,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()

        temperature = torch.exp(log_temperature)

        loss = F.cross_entropy(
            logits_t / temperature,
            labels_t,
        )

        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(
        torch.exp(log_temperature).detach().cpu().item()
    )

    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError(
            f"Invalid fitted temperature: {temperature}"
        )

    return temperature


def main():
    print("=== D2 TEMPERATURE CALIBRATION ===")
    print("D2 only: calibration fitting and diagnostic evaluation.")
    print("D3 threshold/decision-rule selection is NOT performed.")
    print()

    # ------------------------------------------------------------
    # 1. Verify frozen checkpoint
    # ------------------------------------------------------------
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    checkpoint_sha = sha256_file(CHECKPOINT)

    print("Frozen C-Base checkpoint:")
    print(f"  SHA-256: {checkpoint_sha}")

    if checkpoint_sha.lower() != EXPECTED_CKPT_SHA256.lower():
        raise RuntimeError(
            "Frozen C-Base checkpoint hash mismatch."
        )

    print("  CHECKPOINT HASH: PASS")
    print()

    # ------------------------------------------------------------
    # 2. Load fresh inference artifact
    # ------------------------------------------------------------
    if not INPUT_NPZ.exists():
        raise FileNotFoundError(INPUT_NPZ)

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(INPUT_MANIFEST)

    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        inference_manifest = json.load(f)

    if inference_manifest.get("method_boundary", {}).get(
        "dtest_used", True
    ):
        raise RuntimeError(
            "Fresh inference manifest indicates D-Test was used."
        )

    data = np.load(INPUT_NPZ, allow_pickle=False)

    record_ids = data["record_ids"]
    labels = data["labels"].astype(np.int64)
    logits = data["logits"].astype(np.float32)

    if len(record_ids) != N_VALIDATION:
        raise RuntimeError("Unexpected record count.")

    if labels.shape != (N_VALIDATION,):
        raise RuntimeError("Unexpected label shape.")

    if logits.shape != (N_VALIDATION, N_CLASSES):
        raise RuntimeError(
            f"Unexpected logits shape: {logits.shape}"
        )

    if not np.isfinite(logits).all():
        raise RuntimeError("Non-finite logits detected.")

    print("Fresh inference artifact:")
    print(f"  records : {len(record_ids)}")
    print(f"  logits  : {logits.shape}")
    print("  LOAD: PASS")
    print()

    # ------------------------------------------------------------
    # 3. Load frozen D-Cal / D-Select partitions
    # ------------------------------------------------------------
    with open(DCAL_MANIFEST, "r", encoding="utf-8") as f:
        dcal_manifest = json.load(f)

    with open(DSELECT_MANIFEST, "r", encoding="utf-8") as f:
        dselect_manifest = json.load(f)

    dcal_ids = extract_ids(dcal_manifest)
    dselect_ids = extract_ids(dselect_manifest)

    if len(dcal_ids) != DCAL_COUNT:
        raise RuntimeError(
            f"D-Cal count mismatch: {len(dcal_ids)}"
        )

    if len(dselect_ids) != DSELECT_COUNT:
        raise RuntimeError(
            f"D-Select count mismatch: {len(dselect_ids)}"
        )

    if set(dcal_ids) & set(dselect_ids):
        raise RuntimeError("D-Cal/D-Select overlap detected.")

    validation_id_to_index = {
        str(record_id): i
        for i, record_id in enumerate(record_ids)
    }

    try:
        dcal_indices = np.asarray(
            [validation_id_to_index[str(x)] for x in dcal_ids],
            dtype=np.int64,
        )

        dselect_indices = np.asarray(
            [validation_id_to_index[str(x)] for x in dselect_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Partition contains record ID absent from validation population: {exc}"
        )

    if len(np.unique(dcal_indices)) != DCAL_COUNT:
        raise RuntimeError("Duplicate D-Cal indices.")

    if len(np.unique(dselect_indices)) != DSELECT_COUNT:
        raise RuntimeError("Duplicate D-Select indices.")

    if set(dcal_indices) & set(dselect_indices):
        raise RuntimeError("D-Cal/D-Select index overlap.")

    if set(dcal_indices) | set(dselect_indices) != set(
        range(N_VALIDATION)
    ):
        raise RuntimeError(
            "D-Cal/D-Select does not cover exactly the full validation population."
        )

    print("Partition mapping:")
    print(f"  D-Cal    : {len(dcal_indices)}")
    print(f"  D-Select : {len(dselect_indices)}")
    print("  overlap  : 0")
    print("  coverage : 69,867 / 69,867")
    print("  PARTITION MAPPING: PASS")
    print()

    # ------------------------------------------------------------
    # 4. Construct D-Cal and D-Select data
    # ------------------------------------------------------------
    dcal_logits = logits[dcal_indices]
    dcal_labels = labels[dcal_indices]

    dselect_logits = logits[dselect_indices]
    dselect_labels = labels[dselect_indices]

    # ------------------------------------------------------------
    # 5. Baseline diagnostics BEFORE calibration
    # ------------------------------------------------------------
    print("Computing raw D-Cal diagnostics...")
    raw_dcal = evaluate(
        dcal_logits,
        dcal_labels,
        temperature=1.0,
    )

    print("Computing raw D-Select diagnostics...")
    raw_dselect = evaluate(
        dselect_logits,
        dselect_labels,
        temperature=1.0,
    )

    # ------------------------------------------------------------
    # 6. Fit temperature ONLY on D-Cal
    # ------------------------------------------------------------
    print()
    print("Fitting temperature on D-Cal ONLY...")
    temperature = fit_temperature(
        dcal_logits,
        dcal_labels,
    )

    print(f"  Fitted temperature: {temperature:.10f}")

    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Invalid temperature.")

    print("  TEMPERATURE FIT: PASS")
    print()

    # ------------------------------------------------------------
    # 7. Evaluate frozen temperature on D-Cal
    # ------------------------------------------------------------
    print("Evaluating frozen temperature on D-Cal...")
    calibrated_dcal = evaluate(
        dcal_logits,
        dcal_labels,
        temperature,
    )

    # ------------------------------------------------------------
    # 8. Evaluate frozen temperature on D-Select
    # ------------------------------------------------------------
    print("Applying frozen temperature to D-Select...")
    calibrated_dselect = evaluate(
        dselect_logits,
        dselect_labels,
        temperature,
    )

    print("  D-SELECT CALIBRATION: PASS")
    print()

    # ------------------------------------------------------------
    # 9. Explicit D2 boundary checks
    # ------------------------------------------------------------
    # No threshold/coverage/routing rule is derived here.
    #
    # This script only computes diagnostics.
    # ------------------------------------------------------------
    CAL_ROOT.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_NPZ,
        dcal_record_ids=np.asarray(dcal_ids),
        dcal_labels=dcal_labels,
        dcal_raw_logits=dcal_logits,
        dcal_calibrated_probabilities=calibrated_dcal["probabilities"],
        dselect_record_ids=np.asarray(dselect_ids),
        dselect_labels=dselect_labels,
        dselect_raw_logits=dselect_logits,
        dselect_calibrated_probabilities=calibrated_dselect[
            "probabilities"
        ],
    )

    metrics = {
        "phase": "D2",
        "method": "temperature_scaling",
        "temperature": temperature,
        "calibration_fit_population": "D-Cal",
        "calibration_evaluation_population": "D-Select",
        "dcal_count": DCAL_COUNT,
        "dselect_count": DSELECT_COUNT,
        "diagnostic_bins": {
            "count": 15,
            "type": "equal_width",
            "edges": list(np.linspace(0.0, 1.0, 16)),
        },
        "raw": {
            "dcal": {
                "accuracy": raw_dcal["accuracy"],
                "nll": raw_dcal["nll"],
                "ece": raw_dcal["ece_15_equal_width"],
                "brier": raw_dcal["brier_score"],
                "confidence": raw_dcal["confidence"],
            },
            "dselect": {
                "accuracy": raw_dselect["accuracy"],
                "nll": raw_dselect["nll"],
                "ece": raw_dselect["ece_15_equal_width"],
                "brier": raw_dselect["brier_score"],
                "confidence": raw_dselect["confidence"],
            },
        },
        "calibrated": {
            "dcal": {
                "accuracy": calibrated_dcal["accuracy"],
                "nll": calibrated_dcal["nll"],
                "ece": calibrated_dcal["ece_15_equal_width"],
                "brier": calibrated_dcal["brier_score"],
                "confidence": calibrated_dcal["confidence"],
            },
            "dselect": {
                "accuracy": calibrated_dselect["accuracy"],
                "nll": calibrated_dselect["nll"],
                "ece": calibrated_dselect["ece_15_equal_width"],
                "brier": calibrated_dselect["brier_score"],
                "confidence": calibrated_dselect["confidence"],
            },
        },
        "boundary": {
            "dtest_accessed": False,
            "threshold_selected": False,
            "coverage_target_selected": False,
            "abstention_rule_selected": False,
            "human_review_rule_selected": False,
            "open_set_rule_selected": False,
            "phase_c_modified": False,
            "cbase_retrained": False,
        },
    }

    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    reliability = {
        "temperature": temperature,
        "binning": {
            "count": 15,
            "type": "equal_width",
        },
        "dcal": {
            "raw": raw_dcal["reliability_bins"],
            "calibrated": calibrated_dcal["reliability_bins"],
        },
        "dselect": {
            "raw": raw_dselect["reliability_bins"],
            "calibrated": calibrated_dselect["reliability_bins"],
        },
    }

    with open(OUTPUT_RELIABILITY, "w", encoding="utf-8") as f:
        json.dump(reliability, f, indent=2)

    output_sha = sha256_file(OUTPUT_NPZ)

    manifest = {
        "status": "EXECUTED",
        "phase": "D2",
        "stage": "temperature_calibration",
        "seed": SEED,
        "method": "temperature_scaling",
        "temperature": temperature,
        "checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": checkpoint_sha,
            "expected_sha256": EXPECTED_CKPT_SHA256,
            "verified": True,
        },
        "populations": {
            "source_validation": N_VALIDATION,
            "dcal": DCAL_COUNT,
            "dselect": DSELECT_COUNT,
            "dtest_accessed": False,
        },
        "information_flow": [
            "fresh_CBase_inference",
            "DCal_temperature_fit",
            "freeze_temperature",
            "DSelect_evaluation",
        ],
        "outputs": {
            "npz": str(OUTPUT_NPZ),
            "npz_sha256": output_sha,
            "metrics": str(OUTPUT_METRICS),
            "reliability": str(OUTPUT_RELIABILITY),
        },
        "boundary": {
            "d3_threshold_selection": "NOT_PERFORMED",
            "abstention_threshold": "NOT_SELECTED",
            "coverage_target": "NOT_SELECTED",
            "human_review_rule": "NOT_SELECTED",
            "open_set": "NOT_PERFORMED",
            "dtest": "NOT_ACCESSED",
            "phase_c_modified": False,
        },
    }

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------
    # 10. Print compact results
    # ------------------------------------------------------------
    def print_metrics(name, result):
        print(f"{name}:")
        print(f"  Accuracy : {result['accuracy']:.6f}")
        print(f"  NLL      : {result['nll']:.6f}")
        print(f"  ECE      : {result['ece_15_equal_width']:.6f}")
        print(f"  Brier    : {result['brier_score']:.6f}")
        print(
            f"  Mean conf: {result['confidence']['mean']:.6f}"
        )

    print()
    print("=== D2 RESULTS ===")
    print(f"Fitted temperature: {temperature:.10f}")
    print()

    print_metrics("D-Cal RAW", raw_dcal)
    print_metrics("D-Cal CALIBRATED", calibrated_dcal)
    print()
    print_metrics("D-Select RAW", raw_dselect)
    print_metrics("D-Select CALIBRATED", calibrated_dselect)

    print()
    print("Artifacts:")
    print(f"  {OUTPUT_NPZ}")
    print(f"  {OUTPUT_METRICS}")
    print(f"  {OUTPUT_RELIABILITY}")
    print(f"  {OUTPUT_MANIFEST}")

    print()
    print("=== D2 TEMPERATURE CALIBRATION COMPLETE ===")
    print("Temperature was fitted on D-Cal only.")
    print("The fitted temperature was frozen before D-Select evaluation.")
    print("No D3 threshold was selected.")
    print("No coverage target was selected.")
    print("No human-review rule was selected.")
    print("D-Test was not accessed.")
    print("Phase-C artifacts were not modified.")


if __name__ == "__main__":
    main()
