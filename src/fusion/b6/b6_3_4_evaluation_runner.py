from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]

B4_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "abo" / "b4.2" / "text_model.joblib"
TRANSFER_DIR = PROJECT_ROOT / "data" / "representations" / "abo" / "b6_3_3_transfer"
B6_DIR = PROJECT_ROOT / "reports" / "fusion" / "b6"
REPORT_PATH = B6_DIR / "b6_3_4_evaluation_results_v001.json"

SEED = 20260827
TRAIN_COUNT = 70284
TEST_COUNT = 7422
TARGET_CLASS_COUNT = 549
TARGET_FIELD = "product_type"

EXCLUDED_TARGET_CLASSES = frozenset({
    "HAIRBAND",
    "PUNCHING_BAG",
    "SALWAR_SUIT_SET",
    "TREADMILL",
})


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_scalar(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            result = first_scalar(item)
            if result:
                return result
        return ""
    if isinstance(value, dict):
        if "value" in value:
            return first_scalar(value["value"])
        return ""
    return ""


def target_from_record(record: dict) -> str:
    return first_scalar(record.get(TARGET_FIELD))


def text_from_record(record: dict) -> str:
    representation = record.get("b3_representation")
    if not isinstance(representation, dict):
        return ""
    text = representation.get("text")
    if not isinstance(text, dict):
        return ""
    combined_tokens = text.get("combined_tokens")
    if not isinstance(combined_tokens, list):
        return ""
    return " ".join(str(token) for token in combined_tokens)


def load_b3_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
    return records


def locate_b3_json(split: str) -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "abo" / f"{split}.jsonl",
        PROJECT_ROOT / "data" / "processed" / "abo" / f"{split}.json",
        PROJECT_ROOT / "data" / "representations" / "abo" / "b3" / f"{split}.jsonl",
        PROJECT_ROOT / "data" / "representations" / "abo" / "b3" / f"{split}.json",
    ]
    existing = [path for path in candidates if path.exists()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Could not uniquely locate ABO B3 {split} file. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )
    return existing[0]


def load_transfer(split: str, expected_count: int) -> dict:
    npz_path = TRANSFER_DIR / f"{split}.npz"
    metadata_path = TRANSFER_DIR / f"{split}_metadata.json"

    with np.load(npz_path, allow_pickle=False) as archive:
        features = np.asarray(archive["transferred_signal"])
        record_ids = [str(v) for v in np.asarray(archive["record_ids"]).tolist()]

    if features.shape != (expected_count, 3008):
        raise ValueError(
            f"{split} transfer shape mismatch: {features.shape}"
        )
    if len(record_ids) != expected_count or len(set(record_ids)) != expected_count:
        raise ValueError(f"{split} transfer record-ID integrity failure.")
    if not np.isfinite(features).all() or np.any(features < 0):
        raise ValueError(f"{split} transfer numerical integrity failure.")

    return {
        "features": features.astype(np.float32, copy=False),
        "record_ids": record_ids,
        "metadata": load_json(metadata_path),
    }


def filter_common(records: list[dict]) -> tuple[list[dict], np.ndarray]:
    targets = [target_from_record(r) for r in records]
    mask = np.asarray(
        [target not in EXCLUDED_TARGET_CLASSES for target in targets],
        dtype=bool,
    )
    return [r for r, keep in zip(records, mask) if keep], mask


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = np.arange(TARGET_CLASS_COUNT, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def main() -> None:
    print("=" * 72)
    print("B6.3.4 — EVALUATION & ABLATION")
    print("=" * 72)

    exp1_path = B6_DIR / "b6_3_3_exp1_gate2_test_v001.json"
    exp2_path = B6_DIR / "b6_3_3_exp2_gate2_test_v001.json"
    exp1 = load_json(exp1_path)
    exp2 = load_json(exp2_path)

    if exp1.get("status") != "GATE_2_PASS":
        raise ValueError("Frozen Exp1 Gate 2 report is not GATE_2_PASS.")
    if exp2.get("status") != "GATE_2_PASS":
        raise ValueError("Frozen Exp2 Gate 2 report is not GATE_2_PASS.")

    print("LOADING_FROZEN_B4_2_MODEL")
    b4 = joblib.load(B4_MODEL_PATH)
    vectorizer = b4["vectorizer"]
    frozen_classifier = b4["classifier"]
    frozen_label_encoder = b4["label_encoder"]

    if len(frozen_label_encoder.classes_) != 553:
        raise ValueError("Frozen B4.2 artifact does not contain 553 classes.")

    print("LOADING_ABO_B3")
    train = load_b3_records(locate_b3_json("train"))
    test = load_b3_records(locate_b3_json("test"))

    if len(train) != TRAIN_COUNT or len(test) != TEST_COUNT:
        raise ValueError("Unexpected ABO B3 population counts.")

    filtered_train, train_mask = filter_common(train)
    filtered_test, test_mask = filter_common(test)

    print(f"TRAIN_ORIGINAL={len(train)}")
    print(f"TEST_ORIGINAL={len(test)}")
    print(f"TRAIN_FILTERED={len(filtered_train)}")
    print(f"TEST_FILTERED={len(filtered_test)}")

    if len(filtered_train) != 70258 or len(filtered_test) != 7416:
        raise ValueError("Frozen 549-class population boundary failed.")

    train_targets = [target_from_record(r) for r in filtered_train]
    test_targets = [target_from_record(r) for r in filtered_test]
    classes = sorted(set(train_targets))

    if len(classes) != TARGET_CLASS_COUNT:
        raise ValueError(f"Expected 549 train classes, found {len(classes)}.")

    class_to_index = {label: i for i, label in enumerate(classes)}
    test_known_mask = np.asarray(
        [label in class_to_index for label in test_targets],
        dtype=bool,
    )
    y_train = np.asarray(
        [class_to_index[label] for label in train_targets],
        dtype=np.int64,
    )
    y_test = np.asarray(
        [class_to_index[label] for label in np.asarray(test_targets)[test_known_mask]],
        dtype=np.int64,
    )

    print(f"TARGET_CLASSES={len(classes)}")
    print(f"TEST_KNOWN_CLASS_RECORDS={int(test_known_mask.sum())}")
    print(f"TEST_UNSEEN_TARGET_RECORDS={int((~test_known_mask).sum())}")

    if int(test_known_mask.sum()) != 7386:
        raise ValueError("Expected 7386 known-class test records.")

    print("BUILDING_FROZEN_VECTORIZER_FEATURES")
    train_texts = [text_from_record(r) for r in filtered_train]
    test_texts = [text_from_record(r) for r in filtered_test]

    if any(not t for t in train_texts) or any(not t for t in test_texts):
        raise ValueError("Empty B3 text representation detected.")

    X_train = vectorizer.transform(train_texts).tocsr()
    X_test_all = vectorizer.transform(test_texts).tocsr()
    X_test = X_test_all[test_known_mask]

    print(f"TEXT_TRAIN_SHAPE={X_train.shape}")
    print(f"TEXT_TEST_KNOWN_SHAPE={X_test.shape}")
    print(f"TEXT_VOCABULARY_SIZE={len(vectorizer.vocabulary_)}")
    print("VECTORIZER_FIT=False")

    print("TRAINING_B6_3_4_MATCHED_TEXT_ONLY_ABLATION")
    ablation_classifier = SGDClassifier(
        loss="log_loss",
        max_iter=30,
        random_state=SEED,
        n_jobs=1,
        tol=1e-3,
    )
    ablation_classifier.fit(X_train, y_train)
    ablation_predictions = ablation_classifier.predict(X_test)
    text_only_metrics = metrics(y_test, ablation_predictions)

    exp1_metrics = exp1["metrics"]
    exp2_metrics = exp2["metrics"]

    def delta(a, b):
        return {
            "accuracy": float(a["accuracy"] - b["accuracy"]),
            "macro_f1": float(a["macro_f1"] - b["macro_f1"]),
            "weighted_f1": float(a["weighted_f1"] - b["weighted_f1"]),
        }

    exp2_minus_exp1 = delta(exp2_metrics, exp1_metrics)
    exp2_minus_text_only = delta(exp2_metrics, text_only_metrics)
    text_only_minus_exp1 = delta(text_only_metrics, exp1_metrics)

    report = {
        "stage": "B6.3.4",
        "version": "v001",
        "status": "PASS",
        "purpose": "Evaluation and ablation of frozen B6.3.3 cross-dataset transfer result.",
        "frozen_inputs": {
            "exp1_report": str(exp1_path.relative_to(PROJECT_ROOT)),
            "exp2_report": str(exp2_path.relative_to(PROJECT_ROOT)),
            "b4_model": str(B4_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "abo_representation": "B3_v002",
            "target_classes": TARGET_CLASS_COUNT,
            "seed": SEED,
            "gamma_exp1": exp1["gamma"]["value"],
            "gamma_exp2": exp2["gamma"]["value"],
        },
        "population": {
            "train_original": TRAIN_COUNT,
            "train_filtered": len(filtered_train),
            "test_original": TEST_COUNT,
            "test_filtered": len(filtered_test),
            "test_known_class_metric_count": int(test_known_mask.sum()),
            "test_unseen_target_records": int((~test_known_mask).sum()),
            "excluded_target_classes": sorted(EXCLUDED_TARGET_CLASSES),
        },
        "matched_text_only_ablation": {
            "description": (
                "New B6.3.4 ablation trained on the frozen B4.2 TF-IDF "
                "representation with the frozen classifier hyperparameters, "
                "but using the frozen 549-class B6.3.3 target boundary. "
                "This is not the original B4.2 553-class model."
            ),
            "vectorizer_fit": False,
            "classifier": {
                "type": "SGDClassifier",
                "loss": "log_loss",
                "max_iter": 30,
                "random_state": SEED,
                "n_jobs": 1,
                "tol": 1e-3,
            },
            "train_records": len(filtered_train),
            "test_metric_records": int(test_known_mask.sum()),
            "metrics": text_only_metrics,
        },
        "frozen_b6_3_3": {
            "exp1_negative_control": exp1_metrics,
            "exp2_genuine_aligned_transfer": exp2_metrics,
        },
        "metric_deltas": {
            "exp2_minus_exp1": exp2_minus_exp1,
            "exp2_minus_text_only_ablation": exp2_minus_text_only,
            "text_only_ablation_minus_exp1": text_only_minus_exp1,
        },
        "error_analysis": {
            "text_only_classification_report": classification_report(
                y_test,
                ablation_predictions,
                labels=np.arange(TARGET_CLASS_COUNT, dtype=np.int64),
                output_dict=True,
                zero_division=0,
            ),
            "prediction_overlap_with_exp1_exp2": "unavailable because frozen B6.3.3 reports do not contain prediction arrays.",
        },
        "scientific_conclusion": {
            "exp2_vs_exp1": (
                "Genuine aligned transfer is compared with the row-permuted "
                "negative control using the frozen B6.3.3 final test results."
            ),
            "text_only_ablation": (
                "The matched 549-class text-only ablation provides the transfer-"
                "removal reference on the same B6.3.4 evaluation population."
            ),
            "interpretation_rule": (
                "Metric deltas are descriptive evaluation results. They do not "
                "establish causality, broad transferability, or practical superiority."
            ),
        },
        "integrity": {
            "b6_3_3_rerun": False,
            "b6_3_3_modified": False,
            "b3_modified": False,
            "b4_modified": False,
            "b5_modified": False,
            "gamma_search": False,
            "test_used_for_selection": False,
            "split_regeneration": False,
            "record_joins": False,
            "asin_matching": False,
            "semantic_label_crosswalk": False,
            "external_enrichment": False,
            "synthetic_pairing": False,
            "image_download": False,
            "path_guessing": False,
        },
        "known_source_metadata_issue": {
            "exp2_report_objective_string": (
                "The frozen Exp2 Gate 2 JSON objective text says "
                "'row-permuted Rakuten transfer signal', while its execution "
                "metadata correctly records experiment_type='genuine_aligned_transfer', "
                "row_permutation=false, scope='none', and the terminal execution "
                "used genuine aligned transfer."
            ),
            "action": "Do not modify the frozen B6.3.3 report; document the metadata typo only.",
        },
    }

    B6_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("-" * 72)
    print("B6.3.4 TEXT_ONLY_ABLATION")
    print("-" * 72)
    print(f"TEST_ACCURACY={text_only_metrics['accuracy']}")
    print(f"TEST_MACRO_F1={text_only_metrics['macro_f1']}")
    print(f"TEST_WEIGHTED_F1={text_only_metrics['weighted_f1']}")
    print("-" * 72)
    print("B6.3.4 DELTAS")
    print("-" * 72)
    print(f"EXP2_MINUS_EXP1={exp2_minus_exp1}")
    print(f"EXP2_MINUS_TEXT_ONLY={exp2_minus_text_only}")
    print(f"TEXT_ONLY_MINUS_EXP1={text_only_minus_exp1}")
    print(f"REPORT_SAVED={REPORT_PATH}")
    print("=" * 72)
    print("B6.3.4 IMPLEMENTATION=PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
