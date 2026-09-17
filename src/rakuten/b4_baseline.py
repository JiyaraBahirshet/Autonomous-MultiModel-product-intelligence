"""
Rakuten B4.1 — Flat TF-IDF + Linear Classification Baseline

Frozen contract:
- Input: data/representations/rakuten/b3/{train,validation,test}.jsonl
- Text: B3 title_tokens only
- Target: exact complete category_id_path
- TF-IDF fitting: train only
- Validation: model/configuration assessment
- Test: final held-out evaluation only
- No hierarchy semantics are invented
- B3 artifacts are read-only
"""

import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "representations" / "rakuten" / "b3"
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "rakuten" / "b4.1"
REPORT_DIR = PROJECT_ROOT / "reports" / "baseline" / "rakuten"

SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {
    "train": 719_701,
    "validation": 80_299,
    "test": 200_000,
}

RANDOM_STATE = 20260827
REPRESENTATION_VERSION = "v001"
B4_VERSION = "v001"

# Conservative first baseline configuration.
VECTORIZER_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "norm": "l2",
    "lowercase": False,
    "token_pattern": r"(?u)\b\w+\b",
}

# Scalable SGD classifier configuration for large-scale multi-class problems on Windows
CLASSIFIER_CONFIG = {
    "loss": "log_loss",
    "penalty": "l2",
    "alpha": 1e-5,
    "max_iter": 5,
    "learning_rate": "optimal",
    "early_stopping": False,
    "verbose": 1,
    "n_jobs": 1,
    "random_state": RANDOM_STATE,
}


def load_split(split: str):
    path = INPUT_DIR / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)

    texts = []
    targets = []
    record_ids = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)

            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")

            required = {
                "record_id",
                "split",
                "title_tokens",
                "category_id_path",
            }
            missing = required - rec.keys()
            if missing:
                raise ValueError(
                    f"{path}:{line_no}: missing B3 fields {sorted(missing)}"
                )

            # Validate split context against B3 provenance or fallback to file context.
            # Top-level rec["split"] may retain legacy source-stage annotations.
            effective_split = rec.get("b3_provenance", {}).get("source_split", split)
            if effective_split != split:
                raise ValueError(
                    f"{path}:{line_no}: expected split {split!r}, "
                    f"got {effective_split!r}"
                )

            tokens = rec["title_tokens"]
            if not isinstance(tokens, list):
                raise ValueError(
                    f"{path}:{line_no}: title_tokens must be a list"
                )

            # B3 already defines the tokenization. Reconstruct exactly that
            # token sequence; no second tokenizer is applied.
            text = " ".join(str(x) for x in tokens)

            target = rec["category_id_path"]
            if not isinstance(target, str) or not target:
                raise ValueError(
                    f"{path}:{line_no}: category_id_path must be a non-empty string"
                )

            texts.append(text)
            targets.append(target)
            record_ids.append(str(rec["record_id"]))

    return texts, targets, record_ids


def check_split_integrity(split, texts, targets, record_ids):
    if len(record_ids) != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED_COUNTS[split]:,} records, "
            f"got {len(record_ids):,}"
        )

    if len(set(record_ids)) != len(record_ids):
        raise ValueError(f"{split}: duplicate record IDs detected")

    if len(texts) != len(targets) or len(targets) != len(record_ids):
        raise ValueError(f"{split}: text/target/record-ID length mismatch")


def hierarchy_metrics(y_true, y_pred):
    """
    Compare category paths level-by-level.

    Paths are preserved as anonymized IDs separated by '>'.
    For each level that exists in a record, agreement is measured against
    the corresponding predicted level. Exact-path accuracy is reported
    separately.
    """
    max_depth = 0
    correct_by_level = Counter()
    total_by_level = Counter()

    for true_path, pred_path in zip(y_true, y_pred):
        t = true_path.split(">")
        p = pred_path.split(">")

        max_depth = max(max_depth, len(t))
        for i, true_id in enumerate(t, 1):
            total_by_level[i] += 1
            if i <= len(p) and p[i - 1] == true_id:
                correct_by_level[i] += 1

    level_accuracy = {
        str(level): correct_by_level[level] / total_by_level[level]
        for level in sorted(total_by_level)
    }

    return {
        "max_true_depth_observed": max_depth,
        "level_accuracy": level_accuracy,
    }


def evaluate(y_true, y_pred):
    exact = accuracy_score(y_true, y_pred)
    return {
        "records_evaluated": len(y_true),
        "exact_complete_path_accuracy": exact,
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "hierarchy_diagnostics": hierarchy_metrics(y_true, y_pred),
    }


def main():
    print("=" * 70)
    print("RAKUTEN B4.1 — TF-IDF + LINEAR CLASSIFICATION BASELINE")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for split in SPLITS:
        print(f"\nLoading B3 {split} split...")
        data[split] = load_split(split)
        check_split_integrity(split, *data[split])
        print(f"  Records: {len(data[split][0]):,}")
        print(f"  Unique targets: {len(set(data[split][1])):,}")

    train_texts, y_train, train_ids = data["train"]
    val_texts, y_val, val_ids = data["validation"]
    test_texts, y_test, test_ids = data["test"]

    train_classes = set(y_train)
    val_unseen = sorted(set(y_val) - train_classes)
    test_unseen = sorted(set(y_test) - train_classes)

    if val_unseen:
        raise ValueError(
            f"Validation contains {len(val_unseen)} target classes unseen in train"
        )
    if test_unseen:
        raise ValueError(
            f"Test contains {len(test_unseen)} target classes unseen in train"
        )

    print("\nFitting TF-IDF on TRAIN ONLY...")
    vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG)
    X_train = vectorizer.fit_transform(train_texts)

    print(f"  Train matrix: {X_train.shape}")
    print(f"  Vocabulary: {len(vectorizer.vocabulary_):,}")

    print("\nTransforming VALIDATION and TEST with frozen train vocabulary...")
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    print("\nTraining linear classifier on TRAIN ONLY...")
    classifier = SGDClassifier(**CLASSIFIER_CONFIG)
    classifier.fit(X_train, y_train)

    print("\nEvaluating VALIDATION...")
    val_pred = classifier.predict(X_val)
    val_metrics = evaluate(y_val, val_pred)
    print(f"  Exact path accuracy: {val_metrics['exact_complete_path_accuracy']:.6f}")
    print(f"  Macro F1            : {val_metrics['macro_f1']:.6f}")
    print(f"  Weighted F1         : {val_metrics['weighted_f1']:.6f}")

    print("\nEvaluating TEST (held-out)...")
    test_pred = classifier.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)
    print(f"  Exact path accuracy: {test_metrics['exact_complete_path_accuracy']:.6f}")
    print(f"  Macro F1            : {test_metrics['macro_f1']:.6f}")
    print(f"  Weighted F1         : {test_metrics['weighted_f1']:.6f}")

    # Save model artifacts. joblib is imported only after successful training.
    import joblib

    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(classifier, MODEL_DIR / "linear_classifier.joblib")

    # Auditable test predictions.
    prediction_path = REPORT_DIR / "rakuten_b4.1_test_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as f:
        for rid, true_label, pred_label in zip(test_ids, y_test, test_pred):
            f.write(
                json.dumps(
                    {
                        "record_id": rid,
                        "true_category_id_path": true_label,
                        "predicted_category_id_path": pred_label,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    report = {
        "report": "rakuten_b4.1_baseline",
        "report_version": B4_VERSION,
        "status": "PASS",
        "dataset": "Rakuten 2018",
        "stage": "B4.1",
        "baseline_name": "RK-FLAT",
        "task": {
            "description": "Flat supervised classification of the exact complete anonymized category path.",
            "hierarchy_aware_final_model": False,
            "comparison_baseline": True,
        },
        "inputs": {
            "directory": "data/representations/rakuten/b3",
            "representation_version": REPRESENTATION_VERSION,
            "text_source": "title_tokens",
            "target_source": "category_id_path",
            "splits": {
                k: EXPECTED_COUNTS[k] for k in SPLITS
            },
        },
        "fitting_protocol": {
            "tfidf_fit_split": "train",
            "classifier_fit_split": "train",
            "validation_role": "validation/model assessment",
            "test_role": "final held-out evaluation only",
        },
        "vectorizer": VECTORIZER_CONFIG,
        "classifier": CLASSIFIER_CONFIG,
        "data_integrity": {
            "expected_counts": EXPECTED_COUNTS,
            "record_ids_unique_within_splits": True,
            "b3_modified": False,
            "test_used_for_model_selection": False,
            "validation_target_classes_unseen_in_train": len(val_unseen),
            "test_target_classes_unseen_in_train": len(test_unseen),
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "artifacts": {
            "vectorizer": "data/models/rakuten/b4.1/tfidf_vectorizer.joblib",
            "classifier": "data/models/rakuten/b4.1/linear_classifier.joblib",
            "test_predictions": "reports/baseline/rakuten/rakuten_b4.1_test_predictions.jsonl",
        },
        "runtime": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
        },
    }

    report_path = REPORT_DIR / "rakuten_b4.1_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "-" * 70)
    print("B4.1 RAKUTEN SUMMARY")
    print("-" * 70)
    print(f"Train records       : {len(train_ids):,}")
    print(f"Validation records  : {len(val_ids):,}")
    print(f"Test records        : {len(test_ids):,}")
    print(f"Train vocabulary    : {len(vectorizer.vocabulary_):,}")
    print(f"Validation accuracy : {val_metrics['exact_complete_path_accuracy']:.6f}")
    print(f"Test accuracy       : {test_metrics['exact_complete_path_accuracy']:.6f}")
    print("\nSTATUS: PASS")
    print("\nArtifacts:")
    print(f"  {MODEL_DIR}")
    print(f"  {report_path}")
    print(f"  {prediction_path}")


if __name__ == "__main__":
    main()