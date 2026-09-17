from pathlib import Path
import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[3]
RAKUTEN_TRAIN = ROOT / "data" / "representations" / "rakuten" / "b3" / "train.jsonl"
MODEL_PATH = ROOT / "models" / "fusion" / "b6" / "b6_3_3_rakuten_source_model_v001.pkl"

EXPECTED_TRAIN_RECORDS = 719_701
EXPECTED_CLASSES = 3_008
SEED = 20260827

# Frozen implementation choice required to make the 3,008-class
# TF-IDF + SGD source model computationally feasible.
MAX_FEATURES = 100_000


def load_rakuten_train():
    texts = []
    labels = []

    with RAKUTEN_TRAIN.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            record = json.loads(line)

            title = record.get("title")
            if not isinstance(title, str) or not title.strip():
                raise ValueError(
                    f"Invalid/missing title at line {line_number}"
                )

            leaf_id = record.get("leaf_category_id")
            if leaf_id is None:
                raise ValueError(
                    f"Missing leaf_category_id at line {line_number}"
                )

            texts.append(title)
            labels.append(str(leaf_id))

    return texts, np.asarray(labels, dtype=object)


def build_source_signal():
    if not RAKUTEN_TRAIN.exists():
        raise FileNotFoundError(RAKUTEN_TRAIN)

    texts, labels = load_rakuten_train()

    train_count = len(texts)
    unique_classes = np.unique(labels)

    print(f"TRAIN_RECORDS={train_count}")
    print(f"SOURCE_CLASSES={len(unique_classes)}")

    if train_count != EXPECTED_TRAIN_RECORDS:
        raise ValueError(
            f"Rakuten train count mismatch: "
            f"{train_count} != {EXPECTED_TRAIN_RECORDS}"
        )

    if len(unique_classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Rakuten class count mismatch: "
            f"{len(unique_classes)} != {EXPECTED_CLASSES}"
        )


    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        random_state=SEED,
    )

    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )

    print(f"MAX_FEATURES={MAX_FEATURES}")
    print("FITTING_SOURCE_MODEL")

    pipeline.fit(texts, labels)

    fitted_vectorizer = pipeline.named_steps["tfidf"]
    fitted_classifier = pipeline.named_steps["classifier"]

    actual_features = len(fitted_vectorizer.vocabulary_)
    actual_classes = len(fitted_classifier.classes_)

    print(f"ACTUAL_FEATURES={actual_features}")
    print(f"SOURCE_MODEL_K={actual_classes}")

    if actual_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Fitted classifier class count mismatch: "
            f"{actual_classes} != {EXPECTED_CLASSES}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "stage": "B6.3.3",
        "artifact_version": "v001",
        "dataset": "Rakuten",
        "representation_version": "B3_v001",
        "train_records": EXPECTED_TRAIN_RECORDS,
        "source_classes": EXPECTED_CLASSES,
        "tfidf": {
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "max_features": MAX_FEATURES,
            "dtype": "float32",
            "actual_features": actual_features,
        },
        "classifier": {
            "type": "SGDClassifier",
            "loss": "log_loss",
            "penalty": "l2",
            "random_state": SEED,
        },
        "target_dimension": EXPECTED_CLASSES,
    }

    artifact = {
        "pipeline": pipeline,
        "metadata": metadata,
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"SOURCE_MODEL_SAVED={MODEL_PATH}")
    print(f"SOURCE_MODEL_K={actual_classes}")


if __name__ == "__main__":
    build_source_signal()
