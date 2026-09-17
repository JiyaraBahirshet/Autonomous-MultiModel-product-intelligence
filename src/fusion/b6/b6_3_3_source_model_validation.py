from pathlib import Path
import pickle
import numpy as np


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    ROOT
    / "models"
    / "fusion"
    / "b6"
    / "b6_3_3_rakuten_source_model_v001.pkl"
)

EXPECTED_CLASSES = 3008
EXPECTED_FEATURES = 100000
EXPECTED_TRAIN_RECORDS = 719701
EXPECTED_SEED = 20260827


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    print("LOADING_SOURCE_MODEL")

    with MODEL_PATH.open("rb") as f:
        artifact = pickle.load(f)

    if not isinstance(artifact, dict):
        raise ValueError("Artifact is not a dictionary.")

    if "pipeline" not in artifact:
        raise ValueError("Missing pipeline.")

    if "metadata" not in artifact:
        raise ValueError("Missing metadata.")

    pipeline = artifact["pipeline"]
    metadata = artifact["metadata"]

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    actual_features = len(vectorizer.vocabulary_)
    actual_classes = len(classifier.classes_)

    print(f"TRAIN_RECORDS={metadata['train_records']}")
    print(f"SOURCE_CLASSES={actual_classes}")
    print(f"TFIDF_FEATURES={actual_features}")
    print(f"SEED={metadata['classifier']['random_state']}")
    print(f"LOSS={metadata['classifier']['loss']}")
    print(f"PENALTY={metadata['classifier']['penalty']}")

    if metadata["train_records"] != EXPECTED_TRAIN_RECORDS:
        raise ValueError("Train-record count mismatch.")

    if actual_classes != EXPECTED_CLASSES:
        raise ValueError("Source class count mismatch.")

    if actual_features != EXPECTED_FEATURES:
        raise ValueError("TF-IDF feature count mismatch.")

    if metadata["classifier"]["random_state"] != EXPECTED_SEED:
        raise ValueError("Seed mismatch.")

    if metadata["classifier"]["loss"] != "log_loss":
        raise ValueError("Classifier loss mismatch.")

    if metadata["classifier"]["penalty"] != "l2":
        raise ValueError("Classifier penalty mismatch.")

    if not hasattr(classifier, "predict_proba"):
        raise ValueError("Classifier does not expose predict_proba.")

    # Verify coefficient dimensionality.
    coef = classifier.coef_

    if coef.shape != (EXPECTED_CLASSES, EXPECTED_FEATURES):
        raise ValueError(
            f"Coefficient shape mismatch: {coef.shape}"
        )

    if not np.isfinite(coef).all():
        raise ValueError("Non-finite classifier coefficients detected.")

    print(f"COEF_SHAPE={coef.shape}")
    print("PREDICT_PROBA_AVAILABLE=True")
    print("SOURCE_MODEL_VALIDATION=PASS")


if __name__ == "__main__":
    main()