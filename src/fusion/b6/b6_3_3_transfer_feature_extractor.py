from pathlib import Path
import json
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

ABO_DIR = ROOT / "data" / "representations" / "abo" / "b3"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "representations"
    / "abo"
    / "b6_3_3_transfer"
)

EXPECTED_COUNTS = {
    "train": 70284,
    "validation": 69996,
    "test": 7422,
}

EXPECTED_DIMENSION = 3008


def load_records(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            record = json.loads(line)

            if "record_id" not in record:
                raise ValueError(
                    f"Missing record_id at {path}, line {line_number}"
                )

            b3 = record.get("b3_representation")
            if not isinstance(b3, dict):
                raise ValueError(
                    f"Missing b3_representation at {path}, line {line_number}"
                )

            if b3.get("representation_version") != "v002":
                raise ValueError(
                    f"Unexpected B3 representation version at "
                    f"{path}, line {line_number}: "
                    f"{b3.get('representation_version')}"
                )

            text_block = b3.get("text")
            if not isinstance(text_block, dict):
                raise ValueError(
                    f"Missing B3 text block at {path}, line {line_number}"
                )

            combined_tokens = text_block.get("combined_tokens")

            if not isinstance(combined_tokens, list):
                raise ValueError(
                    f"Missing/invalid combined_tokens at "
                    f"{path}, line {line_number}"
                )

            if not combined_tokens:
                raise ValueError(
                    f"Empty combined_tokens at {path}, line {line_number}"
                )

            if not all(isinstance(token, str) for token in combined_tokens):
                raise ValueError(
                    f"Non-string token found at {path}, line {line_number}"
                )

            records.append(record)

    return records


def extract_split(pipeline, split):
    input_path = ABO_DIR / f"{split}.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    records = load_records(input_path)

    expected_count = EXPECTED_COUNTS[split]

    print(f"{split.upper()}_RECORDS={len(records)}")

    if len(records) != expected_count:
        raise ValueError(
            f"{split} count mismatch: "
            f"{len(records)} != {expected_count}"
        )

    texts = [
    " ".join(
        record["b3_representation"]["text"]["combined_tokens"]
      )
    for record in records
    ]
    record_ids = [str(record["record_id"]) for record in records]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            f"Duplicate record IDs detected in {split}"
        )

    print(f"{split.upper()}_TRANSFORMING")

    # IMPORTANT:
    # This calls the already-fitted pipeline.
    # No fit/fit_transform operation is performed.
    probabilities = pipeline.predict_proba(texts)

    probabilities = np.asarray(probabilities, dtype=np.float32)

    print(
        f"{split.upper()}_SHAPE="
        f"{probabilities.shape[0]}x{probabilities.shape[1]}"
    )

    if probabilities.shape != (
        expected_count,
        EXPECTED_DIMENSION,
    ):
        raise ValueError(
            f"{split} feature shape mismatch: "
            f"{probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"Non-finite transfer features in {split}"
        )

    if np.any(probabilities < 0):
        raise ValueError(
            f"Negative transfer probabilities in {split}"
        )

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            f"Probability rows do not sum to 1 in {split}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_path = OUTPUT_DIR / f"{split}.npz"
    metadata_path = OUTPUT_DIR / f"{split}_metadata.json"

    np.savez_compressed(
        feature_path,
        record_ids=np.asarray(record_ids, dtype=str),
        transferred_signal=probabilities,
    )

    metadata = {
        "stage": "B6.3.3",
        "artifact_version": "v001",
        "source_dataset": "Rakuten",
        "source_model": str(
            MODEL_PATH.relative_to(ROOT)
        ),
        "source_representation_version": "B3_v001",
        "target_dataset": "ABO",
        "target_representation_version": "B3_v002",
        "split": split,
        "record_count": expected_count,
        "feature_dimension": EXPECTED_DIMENSION,
        "signal_type": "Rakuten_source_model_predict_proba",
        "fit_performed": False,
        "external_data_used": False,
        "record_join_performed": False,
        "label_transfer_performed": False,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"{split.upper()}_SAVED={feature_path}")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Frozen source model not found: {MODEL_PATH}"
        )

    print("LOADING_FROZEN_RAKUTEN_SOURCE_MODEL")

    with MODEL_PATH.open("rb") as f:
        artifact = pickle.load(f)

    if not isinstance(artifact, dict):
        raise ValueError("Invalid source-model artifact.")

    pipeline = artifact.get("pipeline")

    if pipeline is None:
        raise ValueError("Missing pipeline in source-model artifact.")

    classifier = pipeline.named_steps["classifier"]

    if len(classifier.classes_) != EXPECTED_DIMENSION:
        raise ValueError(
            f"Source model class count is "
            f"{len(classifier.classes_)}; expected "
            f"{EXPECTED_DIMENSION}"
        )

    print(f"SOURCE_MODEL_CLASSES={len(classifier.classes_)}")
    print("SOURCE_MODEL_FIT_REUSE_ONLY=True")

    for split in ("train", "validation", "test"):
        extract_split(pipeline, split)

    print("TRANSFER_FEATURE_EXTRACTION=PASS")


if __name__ == "__main__":
    main()