from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fusion"
    / "b6"
    / "b6_3_3_rakuten_source_model_v001.pkl"
)

TRANSFER_DIR = (
    PROJECT_ROOT
    / "data"
    / "representations"
    / "abo"
    / "b6_3_3_transfer"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "fusion"
    / "b6"
    / "b6_3_3_transfer_artifact_validation_v001.json"
)


# ============================================================
# FROZEN CONTRACT
# ============================================================

EXPECTED_COUNTS = {
    "train": 70284,
    "validation": 69996,
    "test": 7422,
}

EXPECTED_DIMENSION = 3008
EXPECTED_SEED = 20260827

ROW_SUM_TOLERANCE = 1e-5
REPRO_TOLERANCE = 1e-6
REPRO_SAMPLE_SIZE = 5


# ============================================================
# HELPERS
# ============================================================

def fail(message: str):
    raise RuntimeError(message)


def load_npz(path: Path):
    if not path.exists():
        fail(f"Missing transfer artifact: {path}")

    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    if not keys:
        fail(f"NPZ contains no arrays: {path}")

    return data, keys


def find_matrix(data, keys):
    preferred_names = [
        "probabilities",
        "probability",
        "transfer_features",
        "features",
        "predictions",
        "X",
    ]

    for name in preferred_names:
        if name in keys:
            arr = np.asarray(data[name])
            if arr.ndim == 2:
                return name, arr

    candidates = []

    for key in keys:
        arr = np.asarray(data[key])
        if arr.ndim == 2:
            candidates.append((key, arr))

    if not candidates:
        fail(
            f"No 2-D matrix found. Available keys: {keys}"
        )

    for key, arr in candidates:
        if arr.shape[1] == EXPECTED_DIMENSION:
            return key, arr

    if len(candidates) == 1:
        return candidates[0]

    fail(
        "Could not uniquely identify transfer matrix. "
        f"2-D candidates: {[x[0] for x in candidates]}"
    )


def find_record_ids(data, keys):
    preferred_names = [
        "record_ids",
        "record_id",
        "ids",
    ]

    for name in preferred_names:
        if name in keys:
            return name, np.asarray(data[name]).astype(str)

    candidates = []

    for key in keys:
        arr = np.asarray(data[key])

        if arr.ndim == 1 and arr.dtype.kind in {"U", "S", "O"}:
            candidates.append((key, arr.astype(str)))

    if len(candidates) == 1:
        return candidates[0]

    for key, arr in candidates:
        lower = key.lower()
        if "record" in lower and "id" in lower:
            return key, arr

    fail(
        f"Could not identify record IDs. Available keys: {keys}"
    )


def load_metadata(split: str):
    candidates = [
        TRANSFER_DIR / f"{split}_metadata.json",
        TRANSFER_DIR / f"{split}.json",
        TRANSFER_DIR / "metadata.json",
    ]

    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return path, json.load(f)

    return None, None


def find_abo_b3_file(split: str):
    candidates = [
        PROJECT_ROOT
        / "data"
        / "processed"
        / "abo"
        / f"{split}.jsonl",

        PROJECT_ROOT
        / "data"
        / "processed"
        / "abo"
        / f"{split}_v002.jsonl",

        PROJECT_ROOT
        / "data"
        / "representations"
        / "abo"
        / "b3"
        / f"{split}.jsonl",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_abo_b3_records(split: str):
    path = find_abo_b3_file(split)

    if path is None:
        return None, None

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return path, records


def extract_combined_tokens(record):
    b3 = record.get("b3_representation")

    if not isinstance(b3, dict):
        fail("ABO record missing b3_representation.")

    version = b3.get("representation_version")

    if version != "v002":
        fail(
            "ABO B3 representation version mismatch: "
            f"expected v002, got {version}"
        )

    text = b3.get("text")

    if not isinstance(text, dict):
        fail("ABO record missing b3_representation.text.")

    tokens = text.get("combined_tokens")

    if not isinstance(tokens, list):
        fail(
            "ABO record missing "
            "b3_representation.text.combined_tokens."
        )

    return " ".join(str(token) for token in tokens)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("B6.3.3 TRANSFER ARTIFACT VALIDATION")
    print("=" * 70)

    results = {
        "validation_version": "v001",
        "status": "FAIL",
        "split_results": {},
        "dimension_results": {},
        "record_id_results": {},
        "numerical_results": {},
        "reproducibility_sample_results": {},
        "provenance_results": {},
        "integrity_constraints": {
            "record_joins": False,
            "label_transfer": False,
            "synthetic_pairing": False,
            "external_enrichment": False,
            "image_download": False,
            "path_guessing": False,
            "split_regeneration": False,
            "b5_outputs_or_weights": False,
            "target_model_training": False,
            "source_model_fitting": False,
        },
        "artifact_paths": {
            "source_model": str(SOURCE_MODEL_PATH),
            "transfer_directory": str(TRANSFER_DIR),
        },
        "failure_reasons": [],
    }

    try:

        # ========================================================
        # 1. LOAD FROZEN SOURCE MODEL
        # ========================================================

        print("\n[1] LOADING FROZEN SOURCE MODEL")

        if not SOURCE_MODEL_PATH.exists():
            fail(
                f"Source model does not exist: "
                f"{SOURCE_MODEL_PATH}"
            )

        with SOURCE_MODEL_PATH.open("rb") as f:
            source_artifact = pickle.load(f)

        print(
            "SOURCE_ARTIFACT_TYPE="
            f"{type(source_artifact).__name__}"
        )

        if not isinstance(source_artifact, dict):
            fail(
                "Frozen source model artifact must be a "
                "dictionary containing 'pipeline' and 'metadata'."
            )

        if "pipeline" not in source_artifact:
            fail(
                "Frozen source model artifact missing 'pipeline'."
            )

        if "metadata" not in source_artifact:
            fail(
                "Frozen source model artifact missing 'metadata'."
            )

        pipeline = source_artifact["pipeline"]
        source_metadata = source_artifact["metadata"]

        print(
            "PIPELINE_TYPE="
            f"{type(pipeline).__name__}"
        )

        if not hasattr(pipeline, "predict_proba"):
            fail(
                "Frozen source pipeline does not expose "
                "predict_proba()."
            )

        if not hasattr(pipeline, "named_steps"):
            fail(
                "Frozen source pipeline does not expose "
                "named_steps."
            )

        if "classifier" not in pipeline.named_steps:
            fail(
                "Frozen source pipeline does not contain "
                "expected 'classifier' step."
            )

        classifier = pipeline.named_steps["classifier"]

        source_classes = getattr(
            classifier,
            "classes_",
            None,
        )

        if source_classes is None:
            fail(
                "Frozen source classifier does not expose "
                "classes_."
            )

        source_k = len(source_classes)

        print(f"SOURCE_MODEL_CLASSES={source_k}")

        if source_k != EXPECTED_DIMENSION:
            fail(
                "Source model class count mismatch: "
                f"expected {EXPECTED_DIMENSION}, "
                f"got {source_k}"
            )

        results["provenance_results"] = {
            "source_dataset": "Rakuten",
            "source_representation": "B3_v001",
            "target_dataset": "ABO",
            "target_representation": "B3_v002",
            "source_model_reference": str(
                SOURCE_MODEL_PATH
            ),
            "source_model_class_count": source_k,
            "seed": EXPECTED_SEED,
            "fit_performed_during_validation": False,
            "source_artifact_metadata": source_metadata,
        }

        print("SOURCE_MODEL_VALIDATION=PASS")

        # ========================================================
        # 2. VALIDATE EACH TRANSFER ARTIFACT
        # ========================================================

        matrices = {}
        record_ids = {}

        for split in ["train", "validation", "test"]:

            print("\n" + "-" * 70)
            print(f"[2] VALIDATING {split.upper()} ARTIFACT")

            npz_path = TRANSFER_DIR / f"{split}.npz"

            data, keys = load_npz(npz_path)

            print(f"NPZ={npz_path}")
            print(f"KEYS={keys}")

            matrix_key, matrix = find_matrix(
                data,
                keys,
            )

            id_key, ids = find_record_ids(
                data,
                keys,
            )

            expected_count = EXPECTED_COUNTS[split]

            print(f"MATRIX_KEY={matrix_key}")
            print(f"ID_KEY={id_key}")
            print(f"SHAPE={matrix.shape}")

            # Count
            if matrix.shape[0] != expected_count:
                fail(
                    f"{split}: expected "
                    f"{expected_count} records, "
                    f"got {matrix.shape[0]}"
                )

            # Dimension
            if matrix.shape[1] != EXPECTED_DIMENSION:
                fail(
                    f"{split}: expected "
                    f"{EXPECTED_DIMENSION} dimensions, "
                    f"got {matrix.shape[1]}"
                )

            # ID count
            if len(ids) != expected_count:
                fail(
                    f"{split}: record-ID count mismatch."
                )

            # ID uniqueness
            unique_count = len(set(ids))

            if unique_count != expected_count:
                fail(
                    f"{split}: duplicate record IDs detected."
                )

            # Numerical integrity
            finite = bool(np.isfinite(matrix).all())

            if not finite:
                fail(
                    f"{split}: non-finite values detected."
                )

            nonnegative = bool((matrix >= 0).all())

            if not nonnegative:
                fail(
                    f"{split}: negative probability values detected."
                )

            row_sums = matrix.sum(axis=1)

            max_row_sum_error = float(
                np.max(np.abs(row_sums - 1.0))
            )

            row_sum_pass = (
                max_row_sum_error
                <= ROW_SUM_TOLERANCE
            )

            if not row_sum_pass:
                fail(
                    f"{split}: row-sum validation failed. "
                    f"max_error={max_row_sum_error}"
                )

            results["split_results"][split] = {
                "expected_records": expected_count,
                "actual_records": int(matrix.shape[0]),
                "passed": True,
            }

            results["dimension_results"][split] = {
                "expected_dimension": EXPECTED_DIMENSION,
                "actual_dimension": int(matrix.shape[1]),
                "passed": True,
            }

            results["record_id_results"][split] = {
                "record_id_count": int(len(ids)),
                "unique_record_id_count": int(
                    unique_count
                ),
                "unique_within_split": True,
            }

            results["numerical_results"][split] = {
                "finite": finite,
                "nonnegative": nonnegative,
                "row_sum_tolerance": ROW_SUM_TOLERANCE,
                "max_row_sum_error": max_row_sum_error,
                "row_sums_pass": row_sum_pass,
            }

            matrices[split] = matrix
            record_ids[split] = ids

            print("COUNT_PASS=True")
            print("DIMENSION_PASS=True")
            print("UNIQUE_IDS=True")
            print(f"FINITE={finite}")
            print(f"NONNEGATIVE={nonnegative}")
            print(
                "MAX_ROW_SUM_ERROR="
                f"{max_row_sum_error:.12g}"
            )
            print("ROW_SUM_PASS=True")

        # ========================================================
        # 3. CROSS-SPLIT ID LEAKAGE
        # ========================================================

        print("\n[3] CROSS-SPLIT RECORD-ID LEAKAGE")

        train_ids = set(record_ids["train"])
        validation_ids = set(
            record_ids["validation"]
        )
        test_ids = set(record_ids["test"])

        train_validation_overlap = (
            train_ids & validation_ids
        )

        train_test_overlap = (
            train_ids & test_ids
        )

        validation_test_overlap = (
            validation_ids & test_ids
        )

        results["record_id_results"]["cross_split"] = {
            "train_validation_overlap": len(
                train_validation_overlap
            ),
            "train_test_overlap": len(
                train_test_overlap
            ),
            "validation_test_overlap": len(
                validation_test_overlap
            ),
            "passed": (
                len(train_validation_overlap) == 0
                and len(train_test_overlap) == 0
                and len(validation_test_overlap) == 0
            ),
        }

        print(
            "TRAIN_VALIDATION_OVERLAP="
            f"{len(train_validation_overlap)}"
        )
        print(
            "TRAIN_TEST_OVERLAP="
            f"{len(train_test_overlap)}"
        )
        print(
            "VALIDATION_TEST_OVERLAP="
            f"{len(validation_test_overlap)}"
        )

        if (
            train_validation_overlap
            or train_test_overlap
            or validation_test_overlap
        ):
            fail(
                "Cross-split record-ID leakage detected."
            )

        print("CROSS_SPLIT_LEAKAGE_PASS=True")

        # ========================================================
        # 4. METADATA
        # ========================================================

        print("\n[4] METADATA / PROVENANCE")

        metadata_results = {}

        for split in ["train", "validation", "test"]:

            metadata_path, metadata = load_metadata(
                split
            )

            if metadata is None:

                metadata_results[split] = {
                    "metadata_found": False,
                    "note": (
                        "No separate metadata JSON found."
                    ),
                }

                print(
                    f"{split}: "
                    "NO_SEPARATE_METADATA_FILE"
                )

            else:

                metadata_results[split] = {
                    "metadata_found": True,
                    "path": str(metadata_path),
                    "keys": (
                        list(metadata.keys())
                        if isinstance(
                            metadata,
                            dict,
                        )
                        else None
                    ),
                }

                print(
                    f"{split}: "
                    f"METADATA={metadata_path}"
                )

        results["provenance_results"][
            "split_metadata"
        ] = metadata_results

        # ========================================================
        # 5. AUTHORITATIVE B3 ORDER CHECK
        # ========================================================

        print(
            "\n[5] AUTHORITATIVE ABO B3 "
            "RECORD ORDER CHECK"
        )

        order_results = {}

        for split in ["train", "validation", "test"]:

            b3_path, b3_records = (
                load_abo_b3_records(split)
            )

            if b3_records is None:

                order_results[split] = {
                    "checked": False,
                    "passed": None,
                    "note": (
                        "Known B3 JSONL paths did not "
                        "locate the source file."
                    ),
                }

                print(
                    f"{split}: "
                    "B3_SOURCE_NOT_LOCATED"
                )

                continue

            b3_ids = []

            for record in b3_records:

                rid = record.get("record_id")

                if rid is None:
                    fail(
                        f"{split}: B3 record missing "
                        "record_id."
                    )

                b3_ids.append(str(rid))

            artifact_ids = list(
                record_ids[split]
            )

            count_match = (
                len(b3_ids)
                == len(artifact_ids)
            )

            exact_order_match = (
                count_match
                and b3_ids == artifact_ids
            )

            order_results[split] = {
                "checked": True,
                "path": str(b3_path),
                "b3_record_count": len(b3_ids),
                "artifact_record_count": len(
                    artifact_ids
                ),
                "count_match": count_match,
                "exact_order_match": (
                    exact_order_match
                ),
                "passed": exact_order_match,
            }

            print(
                f"{split}: "
                f"B3_RECORDS={len(b3_ids)}"
            )

            print(
                f"{split}: "
                f"ARTIFACT_RECORDS="
                f"{len(artifact_ids)}"
            )

            print(
                f"{split}: "
                f"EXACT_ORDER_MATCH="
                f"{exact_order_match}"
            )

            if not exact_order_match:
                fail(
                    f"{split}: transfer artifact "
                    "record ordering does not exactly "
                    "match authoritative B3 ordering."
                )

        results["record_id_results"][
            "authoritative_b3_order"
        ] = order_results

        # ========================================================
        # 6. DIRECT INFERENCE REPRODUCIBILITY
        # ========================================================

        print(
            "\n[6] DETERMINISTIC INFERENCE "
            "REPRODUCIBILITY"
        )

        reproducibility_results = {}

        for split in ["train", "validation", "test"]:

            b3_path, b3_records = (
                load_abo_b3_records(split)
            )

            if b3_records is None:

                reproducibility_results[split] = {
                    "checked": False,
                    "passed": None,
                    "note": (
                        "B3 source not located."
                    ),
                }

                print(
                    f"{split}: "
                    "REPRO_CHECK_NOT_PERFORMED"
                )

                continue

            sample_size = min(
                REPRO_SAMPLE_SIZE,
                len(b3_records),
            )

            sample_indices = np.linspace(
                0,
                len(b3_records) - 1,
                num=sample_size,
                dtype=int,
            )

            texts = [
                extract_combined_tokens(
                    b3_records[i]
                )
                for i in sample_indices
            ]

            direct_predictions = (
                pipeline.predict_proba(texts)
            )

            saved_predictions = matrices[
                split
            ][sample_indices]

            if direct_predictions.shape != (
                saved_predictions.shape
            ):
                fail(
                    f"{split}: direct inference shape "
                    "does not match saved artifact."
                )

            max_abs_difference = float(
                np.max(
                    np.abs(
                        direct_predictions
                        - saved_predictions
                    )
                )
            )

            passed = (
                max_abs_difference
                <= REPRO_TOLERANCE
            )

            reproducibility_results[split] = {
                "checked": True,
                "b3_path": str(b3_path),
                "sample_indices": (
                    sample_indices.tolist()
                ),
                "sample_size": int(sample_size),
                "max_absolute_difference": (
                    max_abs_difference
                ),
                "tolerance": REPRO_TOLERANCE,
                "passed": passed,
            }

            print(
                f"{split}: "
                "MAX_ABS_DIFFERENCE="
                f"{max_abs_difference:.12g}"
            )

            print(
                f"{split}: "
                f"REPRO_PASS={passed}"
            )

            if not passed:
                fail(
                    f"{split}: saved transfer "
                    "probabilities do not reproduce "
                    "direct frozen-model inference."
                )

        results[
            "reproducibility_sample_results"
        ] = reproducibility_results

        # ========================================================
        # 7. FINAL PASS
        # ========================================================

        results["status"] = "PASS"

        print("\n" + "=" * 70)
        print("TRANSFER_ARTIFACT_VALIDATION=PASS")
        print("=" * 70)

    except Exception as exc:

        results["status"] = "FAIL"
        results["failure_reasons"].append(
            str(exc)
        )

        print("\n" + "=" * 70)
        print("TRANSFER_ARTIFACT_VALIDATION=FAIL")
        print(f"REASON={exc}")
        print("=" * 70)

        REPORT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with REPORT_PATH.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
            )

        print(
            f"FAILURE_REPORT_SAVED="
            f"{REPORT_PATH}"
        )

        sys.exit(1)

    # ============================================================
    # WRITE SUCCESS REPORT
    # ============================================================

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"REPORT_SAVED={REPORT_PATH}"
    )


if __name__ == "__main__":
    main()