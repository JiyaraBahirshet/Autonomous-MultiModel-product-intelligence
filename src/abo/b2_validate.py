import json
import re
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------
# B2 ARTIFACT LOCATIONS
# ----------------------------------------------------------------------

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "abo"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "abo"
    / "b2"
)

SCHEMA_FILE = OUTPUT_DIR / "schema_v001.json"
STATISTICS_FILE = OUTPUT_DIR / "statistics.json"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "validation"
)

REPORT_FILE = (
    REPORT_DIR
    / "abo_b2_validation.json"
)

SPLITS = (
    "train",
    "validation",
    "test",
)

EXPECTED_RECORD_COUNTS = {
    "train": 70284,
    "validation": 69996,
    "test": 7422,
}

EXPECTED_TOTAL_RECORDS = 147702


# ----------------------------------------------------------------------
# B2 CONTRACT
# ----------------------------------------------------------------------

TEXT_FIELDS = {
    "brand",
    "item_name",
    "bullet_point",
    "color",
    "fabric_type",
    "finish_type",
    "item_keywords",
    "item_shape",
    "material",
    "model_name",
    "pattern",
    "product_description",
    "style",
    "title",
    "description",
    "feature",
    "amazon_category_path",
    "amazon_main_category",
    "mave_category",
    "mave_label_source",
    "mave_information_available",
}

HTML_TAG_PATTERN = re.compile(
    r"<[^>]*>"
)


# ----------------------------------------------------------------------
# JSON HELPERS
# ----------------------------------------------------------------------

def load_json(
    path: Path,
) -> dict[str, Any]:

    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return data


# ----------------------------------------------------------------------
# B2 TRANSFORMATION REIMPLEMENTATION
#
# This must exactly reproduce b2_preprocess.py.
# ----------------------------------------------------------------------

def normalize_text(
    value: str,
) -> str:

    if not isinstance(value, str):
        raise ValueError(
            "normalize_text expects a string"
        )

    # Unicode NFC normalization
    normalized = unicodedata.normalize(
        "NFC",
        value,
    )

    # HTML-like tag removal
    normalized = HTML_TAG_PATTERN.sub(
        " ",
        normalized,
    )

    # Whitespace normalization
    normalized = " ".join(
        normalized.split()
    ).strip()

    return normalized


def normalize_text_structure(
    value: Any,
) -> Any:

    if isinstance(value, str):

        return normalize_text(
            value
        )

    if isinstance(value, list):

        return [
            normalize_text_structure(
                item
            )
            for item in value
        ]

    if isinstance(value, dict):

        normalized = {}

        for key, item in value.items():

            # language_tag is metadata.
            # It must remain unchanged.
            if key == "language_tag":

                normalized[key] = item

            else:

                normalized[key] = (
                    normalize_text_structure(
                        item
                    )
                )

        return normalized

    return value


def build_expected_record(
    source_record: dict[str, Any],
    expected_split: str,
) -> dict[str, Any]:

    expected = {}

    for key, value in source_record.items():

        # --------------------------------------------------------------
        # Image references and image metadata are preserved exactly.
        # --------------------------------------------------------------

        if key in {
            "main_image",
            "other_images",
            "image_counts",
        }:

            expected[key] = value

            continue

        # --------------------------------------------------------------
        # Identity / structural metadata preserved exactly.
        # --------------------------------------------------------------

        if key in {
            "record_id",
            "item_id",
            "country",
            "marketplace",
            "domain_name",
            "split",
        }:

            expected[key] = value

            continue

        # --------------------------------------------------------------
        # Designated text fields.
        # --------------------------------------------------------------

        if key in TEXT_FIELDS:

            expected[key] = (
                normalize_text_structure(
                    value
                )
            )

            continue

        # --------------------------------------------------------------
        # Unknown/non-text fields.
        # --------------------------------------------------------------

        expected[key] = value

    # --------------------------------------------------------------
    # B2 provenance.
    # --------------------------------------------------------------

    expected["b2_provenance"] = {
        "processing_version": "v001",
        "source_split": expected_split,
        "source_artifact": (
            f"data/splits/abo/{expected_split}.jsonl"
        ),
    }

    return expected


# ----------------------------------------------------------------------
# IMAGE VALIDATION
# ----------------------------------------------------------------------

def calculate_image_statistics(
    record: dict[str, Any],
) -> tuple[int, int, int, int]:

    references = []

    main_image = record.get(
        "main_image"
    )

    if (
        isinstance(main_image, dict)
        and main_image.get("image_id")
    ):

        references.append(
            main_image
        )

    other_images = record.get(
        "other_images"
    )

    if isinstance(
        other_images,
        list,
    ):

        for image_ref in other_images:

            if (
                isinstance(image_ref, dict)
                and image_ref.get("image_id")
            ):

                references.append(
                    image_ref
                )

    physical_available = 0
    physical_missing = 0

    for image_ref in references:

        flag = image_ref.get(
            "physical_file_available"
        )

        if flag is True:

            physical_available += 1

        elif flag is False:

            physical_missing += 1

        else:

            raise ValueError(
                "Invalid physical_file_available "
                "value in image reference"
            )

    main_present = int(
        isinstance(main_image, dict)
        and bool(
            main_image.get("image_id")
        )
    )

    return (
        len(references),
        physical_available,
        physical_missing,
        main_present,
    )


# ----------------------------------------------------------------------
# SPLIT VALIDATION
# ----------------------------------------------------------------------

def validate_split(
    split: str,
) -> dict[str, Any]:

    source_file = (
        INPUT_DIR
        / f"{split}.jsonl"
    )

    output_file = (
        OUTPUT_DIR
        / f"{split}.jsonl"
    )

    if not source_file.is_file():

        raise FileNotFoundError(
            f"Frozen ABO split not found: "
            f"{source_file}"
        )

    if not output_file.is_file():

        raise FileNotFoundError(
            f"B2 ABO artifact not found: "
            f"{output_file}"
        )

    source_records = 0
    output_records = 0

    source_record_ids = set()
    output_record_ids = set()

    source_item_ids = set()
    output_item_ids = set()

    source_duplicate_record_ids = 0
    output_duplicate_record_ids = 0

    source_duplicate_item_ids = 0
    output_duplicate_item_ids = 0

    malformed_source = 0
    malformed_output = 0

    mismatched_records = 0

    first_mismatch = None

    changed_text_values = 0
    unchanged_text_values = 0

    total_image_references = 0
    image_references_with_files = 0
    image_references_without_files = 0

    records_with_main_image = 0
    records_without_main_image = 0

    # --------------------------------------------------------------
    # Lockstep reading.
    #
    # This ensures the output has the same record ordering and
    # cardinality as the frozen input.
    # --------------------------------------------------------------

    with (
        source_file.open(
            "r",
            encoding="utf-8",
        ) as source,

        output_file.open(
            "r",
            encoding="utf-8",
        ) as output,
    ):

        line_number = 0

        while True:

            source_line = next(
                source,
                None,
            )

            output_line = next(
                output,
                None,
            )

            if (
                source_line is None
                and output_line is None
            ):

                break

            line_number += 1

            # ------------------------------------------------------
            # Length mismatch.
            # ------------------------------------------------------

            if (
                source_line is None
                or output_line is None
            ):

                mismatched_records += 1

                if first_mismatch is None:

                    first_mismatch = {
                        "line": line_number,
                        "reason": (
                            "source/output "
                            "length mismatch"
                        ),
                    }

                continue

            source_line = source_line.rstrip(
                "\r\n"
            )

            output_line = output_line.rstrip(
                "\r\n"
            )

            # ------------------------------------------------------
            # Blank source lines should not exist in canonical
            # artifacts, but preserve the same streaming behavior
            # as preprocessing.
            # ------------------------------------------------------

            if not source_line:

                continue

            if not output_line:

                mismatched_records += 1

                if first_mismatch is None:

                    first_mismatch = {
                        "line": line_number,
                        "reason": "empty output line",
                    }

                continue

            # ------------------------------------------------------
            # Parse source.
            # ------------------------------------------------------

            try:

                source_record = json.loads(
                    source_line
                )

            except json.JSONDecodeError as exc:

                malformed_source += 1

                raise ValueError(
                    f"Malformed source JSON at "
                    f"{source_file}:{line_number}: "
                    f"{exc}"
                ) from exc

            # ------------------------------------------------------
            # Parse output.
            # ------------------------------------------------------

            try:

                output_record = json.loads(
                    output_line
                )

            except json.JSONDecodeError as exc:

                malformed_output += 1

                raise ValueError(
                    f"Malformed B2 JSON at "
                    f"{output_file}:{line_number}: "
                    f"{exc}"
                ) from exc

            if (
                not isinstance(source_record, dict)
                or not isinstance(output_record, dict)
            ):

                raise ValueError(
                    f"Expected JSON objects at "
                    f"line {line_number}"
                )

            source_records += 1
            output_records += 1

            # ------------------------------------------------------
            # Record IDs.
            # ------------------------------------------------------

            source_record_id = str(
                source_record.get(
                    "record_id"
                )
            )

            output_record_id = str(
                output_record.get(
                    "record_id"
                )
            )

            if source_record_id in source_record_ids:

                source_duplicate_record_ids += 1

            else:

                source_record_ids.add(
                    source_record_id
                )

            if output_record_id in output_record_ids:

                output_duplicate_record_ids += 1

            else:

                output_record_ids.add(
                    output_record_id
                )

            # ------------------------------------------------------
            # Item IDs.
            #
            # Duplicate item IDs are NOT automatically failures.
            # The existing B2 statistics explicitly preserve them.
            # ------------------------------------------------------

            source_item_id = str(
                source_record.get(
                    "item_id"
                )
            )

            output_item_id = str(
                output_record.get(
                    "item_id"
                )
            )

            if source_item_id in source_item_ids:

                source_duplicate_item_ids += 1

            else:

                source_item_ids.add(
                    source_item_id
                )

            if output_item_id in output_item_ids:

                output_duplicate_item_ids += 1

            else:

                output_item_ids.add(
                    output_item_id
                )

            # ------------------------------------------------------
            # Exact expected B2 output.
            # ------------------------------------------------------

            expected_record = (
                build_expected_record(
                    source_record,
                    split,
                )
            )

            if output_record != expected_record:

                mismatched_records += 1

                if first_mismatch is None:

                    differing_keys = []

                    for key in sorted(
                        set(expected_record)
                        | set(output_record)
                    ):

                        if (
                            expected_record.get(key)
                            != output_record.get(key)
                        ):

                            differing_keys.append(
                                key
                            )

                    first_mismatch = {
                        "line": line_number,
                        "reason": (
                            "B2 output does not "
                            "match deterministic "
                            "expected transformation"
                        ),
                        "differing_keys": (
                            differing_keys[:20]
                        ),
                    }

            # ------------------------------------------------------
            # Missing-field preservation.
            # ------------------------------------------------------

            for field in TEXT_FIELDS:

                if field not in source_record:

                    if field in output_record:

                        mismatched_records += 1

                        if first_mismatch is None:

                            first_mismatch = {
                                "line": line_number,
                                "reason": (
                                    "B2 fabricated "
                                    f"missing field: {field}"
                                ),
                            }

                    continue

                expected_value = (
                    normalize_text_structure(
                        source_record[field]
                    )
                )

                actual_value = (
                    output_record.get(
                        field
                    )
                )

                if actual_value == source_record[field]:

                    unchanged_text_values += 1

                else:

                    changed_text_values += 1

                if actual_value != expected_value:

                    mismatched_records += 1

                    if first_mismatch is None:

                        first_mismatch = {
                            "line": line_number,
                            "reason": (
                                "text normalization "
                                f"mismatch: {field}"
                            ),
                        }

            # ------------------------------------------------------
            # Image fields must be preserved exactly.
            # ------------------------------------------------------

            for field in (
                "main_image",
                "other_images",
                "image_counts",
            ):

                if (
                    output_record.get(field)
                    != source_record.get(field)
                ):

                    mismatched_records += 1

                    if first_mismatch is None:

                        first_mismatch = {
                            "line": line_number,
                            "reason": (
                                "image/structural "
                                f"field changed: {field}"
                            ),
                        }

            # ------------------------------------------------------
            # Image statistics.
            # ------------------------------------------------------

            (
                references,
                available,
                missing,
                main_present,
            ) = calculate_image_statistics(
                output_record
            )

            total_image_references += (
                references
            )

            image_references_with_files += (
                available
            )

            image_references_without_files += (
                missing
            )

            if main_present:

                records_with_main_image += 1

            else:

                records_without_main_image += 1

    expected_count = (
        EXPECTED_RECORD_COUNTS[split]
    )

    checks = {
        "source_record_count_expected": (
            source_records
            == expected_count
        ),

        "output_record_count_expected": (
            output_records
            == expected_count
        ),

        "source_output_counts_match": (
            source_records
            == output_records
        ),

        "source_record_ids_unique": (
            source_duplicate_record_ids == 0
        ),

        "output_record_ids_unique": (
            output_duplicate_record_ids == 0
        ),

        "record_id_set_preserved": (
            source_record_ids
            == output_record_ids
        ),

        "item_id_set_preserved": (
            source_item_ids
            == output_item_ids
        ),

        "exact_b2_transformation": (
            mismatched_records == 0
        ),

        "malformed_source_zero": (
            malformed_source == 0
        ),

        "malformed_output_zero": (
            malformed_output == 0
        ),
    }

    return {
        "source_records": source_records,
        "output_records": output_records,

        "source_unique_record_ids": (
            len(source_record_ids)
        ),

        "output_unique_record_ids": (
            len(output_record_ids)
        ),

        "source_duplicate_record_ids": (
            source_duplicate_record_ids
        ),

        "output_duplicate_record_ids": (
            output_duplicate_record_ids
        ),

        "source_unique_item_ids": (
            len(source_item_ids)
        ),

        "output_unique_item_ids": (
            len(output_item_ids)
        ),

        "source_duplicate_item_ids": (
            source_duplicate_item_ids
        ),

        "output_duplicate_item_ids": (
            output_duplicate_item_ids
        ),

        "mismatched_records": (
            mismatched_records
        ),

        "first_mismatch": first_mismatch,

        "malformed_source": malformed_source,
        "malformed_output": malformed_output,

        "changed_text_values": (
            changed_text_values
        ),

        "unchanged_text_values": (
            unchanged_text_values
        ),

        "total_image_references": (
            total_image_references
        ),

        "image_references_with_physical_files": (
            image_references_with_files
        ),

        "image_references_missing_physical_files": (
            image_references_without_files
        ),

        "records_with_main_image": (
            records_with_main_image
        ),

        "records_without_main_image": (
            records_without_main_image
        ),

        "checks": checks,

        "status": all(
            checks.values()
        ),
    }


# ----------------------------------------------------------------------
# SCHEMA VALIDATION
# ----------------------------------------------------------------------

def validate_schema(
    schema: dict[str, Any],
) -> dict[str, bool]:

    transformations = schema.get(
        "transformations",
        {},
    )

    provenance = schema.get(
        "provenance",
        {},
    )

    return {
        "dataset_correct": (
            schema.get("dataset")
            == "Amazon Berkeley Objects (ABO)"
        ),

        "stage_correct": (
            schema.get("stage")
            == "B2"
        ),

        "schema_version_correct": (
            schema.get("schema_version")
            == "v001"
        ),

        "processing_version_correct": (
            schema.get("processing_version")
            == "v001"
        ),

        "input_artifact_correct": (
            schema.get("input_artifact")
            == "data/splits/abo/{split}.jsonl"
        ),

        "output_artifact_correct": (
            schema.get("output_artifact")
            == "data/processed/abo/b2/{split}.jsonl"
        ),

        "split_values_correct": (
            schema.get("split_values")
            == list(SPLITS)
        ),

        "translation_disabled": (
            transformations.get(
                "translation"
            )
            is False
        ),

        "lowercasing_disabled": (
            transformations.get(
                "lowercasing"
            )
            is False
        ),

        "synthetic_text_disabled": (
            transformations.get(
                "synthetic_text"
            )
            is False
        ),

        "deduplication_disabled": (
            transformations.get(
                "deduplication"
            )
            is False
        ),

        "split_regeneration_disabled": (
            transformations.get(
                "split_regeneration"
            )
            is False
        ),

        "image_download_disabled": (
            transformations.get(
                "image_download"
            )
            is False
        ),

        "image_path_guessing_disabled": (
            transformations.get(
                "image_path_guessing"
            )
            is False
        ),

        "record_id_preserved": (
            provenance.get(
                "record_id_preserved"
            )
            is True
        ),

        "item_id_preserved": (
            provenance.get(
                "item_id_preserved"
            )
            is True
        ),

        "split_preserved": (
            provenance.get(
                "split_preserved"
            )
            is True
        ),

        "source_split_recorded": (
            provenance.get(
                "source_split_recorded"
            )
            is True
        ),

        "processing_version_recorded": (
            provenance.get(
                "processing_version_recorded"
            )
            is True
        ),

        "raw_data_not_modified": (
            schema.get(
                "raw_data_modified"
            )
            is False
        ),
    }


# ----------------------------------------------------------------------
# STATISTICS VALIDATION
# ----------------------------------------------------------------------

def validate_statistics(
    statistics: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> dict[str, bool]:

    summary = statistics.get(
        "summary",
        {},
    )

    train = results["train"]
    validation = results["validation"]
    test = results["test"]

    total_records = (
        train["output_records"]
        + validation["output_records"]
        + test["output_records"]
    )

    total_image_references = (
        train["total_image_references"]
        + validation["total_image_references"]
        + test["total_image_references"]
    )

    total_duplicate_record_ids = (
        train["output_duplicate_record_ids"]
        + validation["output_duplicate_record_ids"]
        + test["output_duplicate_record_ids"]
    )

    total_duplicate_item_ids = (
        train["output_duplicate_item_ids"]
        + validation["output_duplicate_item_ids"]
        + test["output_duplicate_item_ids"]
    )

    checks = {
        "dataset_correct": (
            statistics.get("dataset")
            == "Amazon Berkeley Objects (ABO)"
        ),

        "stage_correct": (
            statistics.get("stage")
            == "B2"
        ),

        "processing_version_correct": (
            statistics.get("processing_version")
            == "v001"
        ),

        "total_records_match": (
            summary.get("total_records")
            == total_records
        ),

        "total_records_expected": (
            summary.get("total_records")
            == EXPECTED_TOTAL_RECORDS
        ),

        "total_image_references_match": (
            summary.get(
                "total_image_references"
            )
            == total_image_references
        ),

        "duplicate_record_ids_match": (
            summary.get(
                "duplicate_record_ids"
            )
            == total_duplicate_record_ids
        ),

        "duplicate_item_ids_match": (
            summary.get(
                "duplicate_item_ids"
            )
            == total_duplicate_item_ids
        ),

        "record_count_preserved": (
            statistics.get(
                "validation",
                {},
            ).get(
                "record_count_preserved"
            )
            is True
        ),

        "duplicates_removed_false": (
            statistics.get(
                "validation",
                {},
            ).get(
                "duplicates_removed"
            )
            is False
        ),

        "raw_data_modified_false": (
            statistics.get(
                "validation",
                {},
            ).get(
                "raw_data_modified"
            )
            is False
        ),

        "synthetic_values_false": (
            statistics.get(
                "validation",
                {},
            ).get(
                "synthetic_values_created"
            )
            is False
        ),

        "splits_regenerated_false": (
            statistics.get(
                "validation",
                {},
            ).get(
                "splits_regenerated"
            )
            is False
        ),
    }

    return checks


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> int:

    print("=" * 70)
    print("ABO B2 VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------------
    # Required artifacts.
    # --------------------------------------------------------------

    required_files = [
        SCHEMA_FILE,
        STATISTICS_FILE,
    ]

    for split in SPLITS:

        required_files.append(
            INPUT_DIR
            / f"{split}.jsonl"
        )

        required_files.append(
            OUTPUT_DIR
            / f"{split}.jsonl"
        )

    for path in required_files:

        if not path.is_file():

            print(
                f"[FAIL] Required artifact not found: "
                f"{path}"
            )

            return 1

    # --------------------------------------------------------------
    # Load metadata artifacts.
    # --------------------------------------------------------------

    print()
    print("Loading B2 schema...")

    schema = load_json(
        SCHEMA_FILE
    )

    print("Loading B2 statistics...")

    statistics = load_json(
        STATISTICS_FILE
    )

    # --------------------------------------------------------------
    # Schema.
    # --------------------------------------------------------------

    print()
    print("Validating schema...")

    schema_checks = validate_schema(
        schema
    )

    # --------------------------------------------------------------
    # Split lockstep validation.
    # --------------------------------------------------------------

    print()
    print(
        "Validating frozen input -> B2 output..."
    )

    results = {}

    for split in SPLITS:

        print()
        print(
            f"  {split.upper()}"
        )

        result = validate_split(
            split
        )

        results[split] = result

        print(
            f"    Source records : "
            f"{result['source_records']:,}"
        )

        print(
            f"    B2 records     : "
            f"{result['output_records']:,}"
        )

        print(
            f"    Record-ID dup. : "
            f"{result['output_duplicate_record_ids']:,}"
        )

        print(
            f"    Item-ID dup.   : "
            f"{result['output_duplicate_item_ids']:,}"
        )

        print(
            f"    Image refs     : "
            f"{result['total_image_references']:,}"
        )

        print(
            f"    Mismatches     : "
            f"{result['mismatched_records']:,}"
        )

        print(
            f"    Status         : "
            f"{'PASS' if result['status'] else 'FAIL'}"
        )

        if (
            not result["status"]
            and result["first_mismatch"]
        ):

            print(
                "    First mismatch:"
            )

            print(
                json.dumps(
                    result["first_mismatch"],
                    indent=6,
                    ensure_ascii=False,
                )
            )

    # --------------------------------------------------------------
    # Statistics validation.
    # --------------------------------------------------------------

    print()
    print(
        "Validating statistics..."
    )

    statistics_checks = (
        validate_statistics(
            statistics,
            results,
        )
    )

    # --------------------------------------------------------------
    # Overall status.
    # --------------------------------------------------------------

    split_pass = all(
        result["status"]
        for result in results.values()
    )

    schema_pass = all(
        schema_checks.values()
    )

    statistics_pass = all(
        statistics_checks.values()
    )

    overall_pass = (
        split_pass
        and schema_pass
        and statistics_pass
    )

    # --------------------------------------------------------------
    # Validation report.
    # --------------------------------------------------------------

    report = {
        "validator": (
            "abo_b2_validation"
        ),

        "validator_version": "v001",

        "dataset": (
            "Amazon Berkeley Objects (ABO)"
        ),

        "stage": "B2",

        "processing_version": "v001",

        "input_directory": str(
            INPUT_DIR
        ),

        "output_directory": str(
            OUTPUT_DIR
        ),

        "expected": {
            "split_record_counts": (
                EXPECTED_RECORD_COUNTS
            ),

            "total_records": (
                EXPECTED_TOTAL_RECORDS
            ),
        },

        "schema_checks": (
            schema_checks
        ),

        "statistics_checks": (
            statistics_checks
        ),

        "splits": results,

        "overall_status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------------

    total_records = sum(
        result["output_records"]
        for result in results.values()
    )

    total_image_references = sum(
        result["total_image_references"]
        for result in results.values()
    )

    print()
    print("-" * 70)
    print("FINAL SUMMARY")
    print("-" * 70)

    print(
        f"Train        : "
        f"{results['train']['output_records']:,}"
    )

    print(
        f"Validation   : "
        f"{results['validation']['output_records']:,}"
    )

    print(
        f"Test         : "
        f"{results['test']['output_records']:,}"
    )

    print(
        f"Total records: "
        f"{total_records:,}"
    )

    print(
        f"Total image references: "
        f"{total_image_references:,}"
    )

    print(
        "Schema Validation        : "
        f"{'PASS' if schema_pass else 'FAIL'}"
    )

    print(
        "Statistics Validation    : "
        f"{'PASS' if statistics_pass else 'FAIL'}"
    )

    print(
        "Split Lockstep Validation: "
        f"{'PASS' if split_pass else 'FAIL'}"
    )

    print(
        "OVERALL STATUS           : "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    print(
        "Report Output            : "
        f"{REPORT_FILE}"
    )

    return (
        0
        if overall_pass
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )