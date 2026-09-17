import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "splits" / "rakuten"
B2_DIR = PROJECT_ROOT / "data" / "processed" / "rakuten" / "b2"

SCHEMA_FILE = B2_DIR / "schema_v001.json"
STATISTICS_FILE = B2_DIR / "statistics.json"

REPORT_DIR = PROJECT_ROOT / "reports" / "validation"
REPORT_FILE = REPORT_DIR / "rakuten_b2_validation.json"

PROCESSING_VERSION = "v001"
SCHEMA_VERSION = "v001"

SPLITS = (
    "train",
    "validation",
    "test",
)

EXPECTED_COUNTS = {
    "train": 719701,
    "validation": 80299,
    "test": 200000,
}

EXPECTED_TOTAL = 1000000

VALID_SPLIT_ALIASES = {
    "train": {"train"},
    "validation": {"validation", "val", "dev"},
    "test": {"test"},
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def normalize_title_reference(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Rakuten title must be a string")

    return " ".join(value.split()).strip()


def validate_hierarchy_structure(record: dict[str, Any]) -> list[str]:
    """
    Validate internal structural consistency of one processed Rakuten record.
    category_id_path is a '>-separated' string.
    category_path is a list of node strings.
    """
    errors = []

    category_id_path = record.get("category_id_path")
    category_path = record.get("category_path")
    depth = record.get("depth")
    root_category_id = record.get("root_category_id")
    leaf_category_id = record.get("leaf_category_id")

    if not isinstance(category_id_path, str) or not category_id_path.strip():
        errors.append("category_id_path_not_string")

    if not isinstance(category_path, list):
        errors.append("category_path_not_list")

    if not isinstance(depth, int):
        errors.append("depth_not_integer")

    if isinstance(category_path, list) and isinstance(depth, int):
        if len(category_path) != depth:
            errors.append("category_path_depth_mismatch")

    if isinstance(category_id_path, str) and category_id_path.strip():
        id_parts = category_id_path.split(">")
        if isinstance(depth, int) and len(id_parts) != depth:
            errors.append("category_id_path_depth_mismatch")

        if id_parts and str(root_category_id) != id_parts[0]:
            errors.append("root_category_id_mismatch")

        if id_parts and str(leaf_category_id) != id_parts[-1]:
            errors.append("leaf_category_id_mismatch")

    if isinstance(category_path, list) and category_path:
        if str(root_category_id) != str(category_path[0]):
            errors.append("root_category_id_path_mismatch")

        if str(leaf_category_id) != str(category_path[-1]):
            errors.append("leaf_category_id_path_mismatch")

    return errors


def validate_split(split: str) -> dict[str, Any]:
    source_file = INPUT_DIR / f"{split}.jsonl"
    b2_file = B2_DIR / f"{split}.jsonl"

    if not source_file.is_file():
        raise FileNotFoundError(f"Frozen Rakuten split not found: {source_file}")

    if not b2_file.is_file():
        raise FileNotFoundError(f"Rakuten B2 artifact not found: {b2_file}")

    source_records = 0
    b2_records = 0

    source_record_ids = set()
    b2_record_ids = set()

    source_duplicate_record_ids = 0
    b2_duplicate_record_ids = 0

    source_malformed = 0
    b2_malformed = 0

    mismatches = 0
    title_mismatches = 0
    hierarchy_mismatches = 0
    depth_mismatches = 0
    root_mismatches = 0
    leaf_mismatches = 0
    split_mismatches = 0
    frozen_split_errors = 0
    hierarchy_errors = 0

    replacement_character_titles = 0
    short_titles = 0
    changed_titles = 0

    original_title_characters = 0
    processed_title_characters = 0

    valid_aliases = VALID_SPLIT_ALIASES.get(split, {split})

    with source_file.open("r", encoding="utf-8") as src_f, b2_file.open("r", encoding="utf-8") as b2_f:
        for line_num, (src_line, b2_line) in enumerate(zip(src_f, b2_f), start=1):
            src_line = src_line.rstrip("\r\n")
            b2_line = b2_line.rstrip("\r\n")

            if not src_line or not b2_line:
                continue

            try:
                original = json.loads(src_line)
            except json.JSONDecodeError as exc:
                source_malformed += 1
                raise ValueError(f"Malformed JSON in source {source_file}:{line_num}") from exc

            try:
                record = json.loads(b2_line)
            except json.JSONDecodeError as exc:
                b2_malformed += 1
                raise ValueError(f"Malformed JSON in B2 {b2_file}:{line_num}") from exc

            source_records += 1
            b2_records += 1

            src_id = str(original.get("record_id")).strip()
            b2_id = str(record.get("record_id")).strip()

            if src_id in source_record_ids:
                source_duplicate_record_ids += 1
            else:
                source_record_ids.add(src_id)

            if b2_id in b2_record_ids:
                b2_duplicate_record_ids += 1
            else:
                b2_record_ids.add(b2_id)

            if src_id != b2_id:
                mismatches += 1

            # Validate split preservation against source split or valid split aliases
            expected_src_split = original.get("split") or original.get("frozen_split") or split
            record_split = record.get("split")
            if record_split and record_split != expected_src_split and record_split not in valid_aliases:
                split_mismatches += 1
                mismatches += 1

            record_frozen_split = record.get("frozen_split")
            if record_frozen_split and record_frozen_split != expected_src_split and record_frozen_split not in valid_aliases:
                frozen_split_errors += 1
                mismatches += 1

            original_title = original.get("title", "")
            processed_title = record.get("title", "")

            if "\ufffd" in original_title:
                replacement_character_titles += 1

            if len(original_title) <= 3:
                short_titles += 1

            original_title_characters += len(original_title)

            expected_title = normalize_title_reference(original_title)

            if processed_title != expected_title:
                title_mismatches += 1
                mismatches += 1

            if original_title != processed_title:
                changed_titles += 1

            if isinstance(processed_title, str):
                processed_title_characters += len(processed_title)

            hierarchy_fields = (
                "category_id_path",
                "category_path",
                "depth",
                "root_category_id",
                "leaf_category_id",
            )

            hierarchy_changed = False
            for field in hierarchy_fields:
                if canonical_json(original.get(field)) != canonical_json(record.get(field)):
                    hierarchy_changed = True
                    if field == "depth":
                        depth_mismatches += 1
                    elif field == "root_category_id":
                        root_mismatches += 1
                    elif field == "leaf_category_id":
                        leaf_mismatches += 1

            if hierarchy_changed:
                hierarchy_mismatches += 1
                mismatches += 1

            record_hierarchy_errors = validate_hierarchy_structure(record)
            hierarchy_errors += len(record_hierarchy_errors)

    expected_count = EXPECTED_COUNTS[split]

    checks = {
        "source_record_count": source_records == expected_count,
        "b2_record_count": b2_records == expected_count,
        "source_jsonl_valid": source_malformed == 0,
        "b2_jsonl_valid": b2_malformed == 0,
        "source_record_ids_unique": source_duplicate_record_ids == 0,
        "b2_record_ids_unique": b2_duplicate_record_ids == 0,
        "record_id_sets_match": source_record_ids == b2_record_ids,
        "title_transformation_correct": title_mismatches == 0,
        "hierarchy_preserved": hierarchy_mismatches == 0,
        "hierarchy_internal_consistency": hierarchy_errors == 0,
        "split_preserved": split_mismatches == 0,
        "frozen_split_correct": frozen_split_errors == 0,
        "no_unexpected_mismatches": mismatches == 0,
    }

    failed_checks = [check_name for check_name, is_valid in checks.items() if not is_valid]
    if failed_checks:
        print(f"    [FAILED CHECKS for {split.upper()}]: {', '.join(failed_checks)}")

    return {
        "source_records": source_records,
        "b2_records": b2_records,
        "expected_records": expected_count,
        "source_duplicate_record_ids": source_duplicate_record_ids,
        "b2_duplicate_record_ids": b2_duplicate_record_ids,
        "source_malformed_records": source_malformed,
        "b2_malformed_records": b2_malformed,
        "title_mismatches": title_mismatches,
        "hierarchy_mismatches": hierarchy_mismatches,
        "depth_mismatches": depth_mismatches,
        "root_mismatches": root_mismatches,
        "leaf_mismatches": leaf_mismatches,
        "split_mismatches": split_mismatches,
        "frozen_split_errors": frozen_split_errors,
        "hierarchy_errors": hierarchy_errors,
        "replacement_character_titles": replacement_character_titles,
        "short_titles_length_le_3": short_titles,
        "titles_changed_by_whitespace_normalization": changed_titles,
        "original_title_characters": original_title_characters,
        "processed_title_characters": processed_title_characters,
        "mismatches": mismatches,
        "checks": checks,
        "status": "PASS" if not failed_checks else "FAIL",
    }


def validate_statistics(
    statistics: dict[str, Any],
    split_results: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    checks = {}
    actual_total = sum(result["b2_records"] for result in split_results.values())

    checks["total_records"] = statistics.get("total_records") == actual_total == EXPECTED_TOTAL

    for split in SPLITS:
        stored = statistics.get("splits", {}).get(split)
        actual = split_results[split]

        if not isinstance(stored, dict):
            checks[f"{split}_statistics_present"] = False
            continue

        checks[f"{split}_records_processed"] = stored.get("records_processed") == actual["b2_records"]
        checks[f"{split}_unique_record_ids"] = stored.get("unique_record_ids") == (
            actual["b2_records"] - actual["b2_duplicate_record_ids"]
        )
        checks[f"{split}_duplicate_record_ids"] = stored.get("duplicate_record_ids") == actual["b2_duplicate_record_ids"]
        checks[f"{split}_malformed_records"] = stored.get("malformed_records") == actual["b2_malformed_records"]
        checks[f"{split}_replacement_character_titles"] = (
            stored.get("titles_with_replacement_character") == actual["replacement_character_titles"]
        )
        checks[f"{split}_short_titles"] = stored.get("titles_length_le_3") == actual["short_titles_length_le_3"]
        checks[f"{split}_changed_titles"] = (
            stored.get("titles_changed_by_whitespace_normalization") == actual["titles_changed_by_whitespace_normalization"]
        )
        checks[f"{split}_original_title_characters"] = (
            stored.get("original_title_characters") == actual["original_title_characters"]
        )
        checks[f"{split}_processed_title_characters"] = (
            stored.get("processed_title_characters") == actual["processed_title_characters"]
        )
        checks[f"{split}_category_paths_unchanged"] = stored.get("category_paths_changed") == 0
        checks[f"{split}_category_id_paths_unchanged"] = stored.get("category_id_paths_changed") == 0

    calculated_replacement = sum(result["replacement_character_titles"] for result in split_results.values())
    calculated_short = sum(result["short_titles_length_le_3"] for result in split_results.values())
    calculated_changed = sum(result["titles_changed_by_whitespace_normalization"] for result in split_results.values())
    calculated_duplicates = sum(result["b2_duplicate_record_ids"] for result in split_results.values())

    checks["replacement_character_total"] = statistics.get("replacement_character_titles") == calculated_replacement
    checks["short_titles_total"] = statistics.get("short_titles_length_le_3") == calculated_short
    checks["changed_titles_total"] = statistics.get("titles_changed_by_whitespace_normalization") == calculated_changed
    checks["duplicate_record_ids_total"] = statistics.get("duplicate_record_ids") == calculated_duplicates

    invariants = statistics.get("invariants", {})
    checks["record_count_preserved_invariant"] = invariants.get("record_count_preserved") is True
    checks["category_information_preserved_invariant"] = invariants.get("category_information_preserved") is True
    checks["record_ids_preserved_invariant"] = invariants.get("record_ids_preserved") is True
    checks["split_boundaries_preserved_invariant"] = invariants.get("split_boundaries_preserved") is True
    checks["duplicates_preserved_invariant"] = invariants.get("duplicates_preserved") is True
    checks["raw_data_not_modified_invariant"] = invariants.get("raw_data_modified") is False

    failed_stats = [check_name for check_name, is_valid in checks.items() if not is_valid]
    if failed_stats:
        print(f"    [FAILED STATS CHECKS]: {', '.join(failed_stats)}")

    return checks


def validate_schema(schema: dict[str, Any]) -> dict[str, bool]:
    transformations = schema.get("transformations", {})
    checks = {
        "dataset": schema.get("dataset") == "Rakuten 2018",
        "stage": schema.get("stage") == "B2",
        "schema_version": schema.get("schema_version") == SCHEMA_VERSION,
        "processing_version": schema.get("processing_version") == PROCESSING_VERSION,
        "title_transformation": transformations.get("title") == "deterministic whitespace normalization only",
        "category_id_path_preserved": transformations.get("category_id_path") == "preserved exactly",
        "category_path_preserved": transformations.get("category_path") == "preserved exactly",
        "depth_preserved": transformations.get("depth") == "preserved exactly",
        "root_category_preserved": transformations.get("root_category_id") == "preserved exactly",
        "leaf_category_preserved": transformations.get("leaf_category_id") == "preserved exactly",
        "split_preserved": transformations.get("split") == "preserved exactly",
        "record_id_preserved": transformations.get("record_id") == "preserved exactly",
        "no_translation": "translation" not in transformations
        or transformations.get("translation") in (False, "preserved; no translation or ASCII conversion"),
        "raw_data_not_modified": schema.get("raw_data_modified") is False,
    }

    failed_schema = [check_name for check_name, is_valid in checks.items() if not is_valid]
    if failed_schema:
        print(f"    [FAILED SCHEMA CHECKS]: {', '.join(failed_schema)}")

    return checks


def main() -> int:
    print("=" * 70)
    print("RAKUTEN B2 VALIDATION")
    print("=" * 70)

    required_files = [SCHEMA_FILE, STATISTICS_FILE]
    for split in SPLITS:
        required_files.append(INPUT_DIR / f"{split}.jsonl")
        required_files.append(B2_DIR / f"{split}.jsonl")

    for path in required_files:
        if not path.is_file():
            print(f"[FAIL] Required artifact not found: {path}")
            return 1

    print("\nLoading B2 schema...")
    schema = load_json(SCHEMA_FILE)

    print("Loading B2 statistics...")
    statistics = load_json(STATISTICS_FILE)

    print("\nValidating schema...")
    schema_checks = validate_schema(schema)
    schema_pass = all(schema_checks.values())
    print(f"Schema Validation        : {'PASS' if schema_pass else 'FAIL'}")

    print("\nValidating frozen input -> B2 output...")
    split_results = {}

    for split in SPLITS:
        print(f"\n  {split.upper()}")
        result = validate_split(split)
        split_results[split] = result

        print(f"    Source records : {result['source_records']:,}")
        print(f"    B2 records     : {result['b2_records']:,}")
        print(f"    Record-ID dup. : {result['b2_duplicate_record_ids']:,}")
        print(f"    Hierarchy err. : {result['hierarchy_errors']:,}")
        print(f"    Title mismatch : {result['title_mismatches']:,}")
        print(f"    Mismatches     : {result['mismatches']:,}")
        print(f"    Status         : {result['status']}")

    split_lockstep_pass = all(result["status"] == "PASS" for result in split_results.values())

    print("\nValidating statistics...")
    statistics_checks = validate_statistics(statistics, split_results)
    statistics_pass = all(statistics_checks.values())
    print(f"Statistics Validation    : {'PASS' if statistics_pass else 'FAIL'}")

    overall_pass = schema_pass and statistics_pass and split_lockstep_pass

    report = {
        "validator": "rakuten_b2_validation",
        "validator_version": "v001",
        "dataset": "Rakuten 2018",
        "stage": "B2",
        "schema": {"checks": schema_checks, "status": "PASS" if schema_pass else "FAIL"},
        "splits": split_results,
        "statistics": {"checks": statistics_checks, "status": "PASS" if statistics_pass else "FAIL"},
        "summary": {
            "train_records": split_results["train"]["b2_records"],
            "validation_records": split_results["validation"]["b2_records"],
            "test_records": split_results["test"]["b2_records"],
            "total_records": sum(result["b2_records"] for result in split_results.values()),
        },
        "validation": {
            "schema_validation": schema_pass,
            "statistics_validation": statistics_pass,
            "split_lockstep_validation": split_lockstep_pass,
            "overall_status": "PASS" if overall_pass else "FAIL",
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "-" * 70)
    print("FINAL SUMMARY")
    print("-" * 70)
    for split in SPLITS:
        print(f"{split.capitalize():12}: {split_results[split]['b2_records']:,}")

    print(f"Total records : {sum(result['b2_records'] for result in split_results.values()):,}")
    print(f"Schema Validation        : {'PASS' if schema_pass else 'FAIL'}")
    print(f"Statistics Validation    : {'PASS' if statistics_pass else 'FAIL'}")
    print(f"Split Lockstep Validation: {'PASS' if split_lockstep_pass else 'FAIL'}")
    print(f"OVERALL STATUS           : {'PASS' if overall_pass else 'FAIL'}\n")
    print(f"Report Output            : {REPORT_FILE}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())