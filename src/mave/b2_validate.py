import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = PROJECT_ROOT / "data" / "splits" / "mave"
B2_DIR = PROJECT_ROOT / "data" / "processed" / "mave" / "b2"

SCHEMA_PATH = B2_DIR / "schema_v001.json"
STATISTICS_PATH = B2_DIR / "statistics.json"

REPORT_DIR = PROJECT_ROOT / "reports" / "validation"
REPORT_PATH = REPORT_DIR / "mave_b2_validation.json"

EXPECTED_COUNTS = {
    "train": 2_326_033,
    "validation": 290_488,
    "test": 290_837,
}

EXPECTED_TOTAL_RECORDS = 2_907_358

APPROVED_TEXT_FIELDS = {
    "title",
    "description",
    "brand",
    "amazon_category_path",
    "amazon_main_category",
}


def normalize_text(value: str) -> str:
    """Deterministic whitespace normalization used by MAVE B2."""
    return " ".join(value.split())


def normalize_value(value: Any, field_name: str | None = None) -> Any:
    """Apply approved field whitespace normalization recursively."""
    if isinstance(value, str):
        if field_name in APPROVED_TEXT_FIELDS:
            return normalize_text(value)
        return value

    if isinstance(value, list):
        return [normalize_value(item, field_name) for item in value]

    if isinstance(value, dict):
        return {key: normalize_value(item, key) for key, item in value.items()}

    return value


def make_record_id(asin: str) -> str:
    """Calculate expected record_id."""
    digest = hashlib.sha256(asin.encode("utf-8")).hexdigest()[:16]
    return f"mave:{digest}"


def load_json(path: Path) -> Dict[str, Any]:
    """Load and parse JSON file from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return data


def validate_schema(schema: Dict[str, Any]) -> Dict[str, bool]:
    """Validate schema contract adherence."""
    transformations = schema.get("transformations", {})

    return {
        "dataset": schema.get("dataset") == "MAVE",
        "stage": schema.get("stage") == "B2",
        "schema_version": schema.get("schema_version") == "v001",
        "processing_version": schema.get("processing_version") == "v001",
        "record_artifacts": schema.get("record_artifacts") == [
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
        ],
        "approved_text_fields": set(
            transformations.get("approved_text_fields", [])
        ) == APPROVED_TEXT_FIELDS,
        "text_transformation": transformations.get("text")
        == "deterministic whitespace normalization only",
        "learned_statistics": schema.get("learned_statistics") is False,
        "train_only_fitting_required": schema.get("train_only_fitting_required") is False,
        "raw_or_frozen_split_modified": schema.get("raw_or_frozen_split_modified") is False,
    }


def validate_split(
    split: str,
    global_asin_tracker: Set[str],
) -> Tuple[Dict[str, Any], Set[str]]:
    """Stream and validate B2 output against its frozen source in streaming lockstep."""
    source_path = SOURCE_DIR / f"{split}.jsonl"
    b2_path = B2_DIR / f"{split}.jsonl"

    if not source_path.is_file():
        raise FileNotFoundError(f"Source split not found: {source_path}")
    if not b2_path.is_file():
        raise FileNotFoundError(f"B2 split not found: {b2_path}")

    result = {
        "split": split,
        "expected_records": EXPECTED_COUNTS[split],
        "source_records": 0,
        "b2_records": 0,
        "source_malformed": 0,
        "b2_malformed": 0,
        "source_missing_asin": 0,
        "b2_missing_asin": 0,
        "b2_duplicate_asins": 0,
        "cross_split_leakage_count": 0,
        "asin_mismatches": 0,
        "field_mismatches": 0,
        "record_id_errors": 0,
        "provenance_errors": 0,
        "supervision_errors": 0,
        "structure_errors": 0,
        "record_mismatches": 0,
        "validation": {},
    }

    split_asin_seen: Set[str] = set()

    print()
    print("=" * 70)
    print(f"VALIDATING {split.upper()} SPLIT (STREAMING LOCKSTEP)")
    print("=" * 70)

    with (
        source_path.open("r", encoding="utf-8") as src_file,
        b2_path.open("r", encoding="utf-8") as b2_file,
    ):
        line_idx = 0
        while True:
            src_line = src_file.readline()
            b2_line = b2_file.readline()

            if not src_line and not b2_line:
                break

            line_idx += 1

            if bool(src_line) != bool(b2_line):
                result["structure_errors"] += 1
                result["record_mismatches"] += 1
                if src_line:
                    result["source_records"] += 1
                if b2_line:
                    result["b2_records"] += 1
                continue

            src_str = src_line.strip()
            b2_str = b2_line.strip()

            if not src_str and not b2_str:
                continue

            src_rec, b2_rec = None, None
            try:
                src_rec = json.loads(src_str)
            except Exception:
                result["source_malformed"] += 1

            try:
                b2_rec = json.loads(b2_str)
            except Exception:
                result["b2_malformed"] += 1

            if src_rec is not None:
                result["source_records"] += 1
            if b2_rec is not None:
                result["b2_records"] += 1

            if not isinstance(src_rec, dict) or not isinstance(b2_rec, dict):
                result["structure_errors"] += 1
                result["record_mismatches"] += 1
                continue

            src_asin = src_rec.get("asin")
            b2_asin = b2_rec.get("asin")

            if not src_asin:
                result["source_missing_asin"] += 1
            if not b2_asin:
                result["b2_missing_asin"] += 1

            src_asin_str = str(src_asin) if src_asin is not None else ""
            b2_asin_str = str(b2_asin) if b2_asin is not None else ""

            is_asin_mismatch = (src_asin_str != b2_asin_str) or not b2_asin_str
            if is_asin_mismatch:
                result["asin_mismatches"] += 1

            if b2_asin_str:
                if b2_asin_str in split_asin_seen:
                    result["b2_duplicate_asins"] += 1
                else:
                    split_asin_seen.add(b2_asin_str)

                if b2_asin_str in global_asin_tracker:
                    result["cross_split_leakage_count"] += 1

            expected_record_id = make_record_id(src_asin_str) if src_asin_str else ""
            is_record_id_error = b2_rec.get("record_id") != expected_record_id
            if is_record_id_error:
                result["record_id_errors"] += 1

            expected_provenance = {
                "processing_version": "v001",
                "source_asin": src_asin_str,
            }
            is_provenance_error = b2_rec.get("b2_provenance") != expected_provenance
            if is_provenance_error:
                result["provenance_errors"] += 1

            is_field_mismatch = False
            for key, val in src_rec.items():
                expected_val = normalize_value(val, key)
                actual_val = b2_rec.get(key)
                if actual_val != expected_val:
                    is_field_mismatch = True
                    break

            if is_field_mismatch:
                result["field_mismatches"] += 1

            expected_keys = set(src_rec.keys()) | {
                "record_id",
                "b2_provenance",
                "supervision",
            }
            is_structure_error = set(b2_rec.keys()) != expected_keys
            if is_structure_error:
                result["structure_errors"] += 1

            expected_supervision = {
                "mave_information_available": src_rec.get("mave_information_available") is True,
                "label_available": bool(
                    src_rec.get("mave_category") or src_rec.get("mave_attributes")
                ),
            }
            is_supervision_error = b2_rec.get("supervision") != expected_supervision
            if is_supervision_error:
                result["supervision_errors"] += 1

            # Count a record mismatch once at record level if any field/validation checks failed
            if (
                is_asin_mismatch
                or is_field_mismatch
                or is_record_id_error
                or is_provenance_error
                or is_structure_error
                or is_supervision_error
            ):
                result["record_mismatches"] += 1

            if line_idx % 100_000 == 0:
                print(f"  Processed {line_idx:,} records...")

    result["validation"] = {
        "expected_record_count": (
            result["source_records"] == EXPECTED_COUNTS[split]
            and result["b2_records"] == EXPECTED_COUNTS[split]
        ),
        "source_jsonl_valid": result["source_malformed"] == 0,
        "b2_jsonl_valid": result["b2_malformed"] == 0,
        "source_asins_present": result["source_missing_asin"] == 0,
        "b2_asins_present": result["b2_missing_asin"] == 0,
        "b2_asins_unique": result["b2_duplicate_asins"] == 0,
        "cross_split_leakage_free": result["cross_split_leakage_count"] == 0,
        "asin_values_preserved": result["asin_mismatches"] == 0,
        "source_fields_preserved": result["field_mismatches"] == 0,
        "record_ids_valid": result["record_id_errors"] == 0,
        "provenance_valid": result["provenance_errors"] == 0,
        "supervision_valid": result["supervision_errors"] == 0,
        "field_structure_preserved": result["structure_errors"] == 0,
        "records_preserved": result["record_mismatches"] == 0,
    }

    result["overall_valid"] = all(result["validation"].values())
    print(f"Completed {split}. Records: {result['b2_records']:,} | Status: {'PASS' if result['overall_valid'] else 'FAIL'}")
    return result, split_asin_seen


def validate_statistics(statistics: Dict[str, Any]) -> Dict[str, bool]:
    """Validate global and split statistics counts against frozen specs."""
    summary = statistics.get("summary", {})
    split_stats = statistics.get("split_statistics", {})

    checks = {
        "total_records": summary.get("total_records") == EXPECTED_TOTAL_RECORDS,
        "total_unique_asins": summary.get("total_unique_asins") == EXPECTED_TOTAL_RECORDS,
        "missing_asin": summary.get("missing_asin") == 0,
        "duplicate_asins": summary.get("duplicate_asins") == 0,
        "malformed_records": summary.get("malformed_records") == 0,
        "text_values_changed": summary.get("text_values_changed") == 0,
    }

    for split in ("train", "validation", "test"):
        st = split_stats.get(split, {})
        prefix = f"{split}_"
        checks[f"{prefix}records"] = st.get("records_processed") == EXPECTED_COUNTS[split]
        checks[f"{prefix}unique_asins"] = st.get("unique_asins") == EXPECTED_COUNTS[split]
        checks[f"{prefix}missing_asin"] = st.get("missing_asin") == 0
        checks[f"{prefix}duplicate_asins"] = st.get("duplicate_asins") == 0
        checks[f"{prefix}malformed_records"] = st.get("malformed_records") == 0
        checks[f"{prefix}text_values_changed"] = st.get("text_values_changed") == 0

    return checks


def main() -> None:
    print("=" * 70)
    print("MAVE B2 VALIDATION RUNNER")
    print("=" * 70)

    if not B2_DIR.is_dir():
        raise FileNotFoundError(f"B2 directory not found: {B2_DIR}")
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Schema path not found: {SCHEMA_PATH}")
    if not STATISTICS_PATH.is_file():
        raise FileNotFoundError(f"Statistics path not found: {STATISTICS_PATH}")

    schema = load_json(SCHEMA_PATH)
    statistics = load_json(STATISTICS_PATH)

    schema_checks = validate_schema(schema)
    schema_valid = all(schema_checks.values())

    global_asin_tracker: Set[str] = set()
    split_results: Dict[str, Any] = {}
    cross_split_leakage_detected = False

    for split in ("train", "validation", "test"):
        res, split_asins = validate_split(split, global_asin_tracker)
        if res["cross_split_leakage_count"] > 0:
            cross_split_leakage_detected = True
        global_asin_tracker.update(split_asins)
        split_results[split] = res

    splits_valid = all(r["overall_valid"] for r in split_results.values()) and not cross_split_leakage_detected
    statistics_checks = validate_statistics(statistics)
    statistics_valid = all(statistics_checks.values())

    actual_total_records = sum(split_results[s]["b2_records"] for s in ("train", "validation", "test"))
    record_count_valid = actual_total_records == EXPECTED_TOTAL_RECORDS

    validation_summary = {
        "schema_valid": schema_valid,
        "splits_valid": splits_valid,
        "statistics_consistent": statistics_valid,
        "record_count_valid": record_count_valid,
        "cross_split_leakage_free": not cross_split_leakage_detected,
    }

    overall_valid = all(validation_summary.values())

    report = {
        "validator": "mave_b2_validation",
        "validator_version": "v001",
        "dataset": "MAVE",
        "stage": "B2",
        "expected_counts": EXPECTED_COUNTS,
        "actual_total_records": actual_total_records,
        "schema_checks": schema_checks,
        "statistics_checks": statistics_checks,
        "split_results": split_results,
        "validation": validation_summary,
        "overall_valid": overall_valid,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total Records Validated : {actual_total_records:,} / {EXPECTED_TOTAL_RECORDS:,}")
    print(f"Schema Validation       : {'PASS' if schema_valid else 'FAIL'}")
    print(f"Split Validation        : {'PASS' if splits_valid else 'FAIL'}")
    print(f"Statistics Consistency  : {'PASS' if statistics_valid else 'FAIL'}")
    print(f"OVERALL STATUS          : {'PASS' if overall_valid else 'FAIL'}")
    print(f"Report Output           : {REPORT_PATH}")

    if not overall_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()