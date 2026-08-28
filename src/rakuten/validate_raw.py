import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from rakuten.schema import parse_category_path, category_depth


TRAIN_FILE = PROJECT_ROOT / "rakuten-data-challenge" / "rdc-catalog-train.tsv"
TEST_FILE = PROJECT_ROOT / "rakuten-data-challenge" / "rdc-catalog-test.tsv"

REPORT_DIR = PROJECT_ROOT / "reports" / "validation"
REPORT_FILE = REPORT_DIR / "rakuten_raw_validation.json"


def validate_train(path: Path) -> dict:
    total_rows = 0
    malformed_rows = 0
    empty_titles = 0
    empty_category_paths = 0
    invalid_category_paths = 0

    depth_counts = Counter()
    unique_paths = set()

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter="\t")

        for row_number, row in enumerate(reader, start=1):
            total_rows += 1

            if len(row) != 2:
                malformed_rows += 1
                continue

            title, category_path = row

            if not title.strip():
                empty_titles += 1

            if not category_path.strip():
                empty_category_paths += 1
                continue

            try:
                depth = category_depth(category_path)
                parse_category_path(category_path)
                depth_counts[str(depth)] += 1
                unique_paths.add(category_path)
            except (TypeError, ValueError):
                invalid_category_paths += 1

    return {
        "file": str(path),
        "format": "TSV",
        "header_present": False,
        "total_rows": total_rows,
        "malformed_rows": malformed_rows,
        "empty_titles": empty_titles,
        "empty_category_paths": empty_category_paths,
        "invalid_category_paths": invalid_category_paths,
        "unique_complete_category_paths": len(unique_paths),
        "depth_distribution": dict(sorted(depth_counts.items(), key=lambda x: int(x[0]))),
    }


def validate_test(path: Path) -> dict:
    total_data_rows = 0
    malformed_rows = 0
    empty_titles = 0
    empty_category_paths = 0
    invalid_category_paths = 0

    depth_counts = Counter()
    unique_paths = set()

    expected_header = ["Title", "CategoryIdPath"]

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter="\t")

        try:
            header = next(reader)
        except StopIteration:
            return {
                "file": str(path),
                "format": "TSV",
                "header_present": False,
                "header": None,
                "total_data_rows": 0,
                "malformed_rows": 0,
                "empty_titles": 0,
                "empty_category_paths": 0,
                "invalid_category_paths": 0,
                "unique_complete_category_paths": 0,
                "depth_distribution": {},
            }

        header_correct = header == expected_header

        for row_number, row in enumerate(reader, start=2):
            total_data_rows += 1

            if len(row) != 2:
                malformed_rows += 1
                continue

            title, category_path = row

            if not title.strip():
                empty_titles += 1

            if not category_path.strip():
                empty_category_paths += 1
                continue

            try:
                depth = category_depth(category_path)
                parse_category_path(category_path)
                depth_counts[str(depth)] += 1
                unique_paths.add(category_path)
            except (TypeError, ValueError):
                invalid_category_paths += 1

    return {
        "file": str(path),
        "format": "TSV",
        "header_present": True,
        "header": header,
        "header_matches_expected": header_correct,
        "total_data_rows": total_data_rows,
        "malformed_rows": malformed_rows,
        "empty_titles": empty_titles,
        "empty_category_paths": empty_category_paths,
        "invalid_category_paths": invalid_category_paths,
        "unique_complete_category_paths": len(unique_paths),
        "depth_distribution": dict(sorted(depth_counts.items(), key=lambda x: int(x[0]))),
    }


def main() -> int:
    print("=" * 70)
    print("RAKUTEN RAW INPUT VALIDATION")
    print("=" * 70)

    if not TRAIN_FILE.is_file():
        print(f"[FAIL] Train file not found: {TRAIN_FILE}")
        return 1

    if not TEST_FILE.is_file():
        print(f"[FAIL] Test file not found: {TEST_FILE}")
        return 1

    print(f"Train: {TRAIN_FILE}")
    print(f"Test : {TEST_FILE}")
    print()

    print("Scanning train file...")
    train_result = validate_train(TRAIN_FILE)

    print("Scanning test file...")
    test_result = validate_test(TEST_FILE)

    result = {
        "validator": "rakuten_raw_validation",
        "validator_version": "v001",
        "dataset": "Rakuten 2018",
        "train": train_result,
        "test": test_result,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_FILE.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print()
    print("-" * 70)
    print("TRAIN")
    print("-" * 70)
    print(json.dumps(train_result, indent=2))

    print()
    print("-" * 70)
    print("TEST")
    print("-" * 70)
    print(json.dumps(test_result, indent=2))

    print()
    print(f"Validation report written to:")
    print(REPORT_FILE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
