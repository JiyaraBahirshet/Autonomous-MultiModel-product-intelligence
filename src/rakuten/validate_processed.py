import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "rakuten"

TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
TEST_FILE = PROCESSED_DIR / "test.jsonl"
HIERARCHY_FILE = PROCESSED_DIR / "hierarchy.json"
SCHEMA_FILE = PROCESSED_DIR / "schema_v001.json"

REPORT_DIR = PROJECT_ROOT / "reports" / "validation"
REPORT_FILE = REPORT_DIR / "rakuten_processed_validation.json"


EXPECTED_TRAIN_RECORDS = 800_000
EXPECTED_TEST_RECORDS = 200_000
EXPECTED_TOTAL_RECORDS = 1_000_000

EXPECTED_HIERARCHY_NODES = 3_695

EXPECTED_TRAIN_DEPTHS = {
    "1": 8172,
    "2": 2792,
    "3": 228888,
    "4": 344472,
    "5": 166165,
    "6": 45253,
    "7": 4197,
    "8": 61,
}

EXPECTED_TEST_DEPTHS = {
    "1": 2094,
    "2": 780,
    "3": 59356,
    "4": 85153,
    "5": 40948,
    "6": 10806,
    "7": 855,
    "8": 8,
}

EXPECTED_ROOT_CATEGORIES = {
    "1208",
    "1395",
    "1608",
    "2075",
    "2199",
    "2296",
    "3093",
    "3292",
    "3625",
    "3730",
    "4015",
    "4238",
    "4564",
    "92",
}


REQUIRED_RECORD_FIELDS = {
    "record_id",
    "title",
    "category_id_path",
    "category_path",
    "depth",
    "root_category_id",
    "leaf_category_id",
    "split",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def validate_jsonl(
    path: Path,
    expected_split: str,
    expected_count: int,
    expected_depths: dict[str, int],
) -> dict:

    total_records = 0
    malformed_lines = 0

    missing_required_fields = 0
    invalid_record_ids = 0
    duplicate_record_ids = set()

    record_ids = set()

    empty_titles = 0
    empty_category_id_paths = 0

    invalid_category_paths = 0
    invalid_depths = 0
    invalid_root_categories = 0
    invalid_leaf_categories = 0
    invalid_split = 0

    depth_counts = Counter()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            if not isinstance(record, dict):
                malformed_lines += 1
                continue

            total_records += 1

            missing_fields = (
                REQUIRED_RECORD_FIELDS - set(record.keys())
            )

            if missing_fields:
                missing_required_fields += 1

            record_id = record.get("record_id")

            if not isinstance(record_id, str) or not record_id:
                invalid_record_ids += 1
            else:
                if record_id in record_ids:
                    duplicate_record_ids.add(record_id)
                else:
                    record_ids.add(record_id)

            title = record.get("title")

            if not isinstance(title, str) or not title.strip():
                empty_titles += 1

            category_id_path = record.get("category_id_path")

            if (
                not isinstance(category_id_path, str)
                or not category_id_path.strip()
            ):
                empty_category_id_paths += 1

            category_path = record.get("category_path")

            if not isinstance(category_path, list):
                invalid_category_paths += 1
            else:
                if not all(
                    isinstance(node, str) and node.strip()
                    for node in category_path
                ):
                    invalid_category_paths += 1

                expected_depth = len(category_path)

                if record.get("depth") != expected_depth:
                    invalid_depths += 1

                if category_path:
                    if (
                        record.get("root_category_id")
                        != category_path[0]
                    ):
                        invalid_root_categories += 1

                    if (
                        record.get("leaf_category_id")
                        != category_path[-1]
                    ):
                        invalid_leaf_categories += 1

                    depth_counts[str(expected_depth)] += 1

            if record.get("split") != expected_split:
                invalid_split += 1

    return {
        "file": str(path),
        "expected_split": expected_split,
        "total_records": total_records,
        "malformed_lines": malformed_lines,
        "missing_required_fields": missing_required_fields,
        "invalid_record_ids": invalid_record_ids,
        "unique_record_ids": len(record_ids),
        "duplicate_record_ids": len(duplicate_record_ids),
        "empty_titles": empty_titles,
        "empty_category_id_paths": empty_category_id_paths,
        "invalid_category_paths": invalid_category_paths,
        "invalid_depths": invalid_depths,
        "invalid_root_categories": invalid_root_categories,
        "invalid_leaf_categories": invalid_leaf_categories,
        "invalid_split": invalid_split,
        "depth_distribution": dict(
            sorted(
                depth_counts.items(),
                key=lambda item: int(item[0]),
            )
        ),
    }


def validate_hierarchy() -> dict:
    hierarchy = load_json(HIERARCHY_FILE)

    nodes = hierarchy.get("nodes")

    if not isinstance(nodes, dict):
        return {
            "valid_structure": False,
            "node_count": 0,
            "root_categories": [],
            "invalid_nodes": 1,
        }

    root_categories = set()
    invalid_nodes = 0

    for node_id, node in nodes.items():

        if not isinstance(node, dict):
            invalid_nodes += 1
            continue

        if node.get("node_id") != node_id:
            invalid_nodes += 1

        depth = node.get("depth")
        parent = node.get("parent")

        if not isinstance(depth, int) or depth < 1:
            invalid_nodes += 1

        if depth == 1:
            if parent is not None:
                invalid_nodes += 1

            root_categories.add(node_id)

        else:
            if not isinstance(parent, str) or not parent:
                invalid_nodes += 1

    return {
        "valid_structure": invalid_nodes == 0,
        "node_count": len(nodes),
        "root_categories": sorted(root_categories),
        "root_category_count": len(root_categories),
        "invalid_nodes": invalid_nodes,
    }


def validate_schema() -> dict:
    schema = load_json(SCHEMA_FILE)

    return {
        "dataset_correct": (
            schema.get("dataset")
            == "Rakuten 2018"
        ),
        "schema_version_correct": (
            schema.get("schema_version")
            == "v001"
        ),
        "record_fields_present": (
            set(schema.get("record_fields", {}).keys())
            == REQUIRED_RECORD_FIELDS
        ),
        "split_values_correct": (
            schema.get("split_values")
            == ["train", "test"]
        ),
        "separator_correct": (
            schema.get("category_path_separator")
            == ">"
        ),
        "raw_data_not_modified": (
            schema.get("raw_data_modified") is False
        ),
    }


def main() -> int:

    print("=" * 70)
    print("RAKUTEN PROCESSED ARTIFACT VALIDATION")
    print("=" * 70)

    required_files = [
        TRAIN_FILE,
        TEST_FILE,
        HIERARCHY_FILE,
        SCHEMA_FILE,
    ]

    for path in required_files:
        if not path.is_file():
            print(f"[FAIL] Required artifact not found: {path}")
            return 1

    print()
    print("Validating train.jsonl...")

    train = validate_jsonl(
        TRAIN_FILE,
        "train",
        EXPECTED_TRAIN_RECORDS,
        EXPECTED_TRAIN_DEPTHS,
    )

    print("Validating test.jsonl...")

    test = validate_jsonl(
        TEST_FILE,
        "test",
        EXPECTED_TEST_RECORDS,
        EXPECTED_TEST_DEPTHS,
    )

    print("Validating hierarchy.json...")

    hierarchy = validate_hierarchy()

    print("Validating schema_v001.json...")

    schema = validate_schema()

    checks = {
        "train_record_count": (
            train["total_records"]
            == EXPECTED_TRAIN_RECORDS
        ),

        "test_record_count": (
            test["total_records"]
            == EXPECTED_TEST_RECORDS
        ),

        "total_record_count": (
            train["total_records"]
            + test["total_records"]
            == EXPECTED_TOTAL_RECORDS
        ),

        "train_jsonl_valid": (
            train["malformed_lines"] == 0
        ),

        "test_jsonl_valid": (
            test["malformed_lines"] == 0
        ),

        "train_required_fields_present": (
            train["missing_required_fields"] == 0
        ),

        "test_required_fields_present": (
            test["missing_required_fields"] == 0
        ),

        "train_record_ids_valid": (
            train["invalid_record_ids"] == 0
        ),

        "test_record_ids_valid": (
            test["invalid_record_ids"] == 0
        ),

        "train_record_ids_unique": (
            train["duplicate_record_ids"] == 0
        ),

        "test_record_ids_unique": (
            test["duplicate_record_ids"] == 0
        ),

        "train_titles_present": (
            train["empty_titles"] == 0
        ),

        "test_titles_present": (
            test["empty_titles"] == 0
        ),

        "train_category_paths_valid": (
            train["invalid_category_paths"] == 0
        ),

        "test_category_paths_valid": (
            test["invalid_category_paths"] == 0
        ),

        "train_depths_valid": (
            train["invalid_depths"] == 0
        ),

        "test_depths_valid": (
            test["invalid_depths"] == 0
        ),

        "train_root_categories_valid": (
            train["invalid_root_categories"] == 0
        ),

        "test_root_categories_valid": (
            test["invalid_root_categories"] == 0
        ),

        "train_leaf_categories_valid": (
            train["invalid_leaf_categories"] == 0
        ),

        "test_leaf_categories_valid": (
            test["invalid_leaf_categories"] == 0
        ),

        "train_split_values_valid": (
            train["invalid_split"] == 0
        ),

        "test_split_values_valid": (
            test["invalid_split"] == 0
        ),

        "train_depth_distribution": (
            train["depth_distribution"]
            == EXPECTED_TRAIN_DEPTHS
        ),

        "test_depth_distribution": (
            test["depth_distribution"]
            == EXPECTED_TEST_DEPTHS
        ),

        "hierarchy_structure_valid": (
            hierarchy["valid_structure"]
        ),

        "hierarchy_node_count": (
            hierarchy["node_count"]
            == EXPECTED_HIERARCHY_NODES
        ),

        "hierarchy_root_categories": (
            set(hierarchy["root_categories"])
            == EXPECTED_ROOT_CATEGORIES
        ),

        "hierarchy_root_count": (
            hierarchy["root_category_count"] == 14
        ),

        "schema_dataset_correct": (
            schema["dataset_correct"]
        ),

        "schema_version_correct": (
            schema["schema_version_correct"]
        ),

        "schema_record_fields_correct": (
            schema["record_fields_present"]
        ),

        "schema_split_values_correct": (
            schema["split_values_correct"]
        ),

        "schema_separator_correct": (
            schema["separator_correct"]
        ),

        "raw_data_not_modified": (
            schema["raw_data_not_modified"]
        ),
    }

    all_checks_pass = all(checks.values())

    report = {
        "validator": "rakuten_processed_validation",
        "validator_version": "v001",
        "dataset": "Rakuten 2018",

        "artifacts": {
            "train": str(TRAIN_FILE),
            "test": str(TEST_FILE),
            "hierarchy": str(HIERARCHY_FILE),
            "schema": str(SCHEMA_FILE),
        },

        "train": train,
        "test": test,
        "hierarchy": hierarchy,
        "schema": schema,

        "expected": {
            "train_records": EXPECTED_TRAIN_RECORDS,
            "test_records": EXPECTED_TEST_RECORDS,
            "total_records": EXPECTED_TOTAL_RECORDS,
            "hierarchy_nodes": EXPECTED_HIERARCHY_NODES,
            "root_categories": sorted(
                EXPECTED_ROOT_CATEGORIES
            ),
            "train_depth_distribution": (
                EXPECTED_TRAIN_DEPTHS
            ),
            "test_depth_distribution": (
                EXPECTED_TEST_DEPTHS
            ),
        },

        "checks": checks,

        "validation": {
            "all_checks_pass": all_checks_pass,
            "train_duplicate_record_ids": (
                train["duplicate_record_ids"]
            ),
            "test_duplicate_record_ids": (
                test["duplicate_record_ids"]
            ),
        },
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

    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(
        f"Train records           : "
        f"{train['total_records']:,}"
    )

    print(
        f"Test records            : "
        f"{test['total_records']:,}"
    )

    print(
        f"Total records           : "
        f"{train['total_records'] + test['total_records']:,}"
    )

    print(
        f"Train duplicate IDs     : "
        f"{train['duplicate_record_ids']:,}"
    )

    print(
        f"Test duplicate IDs      : "
        f"{test['duplicate_record_ids']:,}"
    )

    print(
        f"Hierarchy nodes         : "
        f"{hierarchy['node_count']:,}"
    )

    print(
        f"Root categories         : "
        f"{hierarchy['root_category_count']:,}"
    )

    print()
    print(
        "VALIDATION STATUS       : "
        f"{'PASS' if all_checks_pass else 'FAIL'}"
    )

    print()
    print(
        f"Report written to: {REPORT_FILE}"
    )

    return 0 if all_checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())