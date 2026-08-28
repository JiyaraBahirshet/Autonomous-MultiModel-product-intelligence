import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = PROJECT_ROOT / "rakuten-data-challenge" / "rdc-catalog-train.tsv"
TEST_FILE = PROJECT_ROOT / "rakuten-data-challenge" / "rdc-catalog-test.tsv"

REPORT_DIR = PROJECT_ROOT / "reports" / "validation"
REPORT_FILE = REPORT_DIR / "rakuten_hierarchy_validation.json"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rakuten.schema import parse_category_path


def collect_paths(path: Path, has_header: bool) -> tuple[Counter, set[str], dict[str, set[str]]]:
    root_counts = Counter()
    complete_paths = set()
    parent_children = defaultdict(set)

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        if has_header:
            next(reader)

        for row in reader:
            if len(row) != 2:
                continue

            _, category_path = row

            try:
                parts = parse_category_path(category_path)
            except (TypeError, ValueError):
                continue

            root_counts[parts[0]] += 1
            complete_paths.add(category_path)

            for index in range(1, len(parts)):
                parent = ">".join(parts[:index])
                child = parts[index]
                parent_children[parent].add(child)

    return root_counts, complete_paths, parent_children


def main() -> int:
    print("=" * 70)
    print("RAKUTEN HIERARCHY INTEGRITY VALIDATION")
    print("=" * 70)

    train_roots, train_paths, train_relationships = collect_paths(
        TRAIN_FILE,
        has_header=False,
    )

    test_roots, test_paths, test_relationships = collect_paths(
        TEST_FILE,
        has_header=True,
    )

    all_roots = set(train_roots) | set(test_roots)

    test_only_paths = test_paths - train_paths

    parent_child_conflicts = []

    all_parents = set(train_relationships) | set(test_relationships)

    for parent in sorted(all_parents):
        train_children = train_relationships.get(parent, set())
        test_children = test_relationships.get(parent, set())

        if (
            train_children
            and test_children
            and not test_children.issubset(train_children)
        ):
            unexpected = sorted(test_children - train_children)

            for child in unexpected:
                parent_child_conflicts.append(
                    {
                        "parent": parent,
                        "test_child": child,
                        "reason": "child relationship not observed in training",
                    }
                )

    result = {
        "validator": "rakuten_hierarchy_validation",
        "validator_version": "v001",
        "dataset": "Rakuten 2018",
        "train": {
            "root_categories": sorted(train_roots),
            "root_category_count": len(train_roots),
            "unique_complete_paths": len(train_paths),
        },
        "test": {
            "root_categories": sorted(test_roots),
            "root_category_count": len(test_roots),
            "unique_complete_paths": len(test_paths),
        },
        "combined": {
            "root_categories": sorted(all_roots),
            "root_category_count": len(all_roots),
            "test_only_complete_paths": len(test_only_paths),
            "test_only_path_examples": sorted(test_only_paths)[:20],
            "parent_child_conflict_count": len(parent_child_conflicts),
            "parent_child_conflict_examples": parent_child_conflicts[:20],
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_FILE.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print()
    print(f"Report written to: {REPORT_FILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
