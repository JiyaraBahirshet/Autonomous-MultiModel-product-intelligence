import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from rakuten.schema import build_record


TRAIN_FILE = (
    PROJECT_ROOT
    / "rakuten-data-challenge"
    / "rdc-catalog-train.tsv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "rakuten-data-challenge"
    / "rdc-catalog-test.tsv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "rakuten"

TRAIN_OUTPUT = OUTPUT_DIR / "train.jsonl"
TEST_OUTPUT = OUTPUT_DIR / "test.jsonl"
HIERARCHY_OUTPUT = OUTPUT_DIR / "hierarchy.json"
SCHEMA_OUTPUT = OUTPUT_DIR / "schema_v001.json"


def normalize_title(title: str) -> str:
    """
    Deterministically normalize title whitespace.

    No semantic text transformations are performed.
    """
    return " ".join(title.split())


def process_train():
    records = 0
    hierarchy = {}

    with (
        TRAIN_FILE.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as source,
        TRAIN_OUTPUT.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as target,
    ):
        for row_number, line in enumerate(source, start=1):
            line = line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"Malformed train row at line {row_number}"
                )

            title, category_id_path = parts

            title = normalize_title(title)

            record = build_record(
                title=title,
                category_id_path=category_id_path,
                split="train",
            )

            record_id = f"train:{records}"

            output = {
                "record_id": record_id,
                "title": record.title,
                "category_id_path": record.category_id_path,
                "category_path": list(record.category_path),
                "depth": record.depth,
                "root_category_id": record.category_path[0],
                "leaf_category_id": record.category_path[-1],
                "split": record.split,
            }

            target.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            add_hierarchy_path(
                hierarchy,
                record.category_path,
            )

            records += 1

    return records, hierarchy


def process_test(hierarchy):
    records = 0

    with (
        TEST_FILE.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as source,
        TEST_OUTPUT.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as target,
    ):
        header = source.readline().rstrip("\r\n")

        if header.split("\t") != ["Title", "CategoryIdPath"]:
            raise ValueError(
                f"Unexpected test header: {header!r}"
            )

        for row_number, line in enumerate(source, start=2):
            line = line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"Malformed test row at line {row_number}"
                )

            title, category_id_path = parts

            title = normalize_title(title)

            record = build_record(
                title=title,
                category_id_path=category_id_path,
                split="test",
            )

            record_id = f"test:{records}"

            output = {
                "record_id": record_id,
                "title": record.title,
                "category_id_path": record.category_id_path,
                "category_path": list(record.category_path),
                "depth": record.depth,
                "root_category_id": record.category_path[0],
                "leaf_category_id": record.category_path[-1],
                "split": record.split,
            }

            target.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            add_hierarchy_path(
                hierarchy,
                record.category_path,
            )

            records += 1

    return records


def add_hierarchy_path(hierarchy, category_path):
    """
    Add every prefix of a complete category path
    to the hierarchy structure.
    """

    for index in range(len(category_path)):
        path = tuple(category_path[: index + 1])
        path_key = ">".join(path)

        parent = (
            ">"
            .join(category_path[:index])
            if index > 0
            else None
        )

        if path_key not in hierarchy:
            hierarchy[path_key] = {
                "node_id": path_key,
                "parent": parent,
                "depth": index + 1,
            }


def build_hierarchy_artifact(hierarchy):
    nodes = {}

    for path_key in sorted(hierarchy):
        nodes[path_key] = hierarchy[path_key]

    return {
        "dataset": "Rakuten 2018",
        "schema_version": "v001",
        "category_path_separator": ">",
        "nodes": nodes,
        "node_count": len(nodes),
    }


def write_schema():
    schema = {
        "dataset": "Rakuten 2018",
        "schema_version": "v001",
        "record_fields": {
            "record_id": "string",
            "title": "string",
            "category_id_path": "string",
            "category_path": "array[string]",
            "depth": "integer",
            "root_category_id": "string",
            "leaf_category_id": "string",
            "split": "string",
        },
        "split_values": [
            "train",
            "test",
        ],
        "category_path_separator": ">",
        "transformations": {
            "title": "deterministic whitespace normalization",
            "category_id_path": "preserved",
            "category_path": "derived by parsing category_id_path",
            "depth": "derived from category_path length",
            "root_category_id": "first category node",
            "leaf_category_id": "last category node",
            "split": "preserved from source dataset",
        },
        "raw_data_modified": False,
    }

    SCHEMA_OUTPUT.write_text(
        json.dumps(
            schema,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    print("=" * 70)
    print("RAKUTEN PHASE A PREPROCESSING")
    print("=" * 70)

    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(TRAIN_FILE)

    if not TEST_FILE.is_file():
        raise FileNotFoundError(TEST_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Processing train...")

    train_count, hierarchy = process_train()

    print(
        f"  Train records written: "
        f"{train_count:,}"
    )

    print()
    print("Processing test...")

    test_count = process_test(hierarchy)

    print(
        f"  Test records written: "
        f"{test_count:,}"
    )

    hierarchy_artifact = build_hierarchy_artifact(
        hierarchy
    )

    HIERARCHY_OUTPUT.write_text(
        json.dumps(
            hierarchy_artifact,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_schema()

    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(
        f"Train records : {train_count:,}"
    )

    print(
        f"Test records  : {test_count:,}"
    )

    print(
        f"Total records : "
        f"{train_count + test_count:,}"
    )

    print(
        f"Hierarchy nodes: "
        f"{len(hierarchy):,}"
    )

    print()
    print("Outputs:")
    print(f"  {TRAIN_OUTPUT}")
    print(f"  {TEST_OUTPUT}")
    print(f"  {HIERARCHY_OUTPUT}")
    print(f"  {SCHEMA_OUTPUT}")


if __name__ == "__main__":
    main()