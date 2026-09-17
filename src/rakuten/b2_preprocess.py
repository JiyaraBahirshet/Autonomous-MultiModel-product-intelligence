import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "splits" / "rakuten"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "rakuten" / "b2"
REPORT_DIR = PROJECT_ROOT / "reports" / "preprocessing"

SPLITS = ("train", "validation", "test")

PROCESSING_VERSION = "v001"
SCHEMA_VERSION = "v001"


def normalize_title(title: str) -> str:
    """
    Deterministic Rakuten title normalization.

    Only whitespace normalization is performed.
    Unicode characters, digits, punctuation, symbols,
    and replacement characters are preserved.
    """
    if not isinstance(title, str):
        raise ValueError("Rakuten title must be a string")

    return " ".join(title.split()).strip()


def process_record(record: dict, expected_split: str) -> dict:
    """
    Apply B2 preprocessing to one frozen Rakuten record.

    Category information and provenance are preserved exactly.
    """

    if not isinstance(record, dict):
        raise ValueError("Rakuten record must be a JSON object")

    required_fields = {
        "record_id",
        "title",
        "category_id_path",
        "category_path",
        "depth",
        "root_category_id",
        "leaf_category_id",
        "split",
    }

    missing = required_fields - set(record.keys())

    if missing:
        raise ValueError(
            f"Missing required fields: {sorted(missing)}"
        )


    processed = dict(record)

    processed["frozen_split"] = expected_split

    original_title = record["title"]
    processed_title = normalize_title(original_title)

    processed["title"] = processed_title

    return processed

def audit_record(
    original: dict,
    processed: dict,
    statistics: dict,
) -> None:
    """
    Collect B2 transformation statistics.
    """

    original_title = original["title"]
    processed_title = processed["title"]

    statistics["records_processed"] += 1

    if "\ufffd" in original_title:
        statistics["titles_with_replacement_character"] += 1

    if len(original_title) <= 3:
        statistics["titles_length_le_3"] += 1

    if original_title != processed_title:
        statistics["titles_changed_by_whitespace_normalization"] += 1

    if original_title == processed_title:
        statistics["titles_unchanged"] += 1

    statistics["original_title_characters"] += len(
        original_title
    )

    statistics["processed_title_characters"] += len(
        processed_title
    )


def process_split(
    split: str,
) -> dict:

    input_file = INPUT_DIR / f"{split}.jsonl"
    output_file = OUTPUT_DIR / f"{split}.jsonl"

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Frozen split not found: {input_file}"
        )

    statistics = {
        "records_processed": 0,
        "malformed_records": 0,
        "titles_with_replacement_character": 0,
        "titles_length_le_3": 0,
        "titles_changed_by_whitespace_normalization": 0,
        "titles_unchanged": 0,
        "original_title_characters": 0,
        "processed_title_characters": 0,
    }

    record_ids = set()
    duplicate_record_ids = 0

    category_paths_changed = 0
    category_id_paths_changed = 0

    with (
        input_file.open("r", encoding="utf-8") as source,
        output_file.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as target,
    ):

        for line_number, line in enumerate(
            source,
            start=1,
        ):

            line = line.rstrip("\r\n")

            if not line:
                continue

            try:
                original = json.loads(line)
            except json.JSONDecodeError as exc:
                statistics["malformed_records"] += 1
                raise ValueError(
                    f"Malformed JSON at {input_file}:{line_number}: "
                    f"{exc}"
                ) from exc

            processed = process_record(
                original,
                split,
            )

            record_id = processed["record_id"]

            if record_id in record_ids:
                duplicate_record_ids += 1
            else:
                record_ids.add(record_id)

            if (
                processed["category_path"]
                != original["category_path"]
            ):
                category_paths_changed += 1

            if (
                processed["category_id_path"]
                != original["category_id_path"]
            ):
                category_id_paths_changed += 1

            audit_record(
                original,
                processed,
                statistics,
            )

            target.write(
                json.dumps(
                    processed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    statistics["unique_record_ids"] = len(record_ids)
    statistics["duplicate_record_ids"] = duplicate_record_ids
    statistics["category_paths_changed"] = category_paths_changed
    statistics["category_id_paths_changed"] = (
        category_id_paths_changed
    )

    return statistics


def build_schema() -> dict:
    return {
        "dataset": "Rakuten 2018",
        "stage": "B2",
        "schema_version": SCHEMA_VERSION,
        "processing_version": PROCESSING_VERSION,
        "input": {
            "source": "data/splits/rakuten/",
            "frozen_splits": [
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
            ],
        },
        "output": {
            "directory": "data/processed/rakuten/b2/",
            "split_artifacts": [
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
            ],
        },
        "transformations": {
            "title": (
                "deterministic whitespace normalization only"
            ),
            "unicode": (
                "preserved; no translation or ASCII conversion"
            ),
            "replacement_character": (
                "preserved and statistically reported"
            ),
            "digits": "preserved",
            "punctuation": "preserved",
            "symbols": "preserved",
            "short_titles": (
                "preserved; no length-based filtering"
            ),
            "duplicates": (
                "preserved; no deduplication"
            ),
            "category_id_path": "preserved exactly",
            "category_path": "preserved exactly",
            "depth": "preserved exactly",
            "root_category_id": "preserved exactly",
            "leaf_category_id": "preserved exactly",
            "split": "preserved exactly",
            "record_id": "preserved exactly",
        },
        "frozen_split": (
    "experimental split determined by the frozen "
    "split artifact location"
),
        "leakage_policy": (
            "No preprocessing statistics are fitted from "
            "validation or test data."
        ),
        "missing_data_policy": (
            "No values are fabricated or imputed."
        ),
        "raw_data_modified": False,
    }


def main() -> int:

    print("=" * 70)
    print("RAKUTEN B2 PREPROCESSING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_statistics = {}

    for split in SPLITS:

        print()
        print(f"Processing frozen {split} split...")

        statistics = process_split(split)

        split_statistics[split] = statistics

        print(
            f"  Records processed : "
            f"{statistics['records_processed']:,}"
        )

        print(
            f"  Replacement char  : "
            f"{statistics['titles_with_replacement_character']:,}"
        )

        print(
            f"  Short titles <=3  : "
            f"{statistics['titles_length_le_3']:,}"
        )

        print(
            f"  Titles changed    : "
            f"{statistics['titles_changed_by_whitespace_normalization']:,}"
        )

        print(
            f"  Duplicate IDs     : "
            f"{statistics['duplicate_record_ids']:,}"
        )

    total_records = sum(
        item["records_processed"]
        for item in split_statistics.values()
    )

    total_replacement = sum(
        item["titles_with_replacement_character"]
        for item in split_statistics.values()
    )

    total_short = sum(
        item["titles_length_le_3"]
        for item in split_statistics.values()
    )

    total_changed = sum(
        item["titles_changed_by_whitespace_normalization"]
        for item in split_statistics.values()
    )

    total_duplicates = sum(
        item["duplicate_record_ids"]
        for item in split_statistics.values()
    )

    statistics = {
        "dataset": "Rakuten 2018",
        "stage": "B2",
        "processing_version": PROCESSING_VERSION,
        "input_directory": str(
            INPUT_DIR.relative_to(PROJECT_ROOT)
        ),
        "output_directory": str(
            OUTPUT_DIR.relative_to(PROJECT_ROOT)
        ),
        "total_records": total_records,
        "replacement_character_titles": total_replacement,
        "short_titles_length_le_3": total_short,
        "titles_changed_by_whitespace_normalization": (
            total_changed
        ),
        "duplicate_record_ids": total_duplicates,
        "splits": split_statistics,
        "invariants": {
            "record_count_preserved": True,
            "category_information_preserved": True,
            "record_ids_preserved": True,
            "split_boundaries_preserved": True,
            "duplicates_preserved": True,
            "raw_data_modified": False,
        },
    }

    schema = build_schema()

    statistics_file = (
        OUTPUT_DIR / "statistics.json"
    )

    schema_file = (
        OUTPUT_DIR / "schema_v001.json"
    )

    report_file = (
        REPORT_DIR / "rakuten_b2_preprocessing.json"
    )

    statistics_file.write_text(
        json.dumps(
            statistics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema_file.write_text(
        json.dumps(
            schema,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = {
        "report": "rakuten_b2_preprocessing",
        "report_version": "v001",
        "dataset": "Rakuten 2018",
        "stage": "B2",
        "status": "COMPLETE",
        "statistics": statistics,
        "schema": schema,
    }

    report_file.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("-" * 70)
    print("B2 RAKUTEN SUMMARY")
    print("-" * 70)

    print(
        f"Total records              : "
        f"{total_records:,}"
    )

    print(
        f"Replacement-char titles    : "
        f"{total_replacement:,}"
    )

    print(
        f"Titles length <=3          : "
        f"{total_short:,}"
    )

    print(
        f"Whitespace-normalized      : "
        f"{total_changed:,}"
    )

    print(
        f"Duplicate record IDs       : "
        f"{total_duplicates:,}"
    )

    print()
    print("Outputs:")
    print(f"  {OUTPUT_DIR}")
    print(f"  {schema_file}")
    print(f"  {statistics_file}")
    print(f"  {report_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
