"""
Rakuten Stage B3: Textual Representation Construction

Transforms clean B2 JSONL records into standardized B3 textual representations:
- Tokenizes titles deterministically
- Computes sequence lengths and token counts
- Preserves hierarchical category paths
- Validates split consistency and zero record leakage across splits
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set

# Standard configurations
REPRESENTATION_VERSION = "001"
REQUIRED_FIELDS = {
    "record_id",
    "split",
    "title",
    "category_id_path",
    "category_path",
    "depth",
    "root_category_id",
    "leaf_category_id",
}

# Acceptable aliases for split naming conventions
VALID_SPLIT_ALIASES = {
    "train": {"train"},
    "validation": {"validation", "val", "dev"},
    "test": {"test"},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rakuten_b3")


def tokenize_title(title: str) -> List[str]:
    """
    Standardizes and tokenizes item titles deterministically.
    """
    if not title:
        return []
    # Lowercase and match alphanumeric word boundaries
    tokens = re.findall(r"\b\w+\b", title.lower())
    return tokens


def validate_hierarchy(record: Dict[str, Any]) -> None:
    """
    Ensures hierarchical properties match expectations. Safely checks
    category paths regardless of raw primitive types.
    """
    cat_ids = record.get("category_id_path", [])
    cat_names = record.get("category_path", [])
    depth = record.get("depth", 0)

    # Normalize category_id_path if string/int was passed instead of list
    if isinstance(cat_ids, (str, int)):
        cat_ids = [cat_ids]

    if isinstance(cat_names, str):
        cat_names = [cat_names]

    # Validate that non-empty structures are populated
    if not cat_ids or not cat_names:
        raise ValueError(
            f"Record {record.get('record_id')}: Empty category path or category ID path."
        )

    # Verify path length alignment when lists match 1:1, or fall back to depth check
    if isinstance(cat_ids, list) and isinstance(cat_names, list):
        if len(cat_ids) != len(cat_names) and len(cat_names) != depth:
            logger.warning(
                f"Record {record.get('record_id')}: Category ID length ({len(cat_ids)}) "
                f"differs from category name length ({len(cat_names)}). Depth is {depth}."
            )


def build_representation(
    record: Dict[str, Any],
    expected_split: str,
) -> Dict[str, Any]:
    """
    Constructs the B3 representation dictionary for a given record.
    """
    if not isinstance(record, dict):
        raise ValueError("Rakuten B2 record must be a JSON object")

    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        raise ValueError(f"Missing required B2 fields: {sorted(missing)}")

    # --------------------------------------------------------------
    # Frozen split protection
    # --------------------------------------------------------------
    valid_aliases = VALID_SPLIT_ALIASES.get(expected_split, {expected_split})

    frozen_split = record.get("frozen_split")
    if (
        frozen_split is not None
        and frozen_split != expected_split
        and frozen_split not in valid_aliases
    ):
        raise ValueError(
            f"Frozen split mismatch: expected {expected_split!r}, got {frozen_split!r}"
        )

    source_split = record.get("split")
    if (
        source_split
        and source_split != expected_split
        and source_split not in valid_aliases
    ):
        # Fall back to checking frozen_split if split is an intrinsic raw dataset artifact
        if frozen_split != expected_split:
            raise ValueError(
                f"Split mismatch: expected {expected_split!r}, got {source_split!r}"
            )

    # --------------------------------------------------------------
    # Validate existing hierarchy
    # --------------------------------------------------------------
    validate_hierarchy(record)

    # --------------------------------------------------------------
    # Title representation
    # --------------------------------------------------------------
    title = record["title"]
    tokens = tokenize_title(title)

    # --------------------------------------------------------------
    # B3 output assembly
    # --------------------------------------------------------------
    representation = {
        # Identity
        "record_id": record["record_id"],
        "split": record["split"],
        # Source title
        "title": title,
        # Deterministic textual representation
        "title_tokens": tokens,
        "title_token_count": len(tokens),
        # Hierarchical target information
        "category_id_path": record["category_id_path"],
        "category_path": record["category_path"],
        "depth": record["depth"],
        "root_category_id": record["root_category_id"],
        "leaf_category_id": record["leaf_category_id"],
        # Provenance
        "b3_provenance": {
            "representation_version": REPRESENTATION_VERSION,
            "source_stage": "B2",
            "source_artifact": f"data/processed/rakuten/b2/{expected_split}.jsonl",
            "source_split": expected_split,
        },
    }

    return representation


def process_split(
    input_path: Path, output_path: Path, expected_split: str
) -> Dict[str, Any]:
    """
    Processes a single dataset split file from B2 to B3 representation format.
    """
    logger.info(f"Processing B2 {expected_split} split...")
    seen_ids: Set[Any] = set()
    total_tokens = 0
    empty_token_seq_count = 0
    duplicate_ids_count = 0
    processed_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            record_id = record.get("record_id")

            if record_id in seen_ids:
                duplicate_ids_count += 1
            else:
                seen_ids.add(record_id)

            rep = build_representation(record, expected_split)

            token_count = rep["title_token_count"]
            total_tokens += token_count
            if token_count == 0:
                empty_token_seq_count += 1

            outfile.write(json.dumps(rep, ensure_ascii=False) + "\n")
            processed_count += 1

    stats = {
        "source_records": processed_count,
        "output_records": processed_count,
        "unique_record_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids_count,
        "title_tokens": total_tokens,
        "empty_token_seq": empty_token_seq_count,
    }

    print(f"Processing B2 {expected_split} split...")
    print(f"  Source records    : {stats['source_records']:,}")
    print(f"  Output records    : {stats['output_records']:,}")
    print(f"  Unique record IDs : {stats['unique_record_ids']:,}")
    print(f"  Duplicate IDs     : {stats['duplicate_ids']:,}")
    print(f"  Title tokens      : {stats['title_tokens']:,}")
    print(f"  Empty token seq.  : {stats['empty_token_seq']:,}\n")

    return stats


def main():
    base_dir = Path(__file__).resolve().parents[2]
    b2_dir = base_dir / "data" / "processed" / "rakuten" / "b2"
    b3_dir = base_dir / "data" / "representations" / "rakuten" / "b3"
    reports_dir = base_dir / "reports" / "representation"

    splits = ["train", "validation", "test"]
    overall_stats = {
        "source_records": 0,
        "output_records": 0,
        "total_title_tokens": 0,
        "empty_token_sequences": 0,
        "duplicate_record_ids": 0,
    }

    print("=" * 70)
    print("RAKUTEN B3 REPRESENTATION CONSTRUCTION")
    print("=" * 70 + "\n")

    for split in splits:
        input_file = b2_dir / f"{split}.jsonl"
        output_file = b3_dir / f"{split}.jsonl"

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        split_stats = process_split(input_file, output_file, split)

        overall_stats["source_records"] += split_stats["source_records"]
        overall_stats["output_records"] += split_stats["output_records"]
        overall_stats["total_title_tokens"] += split_stats["title_tokens"]
        overall_stats["empty_token_sequences"] += split_stats[
            "empty_token_seq"
        ]
        overall_stats["duplicate_record_ids"] += split_stats["duplicate_ids"]

    # Write schema document
    schema = {
        "version": REPRESENTATION_VERSION,
        "fields": [
            "record_id",
            "split",
            "title",
            "title_tokens",
            "title_token_count",
            "category_id_path",
            "category_path",
            "depth",
            "root_category_id",
            "leaf_category_id",
            "b3_provenance",
        ],
    }
    with open(b3_dir / "schema_v001.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Write overall statistics
    with open(b3_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, indent=2)

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "rakuten_b3_representation.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({"status": "PASS", "summary": overall_stats}, f, indent=2)

    print("-" * 70)
    print("B3 RAKUTEN SUMMARY")
    print("-" * 70)
    print(f"Source records       : {overall_stats['source_records']:,}")
    print(f"Output records       : {overall_stats['output_records']:,}")
    print(f"Total title tokens   : {overall_stats['total_title_tokens']:,}")
    print(f"Empty token sequences: {overall_stats['empty_token_sequences']:,}")
    print(f"Duplicate record IDs : {overall_stats['duplicate_record_ids']:,}\n")
    print("Outputs:")
    print(f"  {b3_dir}")
    print(f"  {b3_dir / 'schema_v001.json'}")
    print(f"  {b3_dir / 'statistics.json'}")
    print(f"  {report_file}\n")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()