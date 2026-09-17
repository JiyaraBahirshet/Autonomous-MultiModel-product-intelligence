"""
MAVE Stage B3: Feature / Representation Construction

Purpose
-------
Construct deterministic, dataset-specific representations from the frozen MAVE
B2 artifacts.

MAVE is the project's attribute / text-understanding dataset. This stage:
- preserves all B2 source fields unchanged;
- constructs deterministic token representations for approved text fields;
- records structured presence/type information for MAVE category/attributes;
- preserves metadata_source and mave_label_source distinctions;
- infers the split only from the B2 artifact being processed (MAVE B2 records
  do not contain a split field in the current frozen schema);
- adds explicit B3 provenance;
- performs no learned fitting, vocabulary learning, imputation, or label
  fabrication;
- checks record/ASIN uniqueness across all three output splits.

It does NOT train models.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mave" / "b2"
OUTPUT_DIR = PROJECT_ROOT / "data" / "representations" / "mave" / "b3"
REPORT_DIR = PROJECT_ROOT / "reports" / "representation"

SPLITS = ("train", "validation", "test")
SCHEMA_VERSION = "v001"
REPRESENTATION_VERSION = "v001"

# These are the text-bearing fields already approved by MAVE B2.
TEXT_FIELDS = (
    "title",
    "description",
    "brand",
    "amazon_category_path",
    "amazon_main_category",
    "feature",
)

# These fields carry attribute/category supervision or provenance and must
# remain structurally visible rather than being flattened into arbitrary text.
STRUCTURED_FIELDS = (
    "mave_category",
    "mave_attributes",
    "mave_information_available",
    "mave_label_source",
    "metadata_source",
    "metadata_complete",
    "supervision",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mave_b3")


def extract_text(value: Any) -> list[str]:
    """Recursively extract source strings without changing their semantics."""
    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(extract_text(item))
        return result

    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            # MAVE structures may contain metadata keys. Do not treat a key
            # itself as product text.
            if key in {"language_tag", "source", "label_source"}:
                continue
            result.extend(extract_text(item))
        return result

    return []


def normalize_text(value: Any) -> str:
    """Deterministic whitespace normalization only."""
    strings = extract_text(value)
    if not strings:
        return ""
    return " ".join(" ".join(strings).split()).strip()


def tokenize(value: Any) -> list[str]:
    """Deterministic whitespace tokenization; no learned vocabulary."""
    text = normalize_text(value)
    return text.split() if text else []


def value_type(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def field_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return len(value) > 0
    return True


def text_representation(record: dict[str, Any]) -> dict[str, Any]:
    field_tokens: dict[str, list[str]] = {}
    present_fields: list[str] = []
    combined_tokens: list[str] = []

    for field in TEXT_FIELDS:
        if field not in record:
            continue

        tokens = tokenize(record.get(field))
        if tokens:
            field_tokens[field] = tokens
            present_fields.append(field)
            combined_tokens.extend(tokens)

    return {
        "field_tokens": field_tokens,
        "combined_tokens": combined_tokens,
        "token_count": len(combined_tokens),
        "present_text_fields": present_fields,
    }


def structured_representation(record: dict[str, Any]) -> dict[str, Any]:
    presence: dict[str, bool] = {}
    types: dict[str, str] = {}

    for field in STRUCTURED_FIELDS:
        value = record.get(field)
        presence[field] = field_is_present(value)
        types[field] = value_type(value)

    return {
        "field_presence": presence,
        "field_types": types,
        "mave_category_preserved": "mave_category" in record,
        "mave_attributes_preserved": "mave_attributes" in record,
        "label_source_preserved": "mave_label_source" in record,
        "metadata_source_preserved": "metadata_source" in record,
    }


def image_reference_representation(record: dict[str, Any]) -> dict[str, Any]:
    """
    MAVE is not the project's image-primary dataset, but image URL fields are
    preserved in B2. B3 records their availability without downloading,
    resolving, or fabricating images.
    """
    low = record.get("imageURL")
    high = record.get("imageURLHighRes")

    low_count = len(low) if isinstance(low, list) else int(field_is_present(low))
    high_count = (
        len(high) if isinstance(high, list) else int(field_is_present(high))
    )

    return {
        "imageURL_reference_count": low_count,
        "imageURLHighRes_reference_count": high_count,
        "has_imageURL_reference": low_count > 0,
        "has_imageURLHighRes_reference": high_count > 0,
        "image_download": False,
        "image_path_guessing": False,
        "synthetic_images": False,
    }


def build_representation(record: dict[str, Any], expected_split: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("MAVE B2 record must be a JSON object")

    required = {"asin", "record_id"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"Missing required MAVE B2 fields: {sorted(missing)}")

    asin = str(record["asin"])
    record_id = str(record["record_id"])

    expected_record_id = (
        "mave:"
        + hashlib.sha256(asin.encode("utf-8")).hexdigest()[:16]
    )
    if record_id != expected_record_id:
        raise ValueError(
            f"record_id/ASIN mismatch for {asin}: "
            f"expected {expected_record_id!r}, got {record_id!r}"
        )

    # Current MAVE B2 records do not contain a split field. The split is
    # therefore derived from the frozen artifact filename, not guessed from
    # content. If a future B2 record contains split, it must agree.
    source_split = record.get("split")
    if source_split is not None and source_split != expected_split:
        raise ValueError(
            f"Split mismatch: expected {expected_split!r}, got {source_split!r}"
        )

    out = dict(record)
    out["split"] = expected_split
    out["frozen_split"] = expected_split

    text = text_representation(record)
    structured = structured_representation(record)
    image = image_reference_representation(record)

    out["b3_representation"] = {
        "representation_version": REPRESENTATION_VERSION,
        "text": text,
        "structured": structured,
        "image_references": image,
    }

    out["b3_provenance"] = {
        "representation_version": REPRESENTATION_VERSION,
        "source_stage": "B2",
        "source_artifact": f"data/processed/mave/b2/{expected_split}.jsonl",
        "source_split": expected_split,
        "source_record_id": record_id,
        "source_asin": asin,
        "learned_parameters": False,
        "imputation": False,
        "label_fabrication": False,
    }

    return out


def process_split(split: str) -> tuple[dict[str, Any], set[str]]:
    src = INPUT_DIR / f"{split}.jsonl"
    dst = OUTPUT_DIR / f"{split}.jsonl"

    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {src}")

    stats = {
        "source_records": 0,
        "output_records": 0,
        "unique_asins": 0,
        "duplicate_asins": 0,
        "duplicate_record_ids": 0,
        "total_text_tokens": 0,
        "empty_text_sequences": 0,
        "records_with_mave_information": 0,
        "records_with_mave_category": 0,
        "records_with_mave_attributes": 0,
        "metadata_source_counts": {},
        "mave_label_source_counts": {},
        "text_field_presence": {},
    }

    seen_asins: set[str] = set()
    seen_record_ids: set[str] = set()
    metadata_sources: Counter[str] = Counter()
    label_sources: Counter[str] = Counter()
    text_presence: Counter[str] = Counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as infile, dst.open(
        "w", encoding="utf-8", newline="\n"
    ) as outfile:
        for line_number, line in enumerate(infile, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON at {src}:{line_number}: {exc}"
                ) from exc

            stats["source_records"] += 1

            asin = str(record.get("asin", ""))
            record_id = str(record.get("record_id", ""))

            if asin in seen_asins:
                stats["duplicate_asins"] += 1
            seen_asins.add(asin)

            if record_id in seen_record_ids:
                stats["duplicate_record_ids"] += 1
            seen_record_ids.add(record_id)

            out = build_representation(record, split)
            rep = out["b3_representation"]

            token_count = rep["text"]["token_count"]
            stats["total_text_tokens"] += token_count
            if token_count == 0:
                stats["empty_text_sequences"] += 1

            for field in rep["text"]["present_text_fields"]:
                text_presence[field] += 1

            if record.get("mave_information_available") is True:
                stats["records_with_mave_information"] += 1
            if field_is_present(record.get("mave_category")):
                stats["records_with_mave_category"] += 1
            if field_is_present(record.get("mave_attributes")):
                stats["records_with_mave_attributes"] += 1

            metadata_source = str(record.get("metadata_source", ""))
            label_source = str(record.get("mave_label_source", ""))

            metadata_sources[metadata_source] += 1
            label_sources[label_source] += 1

            outfile.write(
                json.dumps(out, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            stats["output_records"] += 1

    stats["unique_asins"] = len(seen_asins)
    stats["metadata_source_counts"] = dict(metadata_sources)
    stats["mave_label_source_counts"] = dict(label_sources)
    stats["text_field_presence"] = dict(text_presence)

    return stats, seen_asins


def schema() -> dict[str, Any]:
    return {
        "dataset": "MAVE",
        "stage": "B3",
        "schema_version": SCHEMA_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "input_artifact": "data/processed/mave/b2/{split}.jsonl",
        "output_artifact": "data/representations/mave/b3/{split}.jsonl",
        "split_values": list(SPLITS),
        "representation_components": {
            "text": {
                "source_fields": list(TEXT_FIELDS),
                "normalization": "deterministic whitespace normalization only",
                "tokenization": "deterministic whitespace tokenization",
                "lowercasing": False,
                "stemming": False,
                "stopword_removal": False,
                "learned_vocabulary": False,
            },
            "structured": {
                "source_fields": list(STRUCTURED_FIELDS),
                "encoding": "field presence and native source type metadata",
                "source_values_overwritten": False,
                "mave_attributes_flattened": False,
                "mave_category_overwritten": False,
            },
            "image_references": {
                "source_fields": ["imageURL", "imageURLHighRes"],
                "references_preserved": True,
                "image_download": False,
                "image_path_guessing": False,
                "synthetic_images": False,
            },
        },
        "provenance": {
            "asin_preserved": True,
            "record_id_preserved": True,
            "split_preserved_or_artifact_derived": True,
            "frozen_split_added": True,
            "b2_fields_preserved": True,
            "metadata_source_preserved": True,
            "mave_label_source_preserved": True,
        },
        "leakage_policy": (
            "Record-local deterministic representations only; no learned "
            "parameters, corpus fitting, vocabulary fitting, or cross-record "
            "statistics."
        ),
        "missing_data_policy": (
            "Missing source values remain missing. No values are imputed or "
            "fabricated."
        ),
        "raw_data_modified": False,
    }


def main() -> int:
    print("=" * 70)
    print("MAVE B3 REPRESENTATION CONSTRUCTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, dict[str, Any]] = {}
    global_asins: set[str] = set()
    cross_split_asin_leakage = 0

    for split in SPLITS:
        print(f"\nProcessing B2 {split} split...")
        stats, split_asins = process_split(split)

        overlap = global_asins.intersection(split_asins)
        if overlap:
            cross_split_asin_leakage += len(overlap)
            raise ValueError(
                f"Cross-split ASIN leakage detected before completing {split}: "
                f"{len(overlap)} ASINs already appeared in another split."
            )

        global_asins.update(split_asins)
        split_stats[split] = stats

        print(f"  Source records             : {stats['source_records']:,}")
        print(f"  Output records             : {stats['output_records']:,}")
        print(f"  Unique ASINs               : {stats['unique_asins']:,}")
        print(f"  Duplicate ASINs            : {stats['duplicate_asins']:,}")
        print(f"  Duplicate record IDs       : {stats['duplicate_record_ids']:,}")
        print(f"  Total text tokens          : {stats['total_text_tokens']:,}")
        print(f"  Empty text sequences       : {stats['empty_text_sequences']:,}")
        print(f"  MAVE information records   : {stats['records_with_mave_information']:,}")
        print(f"  MAVE category records      : {stats['records_with_mave_category']:,}")
        print(f"  MAVE attribute records     : {stats['records_with_mave_attributes']:,}")

    total_records = sum(x["output_records"] for x in split_stats.values())
    total_tokens = sum(x["total_text_tokens"] for x in split_stats.values())
    total_empty = sum(x["empty_text_sequences"] for x in split_stats.values())
    total_dup_asins = sum(x["duplicate_asins"] for x in split_stats.values())
    total_dup_ids = sum(x["duplicate_record_ids"] for x in split_stats.values())

    statistics = {
        "dataset": "MAVE",
        "stage": "B3",
        "representation_version": REPRESENTATION_VERSION,
        "splits": split_stats,
        "total_records": total_records,
        "total_unique_asins": len(global_asins),
        "total_text_tokens": total_tokens,
        "empty_text_sequences": total_empty,
        "duplicate_asins_within_splits": total_dup_asins,
        "duplicate_record_ids_within_splits": total_dup_ids,
        "cross_split_asin_leakage": cross_split_asin_leakage,
        "invariants": {
            "record_count_preserved": total_records == 2_907_358,
            "asin_uniqueness": total_dup_asins == 0 and cross_split_asin_leakage == 0,
            "record_id_uniqueness_within_splits": total_dup_ids == 0,
            "mave_source_distinctions_preserved": True,
            "missing_values_fabricated": False,
            "learned_parameters": False,
            "raw_data_modified": False,
        },
    }

    schema_file = OUTPUT_DIR / "schema_v001.json"
    statistics_file = OUTPUT_DIR / "statistics.json"
    report_file = REPORT_DIR / "mave_b3_representation.json"

    schema_file.write_text(
        json.dumps(schema(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    statistics_file.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ok = (
        total_records == 2_907_358
        and len(global_asins) == 2_907_358
        and total_dup_asins == 0
        and total_dup_ids == 0
        and cross_split_asin_leakage == 0
        and total_tokens > 0
    )

    report = {
        "report": "mave_b3_representation",
        "report_version": "v001",
        "dataset": "MAVE",
        "stage": "B3",
        "status": "PASS" if ok else "FAIL",
        "schema": schema(),
        "statistics": statistics,
    }

    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "-" * 70)
    print("B3 MAVE SUMMARY")
    print("-" * 70)
    print(f"Source / output records : {total_records:,}")
    print(f"Unique ASINs            : {len(global_asins):,}")
    print(f"Total text tokens       : {total_tokens:,}")
    print(f"Empty text sequences    : {total_empty:,}")
    print(f"Duplicate ASINs         : {total_dup_asins:,}")
    print(f"Duplicate record IDs    : {total_dup_ids:,}")
    print(f"Cross-split ASIN leak   : {cross_split_asin_leakage:,}")
    print(f"\nSTATUS: {'PASS' if ok else 'FAIL'}")
    print("\nOutputs:")
    print(f"  {OUTPUT_DIR}")
    print(f"  {schema_file}")
    print(f"  {statistics_file}")
    print(f"  {report_file}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
