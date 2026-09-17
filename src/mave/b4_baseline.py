"""
MAVE B4.3 — Supervision / Input Feasibility Gate

Purpose
-------
This first implementation step does NOT train a model.

It inspects the frozen MAVE B3 artifacts and determines whether the
approved B4.3 TF-IDF + linear-classification baseline has an unambiguous,
non-leaking supervised target and usable B3 textual input.

The script deliberately stops before model fitting if the target semantics
cannot be established from the existing artifacts.

Non-goals
---------
- No modification of B2/B3 artifacts.
- No label fabrication.
- No conversion of missing MAVE information into a negative class.
- No inference of MAVE labels from Amazon metadata.
- No B5 fusion.
- No downloading or external enrichment.
"""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import sklearn
except ImportError:
    sklearn = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "representations" / "mave" / "b3"
REPORT_DIR = PROJECT_ROOT / "reports" / "baseline" / "mave"
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "mave" / "b4.3"

SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {
    "train": 2_326_033,
    "validation": 290_488,
    "test": 290_837,
}
EXPECTED_TOTAL = 2_907_358
EXPECTED_MAVE_INFORMATION = 7_148
EXPECTED_SEED = 20260827
B3_REPRESENTATION_VERSION = "v001"
B4_VERSION = "v001"

TEXT_KEYS = ("title", "description", "brand", "amazon_category_path",
             "amazon_main_category", "feature")


def nonempty(value: Any) -> bool:
    """Return True only for genuinely populated values."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def get_text_tokens(record: dict[str, Any]) -> list[str]:
    rep = record.get("b3_representation")
    if not isinstance(rep, dict):
        return []

    text = rep.get("text")
    if not isinstance(text, dict):
        return []

    tokens = text.get("combined_tokens")
    if not isinstance(tokens, list):
        return []

    return [str(x) for x in tokens if str(x).strip()]


def target_candidates(record: dict[str, Any]) -> dict[str, bool]:
    """
    Report availability of preserved MAVE target structures.

    IMPORTANT:
    These are observations only. This function does not reinterpret
    missing values as negative labels.
    """
    category = record.get("mave_category")
    attributes = record.get("mave_attributes")

    return {
        "mave_category_nonempty": nonempty(category),
        "mave_attributes_nonempty": nonempty(attributes),
        "mave_label_source_nonempty": nonempty(record.get("mave_label_source")),
        "supervision_label_available": bool(
            isinstance(record.get("supervision"), dict)
            and record["supervision"].get("label_available") is True
        ),
        "mave_information_available": (
            record.get("mave_information_available") is True
        ),
    }


def inspect_split(split: str) -> dict[str, Any]:
    path = INPUT_DIR / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing B3 artifact: {path}")

    total = 0
    duplicate_asins = 0
    duplicate_record_ids = 0
    missing_asins = 0
    split_mismatches = 0
    cross_target_flags = Counter()

    asins: set[str] = set()
    record_ids: set[str] = set()

    mave_info = 0
    category_nonempty = 0
    attributes_nonempty = 0
    label_source_nonempty = 0
    supervision_label_available = 0
    usable_text = 0
    usable_text_with_info = 0
    supervised_candidates = 0
    supervised_with_text = 0
    empty_text_with_info = 0

    category_values = Counter()
    label_sources = Counter()
    metadata_sources = Counter()
    representation_versions = Counter()

    malformed = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed += 1
                raise ValueError(
                    f"Malformed JSON at {path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_no}"
                )

            total += 1

            asin = str(record.get("asin", "")).strip()
            rid = str(record.get("record_id", "")).strip()

            if not asin:
                missing_asins += 1
            elif asin in asins:
                duplicate_asins += 1
            else:
                asins.add(asin)

            if rid in record_ids:
                duplicate_record_ids += 1
            else:
                record_ids.add(rid)

            actual_split = record.get("split")
            frozen_split = record.get("frozen_split")

            if actual_split != split or frozen_split != split:
                split_mismatches += 1

            rep = record.get("b3_representation", {})
            if isinstance(rep, dict):
                version = rep.get("representation_version")
                if version is not None:
                    representation_versions[str(version)] += 1

            flags = target_candidates(record)

            if flags["mave_information_available"]:
                mave_info += 1

            if flags["mave_category_nonempty"]:
                category_nonempty += 1
                category = record.get("mave_category")
                if isinstance(category, str):
                    category_values[category.strip()] += 1
                else:
                    category_values[str(category)] += 1

            if flags["mave_attributes_nonempty"]:
                attributes_nonempty += 1

            if flags["mave_label_source_nonempty"]:
                label_source_nonempty += 1
                label_sources[str(record.get("mave_label_source"))] += 1

            if flags["supervision_label_available"]:
                supervision_label_available += 1

            tokens = get_text_tokens(record)
            has_text = bool(tokens)

            if has_text:
                usable_text += 1

            if flags["mave_information_available"] and has_text:
                usable_text_with_info += 1

            # A conservative candidate definition:
            # explicit MAVE information AND at least one preserved MAVE
            # target structure. Missing values are never converted to labels.
            has_explicit_target = (
                flags["mave_information_available"]
                and (
                    flags["mave_category_nonempty"]
                    or flags["mave_attributes_nonempty"]
                )
            )

            if has_explicit_target:
                supervised_candidates += 1
                if has_text:
                    supervised_with_text += 1
                else:
                    empty_text_with_info += 1

            metadata_sources[str(record.get("metadata_source", ""))] += 1

    return {
        "path": str(path),
        "records": total,
        "unique_asins": len(asins),
        "unique_record_ids": len(record_ids),
        "duplicate_asins": duplicate_asins,
        "duplicate_record_ids": duplicate_record_ids,
        "missing_asins": missing_asins,
        "split_mismatches": split_mismatches,
        "malformed_records": malformed,
        "mave_information_available": mave_info,
        "mave_category_nonempty": category_nonempty,
        "mave_attributes_nonempty": attributes_nonempty,
        "mave_label_source_nonempty": label_source_nonempty,
        "supervision_label_available": supervision_label_available,
        "usable_b3_text": usable_text,
        "usable_b3_text_with_mave_information": usable_text_with_info,
        "supervised_candidates": supervised_candidates,
        "supervised_candidates_with_usable_text": supervised_with_text,
        "supervised_candidates_with_empty_text": empty_text_with_info,
        "category_cardinality": len(category_values),
        "top_categories": category_values.most_common(20),
        "label_source_distribution": dict(label_sources),
        "metadata_source_distribution": dict(metadata_sources),
        "representation_versions": dict(representation_versions),
    }


def evaluate_global_integrity(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Check cross-split leakage by streaming only identifiers.

    B3 is never modified.
    """
    asin_sets: dict[str, set[str]] = {}
    rid_sets: dict[str, set[str]] = {}

    for split in SPLITS:
        path = INPUT_DIR / f"{split}.jsonl"
        asins: set[str] = set()
        rids: set[str] = set()

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                asins.add(str(rec.get("asin", "")).strip())
                rids.add(str(rec.get("record_id", "")).strip())

        asin_sets[split] = asins
        rid_sets[split] = rids

    cross_asin = {}
    cross_rid = {}

    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            cross_asin[f"{left}_vs_{right}"] = len(
                asin_sets[left] & asin_sets[right]
            )
            cross_rid[f"{left}_vs_{right}"] = len(
                rid_sets[left] & rid_sets[right]
            )

    return {
        "cross_split_asin_overlap": cross_asin,
        "cross_split_record_id_overlap": cross_rid,
        "cross_split_asin_leakage": sum(cross_asin.values()),
        "cross_split_record_id_leakage": sum(cross_rid.values()),
    }


def determine_target_status(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Apply the B4.3 specification conservatively.

    We do NOT decide that category or attributes is the final model target
    merely because the field exists. The implementation must establish a
    single unambiguous classification target from the frozen artifacts.
    """
    total_info = sum(
        x["mave_information_available"] for x in splits.values()
    )
    total_category = sum(
        x["mave_category_nonempty"] for x in splits.values()
    )
    total_attributes = sum(
        x["mave_attributes_nonempty"] for x in splits.values()
    )
    total_candidates = sum(
        x["supervised_candidates"] for x in splits.values()
    )
    total_candidate_text = sum(
        x["supervised_candidates_with_usable_text"] for x in splits.values()
    )

    # The project record establishes that all 7,148 fallback records preserve
    # mave_category and mave_attributes. Therefore both structures being
    # present is expected and is NOT enough to select one silently.
    if total_candidates == 0:
        status = "BLOCKED_NO_EXPLICIT_SUPERVISION"
        reason = (
            "No records contain an explicit MAVE-information flag together "
            "with a preserved MAVE category/attribute target."
        )
    elif total_candidate_text == 0:
        status = "BLOCKED_NO_USABLE_B3_INPUT"
        reason = (
            "Explicit MAVE-supervised candidates exist, but none have usable "
            "B3 combined_tokens. B4.3 cannot train a text baseline without "
            "changing the approved B3 input contract."
        )
    elif total_category > 0 and total_attributes > 0:
        status = "BLOCKED_TARGET_AMBIGUITY"
        reason = (
            "Both mave_category and mave_attributes are populated in the "
            "supervised population. The frozen artifacts do not, by "
            "themselves, authorize silently selecting one as the sole "
            "classification target."
        )
    else:
        status = "ELIGIBLE_FOR_TARGET_CONFIRMATION"
        reason = (
            "A single preserved MAVE target structure is observed with "
            "usable B3 text. The target can proceed to implementation "
            "provided its classification semantics are documented."
        )

    return {
        "status": status,
        "reason": reason,
        "total_mave_information_records": total_info,
        "total_category_records": total_category,
        "total_attribute_records": total_attributes,
        "total_supervised_candidates": total_candidates,
        "total_supervised_candidates_with_usable_text": total_candidate_text,
    }


def main() -> int:
    print("=" * 70)
    print("MAVE B4.3 — SUPERVISION / INPUT FEASIBILITY GATE")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, dict[str, Any]] = {}

    for split in SPLITS:
        print(f"\nInspecting frozen B3 {split} split...")
        result = inspect_split(split)
        splits[split] = result

        print(f"  Records                         : {result['records']:,}")
        print(f"  Unique ASINs                    : {result['unique_asins']:,}")
        print(f"  Duplicate ASINs                 : {result['duplicate_asins']:,}")
        print(f"  Duplicate record IDs            : {result['duplicate_record_ids']:,}")
        print(f"  MAVE information available     : {result['mave_information_available']:,}")
        print(f"  MAVE category non-empty        : {result['mave_category_nonempty']:,}")
        print(f"  MAVE attributes non-empty      : {result['mave_attributes_nonempty']:,}")
        print(f"  Usable B3 text                 : {result['usable_b3_text']:,}")
        print(
            "  Supervised candidates           : "
            f"{result['supervised_candidates']:,}"
        )
        print(
            "  Supervised candidates + text   : "
            f"{result['supervised_candidates_with_usable_text']:,}"
        )
        print(
            "  Supervised candidates no text  : "
            f"{result['supervised_candidates_with_empty_text']:,}"
        )

        expected = EXPECTED_COUNTS[split]
        if result["records"] != expected:
            print(
                f"  WARNING: expected {expected:,}, "
                f"observed {result['records']:,}"
            )

    global_integrity = evaluate_global_integrity(splits)
    target = determine_target_status(splits)

    total_records = sum(x["records"] for x in splits.values())
    expected_counts_ok = all(
        splits[s]["records"] == EXPECTED_COUNTS[s] for s in SPLITS
    )

    local_integrity_ok = all(
        splits[s]["duplicate_asins"] == 0
        and splits[s]["duplicate_record_ids"] == 0
        and splits[s]["missing_asins"] == 0
        and splits[s]["split_mismatches"] == 0
        and splits[s]["malformed_records"] == 0
        for s in SPLITS
    )

    leakage_ok = (
        global_integrity["cross_split_asin_leakage"] == 0
        and global_integrity["cross_split_record_id_leakage"] == 0
    )

    # No model is trained by this gate, so no learned parameters are created.
    # This explicit field makes the boundary auditable.
    model_training_performed = False
    tfidf_fitted = False
    test_used_for_fitting = False
    b3_modified = False
    labels_fabricated = False
    b5_fusion_performed = False

    overall_pass = (
        expected_counts_ok
        and total_records == EXPECTED_TOTAL
        and local_integrity_ok
        and leakage_ok
        and target["status"] == "ELIGIBLE_FOR_TARGET_CONFIRMATION"
    )

    report = {
        "report": "mave_b4.3_feasibility",
        "report_version": B4_VERSION,
        "dataset": "MAVE",
        "stage": "B4.3",
        "status": "PASS" if overall_pass else "BLOCKED",
        "feasibility_status": target["status"],
        "feasibility_reason": target["reason"],
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn_version": (
                None if sklearn is None else sklearn.__version__
            ),
            "seed": EXPECTED_SEED,
        },
        "source_artifacts": {
            "train": "data/representations/mave/b3/train.jsonl",
            "validation": "data/representations/mave/b3/validation.jsonl",
            "test": "data/representations/mave/b3/test.jsonl",
            "b3_representation_version": B3_REPRESENTATION_VERSION,
        },
        "expected_counts": EXPECTED_COUNTS,
        "observed_total_records": total_records,
        "counts_match_expected": expected_counts_ok,
        "splits": splits,
        "global_integrity": global_integrity,
        "target_assessment": target,
        "invariants": {
            "b3_artifacts_modified": b3_modified,
            "labels_fabricated": labels_fabricated,
            "missing_labels_converted_to_negative": False,
            "amazon_metadata_used_as_mave_target": False,
            "model_training_performed": model_training_performed,
            "tfidf_fitted": tfidf_fitted,
            "test_used_for_fitting": test_used_for_fitting,
            "b5_fusion_performed": b5_fusion_performed,
        },
        "next_step": (
            "Proceed to B4.3 TF-IDF + linear baseline only after the "
            "target is unambiguously established from the frozen artifacts."
            if overall_pass
            else
            "Do not train B4.3 yet. Resolve the reported target/input "
            "eligibility issue without modifying B3."
        ),
    }

    report_path = REPORT_DIR / "mave_b4.3_feasibility.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "-" * 70)
    print("B4.3 FEASIBILITY SUMMARY")
    print("-" * 70)
    print(f"Total records                 : {total_records:,}")
    print(f"MAVE-information records     : {target['total_mave_information_records']:,}")
    print(f"Category records              : {target['total_category_records']:,}")
    print(f"Attribute records             : {target['total_attribute_records']:,}")
    print(f"Supervised candidates         : {target['total_supervised_candidates']:,}")
    print(
        "Supervised candidates + text : "
        f"{target['total_supervised_candidates_with_usable_text']:,}"
    )
    print(
        f"Cross-split ASIN leakage      : "
        f"{global_integrity['cross_split_asin_leakage']:,}"
    )
    print(
        f"Cross-split record-ID leakage : "
        f"{global_integrity['cross_split_record_id_leakage']:,}"
    )
    print(f"Feasibility status             : {target['status']}")
    print(f"Overall status                 : {'PASS' if overall_pass else 'BLOCKED'}")
    print("\nNo model training was performed.")
    print(f"\nReport:\n  {report_path}")

    # The gate intentionally returns non-zero when implementation should stop.
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
