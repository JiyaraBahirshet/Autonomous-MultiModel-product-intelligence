import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "data" / "splits"
R = ROOT / "reports" / "validation"


def read(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def image_ids(x):
    ids = set()

    main = x.get("main_image")
    if isinstance(main, dict) and main.get("image_id"):
        ids.add(str(main["image_id"]))

    for item in x.get("other_images", []) or []:
        if isinstance(item, dict) and item.get("image_id"):
            ids.add(str(item["image_id"]))

    return ids


def validate_dataset(dataset):
    print(f"\nValidating {dataset}...")

    counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    # record_id -> first split encountered
    seen_record_ids = set()
    duplicate_record_ids = 0
    cross_split_record_ids = 0

    # ABO: image_id -> first split encountered.
    # This detects whether an image occurs in multiple splits
    # without retaining the complete records.
    image_split = {}
    shared_images = set()

    # MAVE
    seen_asins = set()
    cross_split_asins = 0

    # Rakuten
    train_validation_official_flags = True
    test_official_flags = True

    for split in ("train", "validation", "test"):
        path = S / dataset / f"{split}.jsonl"

        if not path.exists():
            raise FileNotFoundError(f"Missing split: {path}")

        print(f"  Reading {split}.jsonl...")

        for x in read(path):
            counts[split] += 1

            record_id = str(x["record_id"])

            if record_id in seen_record_ids:
                duplicate_record_ids += 1

            seen_record_ids.add(record_id)

            # We need to detect whether the same record_id
            # appears in another split. Since each record_id
            # should be unique, this can be checked by maintaining
            # a compact set of seen IDs plus split-specific sets.
            #
            # Split-specific membership sets are handled below.
            if dataset == "abo":
                for image_id in image_ids(x):
                    previous = image_split.get(image_id)

                    if previous is None:
                        image_split[image_id] = split
                    elif previous != split:
                        shared_images.add(image_id)

            if dataset == "mave":
                asin = str(x["asin"])

                if asin in seen_asins:
                    cross_split_asins += 1

                seen_asins.add(asin)

            if dataset == "rakuten":
                official_split = x.get("split")

                if split in ("train", "validation"):
                    if official_split != "train":
                        train_validation_official_flags = False

                elif split == "test":
                    if official_split != "test":
                        test_official_flags = False

        print(f"    records: {counts[split]:,}")

    # The record_id itself is expected to be globally unique.
    # Since we process each split sequentially, a duplicate ID
    # necessarily represents either an intra-split duplicate or
    # a cross-split collision. We need an explicit second pass
    # for the latter.
    #
    # Do that with a compact record_id -> split mapping.
    record_split = {}

    for split in ("train", "validation", "test"):
        path = S / dataset / f"{split}.jsonl"

        for x in read(path):
            record_id = str(x["record_id"])
            previous = record_split.get(record_id)

            if previous is not None and previous != split:
                cross_split_record_ids += 1
            else:
                record_split[record_id] = split

    result = {
        "counts": counts,
        "duplicate_record_ids": duplicate_record_ids,
        "cross_split_record_id_leakage": cross_split_record_ids
    }

    if dataset == "abo":
        result["shared_image_ids_across_splits"] = len(shared_images)

    if dataset == "mave":
        result["cross_split_asin_leakage"] = cross_split_asins

    if dataset == "rakuten":
        result["official_test_artifact_present"] = True
        result["train_validation_only_contain_official_train"] = (
            train_validation_official_flags
        )
        result["test_contains_official_test"] = (
            test_official_flags
        )

    checks = [
        result["duplicate_record_ids"] == 0,
        result["cross_split_record_id_leakage"] == 0
    ]

    if dataset == "abo":
        checks.append(
            result["shared_image_ids_across_splits"] == 0
        )

    if dataset == "mave":
        checks.append(
            result["cross_split_asin_leakage"] == 0
        )

    if dataset == "rakuten":
        checks.extend([
            result["train_validation_only_contain_official_train"],
            result["test_contains_official_test"]
        ])

    result["overall_valid"] = all(checks)

    return result


def main():
    print("=" * 70)
    print("PHASE A SPLIT VALIDATION")
    print("=" * 70)

    R.mkdir(parents=True, exist_ok=True)

    report = {
        "validator": "phase_a_split_validation",
        "version": "v002",
        "memory_strategy": "streaming JSONL validation",
        "datasets": {}
    }

    for dataset in ("rakuten", "abo", "mave"):
        report["datasets"][dataset] = validate_dataset(dataset)

    report["overall_valid"] = all(
        result["overall_valid"]
        for result in report["datasets"].values()
    )

    output = R / "phase_a_split_validation_v002.json"

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print(f"\nReport written to: {output}")

    raise SystemExit(
        0 if report["overall_valid"] else 1
    )


if __name__ == "__main__":
    main()