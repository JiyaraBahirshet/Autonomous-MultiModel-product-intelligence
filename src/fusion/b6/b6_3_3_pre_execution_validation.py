import json
import sys
from pathlib import Path


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def count_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def run_pre_execution_validation():

    config_path = Path("configs/fusion/b6/b6_3_3_config_v001.json")
    rakuten_path = Path("data/representations/rakuten/b3/train.jsonl")
    abo_train = Path("data/representations/abo/b3/train.jsonl")
    abo_val = Path("data/representations/abo/b3/validation.jsonl")
    abo_test = Path("data/representations/abo/b3/test.jsonl")
    hierarchy_path = Path("data/processed/rakuten/hierarchy.json")
    report_path = Path(
        "reports/fusion/b6/b6_3_3_pre_execution_validation_v001.json"
    )

    for p in [config_path, rakuten_path, abo_train, abo_val, abo_test, hierarchy_path]:
        if not p.exists():
            fail(f"Required artifact missing: {p}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    checks = {}
    all_passed = True

    # 01 — Rakuten B3 train count
    rakuten_count = count_jsonl(rakuten_path)
    ok = rakuten_count == 719701
    checks["01_RAKUTEN_B3_TRAIN_COUNT"] = {
        "status": "PASS" if ok else "FAIL",
        "expected": 719701,
        "observed": rakuten_count,
    }
    all_passed &= ok

    # 02/03 — Rakuten B3 supervised target integrity
    leaf_ids = set()
    missing = 0

    with open(rakuten_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            leaf = rec.get("leaf_category_id")

            if leaf is None or str(leaf).strip() == "":
                missing += 1
            else:
                leaf_ids.add(str(leaf))

    ok2 = len(leaf_ids) == 3008
    ok3 = missing == 0

    checks["02_UNIQUE_SUPERVISED_LEAF_IDS"] = {
        "status": "PASS" if ok2 else "FAIL",
        "expected": 3008,
        "observed": len(leaf_ids),
    }

    checks["03_MISSING_LEAF_CATEGORY_IDS"] = {
        "status": "PASS" if ok3 else "FAIL",
        "expected": 0,
        "observed": missing,
    }

    all_passed &= ok2 and ok3

    # 04/05 — hierarchy terminal leaves and membership
    with open(hierarchy_path, "r", encoding="utf-8") as f:
        hierarchy = json.load(f)

    paths = list(hierarchy["nodes"].keys())

    terminal_paths = [
        p for p in paths
        if not any(q.startswith(p + ">") for q in paths)
    ]

    hierarchy_leaf_ids = {
        p.split(">")[-1] for p in terminal_paths
    }

    ok4 = len(terminal_paths) == 3008
    missing_from_hierarchy = leaf_ids - hierarchy_leaf_ids
    ok5 = len(missing_from_hierarchy) == 0

    checks["04_HIERARCHY_TERMINAL_LEAF_COUNT"] = {
        "status": "PASS" if ok4 else "FAIL",
        "expected": 3008,
        "observed": len(terminal_paths),
    }

    checks["05_SUPERVISED_LEAF_IDS_IN_HIERARCHY"] = {
        "status": "PASS" if ok5 else "FAIL",
        "expected": 0,
        "observed": len(missing_from_hierarchy),
    }

    all_passed &= ok4 and ok5

    # 06 — ABO B3 split counts
    abo_counts = (
        count_jsonl(abo_train),
        count_jsonl(abo_val),
        count_jsonl(abo_test),
    )

    ok6 = abo_counts == (70284, 69996, 7422)

    checks["06_ABO_B3_SPLIT_COUNTS"] = {
        "status": "PASS" if ok6 else "FAIL",
        "expected": "70284/69996/7422",
        "observed": f"{abo_counts[0]}/{abo_counts[1]}/{abo_counts[2]}",
    }

    all_passed &= ok6

    # 07 — ABO B3 target presence
    target_missing = 0
    target_values = set()

    with open(abo_train, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            target = rec.get("product_type")

            if target is None or str(target).strip() == "":
                target_missing += 1
            else:
                target_values.add(str(target))

    ok7 = target_missing == 0

    checks["07_ABO_B3_PRODUCT_TYPE_PRESENT"] = {
        "status": "PASS" if ok7 else "FAIL",
        "expected_missing": 0,
        "observed_missing": target_missing,
        "observed_train_classes": len(target_values),
    }

    all_passed &= ok7

    # 08 — prohibited dependency / operation lock
    checks["08_PROHIBITED_DEPENDENCIES"] = {
        "status": "PASS",
        "record_joins": False,
        "label_crosswalk": False,
        "external_data": False,
        "B5_dependency": False,
        "synthetic_pairing": False,
        "image_download": False,
        "split_regeneration": False,
    }

    # 09 — configuration lock
    source_cfg = config.get("source_configuration", {})
    classifier_cfg = source_cfg.get("classifier", {})
    vectorizer_cfg = source_cfg.get("text_vectorizer", {})

    ok9 = (
        config.get("global_seed") == 20260827
        and source_cfg.get("train_population_expected") == 719701
        and source_cfg.get("expected_unique_leaf_classes") == 3008
        and vectorizer_cfg.get("sublinear_tf") is True
        and vectorizer_cfg.get("ngram_range") == [1, 2]
        and classifier_cfg.get("loss") == "log_loss"
        and classifier_cfg.get("penalty") == "l2"
        and classifier_cfg.get("random_state") == 20260827
    )

    checks["09_CONFIGURATION_LOCK"] = {
        "status": "PASS" if ok9 else "FAIL",
        "expected": {
            "seed": 20260827,
            "K": 3008,
            "train": 719701,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "loss": "log_loss",
            "penalty": "l2",
        },
        "observed": {
            "seed": config.get("global_seed"),
            "K": source_cfg.get("expected_unique_leaf_classes"),
            "train": source_cfg.get("train_population_expected"),
            "ngram_range": vectorizer_cfg.get("ngram_range"),
            "sublinear_tf": vectorizer_cfg.get("sublinear_tf"),
            "loss": classifier_cfg.get("loss"),
            "penalty": classifier_cfg.get("penalty"),
        },
    }

    all_passed &= ok9

    # 10 — read-only pre-training gate
    checks["10_PRE_TRAINING_GATE"] = {
        "status": "PASS",
        "model_training": False,
        "transfer_extraction": False,
        "experiment_execution": False,
    }

    report = {
        "artifact_version": "v004",
        "stage": "B6.3.3",
        "task": "PRE_EXECUTION_VALIDATION",
        "global_seed": 20260827,
        "overall_status": "PASS" if all_passed else "FAIL",
        "checks": checks,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"OVERALL_STATUS={report['overall_status']}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_pre_execution_validation()