from pathlib import Path
import json
import platform
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_ROOT / "reports" / "baseline" / "abo" / "abo_b4.2_report.json"
STATS_PATH = PROJECT_ROOT / "data" / "models" / "abo" / "b4.2" / "statistics.json"

TEXT_RESULTS = {
    "status": "complete",
    "target_field": "product_type",
    "target_rule": "first non-empty value",
    "train": {"records": 70284, "classes": 553},
    "validation": {
        "records": 69996, "unseen_target_records": 59,
        "accuracy": 0.9082173956560905,
        "macro_f1": 0.08524528962807053,
        "weighted_f1": 0.8980940672973458
    },
    "test": {
        "records": 7422, "unseen_target_records": 30,
        "accuracy": 0.5684523809523809,
        "macro_f1": 0.09387840495350011,
        "weighted_f1": 0.5050464332987978
    },
    "vocabulary_size": 256395,
    "representation_source": "B3 v002 b3_representation.text.combined_tokens"
}

IMAGE_RESULTS = {
    "status": "complete",
    "target_field": "product_type",
    "target_rule": "first non-empty value",
    "train": {"records": 70284, "physically_eligible_records": 69823, "classes": 549},
    "validation": {
        "records": 69996, "physically_eligible_records": 69926,
        "accuracy": 0.838278, "macro_f1": 0.134345
    },
    "test": {
        "records": 7422, "physically_eligible_records": 7378,
        "accuracy": 0.390961, "macro_f1": 0.166236
    },
    "representation_source": "B3 v002 b3_representation.image.main",
    "preprocessing": {
        "color": "RGB", "resize": "64x64", "interpolation": "bilinear",
        "dtype": "float32", "scaling": "divide by 255",
        "representation": "flattened pixel vector"
    },
    "training": {"epochs": 30, "batch_size": 256, "checkpoint_resume": True}
}

report = {
    "report": "abo_b4.2_baseline",
    "version": "v002",
    "dataset": "ABO",
    "stage": "B4.2",
    "status": "PASS",
    "execution_modality": "all",
    "purpose": "Dataset-specific baseline models before multimodal fusion.",
    "target_contract": {"field": "product_type", "selection_rule": "first non-empty value"},
    "input_representation": {"representation_version": "v002", "source": "data/representations/abo/b3"},
    "split_integrity": {
        "train_records": 70284, "validation_records": 69996, "test_records": 7422,
        "duplicate_record_ids": 0, "cross_split_record_id_leakage": 0, "status": "PASS"
    },
    "baselines": {"text_only": TEXT_RESULTS, "image_only": IMAGE_RESULTS},
    "fusion": {"performed": False, "reserved_for": "B5"},
    "provenance": {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": 20260827
    },
    "integrity_constraints": {
        "B2_modified": False, "B3_modified": False,
        "frozen_splits_regenerated": False, "images_downloaded": False,
        "image_paths_guessed": False, "synthetic_images_created": False
    },
    "finalization": {
        "type": "report_only",
        "training_rerun": False,
        "image_evaluation_rerun": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Consolidates completed B4.2 text-only and image-only runs without rerunning training."
    }
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

stats = {
    "report": "abo_b4.2_baseline", "version": "v002", "status": "PASS",
    "text_only": TEXT_RESULTS, "image_only": IMAGE_RESULTS,
    "fusion_performed": False, "seed": 20260827,
    "split_integrity": report["split_integrity"],
    "finalization": report["finalization"]
}
STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print("B4.2 v002 consolidated report created.")
print(f"Report: {REPORT_PATH}")
print(f"Stats : {STATS_PATH}")
print("Training rerun: NO")
print("Image evaluation rerun: NO")
print("B2/B3 modified: NO")
print("Frozen splits regenerated: NO")
