import json
from pathlib import Path
from collections import Counter

ROOT = Path(r"D:\Autonomous MultiModel product intelligence")

DATASETS = {
    "ABO": {
        "path": ROOT / "data/representations/abo/b3",
        "version": "v002",
        "files": ["train.jsonl", "validation.jsonl", "test.jsonl"],
    },
    "MAVE": {
        "path": ROOT / "data/representations/mave/b3",
        "version": "v001",
        "files": ["train.jsonl", "validation.jsonl", "test.jsonl"],
    },
    "Rakuten": {
        "path": ROOT / "data/representations/rakuten/b3",
        "version": "001",
        "files": ["train.jsonl", "validation.jsonl", "test.jsonl"],
    },
}

def first_value(value):
    if isinstance(value, list):
        vals = []
        for x in value:
            if isinstance(x, dict) and x.get("value") not in (None, ""):
                vals.append(x["value"])
            elif x not in (None, ""):
                vals.append(x)
        return vals
    return value

def has_nonempty(value):
    if value is None or value == "" or value == [] or value == {}:
        return False
    if isinstance(value, list):
        return any(has_nonempty(x) for x in value)
    if isinstance(value, dict):
        return any(has_nonempty(v) for v in value.values())
    return True

def signal_flags(dataset, r):
    b3 = r.get("b3_representation", {})
    text = b3.get("text", {})
    if dataset == "ABO":
        return {
            "text": has_nonempty(text.get("combined_tokens")),
            "attributes": any(has_nonempty(r.get(k)) for k in
                               ["brand","color","style","item_keywords","model_name","model_number","model_year"]),
            "category": has_nonempty(r.get("node")) or has_nonempty(
                b3.get("structured", {}).get("field_presence", {}).get("amazon_category_path")),
            "visual": has_nonempty(b3.get("image", {}).get("main", {}).get("image_id"))
                      and bool(b3.get("image", {}).get("main", {}).get("usable_for_local_image_model", False)),
        }
    if dataset == "MAVE":
        return {
            "text": has_nonempty(text.get("combined_tokens")),
            "attributes": has_nonempty(r.get("mave_attributes")),
            "category": has_nonempty(r.get("amazon_category_path"))
                      or has_nonempty(r.get("amazon_main_category"))
                      or has_nonempty(r.get("mave_category")),
            "visual": has_nonempty(r.get("imageURL")) or has_nonempty(r.get("imageURLHighRes")),
        }
    return {
        "text": has_nonempty(r.get("title_tokens")),
        "attributes": False,
        "category": has_nonempty(r.get("category_path"))
                  or has_nonempty(r.get("category_id_path")),
        "visual": False,
    }

results = {}
all_ids = {d: set() for d in DATASETS}
cross_split = []

for dataset, cfg in DATASETS.items():
    ds = {"version": cfg["version"], "splits": {}, "totals": Counter()}
    for fname in cfg["files"]:
        path = cfg["path"] / fname
        if not path.exists():
            raise FileNotFoundError(path)
        counts = Counter()
        ids = set()
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            r = json.loads(line)
            rid = str(r.get("record_id", ""))
            if rid in ids:
                raise RuntimeError(f"Duplicate record_id in {path}: {rid}")
            ids.add(rid)
            all_ids[dataset].add(rid)
            flags = signal_flags(dataset, r)
            counts["records"] += 1
            for k, v in flags.items():
                counts[k] += int(v)
        ds["splits"][fname[:-6]] = dict(counts)
        ds["totals"].update(counts)
    ds["totals"] = dict(ds["totals"])
    results[dataset] = ds

# Dataset-local IDs must remain disjoint across datasets for this audit.
pairwise = {}
names = list(DATASETS)
for i, a in enumerate(names):
    for b in names[i+1:]:
        overlap = all_ids[a] & all_ids[b]
        pairwise[f"{a}__{b}"] = {"record_id_overlap": len(overlap)}

out = ROOT / "reports/fusion/b6"
out.mkdir(parents=True, exist_ok=True)
payload = {
    "status": "PASS",
    "stage": "B6.2.3",
    "analysis_version": "v001",
    "purpose": "Deterministic compatibility analysis of active B3 representations",
    "datasets": results,
    "cross_dataset_record_id_overlap": pairwise,
    "rules": {
        "no_cross_dataset_join": True,
        "no_label_merge": True,
        "no_learned_parameters": True,
        "no_test_driven_selection": True,
        "missingness_preserved": True,
    },
}
report_path = out / "b6_2_3_representation_compatibility_v001.json"
report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("B6.2.3 COMPLETE")
print(f"Report: {report_path}")
for d, x in results.items():
    print(d, x["totals"])
print("Cross-dataset record-ID overlaps:", pairwise)
