from pathlib import Path
import json

PROJECT_ROOT = Path(r"D:\Autonomous MultiModel product intelligence")
OUT = PROJECT_ROOT / "reports" / "fusion" / "b6"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "b6_2_5_alignment_artifact_validation_v001.json"

SOURCES = {
    "ABO": PROJECT_ROOT / "data/representations/abo/b3",
    "MAVE": PROJECT_ROOT / "data/representations/mave/b3",
    "Rakuten": PROJECT_ROOT / "data/representations/rakuten/b3",
}
EXPECTED = {
    "ABO": {"train":70284,"validation":69996,"test":7422},
    "MAVE": {"train":2326033,"validation":290488,"test":290837},
    "Rakuten": {"train":719701,"validation":80299,"test":200000},
}

def text_available(r):
    t = r.get("b3_representation",{}).get("text",{})
    x = t.get("combined_tokens",[])
    return any(isinstance(v,str) and v.strip() for v in x) if isinstance(x,list) else bool(str(x).strip())

def attrs_available(r, ds):
    b = r.get("b3_representation",{})
    if ds == "MAVE":
        return bool(r.get("mave_attributes") or b.get("structured"))
    if ds == "ABO":
        return bool(b.get("structured"))
    return False

def category_available(r, ds):
    if ds == "Rakuten":
        return bool(r.get("category_id_path") or r.get("category_path") or r.get("node"))
    return bool(r.get("node") or r.get("category") or r.get("product_type"))

def visual_available(r, ds):
    if ds == "ABO":
        m = r.get("b3_representation",{}).get("image",{}).get("main",{})
        return bool(m.get("physical_file_available") and m.get("usable_for_local_image_model"))
    if ds == "MAVE":
        # Reference-only signal; never physical-image availability.
        return bool(r.get("image_url") or r.get("image_urls") or r.get("images") or r.get("image"))
    return False

def scan(ds):
    result = {"splits":{}, "duplicate_record_ids":0, "cross_split_record_id_overlap":{}}
    ids_by_split = {}
    versions = set()
    for split in ("train","validation","test"):
        p = SOURCES[ds] / f"{split}.jsonl"
        if not p.exists():
            raise FileNotFoundError(p)
        ids, dup = set(), 0
        counts = {"records":0,"text":0,"attributes":0,"category":0,"visual":0}
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                counts["records"] += 1
                rid = str(r.get("record_id"))
                if rid in ids: dup += 1
                ids.add(rid)
                b = r.get("b3_representation",{})
                versions.add(str(b.get("representation_version")))
                counts["text"] += text_available(r)
                counts["attributes"] += attrs_available(r,ds)
                counts["category"] += category_available(r,ds)
                counts["visual"] += visual_available(r,ds)
        counts["expected_records"] = EXPECTED[ds][split]
        counts["record_count_match"] = counts["records"] == EXPECTED[ds][split]
        result["splits"][split] = counts
        result["duplicate_record_ids"] += dup
        ids_by_split[split] = ids
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        result["cross_split_record_id_overlap"][f"{a}_{b}"] = len(ids_by_split[a] & ids_by_split[b])
    result["representation_versions"] = sorted(versions)
    result["total_records"] = sum(v["records"] for v in result["splits"].values())
    return result, set().union(*ids_by_split.values())

def main():
    datasets, idsets = {}, {}
    failures = []
    for ds in SOURCES:
        datasets[ds], idsets[ds] = scan(ds)
        for sp,v in datasets[ds]["splits"].items():
            if not v["record_count_match"]: failures.append(f"{ds}:{sp}:count")
        if datasets[ds]["duplicate_record_ids"]: failures.append(f"{ds}:duplicate_ids")
        if any(datasets[ds]["cross_split_record_id_overlap"].values()):
            failures.append(f"{ds}:cross_split_overlap")

    pairs = {}
    names = list(SOURCES)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            pairs[f"{a}__{b}"] = {"record_id_overlap":len(idsets[a] & idsets[b])}
            if pairs[f"{a}__{b}"]["record_id_overlap"]: failures.append(f"{a}__{b}:cross_dataset_overlap")

    report = {
        "status":"PASS" if not failures else "FAIL",
        "stage":"B6.2.5",
        "artifact_version":"v001",
        "purpose":"Validate construction inputs for the common semantic alignment layer without cross-dataset joins.",
        "datasets":datasets,
        "cross_dataset_record_id_overlap":pairs,
        "mave_visual_semantics":"reference_only_not_physical_image",
        "mave_supervision_boundary":7148,
        "controls":{
            "B3_inputs_frozen":True,
            "B5_frozen":True,
            "record_level_cross_dataset_join":False,
            "label_space_merge":False,
            "external_backfill":False,
            "model_training":False,
            "split_regeneration":False
        },
        "failures":failures,
        "next_stage":"B6.3 only after B6.2.5 PASS and review"
    }
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("B6.2.5",report["status"])
    print("Report:",REPORT)
    for ds in SOURCES:
        print(ds,{
            "records":datasets[ds]["total_records"],
            "text":sum(x["text"] for x in datasets[ds]["splits"].values()),
            "attributes":sum(x["attributes"] for x in datasets[ds]["splits"].values()),
            "category":sum(x["category"] for x in datasets[ds]["splits"].values()),
            "visual":sum(x["visual"] for x in datasets[ds]["splits"].values())
        })
    print("Cross-dataset record-ID overlaps:",pairs)

if __name__ == "__main__":
    main()
