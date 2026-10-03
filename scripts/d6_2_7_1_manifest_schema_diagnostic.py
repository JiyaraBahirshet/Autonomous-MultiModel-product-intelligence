import json
from pathlib import Path

ROOT = Path.cwd()

FILES = {
    "D2 fresh inference":
        ROOT / "data/models/abo/phase_d/d2/inference/d2_fresh_inference_manifest.json",
    "D2 calibration":
        ROOT / "data/models/abo/phase_d/d2/calibration/d2_calibration_manifest.json",
    "D3":
        ROOT / "data/models/abo/phase_d/d3/development/d3_selective_prediction_manifest.json",
    "D4":
        ROOT / "data/models/abo/phase_d/d4/development/d4_human_review_routing_manifest.json",
    "D5":
        ROOT / "data/models/abo/phase_d/d5/development/d5_reliability_error_analysis_manifest.json",
}

def show(obj, indent=0, max_depth=5):
    prefix = " " * indent

    if indent // 2 >= max_depth:
        print(prefix + "<max depth>")
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(prefix + f"{k}:")
                show(v, indent + 2, max_depth)
            else:
                print(prefix + f"{k}: {v!r}")

    elif isinstance(obj, list):
        print(prefix + f"[list length={len(obj)}]")
        for i, v in enumerate(obj[:10]):
            print(prefix + f"  [{i}]")
            show(v, indent + 4, max_depth)

    else:
        print(prefix + repr(obj))


for name, path in FILES.items():
    print("=" * 80)
    print(name)
    print(path)
    print("=" * 80)

    if not path.exists():
        print("MISSING")
        continue

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("TOP-LEVEL KEYS:")
    print(list(data.keys()))
    print()

    print("FULL MANIFEST STRUCTURE:")
    show(data, max_depth=4)
    print()

print("=" * 80)
print("D6.2.7.1 DIAGNOSTIC COMPLETE")
print("=" * 80)
