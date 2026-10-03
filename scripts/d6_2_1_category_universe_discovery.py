from pathlib import Path
import hashlib, json, re

ROOT = Path.cwd()
ABO_ROOT = ROOT / "data" / "models" / "abo"
OUTDIR = ABO_ROOT / "phase_d" / "d6" / "feasibility"
OUTDIR.mkdir(parents=True, exist_ok=True)

CBASE = ABO_ROOT / "phase_c" / "c3" / "training" / "cbase_best.pt"
EXPECTED = "3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9"

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

print("="*72)
print("D6.2.1 ABO CATEGORY-UNIVERSE DISCOVERY")
print("="*72)
obs=sha256(CBASE)
print("C-Base SHA:", obs)
print("HASH:", "PASS" if obs==EXPECTED else "FAIL")
if obs != EXPECTED: raise RuntimeError("C-Base hash mismatch. STOP.")

def class_list(x):
    if isinstance(x,dict):
        for k,v in x.items():
            if k in ("classes","target_classes") and isinstance(v,list) and len(v)>=500 and all(isinstance(a,str) for a in v):
                return sorted(set(v))
            z=class_list(v)
            if z: return z
    elif isinstance(x,list):
        for v in x:
            z=class_list(v)
            if z: return z
    return None

man=ABO_ROOT/"phase_c"/"c3"/"training"/"cbase_training_manifest.json"
known=class_list(json.loads(man.read_text(encoding="utf-8")))
if len(known)!=549: raise RuntimeError(f"Expected 549 classes, found {len(known)}")
known_set=set(known)
print("Frozen known classes:",len(known_set))
print()

exts={".json",".jsonl",".csv",".tsv"}
hints=("abo","product","listing","metadata","canonical","category","train","validation","val","test")
files=[]
for p in (ROOT/"data").rglob("*"):
    if p.is_file() and p.suffix.lower() in exts and "phase_d" not in str(p).lower():
        if any(h in str(p).lower() for h in hints): files.append(p)
files=sorted(set(files))
print("Candidate local files:",len(files))

keys={"category","category_name","category_id","product_type","product_category","primary_category","categories"}
hits=[]
for p in files:
    try:
        if p.suffix.lower()==".jsonl":
            lines=[]
            with p.open(encoding="utf-8",errors="replace") as f:
                for _ in range(5):
                    s=f.readline()
                    if not s: break
                    lines.append(s)
            sample="".join(lines)
        else:
            sample=p.read_text(encoding="utf-8",errors="replace")[:2000000]
        low=sample.lower()
        kh=sorted(k for k in keys if f'"{k}"' in low or k in low)
        if kh:
            known_hits=sum(c in sample for c in known[:100])
            hits.append({"path":str(p.relative_to(ROOT)),"keys":kh,"known_hits_first100":known_hits,"size":p.stat().st_size})
    except Exception as e:
        hits.append({"path":str(p.relative_to(ROOT)),"error":str(e)})

hits.sort(key=lambda x:(-x.get("known_hits_first100",0),x["path"]))
print("Category-schema hits:",len(hits))
for x in hits[:50]: print(" -",x)

ev=[]
for p in files:
    if p.suffix.lower() not in {".json",".jsonl"}: continue
    try: txt=p.read_text(encoding="utf-8",errors="replace")
    except Exception: continue
    if "576" in txt:
        for m in list(re.finditer("576",txt))[:5]:
            ev.append({"path":str(p.relative_to(ROOT)),"context":txt[max(0,m.start()-150):m.end()+150].replace("\n"," ")})
print()
print("Literal-576 evidence contexts:",len(ev))
for x in ev[:20]: print(" -",x["path"],":",x["context"])

report={"phase":"D6.2.1","read_only":True,"cbase_sha256":obs,
        "known_class_source":str(man.relative_to(ROOT)),"known_class_count":len(known),
        "known_classes":known,"candidate_files": [str(x.relative_to(ROOT)) for x in files],
        "category_schema_hits":hits,"literal_576_evidence":ev,
        "status":"EVIDENCE_DISCOVERY_COMPLETE",
        "next_gate":"Resolve authoritative full ABO category-bearing source; do not infer candidate set from count alone."}
out=OUTDIR/"d6_2_1_category_universe_discovery.json"
out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
print()
print("Generated:",out)
print("NO FEASIBILITY DECISION. NO INFERENCE. NO D-TEST.")
