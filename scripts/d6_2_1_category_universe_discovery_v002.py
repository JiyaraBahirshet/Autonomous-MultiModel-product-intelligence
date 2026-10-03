import json, hashlib, csv, re
from pathlib import Path

ROOT = Path.cwd()
EXPECTED_SHA = '3ae988f0d4a2e186aa6fc896191cd5739c0092c4c89ff9293f8924c9ba0439d9'
EXPECTED_KNOWN = 549
EXPECTED_FULL = 576

CBASE = ROOT/'data/models/abo/phase_c/c3/training/cbase_best.pt'
EMB_MAN = ROOT/'data/models/abo/phase_c/c3/embeddings/c3_embedding_manifest.json'
TRAIN_MAN = ROOT/'data/models/abo/phase_c/c3/training/cbase_training_manifest.json'
OUT = ROOT/'data/models/abo/phase_d/d6/feasibility'
OUT.mkdir(parents=True, exist_ok=True)

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def loadj(p):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

def find_class_list(x):
    # Explicitly prioritize the known schema: target.classes
    if isinstance(x,dict):
        t=x.get('target')
        if isinstance(t,dict) and isinstance(t.get('classes'),list): return t['classes'], 'target.classes'
        if isinstance(x.get('classes'),list): return x['classes'], 'classes'
        for v in x.values():
            r=find_class_list(v)
            if r: return r
    elif isinstance(x,list):
        # Do not recursively accept arbitrary lists as class lists.
        return None
    return None

print('D6.2.1 v002 ABO CATEGORY-UNIVERSE DISCOVERY')
print('READ-ONLY AUDIT')
print('  - no model inference')
print('  - no D-Test access')
print('  - no threshold selection')
print('  - no artifact modification')
print('  - no synthetic/external data')

print('\n1. FROZEN C-BASE CHECK')
obs=sha256(CBASE)
print('  observed:',obs)
print('  expected:',EXPECTED_SHA)
print('  HASH:', 'PASS' if obs==EXPECTED_SHA else 'FAIL')
if obs != EXPECTED_SHA: raise SystemExit('STOP: C-Base hash mismatch')

print('\n2. FROZEN 549 CLASS-NAME SOURCE')
em=loadj(EMB_MAN)
classes, path = find_class_list(em)
if not classes: raise SystemExit('STOP: no explicit class-name list found in embedding manifest')
classes=[str(c) for c in classes]
print('  source:', EMB_MAN)
print('  field:', path)
print('  class count:', len(classes))
print('  unique count:', len(set(classes)))
if len(classes)!=EXPECTED_KNOWN or len(set(classes))!=EXPECTED_KNOWN:
    raise SystemExit('STOP: frozen class-name universe is not exactly 549 unique classes')

print('\n3. INDEPENDENT C-BASE COUNT CHECK')
tm=loadj(TRAIN_MAN)
print('  integrity.target_classes:', tm.get('integrity',{}).get('target_classes'))
print('  model.num_classes:', tm.get('model',{}).get('num_classes'))
print('  training_config.num_classes:', tm.get('training_config',{}).get('num_classes'))

print('\n4. DISCOVER CATEGORY-BEARING LOCAL ARTIFACTS')
roots=[ROOT/'data', ROOT/'reports', ROOT/'configs']
files=[]
for r in roots:
    if not r.exists(): continue
    for p in r.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.json','.jsonl','.csv','.tsv'}:
            files.append(p)
print('  candidate files:',len(files))

hits=[]
category_keys={'category','categories','product_type','product_category','category_name','category_id','target_category','label','labels'}
for p in files:
    s=''
    try:
        if p.stat().st_size>200*1024*1024: continue
        s=p.read_text(encoding='utf-8',errors='ignore')
    except Exception: continue
    low=s.lower()
    if any(k in low for k in category_keys):
        hits.append(str(p.relative_to(ROOT)))
print('  category-bearing candidates:',len(hits))
for x in hits[:80]: print('   ',x)

print('\n5. EXPLICIT 576 EVIDENCE')
evidence=[]
for rel in hits:
    p=ROOT/rel
    try: s=p.read_text(encoding='utf-8',errors='ignore')
    except: continue
    if re.search(r'\b576\b',s): evidence.append(rel)
print('  files containing literal 576:',len(evidence))
for x in evidence[:80]: print('   ',x)

print('\n6. ATTEMPT STRUCTURED FULL-UNIVERSE EXTRACTION')
# Search JSON objects for explicit category collections, but only accept an exact 576-sized list/set.
candidates=[]
for rel in hits:
    p=ROOT/rel
    if p.suffix.lower()!='.json': continue
    x=loadj(p)
    if x is None: continue
    stack=[(x,'$')]
    while stack:
        obj,path0=stack.pop()
        if isinstance(obj,dict):
            for k,v in obj.items():
                if isinstance(v,list) and len(v)==EXPECTED_FULL:
                    # Only consider lists whose key suggests categories/classes/labels.
                    kl=str(k).lower()
                    if any(z in kl for z in ['categor','class','label','product_type']):
                        vals=[str(a) for a in v]
                        if len(set(vals))==EXPECTED_FULL:
                            candidates.append((rel,path0+'.'+str(k),vals))
                elif isinstance(v,(dict,list)): stack.append((v,path0+'.'+str(k)))
        elif isinstance(obj,list):
            for i,v in enumerate(obj[:1000]):
                if isinstance(v,(dict,list)): stack.append((v,path0+f'[{i}]'))
print('  exact-576 structured candidates:',len(candidates))
for rel,pth,vals in candidates[:20]:
    print('   ',rel,'::',pth)

result={
 'status':'EVIDENCE_COLLECTED',
 'cbase_sha256':obs,
 'frozen_known_class_count':len(classes),
 'frozen_known_classes':classes,
 'independent_counts':{
   'integrity.target_classes':tm.get('integrity',{}).get('target_classes'),
   'model.num_classes':tm.get('model',{}).get('num_classes'),
   'training_config.num_classes':tm.get('training_config',{}).get('num_classes')},
 'category_bearing_files':hits,
 'explicit_576_files':evidence,
 'exact_576_structured_candidates':[
   {'file':rel,'path':pth,'classes':vals} for rel,pth,vals in candidates],
 'candidate_categories_not_yet_established':True,
 'no_inference_performed':True,
 'd_test_accessed':False,
 'open_set_threshold_selected':False,
}
(OUT/'d6_2_1_category_universe_discovery_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('\n7. STATUS')
if candidates:
    print('  Status: FULL_UNIVERSE_CANDIDATE_FOUND — requires provenance/split/exposure validation')
else:
    print('  Status: FULL_UNIVERSE_NOT_FOUND — requires targeted source identification')
print('\nGenerated:')
print(' ',OUT/'d6_2_1_category_universe_discovery_results.json')
print('STOP: no candidate declared FEASIBLE; no open-set inference performed.')
