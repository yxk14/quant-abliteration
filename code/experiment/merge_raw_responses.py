#!/usr/bin/env python3
"""Merge the 300-prompt extension into the RAW response files.

merge_extension.py merged only *.judged.json, leaving the raw *.json at 200
rows. Any later re-judge therefore silently rebuilds 200-row judged files and
regresses every n=500 number in both papers. Merging the raw files removes that
trap. Same rule as merge_extension.py: append, then verify.
"""
import json, shutil, sys
from pathlib import Path
RESP=Path("analysis/responses"); ARCH=Path("analysis/responses_ext_merged")
APPLY="--apply" in sys.argv
def load(p): return json.loads(Path(p).read_text())
done=skip=0; bad=[]
for ext in sorted(ARCH.glob("*-ext__*.json")):
    if ".judged" in ext.name: continue
    fam,cfg=ext.name[:-len(".json")].split("__",1)
    base=RESP/f"{fam[:-len('-ext')]}__{cfg}.json"
    if not base.exists(): bad.append(f"{base.name}: missing base"); continue
    d0=load(base); d1=load(ext); i0,i1=d0["items"],d1["items"]
    if len(i0)==500: skip+=1; continue
    if len(i0)!=200 or len(i1)!=300: bad.append(f"{base.name}: {len(i0)}+{len(i1)}"); continue
    if set(i["prompt"] for i in i0) & set(i["prompt"] for i in i1):
        bad.append(f"{base.name}: duplicate prompts"); continue
    merged=dict(d0); merged["items"]=i0+i1; merged["n_prompts"]=500; merged["extension"]=ext.name
    if APPLY:
        bk=base.with_name(base.name.replace(".json",".raw200.json"))
        if not bk.exists(): shutil.copy2(base,bk)
        base.write_text(json.dumps(merged,indent=1))
        chk=load(base)["items"]
        assert len(chk)==500 and len({i["prompt"] for i in chk})==500, base
    done+=1
print(f"merged={done} already500={skip} problems={bad if bad else 'none'}")
print("APPLIED" if APPLY else "DRY RUN")
