#!/usr/bin/env python3
"""Score every E5 generation file with the keyword classifier and the paper's
judge (gemma-4-12b-it, identical rubric/settings to judge_refusals.py).
Writes e5/gen/<name>.judged.json; skips files already scored."""
import json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *
from judge_lib import *
GEN = CR / "e5" / "gen"
files = [f for f in sorted(GEN.glob("*.json")) if ".judged" not in f.name
         and not f.with_name(f.stem + ".judged.json").exists()]
print(f"{len(files)} files to judge", flush=True)
if files:
    m, tok = load_judge("google/gemma-4-12b-it")
    for f in files:
        d = json.loads(f.read_text()); t0 = time.time()
        prompts = [rubric_prompt(it["prompt"], it["text"]) for it in d["items"]]
        raw = free_generate(m, tok, prompts, batch=8)
        for it, r in zip(d["items"], raw):
            it["judge"] = parse_free(r)
            it["keyword_v1"] = bool(M.classify_v1(it["text"], d["max_new"], it["n_tokens"])[0])
        d["judge_model"] = "google/gemma-4-12b-it"
        f.with_name(f.stem + ".judged.json").write_text(json.dumps(d, indent=1, ensure_ascii=False))
        n = len(d["items"]); ref = sum(i["judge"] == "REFUSAL" for i in d["items"])
        print(f"  {f.stem:36s} judge refusal {100*ref/n:5.1f}%  keyword {100*sum(i['keyword_v1'] for i in d['items'])/n:5.1f}%  "
              f"switch {d['n_switch']:3d}  ({time.time()-t0:.0f}s)", flush=True)
print("judging done")
