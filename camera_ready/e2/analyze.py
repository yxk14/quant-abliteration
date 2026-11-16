#!/usr/bin/env python3
"""E2 analysis, run after GATE C. Reads labels from e2/gateC_labels.csv (the
HTML download) or e2/gateC_packet.labels.csv (the terminal tool)."""
import csv, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *

OUT = CR / "e2"
src = next((p for p in [OUT / "gateC_labels.csv", OUT / "gateC_packet.labels.csv"] if p.exists()), None)
if src is None:
    print("GATE C labels not found (e2/gateC_labels.csv). Nothing to do."); sys.exit(0)
lab2 = {r["item_id"]: r["label"].strip().upper() for r in csv.DictReader(open(src))}
key = json.loads((OUT / "gateC_key.json").read_text())
H = {h["item_id"]: h for h in human_labels()}
ids = [i for i in key if i in lab2]
missing = [i for i in key if i not in lab2]
a1 = [H[i]["human"] for i in ids]; a2 = [lab2[i] for i in ids]
nl = [i for i in ids if key[i]["nonlatin"]]
a1n = [H[i]["human"] for i in nl]; a2n = [lab2[i] for i in nl]
res = dict(n=len(ids), missing=missing, source=str(src),
           kappa_all=kappa(a1, a2), agree_all=agreement(a1, a2),
           n_nonlatin=len(nl), kappa_nonlatin=kappa(a1n, a2n) if nl else None,
           agree_nonlatin=agreement(a1n, a2n) if nl else None,
           # on the 14: native-reader labels vs the translation-based relabels
           native_vs_translation={i: dict(translation_relabel=H[i]["human"], native=lab2[i],
                                           original_label=H[i]["human_original"], judge=H[i]["judge"]) for i in nl},
           disagreements=[dict(item_id=i, annotator1=H[i]["human"], annotator2=lab2[i], judge=H[i]["judge"])
                          for i in ids if H[i]["human"] != lab2[i]])
# Recomputed Table 3 using adjudicated labels: where the two agree, use the
# label; where they disagree, the item is listed for the owner and EXCLUDED
# until adjudicated (no auto-resolution).
agreed = {i for i in ids if H[i]["human"] == lab2[i]}
gold = {h: H[h]["human"] for h in H}
def table3(sub):
    g3 = [gold[i] for i in sub]; j = [H[i]["judge"] for i in sub]
    kw = ["REFUSAL" if H[i]["keyword_v1"] else "COMPLY" for i in sub]; g2 = [binarise(x) for x in g3]
    return dict(n=len(sub), judge_agree=agreement(g3, j), judge_kappa=kappa(g3, j),
                keyword_agree=agreement(g2, kw), keyword_kappa=kappa(g2, kw))
keep = [h for h in H if h not in set(ids) or h in agreed]
res["table3_adjudicated_pending"] = table3(keep)
res["table3_original"] = table3(list(H))
# Owner's adjudication of the disagreements, if recorded (e2/gateC_adjudication.csv):
# the adjudicated label replaces the first annotator's for Table 3 only; the
# 132-item reference used elsewhere is left as the first annotator's labels.
adj_p = OUT / "gateC_adjudication.csv"
if adj_p.exists():
    adj = {r["item_id"]: r["adjudicated"].strip().upper() for r in csv.DictReader(open(adj_p))}
    changed = [i for i in adj if adj[i] != H[i]["human"]]
    gold_adj = dict(gold); gold_adj.update(adj)
    gold = gold_adj
    res["adjudication"] = dict(n=len(adj), reference_labels_changed=changed,
                               changes={i: dict(before=H[i]["human"], after=adj[i], judge=H[i]["judge"]) for i in changed},
                               table3_adjudicated=table3(list(H)))
res["items_to_adjudicate"] = [d["item_id"] for d in res["disagreements"]]
write_json(OUT / "numbers.json", res)
print(json.dumps({k: v for k, v in res.items() if k not in ("native_vs_translation", "disagreements")}, indent=1))
print("items for the owner to adjudicate:", res["items_to_adjudicate"])
