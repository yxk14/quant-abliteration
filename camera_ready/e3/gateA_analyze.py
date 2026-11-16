#!/usr/bin/env python3
"""GATE A analysis: sub-classifier accuracy and kappa against the owner's 40
labels (e3/gateA_packet.labels.csv). Then re-run finish_e3.py to drop the
'provisional' flag."""
import csv, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *
OUT = CR / "e3"
src = OUT / "gateA_packet.labels.csv"
if not src.exists():
    print("GATE A labels not found (e3/gateA_packet.labels.csv). Nothing to do."); sys.exit(0)
P = json.loads((OUT / "gateA_packet.json").read_text())
lab = {r["item_id"]: r["label"] for r in csv.DictReader(open(src))}
its = [it for it in P["items"] if it["item_id"] in lab]
g = [lab[it["item_id"]] for it in its]; p = [it["hidden"]["subclassifier"] for it in its]
res = dict(n=len(its), accuracy=agreement(g, p), kappa=kappa(g, p), confusion=confusion(g, p),
           accuracy_ci=clopper_pearson(sum(x == y for x, y in zip(g, p)), len(its)))
write_json(OUT / "gateA_analysis.json", res)
print(json.dumps(res, indent=1))
