#!/usr/bin/env python3
"""E2 fallback for GATE C: a second MACHINE reader on the 60-item packet.

gemma-2-9b-it under its validated E4 condition C1.2 (labels ordered COMPLY /
REFUSAL / GARBLED; see e4/second_judge.py) reads each item's ORIGINAL text with
no translation, exactly as a Chinese-reading colleague would. Writes
e2/gateC_labels.csv in the format e2/analyze.py expects (item_id,label), plus
e2/second_reader.json with the raw outputs. If a human colleague later labels
packet.html, their CSV should replace gateC_labels.csv (this file is kept as
gateC_labels_gemma2.csv for comparison)."""
import csv, json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *
from judge_lib import *

MODEL, NAME, MAXC, BATCH = "google/gemma-2-9b-it", "gemma-2-9b", 1200, 2
OUT = CR / "e2"
packet = json.loads((OUT / "gateC_packet.json").read_text())
items = packet["items"]; assert len(items) == 60, len(items)

blocks = RUBRIC.split("\n\n")
DEFS = {"REFUSAL": blocks[1], "COMPLY": blocks[2], "GARBLED": blocks[3]}
order = ("COMPLY", "REFUSAL", "GARBLED")
rubric = "\n\n".join([blocks[0]] + [DEFS[o] for o in order] + [blocks[4], blocks[5],
                     f"Reply with exactly one word: {order[0]}, {order[1]}, or {order[2]}."])

t0 = time.time(); m, tok = load_judge(MODEL)
prompts = [rubric.format(prompt=it["prompt"][:400], response=(it["text"][:MAXC] or "(empty)")) for it in items]
raw = free_generate(m, tok, prompts, batch=BATCH)
pred = [parse_free(r) for r in raw]
del m, tok; unload(None)

rec = dict(model=MODEL, name=NAME, condition="C1.2", label_order=list(order), max_resp_chars=MAXC,
           n=len(items), item_ids=[it["item_id"] for it in items], pred=pred, raw=[r[:40] for r in raw],
           minutes=round((time.time() - t0) / 60, 1))
write_json(OUT / "second_reader.json", rec)
for name in ["gateC_labels.csv", "gateC_labels_gemma2.csv"]:
    with open(OUT / name, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["item_id", "label"])
        for it, p in zip(items, pred): w.writerow([it["item_id"], p])
from collections import Counter
print(f"second reader done in {rec['minutes']} min: {Counter(pred)}")
