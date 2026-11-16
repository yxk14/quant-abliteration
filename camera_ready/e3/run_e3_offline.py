#!/usr/bin/env python3
"""E3 part 1 (CPU): tag every keyword miss and keyword false positive in the
English-battery corpus, except that Latin-script misses need the sub-classifier
(run_subclassifier.py). Writes e3/tags_offline.json and the sub-classifier
input list e3/subclassifier_input.json."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *

OUT = CR / "e3"
tags = {}      # "fam|cfg|idx" -> category
need = []      # items for the sub-classifier
counts = Counter()
for fam, cfg in corpus_cells():
    for i, it in enumerate(load_cell(fam, cfg)["items"]):
        kw = bool(it["keyword_v1"]); j = it["judge"]
        key = f"{fam}|{cfg}|{i}"
        if j == "REFUSAL" and not kw:
            if is_nonlatin(it["text"]):
                tags[key] = "language_switch"
            else:
                need.append(dict(key=key, prompt=it["prompt"], text=it["text"]))
        elif kw and j != "REFUSAL":
            tags[key] = {"GARBLED": "degenerate", "UNCLEAR": "unclear_judge",
                         "COMPLY": "disclaimer_then_comply"}[j]
        if key in tags:
            counts[tags[key]] += 1
counts["latin_miss_pending_subclassifier"] = len(need)
print(dict(counts))
write_json(OUT / "tags_offline.json", tags)
write_json(OUT / "subclassifier_input.json", need)
print(f"wrote {len(tags)} offline tags and {len(need)} items for the sub-classifier")
