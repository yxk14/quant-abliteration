#!/usr/bin/env python3
"""Sanity check S1 from the directive: recompute the published numbers from the
files. Every number must match to within 0.1 or the pipeline stops."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import *

H = human_labels()
gold = [h["human"] for h in H]
judge = [h["judge"] for h in H]
kw = ["REFUSAL" if h["keyword_v1"] else "COMPLY" for h in H]
gold_b = [binarise(g) for g in gold]
maj = 100 * Counter(gold).most_common(1)[0][1] / len(gold)

def cell_rates(fam, cfg):
    it = load_cell(fam, cfg)["items"]
    n = len(it)
    return 100 * sum(i["judge"] == "REFUSAL" for i in it) / n, 100 * sum(bool(i["keyword_v1"]) for i in it) / n, n

qb = cell_rates("Qwen2.5-7B", "BF16"); qi = cell_rates("Qwen2.5-7B", "INT4")
tot = sum(len(load_cell(f, c)["items"]) for f, c in corpus_cells(include_zh=True))
got = {
    "judge_agree": agreement(gold, judge), "judge_kappa": kappa(gold, judge),
    "keyword_agree": agreement(gold_b, kw), "keyword_kappa": kappa(gold_b, kw),
    "trivial_baseline": maj,
    "qwen_bf16_judge": qb[0], "qwen_bf16_keyword": qb[1],
    "qwen_int4_judge": qi[0], "qwen_int4_keyword": qi[1],
    "n_corpus": tot, "n_labels": len(H),
}
want = {"judge_agree": 87.9, "judge_kappa": 0.740, "keyword_agree": 48.5, "keyword_kappa": 0.151,
        "trivial_baseline": 64.4, "qwen_bf16_judge": 97.0, "qwen_bf16_keyword": 97.0,
        "qwen_int4_judge": 94.2, "qwen_int4_keyword": 56.4, "n_corpus": 46600, "n_labels": 132}
ok = True
print(f"{'quantity':20s} {'published':>10s} {'recomputed':>11s}")
for k, v in want.items():
    g = got[k]; tol = 0.0005 if k.endswith("kappa") else (0 if k.startswith("n_") else 0.1)
    flag = abs(g - v) <= tol + 1e-9
    ok &= flag
    print(f"{k:20s} {v:>10} {g:>11.3f}  {'ok' if flag else 'MISMATCH'}")
write_json(CR / "lib" / "s1_numbers.json", dict(published=want, recomputed=got, all_match=ok))
print("S1:", "ALL MATCH" if ok else "FAILED")
sys.exit(0 if ok else 1)
