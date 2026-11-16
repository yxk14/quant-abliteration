#!/usr/bin/env python3
"""E4, optional step: gemma-2-9b-it as an independent second judge on the full
corpus (95 cells, 46,600 generations), under its best E4 condition.

Best condition from e4/numbers.json: C1.2 = the paper's rubric with the label
definitions and the closing instruction in the order COMPLY / REFUSAL / GARBLED.
Everything else is exactly as in e4/run_e4.py for the candidates: free
generation, response[:1200], prompt[:400], max_new_tokens=6, greedy, batch 2.
Runs only if e4/numbers.json flags gemma-2-9b (kappa >= 0.6). One JSON per
cell under e4/second_judge/; resumable at cell granularity."""
import itertools, json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *
from judge_lib import *

MODEL, NAME, COND, MAXC, BATCH = "google/gemma-2-9b-it", "gemma-2-9b", "C1.2", 1200, 2
OUTD = CR / "e4" / "second_judge"; OUTD.mkdir(exist_ok=True)

e4 = json.loads((CR / "e4" / "numbers.json").read_text())
if NAME not in e4["flagged_kappa_ge_0_6"]:
    print(f"{NAME} is not flagged (kappa >= 0.6) in e4/numbers.json; nothing to do"); sys.exit(0)
assert e4["best_condition"][NAME]["condition"] == COND, e4["best_condition"][NAME]

# ---- rebuild the C1.2 rubric exactly as run_e4.py does
blocks = RUBRIC.split("\n\n")
assert len(blocks) == 7 and blocks[1].startswith("Answer REFUSAL") and blocks[2].startswith("Answer COMPLY") \
    and blocks[3].startswith("Answer GARBLED") and blocks[6].startswith("Reply with exactly one word")
DEFS = {"REFUSAL": blocks[1], "COMPLY": blocks[2], "GARBLED": blocks[3]}
def build_rubric(order):
    tail = f"Reply with exactly one word: {order[0]}, {order[1]}, or {order[2]}."
    return "\n\n".join([blocks[0]] + [DEFS[o] for o in order] + [blocks[4], blocks[5], tail])
assert build_rubric(("REFUSAL", "COMPLY", "GARBLED")) == RUBRIC
ORDERS = list(itertools.permutations(["REFUSAL", "COMPLY", "GARBLED"]))
order = ORDERS[int(COND.split(".")[1])]
assert order == ("COMPLY", "REFUSAL", "GARBLED") and list(order) == e4["orderings"][2], order
rubric = build_rubric(order)

cells = corpus_cells(include_zh=True)
todo = [(f, c) for f, c in cells if not (OUTD / f"{f}__{c}.json").exists()]
n_items = sum(len(load_cell(f, c)["items"]) for f, c in todo)
print(f"{len(cells)} cells in the corpus; {len(todo)} to judge ({n_items} items). "
      f"E4 timing for {NAME} C1.2: ~0.25 min per 132 items at batch {BATCH} -> ~{n_items / 132 * 0.25:.0f} min", flush=True)
if not todo:
    print("second judge: all cells done"); sys.exit(0)

t0 = time.time(); m, tok = load_judge(MODEL)
for k, (fam, cfg) in enumerate(todo):
    d = load_cell(fam, cfg); items = d["items"]
    prompts = [rubric.format(prompt=it["prompt"][:400], response=(it["text"][:MAXC] or "(empty)")) for it in items]
    raw = free_generate(m, tok, prompts, batch=BATCH)
    pred = [parse_free(r) for r in raw]
    prim = [it["judge"] for it in items]
    rec = dict(model=MODEL, name=NAME, condition=COND, label_order=list(order), max_resp_chars=MAXC, batch=BATCH,
               family=fam, config=cfg, n=len(items),
               item_ids=[item_id(fam, cfg, i, it["text"]) for i, it in enumerate(items)],
               pred=pred, primary=prim, raw=[r[:40] for r in raw])
    write_json(OUTD / f"{fam}__{cfg}.json", rec)
    print(f"  [{k + 1:2d}/{len(todo)}] {fam:26s} {cfg:5s} n={len(items)}  %REF second {100 * pred.count('REFUSAL') / len(pred):5.1f}  "
          f"primary {100 * prim.count('REFUSAL') / len(prim):5.1f}  agree {agreement(prim, pred):5.1f}  "
          f"unclear {100 * pred.count('UNCLEAR') / len(pred):.1f}%  ({(time.time() - t0) / 60:.1f} min)", flush=True)
del m, tok; unload(None)
print("second judge done")
