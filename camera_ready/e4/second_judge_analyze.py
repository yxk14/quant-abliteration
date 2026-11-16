#!/usr/bin/env python3
"""E4, optional step, analysis: gemma-2-9b-it (C1.2) as second judge against the
gemma-4 judge on the full corpus. Writes e4/second_judge.json and
e4/table_second_judge.tex (Table 1's BF16 vs INT4 cells under both judges)."""
import json, sys, pathlib, re
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *

SJ = CR / "e4" / "second_judge"
cells = {}
for f in sorted(SJ.glob("*.json")):
    d = json.loads(f.read_text()); cells[(d["family"], d["config"])] = d
    assert len(d["pred"]) == len(d["primary"]) == d["n"] == len(load_cell(d["family"], d["config"])["items"]), f
if not cells:
    print("no second-judge cells yet"); sys.exit(0)
all_cells = corpus_cells(include_zh=True)
missing = [f"{f}__{c}" for f, c in all_cells if (f, c) not in cells]

def stats(prim, pred):
    b = lambda L: [binarise(x) for x in L]
    pc = lambda L, l: 100 * L.count(l) / len(L)
    return dict(n=len(prim), primary_refusal=pc(prim, "REFUSAL"), second_refusal=pc(pred, "REFUSAL"),
                primary_garbled=pc(prim, "GARBLED"), second_garbled=pc(pred, "GARBLED"),
                primary_unclear=pc(prim, "UNCLEAR"), second_unclear=pc(pred, "UNCLEAR"),
                agree=agreement(prim, pred), kappa=kappa(prim, pred),
                agree_bin=agreement(b(prim), b(pred)), kappa_bin=kappa(b(prim), b(pred)),
                confusion=confusion(prim, pred, labels=["REFUSAL", "COMPLY", "GARBLED", "UNCLEAR"]))

per_cell = {f"{fam}__{cfg}": stats(d["primary"], d["pred"]) for (fam, cfg), d in cells.items()}
P = sum((d["primary"] for d in cells.values()), []); Q = sum((d["pred"] for d in cells.values()), [])
overall = stats(P, Q)
pr = np.array([v["primary_refusal"] for v in per_cell.values()]); sr = np.array([v["second_refusal"] for v in per_cell.values()])
cell_level = dict(n_cells=len(per_cell), pearson_r_refusal_rate=float(np.corrcoef(pr, sr)[0, 1]) if len(pr) > 2 else None,
                  mean_abs_diff_refusal_rate=float(np.abs(pr - sr).mean()), max_abs_diff_refusal_rate=float(np.abs(pr - sr).max()),
                  mean_signed_diff_second_minus_primary=float((sr - pr).mean()),
                  worst_cells=sorted(((k, v["primary_refusal"], v["second_refusal"]) for k, v in per_cell.items()),
                                     key=lambda t: -abs(t[1] - t[2]))[:8])

# ---- Table 1 (tab:classifier) rows: BF16 vs INT4 under both judges
T1 = [("Gemma2-9B", "gemma-2-9b-it"), ("Gemma4-12B", "gemma-4-12b-it"), ("Granite31-8B", "Granite-3.1-8B"),
      ("Granite33-8B", "Granite-3.3-8B"), ("Mistral7B", "Mistral-7B-v0.3"), ("OLMo2-7B", "OLMo-2-7B"),
      ("OLMo3-7B", "OLMo-3-7B"), ("Phi35mini", "Phi-3.5-mini"), ("Phi4-mini", "Phi-4-mini"),
      ("Qwen2.5-7B", "Qwen2.5-7B"), ("Qwen3-4B", "Qwen3-4B"), ("Llama31-8B", "Llama-3.1-8B"),
      ("Qwen2.5-7B-abl", "Qwen2.5-7B (abliterated)")]
rows = []
for fam, name in T1:
    kb, kq = f"{fam}__BF16", f"{fam}__INT4"
    if kb in per_cell and kq in per_cell:
        b, q = per_cell[kb], per_cell[kq]
        rows.append(dict(family=fam, name=name, in_table1=not fam.endswith("-abl"),
                         bf16=dict(primary=b["primary_refusal"], second=b["second_refusal"], agree=b["agree"], kappa=b["kappa"]),
                         int4=dict(primary=q["primary_refusal"], second=q["second_refusal"], agree=q["agree"], kappa=q["kappa"]),
                         delta_primary=q["primary_refusal"] - b["primary_refusal"],
                         delta_second=q["second_refusal"] - b["second_refusal"]))
for r in rows:
    r["delta_gap"] = r["delta_second"] - r["delta_primary"]
t1 = dict(rows=rows, n_rows=len(rows),
          max_abs_delta_gap=max((abs(r["delta_gap"]) for r in rows), default=None),
          worst_delta_gap=max(rows, key=lambda r: abs(r["delta_gap"]))["name"] if rows else None,
          sign_agreement=sum(1 for r in rows if (r["delta_primary"] > 0) == (r["delta_second"] > 0) or
                             (abs(r["delta_primary"]) < 2 and abs(r["delta_second"]) < 2)))

# ---- the 132 reference items: second judge vs human, and vs the E4 C1.2 run
H = human_labels(); idmap = {}
for d in cells.values():
    idmap.update(zip(d["item_ids"], d["pred"]))
got = [(h["human"], idmap[h["item_id"]]) for h in H if h["item_id"] in idmap]
labelled = dict(n_matched=len(got))
if got:
    g, p = zip(*got); labelled.update(agree=agreement(list(g), list(p)), kappa=kappa(list(g), list(p)))
    f = CR / "e4" / "results" / "google_gemma-2-9b-it__C1.2.json"
    if f.exists():
        e4r = json.loads(f.read_text()); e4map = dict(zip(e4r["item_ids"], e4r["pred"]))
        same = [idmap[i] == e4map[i] for i in e4map if i in idmap]
        labelled.update(e4_c1_2_agree=e4r["agree"], e4_c1_2_kappa=e4r["kappa"],
                        pct_identical_to_e4_run=100 * sum(same) / len(same), n_compared=len(same))

# ---- central comparison cells: arbitration by the human labels, and the direction of disagreement
hid = {h["item_id"]: h["human"] for h in H}
central = {}
for fam, cfg in [("Qwen2.5-7B-abl", "BF16"), ("Qwen2.5-7B-abl", "INT4")]:
    d = cells.get((fam, cfg))
    if not d: continue
    items = load_cell(fam, cfg)["items"]
    lab = [(hid[i], p_, q_) for i, p_, q_ in zip(d["item_ids"], d["primary"], d["pred"]) if i in hid]
    dis = [(h, a_, b_) for h, a_, b_ in lab if a_ != b_]
    nl = [(p_, q_) for it, p_, q_ in zip(items, d["primary"], d["pred"]) if is_nonlatin(it["text"])]
    g_, p_, q_ = (list(x) for x in zip(*lab)) if lab else ([], [], [])
    central[f"{fam}__{cfg}"] = dict(n_labelled=len(lab),
        primary_agree=agreement(g_, p_) if lab else None, primary_kappa=kappa(g_, p_) if lab else None,
        second_agree=agreement(g_, q_) if lab else None, second_kappa=kappa(g_, q_) if lab else None,
        n_judges_disagree=len(dis), primary_right=sum(h == a_ for h, a_, b_ in dis), second_right=sum(h == b_ for h, a_, b_ in dis),
        neither_right=sum(h != a_ and h != b_ for h, a_, b_ in dis),
        n_nonlatin=len(nl), nonlatin_primary_refusal=sum(x == "REFUSAL" for x, _ in nl), nonlatin_second_refusal=sum(x == "REFUSAL" for _, x in nl))
oc = overall["confusion"]
direction = dict(primary_refusal_second_comply=oc["REFUSAL"]["COMPLY"], primary_comply_second_refusal=oc["COMPLY"]["REFUSAL"],
                 primary_unclear=sum(oc["UNCLEAR"].values()), self_family_min_agree=min((v["agree"] for k, v in per_cell.items() if k.startswith("Gemma2-9B__")), default=None))

numbers = dict(second_judge="gemma-2-9b-it", condition="C1.2 (label order COMPLY / REFUSAL / GARBLED)",
               primary_judge="gemma-4-12b-it", n_cells=len(cells), n_items=len(P), missing_cells=missing,
               overall=overall, cell_level=cell_level, table1=t1, labelled_items=labelled, central_cells=central, direction=direction, per_cell=per_cell)
write_json(CR / "e4" / "second_judge.json", numbers)

def f1(x): return f"{x:.1f}"
def sg(x): return f"{x:+.1f}"
lines = [f"{tex_escape(r['name'])} & {f1(r['bf16']['primary'])} & {f1(r['bf16']['second'])} & {f1(r['bf16']['agree'])} & "
         f"{f1(r['int4']['primary'])} & {f1(r['int4']['second'])} & {f1(r['int4']['agree'])} & {sg(r['delta_primary'])} & {sg(r['delta_second'])} \\\\"
         for r in rows]
(CR / "e4" / "table_second_judge.tex").write_text(
    "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}l rrr rrr rr@{}}\n\\toprule\n"
    "& \\multicolumn{3}{c}{BF16} & \\multicolumn{3}{c}{INT4} & \\multicolumn{2}{c}{INT4 $-$ BF16} \\\\\n"
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n"
    "model & gemma-4 & gemma-2 & agree & gemma-4 & gemma-2 & agree & gemma-4 & gemma-2 \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
print(f"{len(cells)} cells, {len(P)} items; missing {len(missing)}")
print(f"overall: agree {overall['agree']:.1f} kappa {overall['kappa']:+.3f}; binary agree {overall['agree_bin']:.1f} kappa {overall['kappa_bin']:+.3f}; "
      f"%REFUSAL primary {overall['primary_refusal']:.1f} second {overall['second_refusal']:.1f}; second UNCLEAR {overall['second_unclear']:.1f}% GARBLED {overall['second_garbled']:.1f}%")
print("cell level:", json.dumps({k: v for k, v in cell_level.items() if k != "worst_cells"}))
print("worst cells:", cell_level["worst_cells"][:5])
print("table1:", {k: v for k, v in t1.items() if k != "rows"})
for r in rows:
    print(f"  {r['name']:26s} BF16 {r['bf16']['primary']:5.1f}/{r['bf16']['second']:5.1f}  INT4 {r['int4']['primary']:5.1f}/{r['int4']['second']:5.1f}  delta {r['delta_primary']:+5.1f}/{r['delta_second']:+5.1f}")
print("labelled:", labelled); print("central:", json.dumps(central, indent=0)); print("direction:", direction)
