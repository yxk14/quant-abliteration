#!/usr/bin/env python3
"""E0: population-weighted agreement and kappa for every instrument in Table 3.

Design weights are exact: the two annotation passes are replicated from the
frozen 17,600-row frame (lib/common.py::frame_pool), giving each labelled item
its inclusion probability pi = 1 - (1-p1)(1-p2). Horvitz-Thompson weights are
1/pi. Bootstrap: 2,000 replicates resampling within design strata; census strata
(pi = 1 for every member) are held fixed.

A second, post-stratified estimate reweights to the FULL 46,600-row corpus by
(judge verdict x keyword verdict) cell. It is reported alongside, with the
assumption stated: within a cell, the sampled items are treated as exchangeable
with the population.

Candidate judges from Table 3 have no persisted per-item verdicts; E4 condition
C0 re-runs them and writes e4/c0_preds.json. If that file exists, they are
included here; otherwise their rows are marked pending.
"""
import json, sys, pathlib, random
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *

OUT = CR / "e0"
B = 2000
rng = np.random.default_rng(26)

H = human_labels()
pool, meta = frame_pool()
for h in H:
    r = pool[h["item_id"]]
    h["pi"] = r["pi"]; h["w"] = 1 / r["pi"]
    h["stratum"] = r["stratum"] if h["pass_"] == 1 else r["stratum2"]
gold3 = [h["human"] for h in H]
gold2 = [binarise(g) for g in gold3]
w = [h["w"] for h in H]

# ---- instruments: name -> (pred list aligned to H, gold list, size, n_classes)
inst = {}
inst["gemma-4-12b-it (primary)"] = ([h["judge"] for h in H], gold3, "12.2")
inst["keyword substring"] = (["REFUSAL" if h["keyword_v1"] else "COMPLY" for h in H], gold2, "---")
for fn, nm, sz in [("wildguard_refs.json", "WildGuard", "7.0"),
                   ("rejection_refs.json", "distilroberta-rejection", "0.08")]:
    j = json.loads((DATA / fn).read_text())
    assert j["gold"] == ["REFUSAL" if g == "REFUSAL" else "NOT_REFUSAL" for g in gold3], fn
    inst[nm] = (["REFUSAL" if p == "REFUSAL" else "COMPLY" for p in j["pred"]], gold2, sz)
CANDS = [("Qwen/Qwen3-4B-Instruct-2507", "Qwen3-4B-Instruct", "4.0"),
         ("microsoft/Phi-4-mini-instruct", "Phi-4-mini", "3.8"),
         ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B", "8.0"),
         ("google/gemma-2-9b-it", "gemma-2-9b", "9.2"),
         ("ibm-granite/granite-3.3-8b-instruct", "Granite-3.3-8B", "8.2"),
         ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B", "14.8")]
c0 = CR / "e4" / "c0_preds.json"
c0p = json.loads(c0.read_text()) if c0.exists() else {}
for mid, nm, sz in CANDS:
    if mid in c0p:
        assert c0p[mid]["item_ids"] == [h["item_id"] for h in H]
        inst[nm] = (c0p[mid]["pred"], gold3, sz)
    else:
        inst[nm] = (None, gold3, sz)

# ---- strata for the bootstrap
strata = defaultdict(list)
for i, h in enumerate(H):
    strata[h["stratum"]].append(i)
census = {s for s, idx in strata.items() if all(H[i]["pi"] >= 1 - 1e-12 for i in idx)}

def resample():
    out = []
    for s, idx in strata.items():
        if s in census:
            out += idx
        else:
            out += list(rng.choice(idx, size=len(idx), replace=True))
    return out

boot_idx = [resample() for _ in range(B)]

def est(pred, gold, idx=None):
    idx = idx if idx is not None else range(len(H))
    p = [pred[i] for i in idx]; g = [gold[i] for i in idx]; ww = [w[i] for i in idx]
    return weighted_agreement(g, p, ww), weighted_kappa(g, p, ww)

def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

# ---- post-stratification cells on the full corpus
pop = Counter()
for fam, cfg in corpus_cells(include_zh=False):
    for it in load_cell(fam, cfg)["items"]:
        pop[(it["judge"], int(bool(it["keyword_v1"])))] += 1
N_pop = sum(pop.values())
samp_cells = defaultdict(list)
for i, h in enumerate(H):
    samp_cells[(h["judge"], h["keyword_v1"])].append(i)
covered = sum(pop[c] for c in samp_cells)
def poststrat(pred, gold):
    """Within each (judge, keyword) cell use the HT-weighted sample agreement;
    weight cells by their population counts. Cells with no sampled item are
    dropped from both numerator and denominator (coverage is reported)."""
    num = 0.0; den = 0.0
    for c, idx in samp_cells.items():
        a, _ = est(pred, gold, idx)
        num += pop[c] * a; den += pop[c]
    # kappa: build population-weighted label cross-tab
    labs = sorted(set(gold) | set(pred))
    tab = {g: {p: 0.0 for p in labs} for g in labs}
    for c, idx in samp_cells.items():
        W = sum(w[i] for i in idx)
        for i in idx:
            tab[gold[i]][pred[i]] += pop[c] * w[i] / W
    T = sum(sum(r.values()) for r in tab.values())
    po = sum(tab[l][l] for l in labs) / T
    pe = sum(sum(tab[l].values()) / T * sum(tab[g][l] for g in labs) / T for l in labs)
    k = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return num / den, k

# ---- always-REFUSAL baseline, population weighted
base_ht = weighted_agreement(gold3, ["REFUSAL"] * len(H), w)
base_ht_b = [weighted_agreement([gold3[i] for i in ix], ["REFUSAL"] * len(ix), [w[i] for i in ix]) for ix in boot_idx]
base_ps, _ = poststrat(["REFUSAL"] * len(H), gold3)
raw_base = 100 * Counter(gold3)["REFUSAL"] / len(H)

results = {}
for nm, (pred, gold, sz) in inst.items():
    if pred is None:
        results[nm] = dict(size=sz, pending=True); continue
    a, k = agreement(gold, pred), kappa(gold, pred)
    wa, wk = est(pred, gold)
    bs = [est(pred, gold, ix) for ix in boot_idx]
    pa, pk = poststrat(pred, gold)
    results[nm] = dict(size=sz, pct_refusal=100 * pred.count("REFUSAL") / len(pred),
                       agree=a, kappa=k, w_agree=wa, w_kappa=wk,
                       w_agree_ci=ci([b[0] for b in bs]), w_kappa_ci=ci([b[1] for b in bs]),
                       ps_agree=pa, ps_kappa=pk,
                       w_pct_refusal=weighted_agreement(["REFUSAL"] * len(pred), pred, w),
                       confusion=confusion(gold, pred))
    print(f"{nm:28s} raw {a:5.1f}/{k:+.3f}   HT {wa:5.1f} [{results[nm]['w_agree_ci'][0]:.1f},{results[nm]['w_agree_ci'][1]:.1f}] "
          f"/{wk:+.3f} [{results[nm]['w_kappa_ci'][0]:+.3f},{results[nm]['w_kappa_ci'][1]:+.3f}]   post-strat {pa:5.1f}/{pk:+.3f}")

human_mix_w = {l: weighted_agreement(gold3, [l] * len(H), w) for l in LABELS}
numbers = dict(
    n_labels=len(H), frame=meta, n_population_corpus=N_pop,
    strata_sample={s: len(ix) for s, ix in strata.items()}, census_strata=sorted(census),
    bootstrap_replicates=B, bootstrap_seed=26,
    pi_summary=dict(min=min(h["pi"] for h in H), max=max(h["pi"] for h in H),
                    weights_sum=sum(w), effective_n=sum(w) ** 2 / sum(x * x for x in w)),
    always_refusal=dict(raw=raw_base, ht=base_ht, ht_ci=ci(base_ht_b), poststrat=base_ps),
    human_label_mix_weighted=human_mix_w,
    human_label_mix_raw={l: 100 * Counter(gold3)[l] / len(H) for l in LABELS},
    poststrat_cells={f"{j}|kw={k}": dict(population=pop[(j, k)], sampled=len(ix))
                     for (j, k), ix in samp_cells.items()},
    poststrat_coverage=100 * covered / N_pop,
    poststrat_uncovered={f"{j}|kw={k}": n for (j, k), n in pop.items() if (j, k) not in samp_cells},
    instruments=results,
    items=[dict(item_id=h["item_id"], pass_=h["pass_"], stratum=h["stratum"], pi=h["pi"]) for h in H])
write_json(OUT / "numbers.json", numbers)
print(f"\nalways-REFUSAL: raw {raw_base:.1f}  HT {base_ht:.1f} [{numbers['always_refusal']['ht_ci'][0]:.1f},{numbers['always_refusal']['ht_ci'][1]:.1f}]  post-strat {base_ps:.1f}")
print("weighted human mix:", {k: round(v, 1) for k, v in human_mix_w.items()})
print("effective n:", round(numbers["pi_summary"]["effective_n"], 1), " post-strat coverage %.1f%%" % numbers["poststrat_coverage"], numbers["poststrat_uncovered"])

# ---- Table 3 with weighted columns
def row(nm, r):
    if r.get("pending"):
        return f"{nm} & {r['size']} & --- & --- & --- & \\emph{{pending}} & \\emph{{pending}} \\\\"
    kk = lambda k: f"\\textbf{{{k:+.3f}}}" if k < .2 else f"{k:+.3f}"
    return (f"{nm} & {r['size']} & {r['pct_refusal']:.1f} & {r['agree']:.1f} & {kk(r['kappa'])} & "
            f"{r['w_agree']:.1f} [{r['w_agree_ci'][0]:.0f},{r['w_agree_ci'][1]:.0f}] & "
            f"{kk(r['w_kappa'])} [{r['w_kappa_ci'][0]:+.2f},{r['w_kappa_ci'][1]:+.2f}] \\\\")
names = {"gemma-4-12b-it (primary)": "gemma-4-12b-it \\emph{(primary)}",
         "keyword substring": "\\emph{keyword substring}"}
blocks = [["gemma-4-12b-it (primary)", "keyword substring"],
          [nm for _, nm, _ in CANDS], ["WildGuard", "distilroberta-rejection"]]
body = "\n\\midrule\n".join("\n".join(row(names.get(n, n), results[n]) for n in b) for b in blocks)
(OUT / "table3_weighted.tex").write_text(
    "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}l r r r r r r@{}}\n\\toprule\n"
    "& & & \\multicolumn{2}{c}{sample ($n{=}132$)} & \\multicolumn{2}{c}{population-weighted} \\\\\n"
    "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
    "instrument & Size (B) & \\%REFUSAL & agree\\% & $\\kappa$ & agree\\% [95\\% CI] & $\\kappa$ [95\\% CI] \\\\\n\\midrule\n"
    + body + "\n\\midrule\n"
    f"\\emph{{always-REFUSAL baseline}} & --- & 100.0 & {raw_base:.1f} & 0 & {base_ht:.1f} [{numbers['always_refusal']['ht_ci'][0]:.0f},{numbers['always_refusal']['ht_ci'][1]:.0f}] & 0 \\\\\n"
    "\\bottomrule\n\\end{tabular}\n")
print("wrote", OUT / "numbers.json", OUT / "table3_weighted.tex")
