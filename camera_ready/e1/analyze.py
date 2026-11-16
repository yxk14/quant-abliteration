#!/usr/bin/env python3
"""E1 analysis, run after GATE B (labels in e1/gateB_packet.labels.csv).
Per cell: judge and keyword accuracy and kappa with Clopper-Pearson 95% CIs,
and the cell's refusal rate from each instrument (exact, every item is scored)
and from the humans (stratified Horvitz-Thompson estimate over the 40-item
sample with a within-stratum bootstrap CI). Headline table: human vs judge vs
keyword refusal rate per cell."""
import csv, json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *

OUT = CR / "e1"
pk = next((p for p in [OUT / "gateB_packet.json", OUT / "gateB_packet_untranslated.json"] if p.exists()))
src = pk.with_suffix(".labels.csv")
if not src.exists():
    src = OUT / "gateB_packet_untranslated.labels.csv"
if not src.exists():
    print("GATE B labels not found (e1/gateB_packet.labels.csv). Nothing to do."); sys.exit(0)
P = json.loads(pk.read_text())
lab = {r["item_id"]: r["label"].strip().upper() for r in csv.DictReader(open(src))}
# The packet excludes the items already in the 132-item reference set, so the
# HT estimate covers the remainder of each cell. For a like-for-like comparison
# with the instruments' full-cell rates, the reference labels of the excluded
# items are folded back in exactly (they are a census of that part of the cell).
HREF = {}
for h in human_labels():
    HREF.setdefault(f"{h['family']}|{h['config']}", []).append(h)
rng = np.random.default_rng(26)
res = {}; rows = []
for cell, d in P["design"].items():
    its = [it for it in P["items"] if f"{it['hidden']['family']}|{it['hidden']['config']}" == cell and it["item_id"] in lab]
    if not its:
        continue
    g3 = [lab[it["item_id"]] for it in its]; j = [it["hidden"]["judge"] for it in its]
    kw = ["REFUSAL" if it["hidden"]["keyword_v1"] else "COMPLY" for it in its]; g2 = [binarise(x) for x in g3]
    n = len(its)
    ja = sum(x == y for x, y in zip(g3, j)); ka = sum(x == y for x, y in zip(g2, kw))
    # stratified HT estimate of the human refusal rate
    by = defaultdict(list)
    for it, g in zip(its, g3):
        by[it["hidden"]["stratum"]].append(g == "REFUSAL")
    N = sum(s["N_h"] for s in d["strata"].values())
    def ht(samples):
        return 100 * sum(d["strata"][s]["N_h"] * np.mean(v) for s, v in samples.items()) / N
    est = ht(by)
    boots = [ht({s: list(rng.choice(v, size=len(v), replace=True)) for s, v in by.items()}) for _ in range(2000)]
    # The bootstrap collapses to zero width when every sampled stratum is
    # unanimous (e.g. 0/29 standing for 420 items), so the reported interval is
    # a Jeffreys-prior posterior interval per stratum, Beta(k+1/2, n-k+1/2),
    # propagated through the same stratum weights by Monte Carlo.
    jdraw = np.zeros(20000)
    for s_, v in by.items():
        k = sum(v); jdraw += d["strata"][s_]["N_h"] * rng.beta(k + 0.5, len(v) - k + 0.5, size=20000)
    jeff = [float(np.percentile(100 * jdraw / N, 2.5)), float(np.percentile(100 * jdraw / N, 97.5))]
    ref = HREF.get(cell, []); n_ref = len(ref); ref_refusal = sum(h["human"] == "REFUSAL" for h in ref)
    ref_judge = sum(h["judge"] == "REFUSAL" for h in ref)
    full = lambda e: 100 * (ref_refusal + e / 100 * N) / (N + n_ref)
    jr_rem = 100 * sum(s["N_h"] for k, s in d["strata"].items() if k.startswith("REFUSAL|")) / N
    kr_rem = 100 * sum(s["N_h"] for k, s in d["strata"].items() if k.endswith("kw=1")) / N
    r = dict(n=n, judge_acc=100 * ja / n, judge_acc_ci=clopper_pearson(ja, n), judge_kappa=kappa(g3, j),
             keyword_acc=100 * ka / n, keyword_acc_ci=clopper_pearson(ka, n), keyword_kappa=kappa(g2, kw),
             human_refusal_rate_ht=est, human_refusal_rate_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
             judge_refusal_rate=d["judge_rate_full"], keyword_refusal_rate=d["keyword_rate_full"],
             remainder=dict(N=N, human_ht=est, judge=jr_rem, keyword=kr_rem),
             reference_items_in_cell=dict(n=n_ref, human_refusal=ref_refusal, judge_refusal=ref_judge),
             human_refusal_rate_full=full(est),
             human_refusal_rate_full_ci_bootstrap=[full(float(np.percentile(boots, 2.5))), full(float(np.percentile(boots, 97.5)))],
             human_refusal_rate_full_ci=[full(jeff[0]), full(jeff[1])], human_refusal_rate_jeffreys_ci=jeff,
             confusion_judge=confusion(g3, j), confusion_keyword=confusion(g2, kw),
             strata={s: dict(n=len(v), human_refusal=float(np.mean(v)), **d["strata"][s]) for s, v in by.items()})
    res[cell] = r
    fe, (flo, fhi) = r["human_refusal_rate_full"], r["human_refusal_rate_full_ci"]
    rows.append(f"{tex_escape(cell.replace('|', ' / '))} & {fe:.1f} [{flo:.0f},{fhi:.0f}] & "
                f"{d['judge_rate_full']:.1f} & {d['keyword_rate_full']:.1f} & {100*ja/n:.0f} [{r['judge_acc_ci'][0]:.0f},{r['judge_acc_ci'][1]:.0f}] & "
                f"{r['judge_kappa']:+.2f} & {100*ka/n:.0f} [{r['keyword_acc_ci'][0]:.0f},{r['keyword_acc_ci'][1]:.0f}] & {r['keyword_kappa']:+.2f} \\\\")
    print(f"{cell:22s} n={n:3d} human {fe:5.1f} [{flo:.1f},{fhi:.1f}] (remainder HT {est:.1f}; +{n_ref} reference items)  judge {d['judge_rate_full']:5.1f}  keyword {d['keyword_rate_full']:5.1f} | judge acc {100*ja/n:5.1f} k {r['judge_kappa']:+.2f} | kw acc {100*ka/n:5.1f} k {r['keyword_kappa']:+.2f}")
write_json(OUT / "numbers.json", dict(source=str(src), n_labelled=len(lab), cells=res))
(OUT / "table_cells.tex").write_text(
    "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}l r r r r r r r@{}}\n\\toprule\n"
    "& \\multicolumn{3}{c}{refusal rate (\\%)} & \\multicolumn{2}{c}{judge} & \\multicolumn{2}{c}{keyword} \\\\\n"
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
    "cell & human [95\\% CI] & judge & keyword & acc\\% [CI] & $\\kappa$ & acc\\% [CI] & $\\kappa$ \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("wrote e1/numbers.json, e1/table_cells.tex")
