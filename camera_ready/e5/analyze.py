#!/usr/bin/env python3
"""E5 analysis: numbers.json, fig_first_token.pdf, fig_dose.pdf, table_localization.tex."""
import json, sys, pathlib, re
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e5_lib import *
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 6.5, "axes.linewidth": .6, "pdf.fonttype": 42})
OUT = CR / "e5"

def load(subject, tag):
    f = GEN / f"{subject}__{tag}.judged.json"
    if not f.exists():
        f = GEN / f"{subject}__{tag}.json"
    return json.loads(f.read_text()) if f.exists() else None

def summ(d):
    it = d["items"]; n = len(it); sw = [i for i in it if i["nonlatin"]]; en = [i for i in it if not i["nonlatin"]]
    judged = "judge" in it[0]
    r = dict(n=n, n_switch=len(sw), switch_pct=100 * len(sw) / n, n_matrices=d["quant"]["n_matrices"],
             mean_rel_err=(d["quant"]["rel_err_sum"] / d["quant"]["n_matrices"]) if d["quant"]["n_matrices"] else 0.0,
             hit_cap_pct=100 * sum(i["hit_cap"] for i in it) / n,
             first_token_cjk_pct_of_switched=(100 * sum(i["first_cjk_token_idx"] == 0 for i in sw) / len(sw)) if sw else None,
             median_first_cjk_idx=(float(np.median([i["first_cjk_token_idx"] for i in sw]))) if sw else None)
    if judged:
        rf = lambda L: (100 * sum(i["judge"] == "REFUSAL" for i in L) / len(L)) if L else None
        r.update(judge_refusal_pct=rf(it), judge_refusal_switched=rf(sw), judge_refusal_nonswitched=rf(en),
                 keyword_refusal_pct=100 * sum(i["keyword_v1"] for i in it) / n,
                 judge_garbled_pct=100 * sum(i["judge"] == "GARBLED" for i in it) / n,
                 judge_unclear_pct=100 * sum(i["judge"] == "UNCLEAR" for i in it) / n)
    return r

numbers = dict(subjects={})
for S in ["abl", "base"]:
    runs = {}
    for f in sorted(GEN.glob(f"{S}__*.json")):
        if ".judged" in f.name: continue
        tag = f.name[len(S) + 2:-5]; runs[tag] = summ(load(S, tag))
    numbers["subjects"][S] = runs

# ---- E5.1 first token
ft = {}
for S in ["abl", "base"]:
    for tag in ["bf16", "int4_full"]:
        d = load(S, tag)
        if not d or not d.get("firsttok"): continue
        sw = np.array([i["nonlatin"] for i in d["items"]]); mass = np.array(d["firsttok"]["cjk_mass"])
        ft[f"{S}__{tag}"] = dict(n_switch=int(sw.sum()), cjk_mass_mean=float(mass.mean()),
                                 cjk_mass_mean_switched=float(mass[sw].mean()) if sw.any() else None,
                                 cjk_mass_mean_nonswitched=float(mass[~sw].mean()),
                                 argmax_cjk_n=int(sum(d["firsttok"]["argmax_is_cjk"])),
                                 auroc_mass_predicts_switch=float(roc_auc_score(sw, mass)) if 0 < sw.sum() < len(sw) else None,
                                 first_token_cjk_pct_of_switched=runs_ft if False else summ(d)["first_token_cjk_pct_of_switched"])
# cross: does BF16 first-token mass predict the INT4 switch?
for S in ["abl", "base"]:
    b, q = load(S, "bf16"), load(S, "int4_full")
    if b and q and b.get("firsttok"):
        sw = np.array([i["nonlatin"] for i in q["items"]]); mb = np.array(b["firsttok"]["cjk_mass"])
        if 0 < sw.sum() < len(sw):
            ft[f"{S}__bf16_mass_predicts_int4_switch_auroc"] = float(roc_auc_score(sw, mb))
numbers["first_token"] = ft

# figure: histogram of log10 first-token CJK mass, BF16 vs INT4, split by switch (abl)
b, q = load("abl", "bf16"), load("abl", "int4_full")
if b and q and b.get("firsttok") and q.get("firsttok"):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    bins = np.linspace(-12, 0, 25)
    for ax, (lab, d) in zip(axes, [("BF16", b), ("INT4 RTN", q)]):
        sw = np.array([i["nonlatin"] for i in q["items"]])     # split by whether the INT4 response switched
        m = np.log10(np.clip(np.array(d["firsttok"]["cjk_mass"]), 1e-12, 1))
        ax.hist(m[~sw], bins=bins, color="#0072B2", alpha=.75, label=f"stayed English (n={int((~sw).sum())})")
        ax.hist(m[sw], bins=bins, color="#E69F00", alpha=.85, label=f"switched at INT4 (n={int(sw.sum())})")
        ax.set_title(lab, fontsize=7, loc="left"); ax.set_xlabel("log$_{10}$ CJK mass, first token", fontsize=6.5)
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=6)
    axes[0].set_ylabel("prompts", fontsize=6.5); axes[1].legend(frameon=False, fontsize=5.3, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_first_token.pdf"); plt.close(fig)

# ---- E5.3 dose-response
runs = numbers["subjects"].get("abl", {})
alphas = [(0.0, "bf16"), (0.25, "scale_0.25"), (0.5, "scale_0.50"), (0.75, "scale_0.75"), (1.0, "int4_full"),
          (1.25, "scale_1.25"), (1.5, "scale_1.50")]
dose = [(a, runs[t]["switch_pct"]) for a, t in alphas if t in runs]
dose_garbled = [(a, runs[t]["judge_garbled_pct"]) for a, t in alphas if t in runs and runs[t].get("judge_garbled_pct") is not None]
gauss = [runs[t]["switch_pct"] for t in ["gauss_s0", "gauss_s1", "gauss_s2"] if t in runs]
shuf = [runs[t]["switch_pct"] for t in ["shuffle_s0", "shuffle_s1", "shuffle_s2"] if t in runs]
numbers["dose"] = dict(alpha=dose, alpha_garbled=dose_garbled, gauss=gauss, shuffle=shuf,
                       gauss_garbled=[runs[t].get("judge_garbled_pct") for t in ["gauss_s0", "gauss_s1", "gauss_s2"] if t in runs],
                       shuffle_garbled=[runs[t].get("judge_garbled_pct") for t in ["shuffle_s0", "shuffle_s1", "shuffle_s2"] if t in runs])
if dose:
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot([a for a, _ in dose], [v for _, v in dose], "o-", color="#0072B2", ms=3.5, lw=1.1, label="in Chinese: scaled RTN error, $W+\\alpha(W_{RTN}-W)$")
    for vals, col, lab in [(gauss, "#D55E00", "matched Gaussian (3 seeds)"), (shuf, "#009E73", "shuffled RTN error (3 seeds)")]:
        if vals:
            ax.axhline(np.mean(vals), color=col, lw=1.0, ls="--", label=f"{lab}: {np.mean(vals):.1f}%")
            ax.axhspan(min(vals), max(vals), color=col, alpha=.15, lw=0)
    if dose_garbled:
        ax.plot([a for a, _ in dose_garbled], [v for _, v in dose_garbled], "s:", color="#777777", ms=3, lw=1.0, label="judge GARBLED (incoherent output)")
        hi = max(dose_garbled, key=lambda t: t[1])
        if hi[1] >= 50:
            ax.annotate(f"model collapses\n({hi[1]:.0f}% garbled)", xy=hi, xytext=(hi[0] - 0.45, hi[1] - 20), fontsize=5.4, color="#555555",
                        arrowprops=dict(arrowstyle="-", color="#777777", lw=.6))
    ax.set_xlabel("$\\alpha$ (1.0 = full INT4 RTN)", fontsize=6.8); ax.set_ylabel("responses (%)", fontsize=6.8)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6); fig.tight_layout(); fig.savefig(OUT / "fig_dose.pdf"); plt.close(fig)

# ---- E5.2 localisation table
NICE = {"bf16": "BF16 (none)", "int4_full": "all linear layers (paper's INT4 RTN)", "embed_only": "embed\\_tokens only",
        "lm_head_only": "lm\\_head only", "attn_only": "attention projections only", "mlp_only": "MLP projections only",
        "int4_full_plus_embed_head": "all linear + embed + head"}
def nice(t):
    if t in NICE: return NICE[t]
    mm = re.match(r"layers_(\d+)-(\d+)", t)
    return f"layers {int(mm.group(1))}--{int(mm.group(2))}" if mm else t.replace("_", "\\_")
rows = []
for S, lab in [("abl", "abliterated"), ("base", "base")]:
    rr = numbers["subjects"].get(S, {})
    order = ["bf16", "int4_full", "embed_only", "lm_head_only"] + sorted([t for t in rr if t.startswith("layers_")]) + \
            ["attn_only", "mlp_only", "int4_full_plus_embed_head"]
    for t in order:
        if t not in rr: continue
        r = rr[t]
        jr = f"{r['judge_refusal_pct']:.1f}" if "judge_refusal_pct" in r else "---"
        js = f"{r['judge_refusal_switched']:.1f}" if r.get("judge_refusal_switched") is not None else "---"
        je = f"{r['judge_refusal_nonswitched']:.1f}" if r.get("judge_refusal_nonswitched") is not None else "---"
        kw = f"{r['keyword_refusal_pct']:.1f}" if "keyword_refusal_pct" in r else "---"
        ft0 = f"{r['first_token_cjk_pct_of_switched']:.0f}" if r.get("first_token_cjk_pct_of_switched") is not None else "---"
        gb = f"{r['judge_garbled_pct']:.1f}" if "judge_garbled_pct" in r else "---"
        rows.append(f"{lab} & {nice(t)} & {r['n_matrices']} & {100*r['mean_rel_err']:.2f} & {r['n_switch']} & {r['switch_pct']:.1f} & {ft0} & {jr} & {js} & {je} & {gb} & {kw} \\\\")
(OUT / "table_localization.tex").write_text(
    "\\setlength{\\tabcolsep}{2.2pt}\n\\begin{tabular}{@{}l l r r r r r r r r r r@{}}\n\\toprule\n"
    "& & & rel. err. & \\multicolumn{3}{c}{Chinese} & \\multicolumn{3}{c}{judge refusal \\%} & garbled & keyword \\\\\n"
    "\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
    "subject & modules quantized (INT4 RTN) & \\#mat. & \\% & $n$ & \\% & at tok.\\,0 \\% & all & switched & English & \\% & \\% \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- E5.4
f54 = OUT / "e54.json"
if f54.exists():
    e54 = json.loads(f54.read_text()); numbers["quantizers"] = {k: {kk: vv for kk, vv in v.items() if kk != "first_token"} | 
                                                                 ({"first_token": {a: b for a, b in v["first_token"].items() if a != "kl"}} if "first_token" in v else {})
                                                                 for k, v in e54.items()}
    rows = []
    for k, v in e54.items():
        if "error" in v:
            rows.append(f"{k} & \\multicolumn{{4}}{{l}}{{not loadable}} \\\\"); continue
        rows.append(f"{k} & {100*v['weight_error']['mean']:.2f} & {v['first_token']['kl_mean']:.4f} & {v['first_token']['kl_median']:.4f} & {v['first_token']['cjk_mass_mean']:.2e} & {v['corpus_switch_pct']:.1f} \\\\")
    (OUT / "table_quantizer_kl.tex").write_text(
        "\\begin{tabular}{@{}l r r r r r@{}}\n\\toprule\nquantizer & rel. weight err. \\% & KL mean & KL median & CJK mass (first tok.) & Chinese \\% (corpus) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

write_json(OUT / "numbers.json", numbers)
print(json.dumps({S: {t: (r["n_switch"], r.get("judge_refusal_pct")) for t, r in rr.items()} for S, rr in numbers["subjects"].items()}, indent=1))
print("first token:", json.dumps(ft, indent=1)); print("dose:", numbers["dose"])
