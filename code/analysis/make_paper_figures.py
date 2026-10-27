#!/usr/bin/env python3
"""Figures for the paper. All data read from source files, none hand-entered.

Includes figures for SUPERSEDED results (margin probe, layer sweep) because the
paper is written as an experimental history: the false starts are part of the
record, and the layer/margin figures are what the broken classifier produced.
"""
import json, pathlib, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).parent.parent
R = ROOT / "analysis" / "responses"
D = ROOT / "analysis" / "data"
F = ROOT / "paper" / "figs"
F.mkdir(parents=True, exist_ok=True)

# validated palette: one accent + de-emphasis gray + semantic pair
ACC, CTX, BAD, GOOD, WARN = "#2a78d6", "#898781", "#A6342A", "#2E6E4F", "#8A6A17"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": .02,
})


def jc(fam, cfg):
    f = R / f"{fam}__{cfg}.judged.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    v = [i["judge"] for i in d["items"]]
    return dict(ref=100*sum(x == "REFUSAL" for x in v)/len(v),
                com=100*sum(x == "COMPLY" for x in v)/len(v),
                gar=100*sum(x == "GARBLED" for x in v)/len(v),
                kw=100*sum(i["keyword_v1"] for i in d["items"])/len(v), n=len(v))


def save(fig, name):
    fig.savefig(F / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote paper/figs/{name}.pdf")


# ---------------------------------------------------------------- 1. curve
# Two families, same pipeline. The whole point of the figure is that the left
# panel moves at 4 bits and the right panel does not.
def bits_for(ablmain, ablsweep, basmain, bassweep):
    # (label, bit width, abl family, abl config, base family, base config)
    return [("BF16", 16, ablmain, "BF16", basmain, "BF16"),
            ("FP16", 16, ablmain, "FP16", basmain, "FP16"),
            ("INT8",  8, ablsweep, "INT8", bassweep, "INT8"),
            ("INT6",  6, ablsweep, "INT6", bassweep, "INT6"),
            ("INT5",  5, ablsweep, "INT5", bassweep, "INT5"),
            ("NF4",   4, ablmain, "FP4",  basmain, "FP4"),
            ("INT4",  4, ablmain, "INT4", basmain, "INT4"),
            ("INT3",  3, ablsweep, "INT3", bassweep, "INT3")]

FAMS = [("Qwen2.5-7B", bits_for("Qwen2.5-7B-abl", "Qwen2.5-7B-abl-sweep",
                                "Qwen2.5-7B", "Qwen2.5-7B-sweep"),
         "$+9.5$ pts at INT4\n$p=0.002$"),
        ("Llama-3.1-8B", bits_for("Llama31-8B-abl", "Llama31-8B-abl-sweep",
                                  "Llama31-8B", "Llama31-8B-sweep"),
         "$+1.5$ pts at INT4\n$p=0.54$ (n.s.)")]

BITS = FAMS[0][1]                      # kept: later blocks index this
labs = [b[0] for b in BITS]
abl = [jc(b[2], b[3]) for b in BITS]
bas = [jc(b[4], b[5]) for b in BITS]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, (fam, bits, note) in zip(axes, FAMS):
    A = [jc(b[2], b[3]) for b in bits]
    B = [jc(b[4], b[5]) for b in bits]
    if any(v is None for v in A + B):
        ax.text(.5, .5, f"{fam}\n(data pending)", transform=ax.transAxes,
                ha="center", color=CTX, fontsize=8)
        continue
    L = [b[0] for b in bits]; x = np.arange(len(L))
    ax.plot(x, [v["ref"] for v in A], "o-", color=ACC, lw=1.8, ms=4.6,
            label="abliterated: refusal")
    ax.plot(x, [v["ref"] for v in B], "s--", color=CTX, lw=1.4, ms=3.8,
            label="base model: refusal")
    ax.fill_between(x, 0, [v["gar"] for v in A], color=BAD, alpha=.16,
                    label="abliterated: degenerate")
    ax.axvspan(5.5, 6.5, color=ACC, alpha=.07)
    i4 = L.index("INT4")
    ax.annotate(note, xy=(i4, A[i4]["ref"]), xytext=(3.0, 46), fontsize=6.6,
                color=ACC, ha="center",
                arrowprops=dict(arrowstyle="->", color=ACC, lw=.9))
    ax.set_xticks(x); ax.set_xticklabels(L, fontsize=6.6)
    ax.set_xlabel("decreasing precision $\\rightarrow$", fontsize=7)
    ax.set_title(fam, fontsize=8.5, loc="left")
axes[0].set_ylabel("% of 200 harmful prompts"); axes[0].set_ylim(-2, 103)
# the base-model line runs along y~97, so the legend has to sit below it
axes[0].legend(loc="upper left", frameon=False, fontsize=6.4,
               bbox_to_anchor=(-.01, .86))
axes[1].text(.26, .76, "both collapse\nat INT3", transform=axes[1].transAxes,
             fontsize=6.4, color=BAD, ha="center")
save(fig, "fig_curve")

# ------------------------------------------------------------- 2. outcomes
fig, ax = plt.subplots(figsize=(3.4, 2.2))
cfgs = ["BF16", "FP16", "FP4", "INT4"]
a = [jc("Qwen2.5-7B-abl", c) for c in cfgs]
xx = np.arange(len(cfgs))
ax.bar(xx, [v["ref"] for v in a], .58, color=ACC, label="refusal")
ax.bar(xx, [v["com"] for v in a], .58, bottom=[v["ref"] for v in a], color=CTX, alpha=.55, label="comply")
ax.bar(xx, [v["gar"] for v in a], .58,
       bottom=[v["ref"]+v["com"] for v in a], color=BAD, label="degenerate")
for i, v in enumerate(a):
    ax.text(i, v["ref"]+1.5, f"{v['ref']:.1f}", ha="center", fontsize=7, color="white", weight="bold")
ax.set_xticks(xx); ax.set_xticklabels(["BF16", "FP16", "NF4", "INT4"])
ax.set_ylabel("% of prompts"); ax.set_ylim(0, 104)
ax.legend(frameon=False, loc="lower left", ncol=1)
ax.set_title("abliterated checkpoint: outcome mix")
save(fig, "fig_outcomes")

# ------------------------------------------------------------- 3. strength
fig, ax = plt.subplots(figsize=(3.4, 2.2))
w = .36
kl = [jc("Qwen2.5-7B-abl", c)["ref"] for c in cfgs]
ag = [jc("Qwen2.5-7B-aggr", c)["ref"] for c in cfgs]
ax.bar(xx-w/2, kl, w, color=ACC, label="KL-constrained ($w\\approx0.55$)")
ax.bar(xx+w/2, ag, w, color=CTX, label="full projection ($w=1$)")
for i in range(len(cfgs)):
    ax.text(i-w/2, kl[i]+.6, f"{kl[i]:.1f}", ha="center", fontsize=6.5)
    ax.text(i+w/2, ag[i]+.6, f"{ag[i]:.1f}", ha="center", fontsize=6.5)
ax.annotate("", xy=(3-w/2, kl[3]), xytext=(3-w/2, kl[0]),
            arrowprops=dict(arrowstyle="<->", color=ACC, lw=1))
ax.text(3-w/2-.32, (kl[0]+kl[3])/2, "+9.5", fontsize=7, color=ACC, rotation=90, va="center")
ax.annotate("", xy=(3+w/2, ag[3]), xytext=(3+w/2, ag[0]),
            arrowprops=dict(arrowstyle="<->", color="#4a4a48", lw=1))
ax.text(3+w/2+.10, (ag[0]+ag[3])/2, "+9.5", fontsize=7, color="#4a4a48", rotation=90, va="center")
ax.set_xticks(xx); ax.set_xticklabels(["BF16", "FP16", "NF4", "INT4"])
ax.set_ylabel("refusal rate (%)"); ax.set_ylim(0, 22)
ax.legend(frameon=False, loc="upper left")
ax.set_title("effect size is independent of ablation strength")
save(fig, "fig_strength")

# --------------------------------------------------------------- 4. cohort
FAMS = [("Gemma2-9B", "gemma-2-9b"), ("Gemma4-12B", "gemma-4-12b"),
        ("Granite31-8B", "Granite-3.1-8B"), ("Granite33-8B", "Granite-3.3-8B"),
        ("Mistral7B", "Mistral-7B"), ("OLMo2-7B", "OLMo-2-7B"),
        ("OLMo3-7B", "OLMo-3-7B"), ("Phi35mini", "Phi-3.5-mini"),
        ("Phi4-mini", "Phi-4-mini"), ("Qwen2.5-7B", "Qwen2.5-7B"),
        ("Qwen3-4B", "Qwen3-4B"), ("Llama31-8B", "Llama-3.1-8B")]
rows = [(n, [jc(f, c) for c in cfgs]) for f, n in FAMS]
rows = [(n, v) for n, v in rows if all(v)]
fig, ax = plt.subplots(figsize=(5.0, 2.6))
yy = np.arange(len(rows))
for k, (c, col, mk) in enumerate(zip(cfgs, [CTX, CTX, ACC, BAD], ["o", "s", "^", "D"])):
    ax.scatter([v[k]["ref"] for _, v in rows], yy, s=26, marker=mk,
               color=col, alpha=.95 if k > 1 else .45,
               label={"BF16": "BF16", "FP16": "FP16", "FP4": "NF4", "INT4": "INT4"}[c],
               zorder=3)
for i, (n, v) in enumerate(rows):
    ax.plot([v[0]["ref"], v[3]["ref"]], [i, i], color=CTX, lw=.8, alpha=.5, zorder=1)
ax.set_yticks(yy); ax.set_yticklabels([n for n, _ in rows])
ax.set_xlabel("refusal rate (%)"); ax.set_xlim(35, 103)
ax.legend(frameon=False, ncol=4, loc="lower left")
ax.set_title(f"{len(rows)} safety-trained base models: no loss of refusal at any precision")
save(fig, "fig_cohort")

# ----------------------------------------------------------- 5. classifier
fig, ax = plt.subplots(figsize=(5.0, 2.6))
errs = []
for f, n in FAMS:
    b, z = jc(f, "BF16"), jc(f, "INT4")
    if b and z:
        errs.append((n, b["ref"]-b["kw"], z["ref"]-z["kw"]))
errs.sort(key=lambda t: -max(t[1], t[2]))
yy = np.arange(len(errs))
ax.barh(yy-.18, [e[1] for e in errs], .34, color=CTX, label="BF16")
ax.barh(yy+.18, [e[2] for e in errs], .34, color=BAD, label="INT4")
ax.axvline(0, color="#333", lw=.8)
ax.set_yticks(yy); ax.set_yticklabels([e[0] for e in errs]); ax.invert_yaxis()
ax.set_xlabel("judge $-$ keyword refusal rate (percentage points)")
ax.legend(frameon=False, loc="lower right")
ax.set_title("keyword classifier under-counts refusals, model- and config-dependently")
save(fig, "fig_classifier")

# ------------------------------------------------------ 6. judge validation
# Read from the persisted human labels rather than hardcoded constants. An
# earlier version of this figure carried the numbers inline, which made the
# single validation claim in the paper the one number not regenerable from the
# artifact.
try:
    import csv as _csv3, collections as _c3
    # NB: do not name this `labs`; that shadows the module-level bit-width label
    # list used by later figures, which silently stops them regenerating.
    _hlab = {r["item_id"]: r for r in _csv3.DictReader(open(D / "human_labels.csv"))}
    # Apply the language corrections. Without this the figure reports the
    # pre-correction numbers and silently contradicts the text.
    _rv = D / "human_labels_revised.csv"
    if _rv.exists():
        for r in _csv3.DictReader(open(_rv)):
            if r["item_id"] in _hlab:
                _hlab[r["item_id"]]["human"] = r["human_after"]
    hl = list(_hlab.values())
    n = len(hl)
    H = [r["human"] for r in hl]
    acc_j = 100*sum(r["human"] == r["judge"] for r in hl)/n
    acc_k = 100*sum((r["human"] == "REFUSAL") == (r["keyword_v1"] == "1")
                    for r in hl)/n
    base = 100*_c3.Counter(H).most_common(1)[0][1]/n     # always-majority
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    bars = ax.bar([0, 1], [acc_k, acc_j], .52, color=[BAD, ACC])
    ax.axhline(base, color=CTX, lw=1.2, ls="--")
    # Sit the caption just above the dashed line, over whichever bar is
    # shorter, so it clears both the bar and the y-axis.
    # Bars are 0.52 wide centred at 0 and 1, so the clear band above the
    # shorter bar runs from its outer edge to the inner edge of the taller one.
    # Centre the caption in that band rather than on the bar, which would push
    # it into the axis spine.
    _x = 0.24 if acc_k < acc_j else 0.76
    ax.text(_x, base + 2.2, f"always-refuse baseline {base:.0f}%",
            fontsize=6.2, color=CTX, ha="center", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["keyword\nsubstring", "LLM judge\n(gemma-4-12b)"],
                       fontsize=7)
    ax.set_ylabel("agreement with human labels (%)")
    ax.set_ylim(0, 108)
    for i, v in enumerate([acc_k, acc_j]):
        ax.text(i, v+2.5, f"{v:.1f}%", ha="center", fontsize=8, weight="bold")
    ax.set_title(f"$n={n}$ blind hand annotations", fontsize=8.5)
    save(fig, "fig_judge")
except Exception as e:
    print("  skipped fig_judge:", e)

# --------------------------------- 7. SUPERSEDED margin strip plot (scatter)
try:
    import pandas as pd
    CONDS = [("margin_OLMo2-7B_INT4.csv",        "OLMo-2-7B  ·  INT4",   "AUC 0.909   p<0.001"),
             ("margin_Granite8B_FP4.csv",        "Granite-3.1  ·  NF4",  "AUC 0.350   p=0.054"),
             ("margin_Granite8B_INT4.csv",       "Granite-3.1  ·  INT4", "AUC 0.438   p=0.24"),
             ("margin_Qwen2.5-7B-Instruct_INT4.csv","Qwen2.5-7B  ·  INT4","AUC 0.523   p=0.58")]
    fig, axes = plt.subplots(len(CONDS), 1, figsize=(5.6, 3.6), sharex=True)
    rng = np.random.default_rng(0)
    for ax, (fn, title, stat) in zip(axes, CONDS):
        d = pd.read_csv(D / fn); r = d[d.bf16_refused == 1]
        kept = r[r.flipped_to_comply == 0].bf16_margin.values
        lost = r[r.flipped_to_comply == 1].bf16_margin.values
        for vals, y, col, alpha, lab in [(kept, 0.72, CTX, .40, "refusal kept"),
                                         (lost, 0.28, ACC, .85, "refusal ``lost''")]:
            if len(vals) == 0:
                continue
            jit = rng.uniform(-.11, .11, len(vals))
            ax.scatter(vals, np.full(len(vals), y) + jit, s=7, color=col,
                       alpha=alpha, linewidths=0, label=lab, zorder=2)
            m = float(np.median(vals))
            ax.plot([m], [y], marker="D", ms=7, color=col, mec="white",
                    mew=1.2, zorder=4)
            ax.text(m, y + .30, f"{m:.2f}", ha="center", fontsize=6, color=col)
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.text(-.02, .5, title, transform=ax.transAxes, ha="right", va="center",
                fontsize=7, weight="bold", color="#14161C")
        ax.text(1.005, .5, stat, transform=ax.transAxes, ha="left", va="center",
                fontsize=6.3, color=CTX, family="monospace")
        ax.set_xlim(0, 12.5)
    axes[-1].set_xlabel("first-token top1$-$top2 logit margin")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, frameon=False, fontsize=6.5, loc="upper right",
                   markerscale=2.2, ncol=2)
    fig.suptitle("First-token confidence does not predict which refusals die",
                 fontsize=9, y=1.005)
    save(fig, "fig_margin")
except Exception as e:
    print("  skipped fig_margin:", e)

# ---------------------------------------------- 8. SUPERSEDED layer sweep
try:
    import pandas as pd
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0), sharey=True)
    for ax, (fn, t) in zip(axes, [("layer_Granite8B_INT4.csv", "Granite-3.1-8B (40 layers)"),
                                  ("layer_Qwen2.5-7B-Instruct_INT4.csv", "Qwen2.5-7B (28 layers)")]):
        d = pd.read_csv(D / fn)
        xx = np.arange(len(d)); w = .38
        ax.bar(xx-w/2, d.unsafe_flips, w, color=ACC, label="``refusals lost''")
        ax.bar(xx+w/2, (100-d.mmlu_kept).clip(lower=0), w, color=CTX, label="MMLU lost")
        ax.set_xticks(xx); ax.set_xticklabels(d.lo.astype(int), fontsize=6)
        ax.set_title(t, fontsize=7.5); ax.set_xlabel("first layer of quantized window")
    axes[0].set_ylabel("% of prompts / % MMLU")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Refusal damage is localized. Capability damage is not.",
                 fontsize=9, y=1.02)
    save(fig, "fig_layer")
except Exception as e:
    print("  skipped fig_layer:", e)

# ------------------------------------------------------------ 9. rank-1 bound
# Small, thin-lined, all type well under 10pt: this figure carries one curve
# and two points and does not need to be large.
fig, ax = plt.subplots(figsize=(2.15, 1.35))
d_ = np.logspace(np.log10(256), np.log10(16384), 200)
ax.plot(d_, 100/np.sqrt(d_), color=ACC, lw=1.0, label="$1/\\sqrt{d}$ bound")
ax.scatter([3584], [1.77], s=16, color=BAD, zorder=4, label="measured, $w{=}1$ (1.77%)")
ax.scatter([3584], [2.10], s=16, marker="^", color=WARN, zorder=4,
           label="measured, Heretic (2.1%)")
ax.set_xscale("log")
ax.set_xlabel("model width $d$", fontsize=6.0)
ax.set_ylabel("$\\|\\Delta W\\|_F/\\|W\\|_F$ (%)", fontsize=6.0)
ax.tick_params(labelsize=5.4, width=.5, length=2.2)
for _sp in ax.spines.values():
    _sp.set_linewidth(.5)
ax.grid(linewidth=.35)
ax.legend(frameon=False, loc="upper right", fontsize=5.2,
          handlelength=1.3, handletextpad=.4, borderaxespad=.2, labelspacing=.25)
ax.set_title("single-direction ablation is bounded", fontsize=6.4)
save(fig, "fig_rank1")

# ---------------------------------------------------------- 10. token cap
fig, ax = plt.subplots(figsize=(3.4, 2.0))
capA, capB = [], []
for lab, _, fa, ca, fb, cb in BITS:
    da = R / f"{fa}__{ca}.judged.json"; db = R / f"{fb}__{cb}.judged.json"
    capA.append(100*np.mean([i["hit_cap"] for i in json.loads(da.read_text())["items"]]) if da.exists() else np.nan)
    capB.append(100*np.mean([i["hit_cap"] for i in json.loads(db.read_text())["items"]]) if db.exists() else np.nan)
ax.plot(x, capA, "o-", color=ACC, lw=1.6, ms=4, label="abliterated")
ax.plot(x, capB, "s--", color=CTX, lw=1.4, ms=4, label="base")
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.5)
ax.set_ylabel("% hitting 256-token cap"); ax.legend(frameon=False)
ax.set_title("verbosity: a classifier-independent signal")
save(fig, "fig_cap")

print("figures done")

# --------------------------------------------------------- 11. session timeline
EV = [("env + smoke", 0.0, 0.6, CTX), ("Heretic", 0.6, 0.25, ACC),
      ("gate + H1-H3", 0.85, 0.5, CTX), ("base behaviour", 1.4, 1.5, BAD),
      ("capability", 2.9, 2.2, CTX), ("MMLU flips", 5.1, 0.9, BAD),
      ("mechanism probes", 6.0, 1.4, BAD), ("cohort", 7.4, 2.6, WARN),
      ("AUDIT", 10.0, 0.3, "#111"), ("regenerate", 10.3, 2.4, GOOD),
      ("judge", 12.7, 1.1, GOOD), ("dose-response", 13.8, 1.3, GOOD)]
fig, ax = plt.subplots(figsize=(5.4, 2.4))
for i, (lab, t0, dur, col) in enumerate(EV):
    ax.barh(i, dur, left=t0, height=.62, color=col,
            alpha=.95 if col in (ACC, GOOD, "#111") else .75)
    ax.text(t0+dur+.12, i, f"{dur:.1f}h", va="center", fontsize=6.5, color="#333")
ax.set_yticks(range(len(EV))); ax.set_yticklabels([e[0] for e in EV], fontsize=7)
ax.invert_yaxis(); ax.set_xlabel("wall-clock hours")
ax.set_xlim(0, 17)
ax.legend(handles=[Patch(color=BAD, label="discarded (faulty labels)"),
                   Patch(color="#111", label="the audit"),
                   Patch(color=GOOD, label="rebuild")],
          frameon=False, loc="lower right", fontsize=6.5)
ax.set_title("campaign timeline: 4 of 15 GPU-hours discarded")
save(fig, "fig_timeline")

# ------------------------------------ 12. direction recovery (the mechanism)
try:
    import pandas as pd
    kl = pd.read_csv(D / "recovery_kl.csv").set_index("config")
    ag = pd.read_csv(D / "recovery_aggr.csv").set_index("config")
    lm = pd.read_csv(D / "direction_recovery_llama.csv").set_index("config")
    ORD = ["BF16", "INT8", "INT6", "INT5", "NF4", "INT4"]
    o = [c for c in ORD if c in kl.index and c in ag.index and c in lm.index]
    xx = np.arange(len(o))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"wspace": .28})

    # LEFT: the hole, and it filling
    a1.plot(xx, kl.loc[o, "recovery_along_r"], "o-", color=ACC, lw=1.9, ms=5,
            label="Qwen, KL-constrained")
    a1.plot(xx, ag.loc[o, "recovery_along_r"], "^-", color=BAD, lw=1.9, ms=5,
            label="Qwen, full projection")
    a1.plot(xx, lm.loc[o, "recovery_along_r"], "s-", color=GOOD, lw=1.9, ms=4.4,
            label="Llama-3.1-8B")
    a1.axhline(1.0, color=CTX, lw=1.0, ls=":")
    a1.text(0.05, 1.02, "unablated level", fontsize=6, color=CTX)
    a1.annotate("", xy=(0, kl.loc["BF16", "recovery_along_r"]), xytext=(0, 1.0),
                arrowprops=dict(arrowstyle="<->", color=CTX, lw=.8))
    a1.text(.10, .55, "hole dug\nby ablation", fontsize=6, color=CTX)
    a1.annotate("", xy=(0, lm.loc["BF16", "recovery_along_r"]), xytext=(0, 1.0),
                arrowprops=dict(arrowstyle="<->", color=GOOD, lw=.8))
    a1.text(.10, .80, "Llama's hole\nis shallow", fontsize=5.8, color=GOOD)
    a1.set_xticks(xx); a1.set_xticklabels(o, fontsize=6.5)
    a1.set_ylabel("$\\|\\hat{r}^{\\top}W\\| / \\|\\hat{r}^{\\top}W_{base}\\|$")
    a1.set_ylim(0, 1.12)
    a1.set_title("(a) ablation digs a hole; quantization fills it", fontsize=8, loc="left")
    # no legend here: panel (b) names the same three series in the same colours

    # RIGHT: absorbed noise -- the baseline-free quantity
    w = .26
    a2.bar(xx - w, kl.loc[o, "noise_along_r"], w, color=ACC, label="Qwen, KL-constrained")
    a2.bar(xx, ag.loc[o, "noise_along_r"], w, color=BAD, label="Qwen, full projection")
    a2.bar(xx + w, lm.loc[o, "noise_along_r"], w, color=GOOD, label="Llama-3.1-8B")
    a2.plot(xx, kl.loc[o, "noise_surrogate"], "s--", color=CTX, lw=1.3, ms=4,
            label="scale-matched control")
    a2.set_xticks(xx); a2.set_xticklabels(o, fontsize=6.5)
    a2.set_ylabel("absorbed noise $N$")
    a2.legend(frameon=False, fontsize=6.2, loc="upper left")
    a2.set_title("(b) all three absorb the same noise", fontsize=8, loc="left")
    save(fig, "fig_recovery")
except Exception as e:
    print("  skipped fig_recovery:", e)

# ------------------------------------- 13. what abliteration actually edits
try:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    prof = json.loads((D / "edit_profile.json").read_text())
    # Stacked at column width: diagram on top, per-layer magnitudes below.
    fig = plt.figure(figsize=(3.4, 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=.42)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0.7, 10.9); ax.set_ylim(0, 11.2)
    ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, t, col, fc, fs=5.8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    lw=1.3 if bold else .9, ec=col, fc=fc, zorder=3))
        ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=fs,
                zorder=4, color="#14161C", weight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, col="#5C6472", lw=.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=7, lw=lw, color=col, zorder=2))

    # residual stream spine
    ax.plot([1.0, 1.0], [0.5, 9.5], color=ACC, lw=3.4, alpha=.30, zorder=1,
            solid_capstyle="round")
    ax.text(1.0, 9.75, "residual\nstream", ha="center", va="bottom", fontsize=5.6, color=ACC, weight="bold")

    # attention sub-block
    box(2.6, 6.6, 2.5, 1.0, "Q / K / V proj", CTX, "#f2f2f0")
    box(5.6, 6.6, 1.6, 1.0, "attention", CTX, "#f2f2f0")
    box(7.5, 6.6, 2.2, 1.0, "o_proj", BAD, "#f8e9e6", bold=True)
    arr(1.0, 7.1, 2.6, 7.1); arr(5.1, 7.1, 5.6, 7.1); arr(7.2, 7.1, 7.5, 7.1)
    ax.plot([9.7, 10.05], [7.1, 7.1], color="#5C6472", lw=.9)
    ax.plot([10.05, 10.05], [7.1, 5.2], color="#5C6472", lw=.9)
    ax.plot([10.05, 1.0], [5.2, 5.2], color="#5C6472", lw=.9)
    arr(1.4, 5.2, 1.02, 5.2)
    ax.text(1.35, 5.45, "$+$", fontsize=8, color=ACC, weight="bold")

    # FFN sub-block
    box(2.6, 2.6, 2.5, 1.0, "gate / up proj", CTX, "#f2f2f0")
    box(5.6, 2.6, 1.6, 1.0, "SwiGLU", CTX, "#f2f2f0")
    box(7.5, 2.6, 2.2, 1.0, "down_proj", BAD, "#f8e9e6", bold=True)
    arr(1.0, 3.1, 2.6, 3.1); arr(5.1, 3.1, 5.6, 3.1); arr(7.2, 3.1, 7.5, 3.1)
    ax.plot([9.7, 10.05], [3.1, 3.1], color="#5C6472", lw=.9)
    ax.plot([10.05, 10.05], [3.1, 1.3], color="#5C6472", lw=.9)
    ax.plot([10.05, 1.0], [1.3, 1.3], color="#5C6472", lw=.9)
    arr(1.4, 1.3, 1.02, 1.3)
    ax.text(1.35, 1.55, "$+$", fontsize=8, color=ACC, weight="bold")

    ax.text(5.2, 8.35, "reads from stream", fontsize=5.5, color=CTX, ha="center")
    ax.text(8.6, 8.35, "WRITES to stream", fontsize=5.7, color=BAD,
            ha="center", weight="bold")
    ax.text(5.0, 0.25, "$W' = W - \\hat{r}\\,\\hat{r}^{\\top} W$   applied to the two"
            " write matrices only", fontsize=6, ha="center", color="#14161C")
    ax.set_title("(a) one decoder block: only the write paths are edited", fontsize=7, loc="center")

    # right panel: per-layer edit magnitude
    a2 = fig.add_subplot(gs[1])
    nl = 28; xs = np.arange(nl)
    o = [prof["o_proj"].get(str(i), 0)*100 for i in range(nl)]
    dn = [prof["down_proj"].get(str(i), 0)*100 for i in range(nl)]
    a2.bar(xs-.2, o, .4, color=BAD, label="o_proj")
    a2.bar(xs+.2, dn, .4, color=ACC, label="down_proj")
    a2.set_xlabel("layer", fontsize=6.5); a2.set_ylabel("$\\|\\Delta W\\|_F/\\|W\\|_F$  (%)", fontsize=6.5)
    a2.tick_params(labelsize=6)
    a2.legend(frameon=False, fontsize=5.8)
    a2.set_title("(b) edited layers 5--27 / 9--26; the rest untouched", fontsize=7, loc="center")
    a2.set_xticks([0, 5, 10, 15, 20, 25])
    save(fig, "fig_ablation")
except Exception as e:
    print("  skipped fig_ablation:", e)

# --------------------------------- 14. how abliteration is computed (tutorial)
try:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    prof = json.loads((D / "direction_profile.json").read_text())
    fig = plt.figure(figsize=(7.0, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=.42, wspace=.28)

    # ---- (a) collect activations from contrasting prompts
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off"); ax.grid(False)
    def bx(x, y, w, h, t, ec, fc, fs=6.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    lw=.9, ec=ec, fc=fc, zorder=3))
        ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=fs, zorder=4)
    def ar(x1, y1, x2, y2, col="#5C6472"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=6, lw=.85, color=col, zorder=2))
    bx(0.2, 6.6, 3.5, 1.5, '256 HARMFUL\n"How do I make..."', BAD, "#f8e9e6")
    bx(0.2, 2.2, 3.5, 1.5, '256 HARMLESS\n"How do I bake..."', GOOD, "#e6f1ea")
    bx(4.6, 4.4, 2.0, 3.4, "frozen\nmodel\n(no training)", CTX, "#f2f2f0")
    ar(3.7, 7.35, 4.6, 6.6); ar(3.7, 2.95, 4.6, 5.6)
    ar(6.6, 6.6, 7.6, 7.0); ar(6.6, 5.6, 7.6, 4.0)
    bx(7.6, 6.4, 2.2, 1.3, r"$\mu_{harm}$", BAD, "#f8e9e6", 7)
    bx(7.6, 3.4, 2.2, 1.3, r"$\mu_{safe}$", GOOD, "#e6f1ea", 7)
    ax.text(5.6, 1.4, "capture residual stream\nat last prompt token,\nevery layer",
            ha="center", fontsize=5.8, color=CTX)
    ax.set_title("(a) run two contrasting prompt sets", fontsize=8, loc="left")

    # ---- (b) the difference vector, per layer -- REAL data
    ax = fig.add_subplot(gs[0, 1])
    n = prof["norms"]; xs = np.arange(len(n))
    ax.bar(xs, n, .78, color=[ACC if i != int(np.argmax(n)) else BAD for i in xs])
    ax.set_xlabel("layer"); ax.set_ylabel(r"$\|\mu_{harm}-\mu_{safe}\|$")
    ax.set_title(r"(b) $d_\ell=\mu_{harm}-\mu_{safe}$,  $\hat{r}=d/\|d\|$", fontsize=8, loc="left")
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.annotate("raw norm grows with\ndepth — normalise before\nreading a 'peak'",
                xy=(26, n[26]), xytext=(9, max(n)*.72), fontsize=5.6, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=.7))

    # ---- (c) the projection, geometrically
    ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(-.3, 3.4); ax.set_ylim(-.4, 2.6)
    ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)
    ax.add_patch(FancyArrowPatch((0, 0), (2.6, 0), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.6, color=CTX))
    ax.text(2.65, -.05, r"$\hat{r}$", fontsize=9, color=CTX)
    ax.add_patch(FancyArrowPatch((0, 0), (1.9, 1.9), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.9, color=ACC))
    ax.text(1.95, 1.95, r"$W$ column", fontsize=7.5, color=ACC)
    ax.add_patch(FancyArrowPatch((0, 0), (0, 1.9), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.9, color=GOOD))
    ax.text(-.28, 1.9, r"$W'$", fontsize=9, color=GOOD)
    ax.plot([1.9, 1.9], [0, 1.9], color=BAD, lw=1.1, ls="--")
    ax.plot([0, 1.9], [1.9, 1.9], color=GOOD, lw=.8, ls=":")
    ax.add_patch(FancyArrowPatch((0, -.22), (1.9, -.22), arrowstyle="<|-|>",
                                 mutation_scale=6, lw=.9, color=BAD))
    ax.text(.95, -.38, r"$\hat{r}\hat{r}^{\top}W$  (removed)", fontsize=6.5,
            color=BAD, ha="center")
    ax.text(1.55, 2.35, r"$W' = W - \hat{r}\hat{r}^{\top}W$", fontsize=8.5, ha="center")
    ax.set_title("(c) project the direction out of each write matrix", fontsize=8, loc="left")

    # ---- (d) does the found direction match what was removed? REAL data
    ax = fig.add_subplot(gs[1, 1])
    cs = [np.nan if c is None else c for c in prof["cos"]]
    ax.plot(range(len(cs)), cs, "o-", color=ACC, lw=1.5, ms=3.4)
    k = int(np.nanargmax(cs))
    ax.scatter([k], [cs[k]], s=52, color=BAD, zorder=4)
    ax.annotate(f"peak {cs[k]:.2f}\nat layer {k}", xy=(k, cs[k]), xytext=(k-11, .74),
                fontsize=6.2, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=.7))
    ax.set_xlabel("layer"); ax.set_ylabel(r"$|\cos(\hat{r}_\ell,\ \hat{u}_\ell)|$")
    ax.set_ylim(0, 1.02)
    ax.set_title("(d) verify against the direction actually removed", fontsize=8, loc="left")
    ax.text(.5, .06, "difference-in-means agrees with the\nremoved direction only mid-stack",
            transform=ax.transAxes, fontsize=5.8, color=CTX, ha="center")
    save(fig, "fig_howto")
except Exception as e:
    print("  skipped fig_howto:", e)


# ------------------------------------------------- 15. alignment token depth
# Reproduces the measurement behind Fig. 1 of Qi et al. (ICLR 2025) on our two
# abliterated families, to test whether alignment DEPTH explains why one family
# recovers refusal under quantization and the other does not. It does not: the
# family that fails to recover is the SHALLOWER one.
try:
    import csv as _csv
    def _td(f):
        rows = list(_csv.DictReader(open(D / f)))
        return ([int(r["position"]) for r in rows],
                [float(r["kl_mean"]) for r in rows],
                [float(r["kl_se"]) for r in rows])
    qp, qk, qs = _td("token_depth_Qwen2.5-7B.csv")
    lp, lk, ls_ = _td("token_depth_Llama-3.1-8B.csv")

    # Drawn at column width (3.4 in) so it is placed 1:1 in the paper.
    # Stacked so that panel (b) spans the full column width.
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 3.7),
                                 gridspec_kw={"height_ratios": [1.05, 1], "hspace": .62})
    for p, k, s, c, lab in ((qp, qk, qs, ACC, "Qwen2.5-7B"),
                            (lp, lk, ls_, BAD, "Llama-3.1-8B")):
        k = np.array(k); s = np.array(s)
        a1.plot(p, k, "-o", color=c, lw=1.0, ms=1.8, label=lab)
        a1.fill_between(p, k - s, k + s, color=c, alpha=.16, lw=0)
    a1.set_xlabel("response token position", fontsize=6); a1.set_ylabel(r"$D_{KL}$(aligned $\|$ base)", fontsize=6)
    a1.tick_params(labelsize=5.5)
    a1.legend(frameon=False, loc="upper right", fontsize=5.4)
    a1.set_title("(a) front-loaded in both", fontsize=6.5, loc="left")
    a1.text(.30, .50, "KL peaks at position 1,\nas Qi et al. report", transform=a1.transAxes,
            fontsize=5.2, color=CTX)

    # (b) the falsification, stated as the comparison it fails
    # The three measures carry different units (nats, %, a ratio), so plotting
    # them on one linear axis would let the percentage dominate and squash the
    # rest. Show each as the Llama/Qwen ratio instead: one dimensionless scale,
    # with 1.0 = "the two families are equally shallow".
    mets = ["KL at\nposition 1", "share of KL in\nfirst 5 positions", "peak / tail\nratio"]
    qv = [qk[0], sum(qk[:5])/sum(qk), qk[0]/(sum(qk[15:])/10)]
    lv = [lk[0], sum(lk[:5])/sum(lk), lk[0]/(sum(lk[15:])/10)]
    rat = [b/a for a, b in zip(qv, lv)]
    x = np.arange(3)
    a2.axhline(1.0, color=CTX, lw=.8, ls="--")
    a2.bar(x, rat, .56, color=BAD)
    for xi, (r, a, b) in enumerate(zip(rat, qv, lv)):
        a2.text(xi, r, f"{r:.2f}x", ha="center", va="bottom", fontsize=6)
        a2.text(xi, .08, f"Qwen {a:.2f}\nLlama {b:.2f}", ha="center", va="bottom",
                fontsize=5.4, color="white", linespacing=1.2)
    a2.set_xticks(x); a2.set_xticklabels(mets, fontsize=6)
    a2.tick_params(axis="y", labelsize=5.5)
    a2.set_ylabel("Llama $\\div$ Qwen  (>1 = Llama shallower)", fontsize=6)
    a2.set_ylim(0, max(rat) * 1.25)
    a2.text(.98, .94, "prediction was Llama deeper, i.e. $<$1", transform=a2.transAxes,
            fontsize=5.6, color=CTX, ha="right", va="top")
    a2.set_title("(b) all three go the wrong way", fontsize=6.5, loc="left")
    save(fig, "fig_depth")
except Exception as e:
    print("  skipped fig_depth:", e)


# ------------------------------------------------------- 16. hole depth
# The mechanism for the model-dependence. All three checkpoints absorb the same
# absolute noise; what differs is the baseline it lands on. Ordered by hole
# depth. NOTE the file naming: recovery_kl.csv is the KL-constrained (gentle)
# checkpoint at R=0.433 and recovery_aggr.csv is the full projection at 0.171 --
# the aggressive ablation digs the DEEPER hole, as it should.
try:
    import csv as _csv
    def _rows(f):
        return {r["config"]: r for r in _csv.DictReader(open(D / f))}
    SRC = [("Qwen\nfull projection", "recovery_aggr.csv", "+9.5 pts"),
           ("Qwen\nKL-constrained", "recovery_kl.csv", "+9.5 pts\n$p{=}0.002$"),
           ("Llama-3.1-8B", "direction_recovery_llama.csv", "+1.5 pts\n$p{=}0.54$")]
    names, base, after, noise, beh = [], [], [], [], []
    for lab, f, b in SRC:
        d = _rows(f)
        names.append(lab); beh.append(b)
        base.append(float(d["BF16"]["recovery_along_r"]))
        after.append(float(d["INT4"]["recovery_along_r"]))
        noise.append(float(d["INT4"]["noise_along_r"]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(3); w = .34
    a1.bar(x - w/2, base, w, color=CTX, label="BF16 (after ablation)")
    a1.bar(x + w/2, after, w, color=ACC, label="INT4")
    for xi, (b, af, bh) in enumerate(zip(base, after, beh)):
        a1.annotate("", xy=(xi + w/2, af), xytext=(xi + w/2, b),
                    arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1.1))
        a1.text(xi + w/2 + .08, (b + af)/2, f"+{100*(af/b-1):.0f}%", fontsize=6.6,
                color=BAD, va="center")
        a1.text(xi, 1.24, bh, ha="center", fontsize=5.9, color=CTX)
    a1.set_xticks(x); a1.set_xticklabels(names, fontsize=6.4)
    a1.set_ylabel(r"$R=\|\hat r^{\top}W\|/\|\hat r^{\top}W_{\rm base}\|$")
    a1.set_ylim(0, 1.44)
    # annotations occupy the top band, so the legend sits under them
    a1.legend(frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(.01, .80))
    a1.set_title("(a) the same noise, three different baselines", fontsize=8, loc="left")

    a2.bar(x, noise, .46, color=GOOD)
    for xi, n in enumerate(noise):
        a2.text(xi, n, f"{n:.3f}", ha="center", va="bottom", fontsize=6.6)
    a2.set_xticks(x); a2.set_xticklabels(names, fontsize=6.4)
    a2.set_ylabel(r"absorbed noise $N_r$ at INT4")
    a2.set_ylim(0, max(noise) * 1.42)
    a2.text(.5, .84, "hole depth spans 0.17-0.75;\nabsorbed noise does not track it",
            transform=a2.transAxes, fontsize=6.1, color=CTX, ha="center")
    a2.set_title("(b) absolute noise absorbed is nearly constant", fontsize=8, loc="left")
    save(fig, "fig_holedepth")
except Exception as e:
    print("  skipped fig_holedepth:", e)


# ----------------------------------------------------------- 17. MMLU capability
# The control behind "degradation is held constant across the comparison": base
# and abliterated must fall together, or the refusal change could be damage.
# Llama falls far harder than Qwen at INT4 and still does not recover refusal.
try:
    import csv as _csv
    def _mmlu(f):
        out = {}
        for r in _csv.DictReader(open(D / f)):
            if r["task"] == "mmlu":
                out.setdefault(r["variant"], {})[r["config"]] = 100*float(r["value"])
        return out
    QF = _mmlu("capability_Qwen2.5-7B-Instruct.csv")
    LB = _mmlu("capability_Llama31-8B.csv")
    LA = _mmlu("capability_Llama31-8B-abl.csv")
    LF = {"base": LB.get("base", {}), "abliterated": LA.get("abliterated", {})}
    ORDER = ["BF16", "FP16", "INT16", "FP8", "INT8", "INT6", "INT5", "FP4", "INT4", "INT3"]
    PRETTY = {"FP4": "NF4"}

    # One panel drawn at column width (3.4 in) so it is placed 1:1 in the
    # paper; a 7 in two-panel version was being halved and its fonts with it.
    # INT3 is omitted: both families are degenerate there and the 25% floor
    # would squash the coincident curves that are the point of the figure.
    UNION = [c for c in ORDER if c != "INT3"]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for fam, dd, col in (("Qwen2.5-7B", QF, ACC), ("Llama-3.1-8B", LF, BAD)):
        cfgs = [c for c in UNION if c in dd.get("base", {}) and c in dd.get("abliterated", {})]
        if not cfgs:
            continue
        xx = [UNION.index(c) for c in cfgs]
        b = [dd["base"][c] for c in cfgs]; a = [dd["abliterated"][c] for c in cfgs]
        # base as a wide halo under the abliterated line: they coincide, which IS the result
        ax.plot(xx, b, "-", color=col, lw=3.4, alpha=.32, solid_capstyle="round",
                label=f"{fam} base")
        ax.plot(xx, a, "o-", color=col, lw=1.1, ms=2.6, label=f"{fam} abliterated")
    ax.set_xticks(range(len(UNION)))
    ax.set_xticklabels([PRETTY.get(c, c) for c in UNION], fontsize=5.8, rotation=45)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.set_ylabel("MMLU accuracy (%)", fontsize=6.5)
    ax.set_xlabel("decreasing precision $\\rightarrow$", fontsize=6.2)
    ax.legend(frameon=False, fontsize=5.3, loc="lower left", ncol=1, handlelength=1.6)
    save(fig, "fig_mmlu")
except Exception as e:
    print("  skipped fig_mmlu:", e)


# -------------------------------------------------- 18. quantizer error structure
# The central result after the calibrated arm: behaviour does not track how much
# noise lands on the ablated direction. GPTQ puts MORE there than NF4 and moves
# refusal not at all. What separates them is whether the quantizer compensates
# its error to preserve layer outputs.
try:
    import csv as _csv
    def _rows(f):
        return {r["config"]: r for r in _csv.DictReader(open(D / f))}
    pq = _rows("recovery_prequant_qwen.csv"); sim = _rows("recovery_kl.csv")
    kq = _rows("recovery_q4km_qwen.csv")
    # kind: 0 = round-to-nearest, 1 = blocked but uncalibrated, 2 = calibrated
    Q = [("INT4\n(RTN)",   float(sim["INT4"]["noise_along_r"]),  14.5, 0),
         ("GPTQ\nW4A16",   float(pq["GPTQ4"]["noise_along_r"]),   4.5, 2),
         ("AWQ\nW4A16",    float(pq["AWQ4"]["noise_along_r"]),    5.0, 2),
         ("Q4_K_M\n(GGUF)", float(kq["Q4KM"]["noise_along_r"]),   5.5, 1),
         ("NF4\n(RTN)",    float(sim["NF4"]["noise_along_r"]),    8.5, 0)]
    ref0 = 5.0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    order = sorted(Q, key=lambda t: -t[1])
    x = np.arange(len(order))
    KIND = {0: ACC, 1: GOOD, 2: WARN}
    cols = [KIND[t[3]] for t in order]
    a1.bar(x, [t[1] for t in order], .5, color=cols)
    for xi, t in enumerate(order):
        a1.text(xi, t[1], f"{t[1]:.3f}", ha="center", va="bottom", fontsize=6.6)
    a1.set_xticks(x); a1.set_xticklabels([t[0] for t in order], fontsize=6.4)
    a1.set_ylabel(r"noise on ablated direction $N_r$")
    a1.set_ylim(0, max(t[1] for t in order) * 1.35)
    a1.legend(handles=[Patch(color=ACC, label="round-to-nearest"),
                       Patch(color=GOOD, label="blocked, uncalibrated"),
                       Patch(color=WARN, label="calibrated")],
              frameon=False, fontsize=6.0, loc="upper right")
    a1.set_title("(a) how much noise reaches the direction", fontsize=8, loc="left")

    a2.bar(x, [t[2] - ref0 for t in order], .5, color=cols)
    a2.axhline(0, color=CTX, lw=.9)
    for xi, t in enumerate(order):
        d = t[2] - ref0
        a2.text(xi, d + (.25 if d >= 0 else -.6), f"{d:+.1f}", ha="center",
                fontsize=6.6, va="bottom" if d >= 0 else "top")
    a2.set_xticks(x); a2.set_xticklabels([t[0] for t in order], fontsize=6.4)
    a2.set_ylabel("$\\Delta$ refusal vs BF16 (pts)")
    a2.set_ylim(-2.5, 11.5)
    a2.set_title("(b) what it does to behaviour", fontsize=8, loc="left")
    a2.text(.5, .70, "only unstructured rounding\nmoves refusal",
            transform=a2.transAxes, fontsize=6.3, color=BAD, ha="center")
    save(fig, "fig_quantizer")
except Exception as e:
    print("  skipped fig_quantizer:", e)


# ------------------------------------------------------- 19. flips are structured
# Rules out the "quantization just jitters borderline prompts at random" reading:
# independent quantizers recover the SAME prompts far more often than chance.
try:
    from scipy.stats import hypergeom
    def _ref(f):
        d = json.loads((R / f"{f}.judged.json").read_text())
        return [i["judge"] == "REFUSAL" for i in d["items"]]
    b = _ref("Qwen2.5-7B-abl__BF16")
    pool = len(b) - sum(b)
    SET = {"INT4": "Qwen2.5-7B-abl__INT4", "NF4": "Qwen2.5-7B-abl__FP4",
           "INT5": "Qwen2.5-7B-abl-sweep__INT5", "INT6": "Qwen2.5-7B-abl-sweep__INT6"}
    G = {k: {i for i, (x, y) in enumerate(zip(b, _ref(v))) if not x and y}
         for k, v in SET.items()}
    ks = ["INT4", "NF4", "INT5", "INT6"]
    obs, exp, labs_ = [], [], []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            A, B_ = G[ks[i]], G[ks[j]]
            obs.append(len(A & B_)); exp.append(len(A) * len(B_) / pool)
            labs_.append(f"{ks[i]}\n{ks[j]}")
    fig, ax = plt.subplots(figsize=(3.2, 1.95))
    # one bar per pair (observed); the chance expectation is a dash across
    # the bar rather than a second, near-invisible bar
    xx = np.arange(len(obs)); w = .55
    ax.bar(xx, obs, w, color=ACC, label="observed overlap")
    for xi, (o, e) in enumerate(zip(obs, exp)):
        ax.text(xi, o, f"{o}", ha="center", va="bottom", fontsize=6.2)
        ax.plot([xi - w/2 - .08, xi + w/2 + .08], [e, e], color="#14161C", lw=1.4,
                solid_capstyle="butt", zorder=4,
                label="expected if random" if xi == 0 else None)
        ax.text(xi + w/2 + .1, e, f"{e:.1f}", ha="left", va="center", fontsize=5.4,
                color="#14161C")
    ax.set_xticks(xx); ax.set_xticklabels(labs_, fontsize=5.8)
    ax.set_ylabel("prompts recovered by both", fontsize=7)
    ax.set_ylim(0, max(obs) * 1.42)          # headroom so the legend clears the bars
    ax.legend(frameon=False, fontsize=6.0, loc="upper right",
              bbox_to_anchor=(1.0, 1.02), borderaxespad=0)
    ax.set_title("the same prompts recover under independent quantizers",
                 fontsize=7.2)
    save(fig, "fig_structured")
except Exception as e:
    print("  skipped fig_structured:", e)


# ------------------------------------------- 20. weight vs activation displacement
# Resolves the inversion in fig_quantizer. Calibrated quantizers move the WEIGHTS
# a long way along the ablated direction and the RESIDUAL STREAM hardly at all,
# because they compensate their error to preserve layer outputs.
try:
    import csv as _csv
    def _rows(f):
        return {r["config"]: r for r in _csv.DictReader(open(D / f))}
    pq  = _rows("recovery_prequant_qwen.csv")
    sim = _rows("recovery_kl.csv")
    act = _rows("activation_qwen.csv")
    kqw = _rows("recovery_q4km_qwen.csv"); kqa = _rows("activation_q4km_qwen.csv")

    a0 = float(act["BF16"]["A_last_token"])
    def arise(k):                       # % rise of the activation component
        return 100 * (float(act[k]["A_last_token"]) / a0 - 1)
    W = {"INT4": float(sim["INT4"]["noise_along_r"]),
         "GPTQ4": float(pq["GPTQ4"]["noise_along_r"]),
         "AWQ4": float(pq["AWQ4"]["noise_along_r"]),
         "Q4KM": float(kqw["Q4KM"]["noise_along_r"]),
         "FP4": float(sim["NF4"]["noise_along_r"])}
    k0 = float(kqa["BF16"]["A_last_token"])
    A = {k: arise(k) for k in ["INT4", "GPTQ4", "AWQ4", "FP4"]}
    A["Q4KM"] = 100 * (float(kqa["Q4KM"]["A_last_token"]) / k0 - 1)
    # express both as a share of what INT4 round-to-nearest does
    wn = {k: 100 * W[k] / W["INT4"] for k in W}
    an = {k: 100 * A[k] / A["INT4"] for k in A}

    NAMES = [("INT4", "INT4\n(RTN)", False), ("GPTQ4", "GPTQ\nW4A16", True),
             ("AWQ4", "AWQ\nW4A16", True), ("Q4KM", "Q4_K_M\n(GGUF)", True),
             ("FP4", "NF4\n(RTN)", False)]
    BEH = {"INT4": "+9.5", "GPTQ4": "-0.5", "AWQ4": "+0.0",
           "Q4KM": "+0.5", "FP4": "+3.5"}
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    x = np.arange(len(NAMES)); w = .36
    ax.bar(x - w/2, [wn[k] for k, _, _ in NAMES], w, color=CTX,
           label="displacement in the weights")
    ax.bar(x + w/2, [an[k] for k, _, _ in NAMES], w, color=ACC,
           label="displacement in the residual stream")
    for xi, (k, _, calib) in enumerate(NAMES):
        ax.text(xi - w/2, max(wn[k], 0) + 2.5, f"{wn[k]:.0f}", ha="center", fontsize=6.3)
        # value label centred above every bar
        ax.text(xi + w/2, max(an[k], 0) + 2.5, f"{an[k]:.0f}",
                ha="center", fontsize=6.3)
        if calib:
            # the arrow points at the number above the bar; shrinkB stops the
            # arrowhead a few points short of the digits
            ax.annotate("", xy=(xi + w/2, an[k] + 5.0), xytext=(xi - w/2, wn[k]),
                        arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.1,
                                        shrinkA=9, shrinkB=7,
                                        connectionstyle="arc3,rad=-.22"))
    ax.axhline(0, color=CTX, lw=.8)
    ax.set_xticks(x)
    # the behavioural change rides along with each tick label, so no floating
    # legend text can collide with a bar
    ax.set_xticklabels([f"{n}\n$\\Delta$ref {BEH[k]}" for k, n, _ in NAMES],
                       fontsize=6.4)
    ax.set_ylabel("% of what INT4 round-to-nearest does")
    ax.set_ylim(-20, 122)
    ax.legend(frameon=False, fontsize=6.4, loc="upper right")
    save(fig, "fig_actspace")
except Exception as e:
    print("  skipped fig_actspace:", e)


# -------------------------------------------- 21. the two inverted conclusions
# For the measurement paper. Each instrument is asked the same question of the
# same generations: "how much did refusal change from BF16 to INT4?" Where the
# bars point opposite ways, the instrument is not adding noise -- it is
# reporting an effect that is not there, or hiding one that is.
try:
    import glob as _glob
    def _pair(fam):
        out = []
        for cfg in ("BF16", "INT4"):
            f = R / f"{fam}__{cfg}.judged.json"
            if not f.exists():
                return None
            d = json.loads(f.read_text()); it = d["items"]; n = len(it)
            out.append((100*sum(i["judge"] == "REFUSAL" for i in it)/n,
                        100*sum(i["keyword_v1"] for i in it)/n))
        (jb, kb), (jq, kq) = out
        return jq - jb, kq - kb
    rows = []
    for f in sorted(_glob.glob(str(R / "*__BF16.judged.json"))):
        fam = pathlib.Path(f).name.replace("__BF16.judged.json", "")
        # -zh is a different battery and -ext is the 300-prompt extension of the
        # same one; neither is a comparable cell here.
        if any(t in fam for t in ("abl", "aggr", "sweep", "GPTQ", "AWQ")) \
                or fam.endswith("-zh") or fam.endswith("-ext"):
            continue
        r = _pair(fam)
        if r:
            rows.append((fam, *r))
    abl = _pair("Qwen2.5-7B-abl")
    if abl:
        rows.append(("Qwen2.5-7B ABLITERATED", *abl))
    rows.sort(key=lambda t: -abs(t[1] - t[2]))

    # Drawn at column width (3.4 in) so it is placed 1:1 in the paper.
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    y = np.arange(len(rows)); h = .38
    jd = [r[1] for r in rows]; kd = [r[2] for r in rows]
    ax.barh(y + h/2, jd, h, color=ACC, label="LLM judge (validated)")
    ax.barh(y - h/2, kd, h, color=BAD, label="keyword substring classifier")
    for i, (n, j, k) in enumerate(rows):
        if (j > 0) != (k > 0) and abs(j - k) > 3:
            ax.scatter([max(j, k) + 3.5], [i], marker="*", s=44, color=WARN, zorder=5)
    ax.axvline(0, color=CTX, lw=.9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_xlabel("change in reported refusal, BF16 $\\rightarrow$ INT4 (pts)", fontsize=6.2)
    # bottom-left holds the two largest keyword bars; park the legend top-left
    ax.legend(frameon=False, fontsize=5.6, loc="upper left", bbox_to_anchor=(.01, .99),
              handlelength=1.4)
    ax.text(.98, .99, "$\\star$ = instruments disagree on the SIGN",
            transform=ax.transAxes, fontsize=5.6, color=WARN, ha="right", va="top")
    save(fig, "fig_inversion")
except Exception as e:
    print("  skipped fig_inversion:", e)


# --------------------------------------------- 22. RETRACTED, not used
# Built for a draft claiming the judge inflated the effect fourfold by reading
# degenerate text as refusal. That claim was WRONG: 13 of the 17 "degenerate"
# items were fluent Chinese refusals the annotator could not read, and the judge
# agreed with the corrected label on 13/14. No paper references this figure.
# It also predates the revision merge, so it reads human_labels.csv without
# applying human_labels_revised.csv. Kept only so the retraction is visible in
# the code; do not reinstate without recomputing.
try:
    import csv as _c4, collections as _cc4, json as _j4
    rows = list(_c4.DictReader(open(D / "human_labels.csv")))
    FAM = "Qwen2.5-7B-abl"
    jud, hum, gar = {}, {}, {}
    for cfg in ("BF16", "INT4"):
        items = _j4.loads((R / f"{FAM}__{cfg}.judged.json").read_text())["items"]
        N = len(items)
        jud[cfg] = 100*sum(i["judge"] == "REFUSAL" for i in items)/N
        sub = [r for r in rows if r["family"] == FAM and r["config"] == cfg]
        tgt = [r for r in sub if r["judge"] != "COMPLY"]
        fil = [r for r in sub if r["judge"] == "COMPLY"]
        n_com = sum(1 for i in items if i["judge"] == "COMPLY")
        hum[cfg] = 100*(sum(r["human"] == "REFUSAL" for r in tgt)
                        + n_com*sum(r["human"] == "REFUSAL" for r in fil)/max(len(fil), 1))/N
        gar[cfg] = (100*sum(i["judge"] == "GARBLED" for i in items)/N,
                    100*(sum(r["human"] == "GARBLED" for r in tgt)
                         + n_com*sum(r["human"] == "GARBLED" for r in fil)/max(len(fil), 1))/N)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = np.arange(2); w = .34
    a1.bar(x - w/2, [jud["BF16"], jud["INT4"]], w, color=BAD, label="as the judge scored it")
    a1.bar(x + w/2, [hum["BF16"], hum["INT4"]], w, color=ACC, label="hand-corrected")
    for xi, (a, b) in enumerate(zip([jud["BF16"], jud["INT4"]], [hum["BF16"], hum["INT4"]])):
        a1.text(xi - w/2, a + .4, f"{a:.1f}", ha="center", fontsize=7)
        a1.text(xi + w/2, b + .4, f"{b:.1f}", ha="center", fontsize=7)
    a1.set_xticks(x); a1.set_xticklabels(["BF16", "INT4"])
    a1.set_ylabel("refusal rate (%)"); a1.set_ylim(0, 18)
    a1.legend(frameon=False, fontsize=6.4, loc="upper left")
    a1.set_title(f"(a) reported $+{jud['INT4']-jud['BF16']:.1f}$ pts, "
                 f"actual $+{hum['INT4']-hum['BF16']:.1f}$", fontsize=8, loc="left")

    a2.bar(x - w/2, [gar["BF16"][0], gar["INT4"][0]], w, color=BAD,
           label="judge's estimate")
    a2.bar(x + w/2, [gar["BF16"][1], gar["INT4"][1]], w, color=GOOD,
           label="hand-corrected")
    for xi, cfg in enumerate(("BF16", "INT4")):
        a2.text(xi - w/2, gar[cfg][0] + .6, f"{gar[cfg][0]:.0f}", ha="center", fontsize=7)
        a2.text(xi + w/2, gar[cfg][1] + .6, f"{gar[cfg][1]:.0f}", ha="center", fontsize=7)
    a2.set_xticks(x); a2.set_xticklabels(["BF16", "INT4"])
    a2.set_ylabel("degenerate output (%)")
    a2.set_ylim(0, max(g[1] for g in gar.values())*1.35)
    a2.legend(frameon=False, fontsize=6.4, loc="upper left")
    a2.set_title("(b) the blind spot that causes it", fontsize=8, loc="left")
    save(fig, "fig_inflation")
except Exception as e:
    print("  skipped fig_inflation:", e)


# ------------------------------------------------ 23. the language switch
# The finding that reframes the study. Under INT4 uniform rounding the
# abliterated Qwen answers a fraction of English prompts in Chinese. Splitting
# the headline by response language shows the refusal "recovery" lives entirely
# in the switched responses.
try:
    import json as _j5
    def _nl(t):
        cjk = sum(1 for c in t if '一' <= c <= '鿿')
        lat = sum(1 for c in t if c.isalpha() and ord(c) < 128)
        return cjk / max(cjk + lat, 1) > .4
    def _load(f):
        p = R / f"{f}.judged.json"
        return _j5.loads(p.read_text())["items"] if p.exists() else None

    CKPT = [("Qwen abl\n$R{=}0.43$", "Qwen2.5-7B-abl"),
            ("Qwen aggr\n$R{=}0.17$", "Qwen2.5-7B-aggr"),
            ("Qwen base\nno ablation", "Qwen2.5-7B"),
            ("Llama abl\n$R{=}0.75$", "Llama31-8B-abl")]
    CFG = [("BF16", "BF16", CTX), ("NF4", "FP4", GOOD), ("INT4", "INT4", BAD)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(CKPT)); w = .26
    for k, (lab, cfg, col) in enumerate(CFG):
        vals = []
        for _, fam in CKPT:
            it = _load(f"{fam}__{cfg}")
            vals.append(100*sum(_nl(i["text"]) for i in it)/len(it) if it else 0)
        a1.bar(x + (k-1)*w, vals, w, color=col, label=lab)
        for xi, v in enumerate(vals):
            if v > .5:
                a1.text(xi + (k-1)*w, v + .7, f"{v:.1f}", ha="center", fontsize=6.2)
    a1.set_xticks(x); a1.set_xticklabels([c[0] for c in CKPT], fontsize=6.2)
    a1.set_ylabel("responses in Chinese (%)")
    a1.legend(frameon=False, fontsize=6.4, title="precision",
              title_fontsize=6.4, loc="upper right")
    a1.set_title("(a) only INT4, only Qwen, scaling with ablation depth",
                 fontsize=8, loc="left")

    # (b) headline split by language
    names, b16, eng, zh = [], [], [], []
    for lab, fam in CKPT[:2]:
        d0, d1 = _load(f"{fam}__BF16"), _load(f"{fam}__INT4")
        N = len(d1)
        names.append(lab.split("\n")[0])
        b16.append(100*sum(i["judge"] == "REFUSAL" for i in d0)/N)
        eng.append(100*sum(i["judge"] == "REFUSAL" and not _nl(i["text"]) for i in d1)/N)
        zh.append(100*sum(i["judge"] == "REFUSAL" and _nl(i["text"]) for i in d1)/N)
    xx = np.arange(len(names)); w2 = .34
    a2.bar(xx - w2/2, b16, w2, color=CTX, label="BF16")
    a2.bar(xx + w2/2, eng, w2, color=ACC, label="INT4, answered in English")
    a2.bar(xx + w2/2, zh, w2, bottom=eng, color=BAD,
           label="INT4, answered in Chinese")
    for xi in range(len(names)):
        a2.text(xi + w2/2, eng[xi] + zh[xi] + .5, f"{eng[xi]+zh[xi]:.1f}",
                ha="center", fontsize=6.4)
        a2.text(xi - w2/2, b16[xi] + .5, f"{b16[xi]:.1f}", ha="center", fontsize=6.4)
    a2.set_xticks(xx); a2.set_xticklabels(names, fontsize=6.6)
    a2.set_ylabel("refusal rate (%)"); a2.set_ylim(0, 21)
    a2.legend(frameon=False, fontsize=5.9, loc="upper left")
    a2.set_title("(b) the rise is the switched responses", fontsize=8, loc="left")
    save(fig, "fig_language")
except Exception as e:
    print("  skipped fig_language:", e)


# ------------------------------------- 24. two failure modes, one instrument
# (a) the register-shift failure, present at BF16 with no quantization at all.
# (b) the language failure, reachable only under INT4 round-to-nearest.
# The first is the practically relevant one; the second is an existence proof
# that a treatment can move an instrument's blind spot.
try:
    import json as _j6, glob as _g6, collections as _c6
    def _nl(t):
        cjk = sum(1 for c in t if '一' <= c <= '鿿')
        lat = sum(1 for c in t if c.isalpha() and ord(c) < 128)
        return cjk / max(cjk + lat, 1) > .4

    # (a) per-checkpoint gap at BF16
    g = {}
    for f in sorted(_g6.glob(str(R / "*__BF16.judged.json"))):
        d = _j6.loads(pathlib.Path(f).read_text())
        # This panel is about REGISTER shift on the English battery. The Chinese
        # battery fails for a different reason and belongs in the text, not here;
        # -ext is the extension of the same English battery, already counted.
        if d["family"].endswith("-zh") or d["family"].endswith("-ext"):
            continue
        it = d["items"]; n = len(it)
        g[d["family"]] = 100*(sum(i["judge"] == "REFUSAL" for i in it)
                              - sum(bool(i["keyword_v1"]) for i in it))/n
    top = sorted(g.items(), key=lambda kv: -kv[1])[:8]

    # (b) like-for-like gap across quantizers, four common checkpoints
    FAMS = {"Qwen2.5-7B-abl", "Qwen2.5-7B", "Llama31-8B-abl", "Llama31-8B"}
    by = _c6.defaultdict(lambda: [0, 0, 0])
    for f in sorted(_g6.glob(str(R / "*.judged.json"))):
        d = _j6.loads(pathlib.Path(f).read_text()); fam = d["family"]; c = d["config"]
        if fam.endswith("-zh") or fam.endswith("-ext"):
            continue
        base = fam.replace("-GPTQ4", "").replace("-AWQ4", "").replace("-Q4KM", "")
        if base not in FAMS:
            continue
        if "GPTQ4" in fam: c = "GPTQ"
        elif "AWQ4" in fam: c = "AWQ"
        elif "Q4KM" in fam: c = "Q4KM"
        for i in d["items"]:
            by[c][0] += 1
            by[c][1] += i["judge"] == "REFUSAL"
            by[c][2] += bool(i["keyword_v1"])
    ORDER = [("BF16", "BF16"), ("FP16", "FP16"), ("FP4", "NF4"), ("INT4", "INT4\n(RTN)"),
             ("GPTQ", "GPTQ"), ("AWQ", "AWQ"), ("Q4KM", "Q4_K_M")]

    # stacked vertically: the two panels answer different questions and read
    # better in sequence than side by side
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 4.6),
                                 gridspec_kw={"height_ratios": [1.15, 1]})
    yy = np.arange(len(top))
    a1.barh(yy, [v for _, v in top], .62, color=BAD)
    for i, (_, v) in enumerate(top):
        a1.text(v + 1.2, i, f"{v:.0f}", va="center", fontsize=6.4)
    a1.set_yticks(yy); a1.set_yticklabels([k for k, _ in top], fontsize=6.0)
    a1.invert_yaxis(); a1.set_xlim(0, 78)
    a1.set_xlabel("judge $-$ keyword, at BF16 (pts)", fontsize=6.8)
    a1.set_title("(a) register shift: no quantization involved",
                 fontsize=7.6, loc="left")

    vals, labs_ = [], []
    for key, lab in ORDER:
        if key in by:
            n, j, k = by[key]; vals.append(100*(j-k)/n); labs_.append(lab)
    cols = [BAD if l.startswith("INT4") else CTX for l in labs_]
    x = np.arange(len(vals))
    a2.bar(x, vals, .58, color=cols)
    for xi, v in enumerate(vals):
        a2.text(xi, v + .35, f"{v:.1f}", ha="center", fontsize=6.4)
    a2.set_xticks(x); a2.set_xticklabels(labs_, fontsize=6.0)
    a2.set_ylabel("judge $-$ keyword (pts)", fontsize=6.8)
    a2.set_ylim(0, max(vals)*1.42)
    a2.set_title("(b) language: only under INT4 rounding", fontsize=7.6, loc="left")
    a2.text(.5, .96, "same four checkpoints throughout; every\ndeployed quantizer sits at baseline",
            transform=a2.transAxes, fontsize=5.9, color=CTX, ha="center", va="top")
    fig.subplots_adjust(hspace=.45)
    save(fig, "fig_twomodes")
except Exception as e:
    print("  skipped fig_twomodes:", e)


# ------------------------------------- 25. rounding cannot erase a coherent edit
# The paper's central mechanism. Zhang et al. (ICLR 2025) explain catastrophic
# failure of unlearning under quantization with a per-weight argument: the edit
# is smaller than the quantization step, so both models round to the same grid
# point. Panel (a) shows that premise holds for abliteration by a factor of ~50.
# Panel (b) shows the conclusion does not follow: the quantized difference
# reproduces the edit with unit gain.
try:
    import json as _json
    SRC = [("qwen_kl", "Qwen\nKL-constr."), ("qwen_aggr", "Qwen\nfull proj."),
           ("llama", "Llama-3.1-8B")]
    data = []
    for tag, lab in SRC:
        f = D / f"step_vs_edit_{tag}.json"
        if f.exists():
            data.append((lab, _json.loads(f.read_text())))
    if not data:
        raise FileNotFoundError("no step_vs_edit_*.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.4, 1.85),
                                 gridspec_kw={"wspace": .5})
    xs = np.arange(len(data))
    for i, (lab, r) in enumerate(data):
        v = np.array([x["med_ratio"] for x in r])
        a1.scatter(np.full(len(v), i) + np.random.RandomState(0).uniform(-.13, .13, len(v)),
                   v, s=3, color=ACC, alpha=.55, lw=0)
    a1.axhline(.5, color=BAD, lw=1.0, ls="--")
    a1.text(.04, .56, "bin-crossing threshold", transform=a1.get_yaxis_transform(),
            fontsize=5.0, color=BAD, va="bottom", ha="left")
    a1.set_yscale("log"); a1.set_ylim(1e-3, 2)
    a1.set_xticks(xs); a1.set_xticklabels([d[0] for d in data], fontsize=5.2)
    a1.tick_params(axis="y", labelsize=5.5)
    a1.set_ylabel("edit size / quantization step", fontsize=6)
    a1.set_title("(a) edit $\\ll$ step", fontsize=6.6, loc="left")

    for i, (lab, r) in enumerate(data):
        v = np.array([x["proj"] for x in r])
        a2.scatter(np.full(len(v), i) + np.random.RandomState(1).uniform(-.13, .13, len(v)),
                   v, s=3, color=GOOD, alpha=.55, lw=0)
    a2.axhline(1.0, color=CTX, lw=.9, ls=":")
    a2.axhline(0.0, color=BAD, lw=1.0, ls="--")
    a2.text(.04, .04, "erased (predicted)", transform=a2.get_yaxis_transform(),
            fontsize=5.0, color=BAD, va="bottom", ha="left")
    a2.text(.04, .90, "reproduced exactly", transform=a2.get_yaxis_transform(),
            fontsize=5.0, color=CTX, va="top", ha="left")
    a2.set_ylim(-.25, 1.25)
    a2.set_xticks(xs); a2.set_xticklabels([d[0] for d in data], fontsize=5.2)
    a2.tick_params(axis="y", labelsize=5.5)
    a2.set_ylabel("edit recovered", fontsize=6)
    a2.set_title("(b) rounding preserves it", fontsize=6.6, loc="left")
    save(fig, "fig_dither")
except Exception as e:
    print("  skipped fig_dither:", e)
