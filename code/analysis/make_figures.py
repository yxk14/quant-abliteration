#!/usr/bin/env python3
"""Figures for the paper. Reads analysis/data/*.csv, writes paper/figs/*.pdf.
Works identically on synthetic placeholder data and on real measurements."""
import pandas as pd, numpy as np, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = pathlib.Path(__file__).parent / "data"
F = pathlib.Path(__file__).parent.parent / "paper" / "figs"; F.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10.5, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.8,
    "grid.linewidth": 0.4, "lines.linewidth": 1.3, "figure.dpi": 200,
    "axes.grid": True, "grid.alpha": 0.30, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
C_BASE, C_ABL = "#4a9c2f", "#a4308f"      # green / magenta, echoing prior art
C_ACC = "#b3202c"

# ------------------------------------------------------------------ Fig 1: cliff
sw = pd.read_csv(D/"sweep_cliff.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), sharey=True)
for ax, fmt in zip(axes, ["INT", "FP"]):
    d = sw[sw.fmt == fmt]
    for var, c, ls, mk in [("base", C_BASE, "-", "o"), ("abliterated", C_ABL, "--", "o")]:
        g = d[d.variant == var].groupby("compression_pct").mmlu_loss_pp
        mu, sd = g.mean(), g.std()
        ax.plot(mu.index, mu.values, ls, color=c, marker=mk, ms=3.6,
                mfc="white", mew=1.1, label=f"{var}")
        ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.13, lw=0)
    ax.axhline(-1.0, color="0.45", lw=0.6, ls=":")
    # knee annotation
    for var, c in [("base", C_BASE), ("abliterated", C_ABL)]:
        g = d[d.variant == var].groupby("compression_pct").mmlu_loss_pp.mean()
        knee = g[g < -1.0].index.min()
        if not np.isnan(knee):
            ax.axvline(knee, color=c, lw=0.7, ls="-.", alpha=0.65)
    ax.set_xlabel("Compression rate (%)"); ax.set_title(f"{fmt} formats")
    ax.set_xlim(20, 92); ax.set_ylim(-9.5, 0.7)
    ax.set_xticks([25, 37.5, 50, 62.5, 75, 87.5])
    ax.set_xticklabels(["25", "37.5", "50", "62.5", "75", "87.5"])
axes[0].set_ylabel("MMLU accuracy loss (pp)")
axes[0].legend(loc="lower left", frameon=False)
axes[1].text(0.97, 0.06, "dotted: 1 pp tolerance\ndash-dot: knee", transform=axes[1].transAxes,
             ha="right", va="bottom", fontsize=8, color="0.35")
fig.savefig(F/"fig_cliff.pdf"); plt.close(fig)

# ------------------------------------------- Fig 2: H1 outliers + H2 error
h1 = pd.read_csv(D/"h1_outliers.csv"); h2 = pd.read_csv(D/"h2_quant_error.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
ax = axes[0]
for var, c, ls in [("base", C_BASE, "-"), ("abliterated", C_ABL, "--")]:
    g = h1[h1.variant == var].groupby("layer")["kurtosis"]
    ax.plot(g.mean().index, g.mean().values, ls, color=c, label=var)
    ax.fill_between(g.mean().index, g.mean()-g.std(), g.mean()+g.std(), color=c, alpha=0.13, lw=0)
ax2 = ax.twinx(); ax2.grid(False)
rn = h1[h1.variant == "base"].groupby("layer").rdir_norm.mean()
ax2.fill_between(rn.index, 0, rn.values, color="0.55", alpha=0.16, lw=0)
ax2.set_ylabel(r"$\|\hat r\|$", color="0.45", fontsize=8.5, labelpad=1)
ax2.tick_params(axis="y", colors="0.5", labelsize=8); ax2.set_ylim(0, 3.2)
ax.set_xlabel("Transformer layer"); ax.set_ylabel("Excess kurtosis of $W_o$")
ax.set_title("H1: outlier amplification")
ax.legend(loc="lower center", frameon=False, ncol=2, fontsize=8.5)
ax = axes[1]
w = [16, 8, 4]; x = np.arange(3); bw = 0.36
for i, (var, c) in enumerate([("base", C_BASE), ("abliterated", C_ABL)]):
    vals = [h2[(h2.variant == var) & (h2.width == ww)].rel_frob_err.mean() for ww in w]
    ax.bar(x + (i - 0.5)*bw, vals, bw, color=c, alpha=0.85, label=var,
           edgecolor="white", linewidth=0.5)
for j, ww in enumerate(w):
    b = h2[(h2.variant == "base") & (h2.width == ww)].rel_frob_err.mean()
    a = h2[(h2.variant == "abliterated") & (h2.width == ww)].rel_frob_err.mean()
    ax.text(j, a*1.35, f"{a/b:.2f}×", ha="center", fontsize=8.5, color=C_ABL)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{ww}-bit" for ww in w])
ax.set_ylabel(r"rel. $\|W-Q(W)\|_F$"); ax.set_title("H2: error amplification")
ax.legend(loc="upper left", frameon=False)
fig.subplots_adjust(wspace=0.42)
fig.savefig(F/"fig_mechanism.pdf"); plt.close(fig)

# ---------------------------------- Fig: H3 directional recovery (merged)
# One figure, not two: format breakdown and bootstrap intervals on the same
# axes. All type at 8 pt and strokes kept light, so the panel reads as data
# rather than as a heavy diagram.
h3 = pd.read_csv(D/"h3_directional.csv")
bt = pd.read_csv(D/"rho_bootstrap.csv")
_rc = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
       "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
       "axes.linewidth": 0.6, "grid.linewidth": 0.3}
with plt.rc_context(_rc):
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    style = [("base", "INT", C_BASE, "-", "s"), ("base", "FP", C_BASE, "--", "s"),
             ("abliterated", "INT", C_ABL, "-", "o"), ("abliterated", "FP", C_ABL, "--", "o")]
    for var, fmt, c, ls, mk in style:
        d = bt[(bt.variant == var) & (bt.fmt == fmt)].groupby("width").agg(
            m=("rho_mean", "mean"), lo=("rho_ci_lo", "mean"), hi=("rho_ci_hi", "mean"))
        ax.errorbar(d.index, d.m, yerr=[d.m - d.lo, d.hi - d.m], fmt=mk, linestyle=ls,
                    color=c, ms=3.2, mfc="white", mew=0.8, capsize=1.8, lw=0.9,
                    elinewidth=0.6, label=f"{var} \u00b7 {fmt}")
    ax.axhspan(0.86, 1.14, color="0.5", alpha=0.14, lw=0, zorder=0)
    ax.axhline(1.0, color="0.4", lw=0.55, ls=":", zorder=1)
    ax.annotate("isotropic null (95%)", xy=(11.5, 1.155), fontsize=8, color="0.4")
    ax.invert_xaxis()
    ax.set_xticks([16, 8, 4])
    ax.set_xlim(17.4, 2.6)
    ax.set_xlabel("Weight bit width")
    ax.set_ylabel(r"$\rho$   (refusal-direction recovery)")
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.8,
              borderaxespad=0.2, labelspacing=0.22, handletextpad=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(width=0.6, length=2.5)
    fig.savefig(F/"fig_rho.pdf")
    plt.close(fig)

# ------------------------------------------------- Fig 4: intervention arms
arms = pd.read_csv(D/"arms_interventions.csv")
order = ["refusal-ablated", "norm-preserving", "outlier-protected", "random-ablated"]
lbl = {"refusal-ablated": "refusal-ablated", "norm-preserving": "norm-preserving",
       "outlier-protected": "outlier-protected", "random-ablated": "random-direction"}
cols = [C_ABL, "#c96a1e", "#2b7bb9", C_BASE]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
x = np.arange(3); w3 = [16, 8, 4]; bw = 0.2
for i, arm in enumerate(order):
    v = [arms[(arms.arm == arm) & (arms.width == ww)].afi.mean() for ww in w3]
    axes[0].bar(x + (i - 1.5)*bw, v, bw, color=cols[i], label=lbl[arm],
                edgecolor="white", linewidth=0.5)
    v2 = [arms[(arms.arm == arm) & (arms.width == ww)].refusal_rate.mean() for ww in w3]
    axes[1].bar(x + (i - 1.5)*bw, v2, bw, color=cols[i], edgecolor="white", linewidth=0.5)
axes[0].axhline(1.0, color="0.35", lw=0.7, ls=":")
axes[0].set_ylabel("Abliteration Fragility Index"); axes[0].set_ylim(0.95, 2.05)
axes[1].set_ylabel("Refusal rate (%)")
for a in axes:
    a.set_xticks(x); a.set_xticklabels([f"{ww}-bit" for ww in w3]); a.set_xlabel("Weight bit width")
axes[0].legend(loc="upper center", bbox_to_anchor=(1.06, 1.24), frameon=False,
               fontsize=8, ncol=4, columnspacing=1.4, handlelength=1.2,
               handletextpad=0.5)
fig.savefig(F/"fig_arms.pdf"); plt.close(fig)

# ------------------------------------------------- Fig 5: proxy failure
px = pd.read_csv(D/"proxy_test.csv")
fig, ax = plt.subplots(figsize=(3.45, 2.2))
y = np.arange(len(px))[::-1]
ax.barh(y + 0.19, px.r_pooled.abs(), 0.36, color="0.72", label="pooled |r|")
ax.barh(y - 0.19, px.r_partial.abs(), 0.36,
        color=[C_ABL if s == "yes" else "0.85" for s in px.screens],
        edgecolor="0.4", linewidth=0.5, label="partial |r| (width controlled)")
ax.set_yticks(y); ax.set_yticklabels(px.predictor, fontsize=8.5)
ax.set_xlabel("|Pearson r| with refusal drift"); ax.set_xlim(0, 1.0)
ax.axvline(0.5, color="0.45", lw=0.6, ls=":")
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.savefig(F/"fig_proxy.pdf"); plt.close(fig)

print("figures written to", F)
for f in sorted(F.glob("*.pdf")): print("  ", f.name)

# ------------------------------------------- Fig 6: native vs simulated formats
nv = pd.read_csv(D/"native_vs_sim.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
ax = axes[0]
fmts = ["FP8E4M3-sim", "FP8E4M3-nat", "NF4", "NVFP4"]
x = np.arange(len(fmts)); bw = 0.36
for i, (v, c) in enumerate([("base", C_BASE), ("abliterated", C_ABL)]):
    vals = [nv[(nv.fmt == f) & (nv.variant == v)].rel_frob_err.mean() for f in fmts]
    ax.bar(x + (i-0.5)*bw, vals, bw, color=c, label=v, edgecolor="white", lw=0.5)
ax.set_yscale("log"); ax.set_xticks(x)
ax.set_xticklabels(["FP8\nsim", "FP8\nnative", "NF4\ncodebook", "NVFP4\nnative"], fontsize=8.5)
ax.set_ylabel(r"rel. $\|W-Q(W)\|_F$")
ax.set_title("Simulated vs native formats"); ax.legend(frameon=False, loc="upper left")
ax = axes[1]
ratios = []
for v in ["base", "abliterated"]:
    d = nv[nv.variant == v]
    ratios.append(d[d.fmt == "NF4"].rel_frob_err.mean() /
                  d[d.fmt == "NVFP4"].rel_frob_err.mean())
bars = ax.bar([0, 1], ratios, 0.5, color=[C_BASE, C_ABL], edgecolor="white", lw=0.6)
ax.axhline(1.0, color="0.35", lw=0.8, ls=":")
for b, r in zip(bars, ratios):
    ax.text(b.get_x()+b.get_width()/2, r+0.03, f"{r:.2f}", ha="center", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "abliterated"])
ax.set_ylabel("NF4 error / NVFP4 error"); ax.set_ylim(0, 1.45)
ax.set_title("Codebook advantage inverts")
ax.text(0.5, 0.06, "below 1: codebook wins\nabove 1: hardware float wins",
        transform=ax.transAxes, ha="center", fontsize=8, color="0.35")
fig.subplots_adjust(wspace=0.35)
fig.savefig(F/"fig_native.pdf"); plt.close(fig)

print("  fig_native.pdf")
