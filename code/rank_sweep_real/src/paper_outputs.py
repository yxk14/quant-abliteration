#!/usr/bin/env python3
"""Camera-ready table and figure for the real-direction rank sweep.

  python3 paper_outputs.py --main out/rank_sweep_qwen7b.csv \
      --control out/rank_sweep_qwen7b_control.csv \
      --table ../paper/tables/tab_rank_real.tex \
      --fig   ../paper/figs/fig_rank_real

Table: one row per rank k; columns are the norm-matching coefficient, the
INT4 dither gain, the in-subspace SNR under INT4 / NF4 / FP8, and the
in-subspace SNR of the control arm (alpha = 1, edit NOT norm-matched).
Figure: (a) in-subspace SNR vs k for every scheme plus the control arm and the
k^{-1/2} reference; (b) dither gain vs k.
All numbers are means over the 54 edited matrices (27 layers x o_proj,
down_proj). Nothing is typed by hand.
"""
import argparse
import os

import numpy as np
import pandas as pd


def fit_loglog(k, y):
    b, a = np.polyfit(np.log(k), np.log(y), 1)
    pred = a + b * np.log(k)
    ss_res = ((np.log(y) - pred) ** 2).sum()
    ss_tot = ((np.log(y) - np.log(y).mean()) ** 2).sum()
    return b, 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--fig", required=True, help="path without extension")
    args = ap.parse_args()

    df = pd.read_csv(args.main)
    df = df[df.norm_matched == 1]
    ctl = pd.read_csv(args.control)
    ctl = ctl[ctl.scheme == "int4"]

    snr = df.pivot_table(index="k", columns="scheme", values="snr_subspace",
                         aggfunc="mean")
    pi = df.pivot_table(index="k", columns="scheme", values="gain_pi",
                        aggfunc="mean")
    alpha = df[df.scheme == "int4"].groupby("k").alpha.mean()
    csnr = ctl.groupby("k").snr_subspace.mean()
    crel = ctl.groupby("k").rel_edit.mean()
    ks = sorted(snr.index)

    fits = {s: fit_loglog(np.array(ks, float), snr.loc[ks, s].values)
            for s in ("int4", "nf4", "fp8", "int8")}
    cfit = fit_loglog(np.array(ks, float), csnr.loc[ks].values)
    n_mat = df[df.scheme == "int4"].groupby(["layer", "module"]).ngroups

    # ------------------------------------------------------------------ table
    rows = [r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}r r r r r r r@{}}",
            r"\toprule",
            r"& & & \multicolumn{3}{c}{SNR in edited subspace} & control \\",
            r"\cmidrule(lr){4-6}",
            r"$k$ & $\alpha_k$ & gain $\pi$ & INT4 & NF4 & FP8 & INT4, $\alpha{=}1$ \\",
            r"\midrule"]
    for k in ks:
        rows.append(f"{k} & {alpha[k]:.3f} & {pi.loc[k,'int4']:.3f} & "
                    f"{snr.loc[k,'int4']:.2f} & {snr.loc[k,'nf4']:.2f} & "
                    f"{snr.loc[k,'fp8']:.2f} & {csnr[k]:.2f} \\\\")
    rows += [r"\midrule",
             r"fitted exponent & & & "
             + " & ".join(f"${fits[s][0]:.2f}$" for s in ("int4", "nf4", "fp8"))
             + f" & ${cfit[0]:.2f}$ \\\\",
             r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(args.table), exist_ok=True)
    with open(args.table, "w") as f:
        f.write("\n".join(rows) + "\n")

    # ----------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7,
                         "legend.fontsize": 6, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95))
    kk = np.array(ks, float)
    style = {"int4": ("o", "INT4"), "nf4": ("s", "NF4"), "fp8": ("^", "FP8")}
    ax = axes[0]
    for s, (mk, lab) in style.items():
        g = df[df.scheme == s].groupby("k").snr_subspace.agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker=mk, ms=3,
                    lw=1, capsize=1.5, label=lab)
    ax.plot(kk, snr.loc[ks, "int4"].values[0] * (kk / kk[0]) ** -0.5,
            "k--", lw=0.8, label=r"$k^{-1/2}$")
    ax.plot(csnr.index, csnr.values, color="0.45", ls=":", lw=1.2,
            label=r"control ($\alpha{=}1$)")
    ax.axhline(1.0, color="0.7", lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("edit rank $k$")
    ax.set_ylabel("SNR in edited subspace")
    ax.set_title("(a) SNR, edit norm fixed", fontsize=7)

    ax = axes[1]
    for s, (mk, lab) in style.items():
        g = df[df.scheme == s].groupby("k").gain_pi.agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker=mk, ms=3,
                    lw=1, capsize=1.5, label=lab)
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.9, 1.1)
    ax.set_xlabel("edit rank $k$")
    ax.set_ylabel(r"gain $\pi$")
    ax.set_title("(b) gain is flat in $k$", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               handlelength=1.4, columnspacing=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.fig}.{ext}", dpi=220, bbox_inches="tight")

    # ---------------------------------------------------------------- summary
    print(f"matrices: {n_mat}; ranks: {ks}")
    for s, (b, r2) in fits.items():
        print(f"{s}: SNR {snr.loc[ks[0], s]:.2f} -> {snr.loc[ks[-1], s]:.2f}, "
              f"exponent {b:.3f} (R2 {r2:.4f})")
    print(f"control int4: SNR {csnr[ks[0]]:.2f} -> {csnr[ks[-1]]:.2f}, "
          f"exponent {cfit[0]:.3f}; rel edit {crel[ks[0]]:.3f} -> {crel[ks[-1]]:.3f}")
    print(f"int4 gain range {pi['int4'].min():.4f}..{pi['int4'].max():.4f}")
    cross = np.exp(np.interp(0.0, np.log(snr.loc[ks, 'int4'].values)[::-1],
                             np.log(kk)[::-1]))
    print(f"int4 SNR crosses 1 at k ~ {cross:.0f}")
    print(f"wrote {args.table} and {args.fig}.pdf")


if __name__ == "__main__":
    main()
