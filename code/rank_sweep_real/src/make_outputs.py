#!/usr/bin/env python3
"""Stage 3 (CPU): turn the sweep CSV into the camera-ready table and figure.

  python3 make_outputs.py --csv ../out/rank_sweep_qwen.csv --tag qwen \
                          --outdir ../out

Writes:
  <outdir>/table_rank_real_<tag>.tex   -- drop-in replacement for Table 8
  <outdir>/fig_rank_snr_<tag>.pdf/.png -- SNR and gain vs rank
  <outdir>/summary_<tag>.txt           -- numbers to quote in the text
"""
import argparse
import os

import numpy as np
import pandas as pd


def fit_loglog(k, y):
    """Return exponent b in y ~ a * k^b, plus R^2."""
    lk, ly = np.log(k), np.log(y)
    b, a = np.polyfit(lk, ly, 1)
    pred = a + b * lk
    ss_res = ((ly - pred) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    return b, 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--scheme", default="int4", help="scheme for the main table")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = df[df.norm_matched == 1]
    main_df = df[df.scheme == args.scheme]
    if main_df.empty:
        raise SystemExit(f"no rows with scheme={args.scheme}")

    # aggregate across layers and modules: mean +- std over edited matrices
    g = main_df.groupby("k").agg(
        alpha=("alpha", "mean"),
        pi=("gain_pi", "mean"), pi_sd=("gain_pi", "std"),
        snr=("snr_subspace", "mean"), snr_sd=("snr_subspace", "std"),
        glob=("snr_global", "mean"),
        surv=("survive_sub", "mean"),
        n=("gain_pi", "size"),
    ).reset_index().sort_values("k")

    b, r2 = fit_loglog(g.k.values, g.snr.values)

    # ----------------------------------------------------------------- table
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Norm-matched rank sweep on \emph{real} weights and \emph{real}",
        r"activation directions (" + args.tag + r", \textsc{" + args.scheme.upper() + r"}).",
        r"Each row holds $\lVert\Delta W\rVert_F$ fixed and spreads it over $k$",
        r"directions. Mean over all edited matrices ($n$ per row); $\pi$ is the",
        r"dither gain, SNR is measured inside the ablated subspace.}",
        r"\label{tab:rank-real}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$k$ & $\alpha_k$ & gain $\pi$ & SNR$_{\mathrm{sub}}$ & survival \\",
        r"\midrule",
    ]
    for _, r in g.iterrows():
        sd = "" if np.isnan(r.pi_sd) else f"$\\pm${r.pi_sd:.3f}"
        tex.append(f"{int(r.k)} & {r.alpha:.4f} & {r.pi:.3f}{sd} & "
                   f"{r.snr:.2f} & {r.surv:.2f} \\\\")
    tex += [
        r"\midrule",
        r"\multicolumn{5}{l}{\footnotesize SNR$_{\mathrm{sub}} \propto k^{"
        + f"{b:.2f}" + r"}$ "
        + f"($R^2={r2:.3f}$; $k^{{-1/2}}$ predicted)" + r"} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = os.path.join(args.outdir, f"table_rank_real_{args.tag}.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex) + "\n")

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for scheme, mk in [("int4", "o"), ("nf4", "s"), ("fp8", "^"), ("int8", "d")]:
        sub = df[df.scheme == scheme]
        if sub.empty:
            continue
        gg = sub.groupby("k").snr_subspace.agg(["mean", "std"]).reset_index()
        ax.errorbar(gg.k, gg["mean"], yerr=gg["std"], marker=mk, ms=4,
                    lw=1.2, capsize=2, label=scheme.upper())
    kk = np.array(sorted(main_df.k.unique()), dtype=float)
    ref = g.snr.values[0] * (kk / kk[0]) ** -0.5
    ax.plot(kk, ref, "k--", lw=1.0, label=r"$k^{-1/2}$")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("edit rank $k$ (norm-matched)")
    ax.set_ylabel(r"in-subspace SNR")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.set_title("(a) the edit stops surviving", fontsize=8)

    ax = axes[1]
    for scheme, mk in [("int4", "o"), ("nf4", "s"), ("fp8", "^"), ("int8", "d")]:
        sub = df[df.scheme == scheme]
        if sub.empty:
            continue
        gg = sub.groupby("k").gain_pi.agg(["mean", "std"]).reset_index()
        ax.errorbar(gg.k, gg["mean"], yerr=gg["std"], marker=mk, ms=4,
                    lw=1.2, capsize=2, label=scheme.upper())
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.9, 1.1)
    ax.set_xlabel("edit rank $k$ (norm-matched)")
    ax.set_ylabel(r"dither gain $\pi$")
    ax.set_title(r"(b) but it is still transmitted unbiased", fontsize=8)
    for a in axes:
        a.tick_params(labelsize=7)
        a.xaxis.label.set_size(8); a.yaxis.label.set_size(8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"fig_rank_snr_{args.tag}.{ext}"),
                    dpi=200, bbox_inches="tight")

    # --------------------------------------------------------------- summary
    lines = [
        f"model tag        : {args.tag}",
        f"rows in csv      : {len(df)}",
        f"edited matrices  : {main_df.groupby(['layer','module']).ngroups}",
        f"layers           : {sorted(main_df.layer.unique())}",
        f"modules          : {sorted(main_df.module.unique())}",
        f"schemes          : {sorted(df.scheme.unique())}",
        "",
        f"SNR_sub exponent : {b:+.3f}  (R^2={r2:.4f}); analytic prediction -0.5",
        f"SNR_sub k=1      : {g.snr.values[0]:.3f}",
        f"SNR_sub k={int(g.k.values[-1]):<4d}   : {g.snr.values[-1]:.3f}",
        f"SNR_sub drop     : {g.snr.values[0]/g.snr.values[-1]:.1f}x",
        f"crosses SNR=1 at : k ~ {_cross(g):.0f}",
        "",
        f"gain pi range    : [{g.pi.min():.4f}, {g.pi.max():.4f}] "
        f"(max deviation from 1: {max(abs(g.pi.min()-1), abs(g.pi.max()-1))*100:.2f}%)",
        f"global SNR range : [{g.glob.min():.3f}, {g.glob.max():.3f}] "
        "(flat in k by construction -- whole-matrix noise, not the edit)",
        f"rel edit size    : {main_df.rel_edit.mean():.4f} of ||W||_F",
    ]
    sm = os.path.join(args.outdir, f"summary_{args.tag}.txt")
    with open(sm, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote:\n  {tex_path}\n  "
          f"{os.path.join(args.outdir, f'fig_rank_snr_{args.tag}.pdf')}\n  {sm}")


def _cross(g):
    k, s = g.k.values.astype(float), g.snr.values
    if s.min() > 1 or s.max() < 1:
        return float("nan")
    return float(np.exp(np.interp(0.0, np.log(s)[::-1], np.log(k)[::-1])))


if __name__ == "__main__":
    main()
