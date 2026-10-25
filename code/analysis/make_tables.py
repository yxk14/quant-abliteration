#!/usr/bin/env python3
"""Emit LaTeX table bodies from analysis/data/*.csv into paper/tables/.
main.tex \input's these, so regenerating data + rerunning this updates the
paper with zero manual editing."""
import pandas as pd, pathlib
D = pathlib.Path(__file__).parent/"data"
T = pathlib.Path(__file__).parent.parent/"paper"/"tables"; T.mkdir(parents=True, exist_ok=True)

HDR = {
 "tab_main.tex": ("llrrrrrr",
   r"Variant & Cfg & $D_{\mathrm{KL}}$ & MMLU & GSM8K & Ref. & Compl. & Adj.\\,ref."),
 "tab_afi.tex": ("lrrr",
   r"Config & Weight-err. ratio & AFI & $\rho$"),
 "tab_arms.tex": ("@{}p{0.30\\columnwidth}rrrr@{}",
   r"Arm & AFI$_8$ & AFI$_4$ & $\rho_4$ & Ref. (\%)"),
 "tab_proxy.tex": ("@{}lrrrc@{}",
   r"Predictor & \makecell{$r$\\pooled} & \makecell{$r$\\partial} & $p$ & Screens?"),
 "tab_additivity.tex": ("lrrrrr",
   r"Config & Abl. leg & Quant. leg & Additive & Observed & Excess"),
}

def emit(name, rows):
    """Emit a COMPLETE tabular. \\input opens a group, which breaks booktabs'
    \\noalign if only the body is imported -- so the environment ships whole."""
    cols, hdr = HDR[name]
    body = "\n".join(rows).rstrip()
    tex = ("\\setlength{\\tabcolsep}{3.5pt}\n"
           "\\begin{tabular}{" + cols + "}\n\\toprule\n" + hdr +
           " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    (T/name).write_text(tex)

# ---- Table: main results, Llama-3.1-8B (representative)
m = pd.read_csv(D/"main_results.csv")
rows = []
for v in ["base", "abliterated"]:
    for cfg in ["FP16", "INT16", "FP8", "INT8", "FP4", "INT4"]:
        d = m[(m.variant == v) & (m.config == cfg)]
        rows.append(f"{v if cfg=='FP16' else ''} & {cfg} & {d.kld.mean():.4f} & "
                    f"{d.mmlu.mean():.1f} & {d.gsm8k.mean():.1f} & "
                    f"{d.refusal_rate.mean():.1f} & {d.completion_rate.mean():.1f} & "
                    f"{d.adj_refusal_rate.mean():.1f} \\\\")
    if v == "base": rows.append("\\midrule")
emit("tab_main.tex", rows)

# ---- Table: AFI by width and format
afi = []
for cfg in ["FP16", "INT16", "FP8", "INT8", "FP4", "INT4"]:
    b = m[(m.variant=="base") & (m.config==cfg)].kld.mean()
    a = m[(m.variant=="abliterated") & (m.config==cfg)].kld.mean()
    h2 = pd.read_csv(D/"h2_quant_error.csv")
    w = int(cfg[-2:]) if cfg[-2:].isdigit() else int(cfg[-1])
    eb = h2[(h2.variant=="base") & (h2.config==cfg)].rel_frob_err.mean()
    ea = h2[(h2.variant=="abliterated") & (h2.config==cfg)].rel_frob_err.mean()
    h3 = pd.read_csv(D/"h3_directional.csv")
    r = h3[(h3.variant=="abliterated") & (h3.config==cfg)].rho_refusal.mean()
    afi.append(f"{cfg} & {ea/eb:.2f} & {a/b:.2f} & {r:.2f} \\\\")
emit("tab_afi.tex", afi)

# ---- Table: intervention arms at 4-bit
ar = pd.read_csv(D/"arms_interventions.csv")
lab = {
    "refusal-ablated":   "Refusal-ablated",
    "random-ablated":    "Random-direction",
    "norm-preserving":   "Norm-preserving",
    "outlier-protected": "Outlier-protected",
}
rows = []
for arm in ["refusal-ablated","random-ablated","norm-preserving","outlier-protected"]:
    d4 = ar[(ar.arm==arm) & (ar.width==4)]
    d8 = ar[(ar.arm==arm) & (ar.width==8)]
    rows.append(f"{lab[arm]} & {d8.afi.mean():.2f} & {d4.afi.mean():.2f} & "
                f"{d4.rho_refusal.mean():.2f} & {d4.refusal_rate.mean():.1f} \\\\")
emit("tab_arms.tex", rows)

# ---- Table: proxy / screening test
px = pd.read_csv(D/"proxy_test.csv")
CHK, DASH, EOL = "\\checkmark", "--", " \\\\"
rows = []
TEXLBL = {"Delta MMLU (pp)": r"$\Delta$MMLU (pp)",
          "Delta GSM8K (pp)": r"$\Delta$GSM8K (pp)",
          "WikiText-2 PPL": r"WikiText-2 PPL",
          "KL divergence": r"$D_{\mathrm{KL}}$ vs.\ reference",
          "rho (refusal recovery)": r"$\rho$ (refusal recovery)"}
for r in px.itertuples():
    mark = CHK if r.screens == "yes" else DASH
    rows.append(f"{TEXLBL.get(r.predictor, r.predictor)} & {r.r_pooled:+.2f} & {r.r_partial:+.2f} & "
                f"{r.p_partial:.3f} & {mark}" + EOL)
emit("tab_proxy.tex", rows)

# ---- Table: super-additivity
h5 = pd.read_csv(D/"h5_additivity.csv")
rows = []
for cfg in ["INT16","INT8","INT4"]:
    d = h5[h5.config==cfg]
    rows.append(f"{cfg} & {d.kld_ablation_leg.mean():.3f} & {d.kld_quant_leg.mean():.4f} & "
                f"{d.kld_additive_pred.mean():.3f} & {d.kld_joint_observed.mean():.3f} & "
                f"{d.excess_pct.mean():+.1f}\\% \\\\")
emit("tab_additivity.tex", rows)
# native vs simulated
nv = pd.read_csv(D/"native_vs_sim.csv")
rows = []
for f in ["FP8E4M3-sim", "FP8E4M3-nat", "NF4", "NVFP4"]:
    b = nv[(nv.fmt==f)&(nv.variant=="base")]
    a = nv[(nv.fmt==f)&(nv.variant=="abliterated")]
    rows.append(f"{f.replace('_','-')} & {'yes' if bool(a.native.iloc[0]) else 'no'} & "
                f"{b.rel_frob_err.mean():.4f} & {a.rel_frob_err.mean():.4f} & "
                f"{a.rel_frob_err.mean()/b.rel_frob_err.mean():.2f}")
HDR["tab_native.tex"] = ("llrrr",
    r"Format & Native & Base err. & Abl. err. & Ratio")
emit("tab_native.tex", [r + " \\\\" for r in rows])
print("tables written to", T)
for f in sorted(T.glob("*.tex")): print("  ", f.name)
