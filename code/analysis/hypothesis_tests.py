#!/usr/bin/env python3
"""Evaluate H1-H5 against their falsification criteria.

The paper states a criterion for each hypothesis; this is the code that
actually checks them. Run after the experiment, on real CSVs:

    python3 analysis/hypothesis_tests.py            # prints a verdict table
    python3 analysis/hypothesis_tests.py --tex      # also writes tables/tab_verdicts.tex

Every test reports SUPPORTED / FALSIFIED / INCONCLUSIVE plus the numbers, so a
null result is as legible as a positive one.
"""
import argparse, pathlib, sys
import numpy as np, pandas as pd
from scipy import stats

D = pathlib.Path(__file__).parent / "data"
T = pathlib.Path(__file__).parent.parent / "paper" / "tables"
ALPHA = 0.05


def _load(name):
    p = D / name
    return pd.read_csv(p) if p.exists() else None


def verdict(ok, inconclusive=False):
    return "INCONCLUSIVE" if inconclusive else ("SUPPORTED" if ok else "FALSIFIED")


# ------------------------------------------------------------------- H1
def h1():
    d = _load("h1_outliers.csv")
    if d is None: return dict(h="H1", v="INCONCLUSIVE", detail="no h1_outliers.csv")
    piv = d.pivot_table(index=["model", "layer"], columns="variant",
                        values=["kurtosis", "rdir_norm"])
    k_b, k_a = piv[("kurtosis", "base")], piv[("kurtosis", "abliterated")]
    t, p = stats.ttest_rel(k_a, k_b)
    dk = (k_a - k_b).values
    rn = piv[("rdir_norm", "abliterated")].values
    # criterion 2: the kurtosis increase must track ||r_hat|| across layers
    r_spatial, p_spatial = stats.pearsonr(dk, rn)
    ok = (t > 0 and p < ALPHA) and (r_spatial > 0 and p_spatial < ALPHA)
    return dict(h="H1", v=verdict(ok),
                detail=f"kurtosis +{dk.mean():.3f} (t={t:.1f}, p={p:.2g}); "
                       f"spatial corr with |r| r={r_spatial:.2f} (p={p_spatial:.2g})")


# ------------------------------------------------------------------- H2
def h2():
    d = _load("h2_quant_error.csv")
    if d is None: return dict(h="H2", v="INCONCLUSIVE", detail="no h2_quant_error.csv")
    piv = d.pivot_table(index=["model", "config", "layer"], columns="variant",
                        values=["rel_frob_err", "kurtosis"]).dropna()
    ratio = (piv[("rel_frob_err", "abliterated")] /
             piv[("rel_frob_err", "base")])
    # criterion 1: ratio > 1, and grows as width falls
    widths = [int(c[3:]) if c[:3] == "INT" else int(c[2:])
              for c in piv.index.get_level_values("config")]
    rw = pd.DataFrame({"w": widths, "r": ratio.values}).groupby("w").r.mean()
    monotone = all(rw.loc[a] <= rw.loc[b] for a, b in zip(sorted(rw.index)[1:],
                                                          sorted(rw.index)[:-1]))
    t, p = stats.ttest_1samp(ratio.values, 1.0)
    # criterion 2 (the causal link): does delta-kurtosis mediate delta-error?
    dk = (piv[("kurtosis", "abliterated")] - piv[("kurtosis", "base")]).values
    de = (piv[("rel_frob_err", "abliterated")] - piv[("rel_frob_err", "base")]).values
    med_r, med_p = stats.pearsonr(dk, de)
    ok = (ratio.mean() > 1 and p < ALPHA) and monotone and \
         (med_r > 0 and med_p < ALPHA)
    return dict(h="H2", v=verdict(ok),
                detail=f"err ratio {ratio.mean():.3f} (p={p:.2g}), "
                       f"monotone in width: {monotone}; "
                       f"mediation dK->dErr r={med_r:.2f} (p={med_p:.2g})")


# ------------------------------------------------------------------- H3
def h3():
    d = _load("h3_directional.csv")
    if d is None: return dict(h="H3", v="INCONCLUSIVE", detail="no h3_directional.csv")
    a = d[d.variant == "abliterated"]; b = d[d.variant == "base"]
    out_of_null = (a.rho_refusal > a.rho_null_p95).mean()
    base_in_null = (b.rho_refusal <= b.rho_null_p95).mean() if len(b) else np.nan
    ok = out_of_null > 0.5 and (a.p_value < ALPHA).mean() > 0.5
    if len(b):
        ok = ok and base_in_null > 0.5   # base must NOT show the effect
    return dict(h="H3", v=verdict(ok),
                detail=f"abliterated rho={a.rho_refusal.mean():.2f}, "
                       f"{out_of_null:.0%} above null p95; "
                       f"base rho={b.rho_refusal.mean():.2f} "
                       f"({base_in_null:.0%} inside null)" if len(b) else "")


# ------------------------------------------------------------------- H4
def h4():
    beh, dir3 = _load("behavior_raw.csv"), _load("h3_directional.csv")
    if beh is None: return dict(h="H4", v="INCONCLUSIVE", detail="no behavior_raw.csv")
    a = beh[beh.variant == "abliterated"].merge(
        dir3[dir3.variant == "abliterated"][["model", "config", "rho_refusal"]],
        on=["model", "config"], how="left")
    cap = _load("capability_raw.csv")
    preds = {"rho": a.rho_refusal}
    if cap is not None:
        mm = cap[(cap.task.str.contains("mmlu")) & (cap.variant == "abliterated")]
        mm = mm.groupby(["model", "config"]).value.mean().rename("mmlu").reset_index()
        a = a.merge(mm, on=["model", "config"], how="left")
        preds["mmlu"] = a.get("mmlu")
    # partial correlation controlling for bit width
    def resid(y, g):
        y = np.asarray(y, float); out = y.copy()
        for u in np.unique(g): out[g == u] = y[g == u] - y[g == u].mean()
        return out
    ry = resid(a.refusal_rate.values, a.width.values)
    res = {}
    for k, v in preds.items():
        if v is None or pd.isna(v).all(): continue
        msk = ~pd.isna(v)
        res[k] = stats.pearsonr(resid(np.asarray(v)[msk], a.width.values[msk]),
                                ry[msk])
    if "rho" not in res: return dict(h="H4", v="INCONCLUSIVE", detail="no rho merge")
    rr, pr = res["rho"]
    beats = all(abs(rr) > abs(v[0]) for k, v in res.items() if k != "rho")
    ok = pr < ALPHA and beats
    extra = "; ".join(f"{k} r={v[0]:+.2f}" for k, v in res.items())
    return dict(h="H4", v=verdict(ok, inconclusive=len(res) < 2),
                detail=f"partial (width controlled): {extra}")


# ------------------------------------------------------------------- H5
def h5():
    d = _load("h5_additivity.csv")
    if d is None: return dict(h="H5", v="INCONCLUSIVE", detail="no h5_additivity.csv")
    t, p = stats.ttest_1samp(d.excess_pct.values, 0.0)
    by_w = d.groupby("width").excess_pct.mean()
    grows = list(by_w.sort_index(ascending=False).values) == \
            sorted(by_w.values)  # excess should grow as width falls
    ok = d.excess_pct.mean() > 0 and p < ALPHA
    return dict(h="H5", v=verdict(ok),
                detail=f"excess {d.excess_pct.mean():+.1f}% (t={t:.1f}, p={p:.2g}); "
                       f"by width {dict(by_w.round(1))}; grows as width falls: {grows}")


CLAIMS = {
 "H1": "Abliteration inflates weight outliers, tracking ||r_hat||",
 "H2": "-> larger reconstruction error, mediated by that outlier growth",
 "H3": "-> quantization error is aimed at r_hat, not isotropic",
 "H4": "-> rho predicts refusal drift better than capability metrics",
 "H5": "Joint displacement is super-additive",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tex", action="store_true")
    a = ap.parse_args()
    rows = [h1(), h2(), h3(), h4(), h5()]
    print(f"\n{'':4s} {'verdict':13s} claim / evidence")
    print("-" * 96)
    for r in rows:
        print(f"{r['h']:4s} {r['v']:13s} {CLAIMS[r['h']]}")
        print(f"{'':18s} {r['detail']}")
    n_ok = sum(r["v"] == "SUPPORTED" for r in rows)
    print("-" * 96)
    print(f"{n_ok}/5 supported, "
          f"{sum(r['v']=='FALSIFIED' for r in rows)} falsified, "
          f"{sum(r['v']=='INCONCLUSIVE' for r in rows)} inconclusive")
    if a.tex:
        T.mkdir(parents=True, exist_ok=True)
        body = "\n".join(
            f"{r['h']} & {CLAIMS[r['h']][:52]} & \\textsc{{{r['v'].lower()}}} \\\\"
            for r in rows)
        (T / "tab_verdicts.tex").write_text(
            "\\begin{tabular}{lll}\n\\toprule\n & Claim & Verdict \\\\\n"
            "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
        print(f"wrote {T/'tab_verdicts.tex'}")
