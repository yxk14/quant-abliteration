#!/usr/bin/env python3
"""
gen_synthetic.py -- SYNTHETIC PLACEHOLDER DATA GENERATOR

!!  THE NUMBERS THIS SCRIPT PRODUCES ARE NOT MEASUREMENTS.  !!

It emits placeholder result tables with the shape, units and plausible
magnitudes of the real experiment, so the analysis + figure + paper
pipeline can be validated end-to-end before any GPU time is spent.

To use real data: run experiment/run_experiment.py, which writes CSVs with
IDENTICAL column names to analysis/data/. Then rebuild:  ./build.sh --final

Generative model is documented per-hypothesis below. Effect sizes are
anchored to published values where they exist (see refs in paper/refs.bib):
  - Quantization KLD by bit width: smcleod (2026) MLX Qwen3.6-27B
        8-bit 0.0139 | 6-bit 0.0287 | 4-bit 0.1100
  - Abliteration KLD range: Heretic cross-arch study, 0.043-1.646
  - Abliteration GSM8K cost: up to -18.81 pp (Yi-1.5-9B)
  - Safety-neuron quantization error absorption: 1.39x  (arXiv 2606.10154)
  - Abliterated LLaMA-3.1-8B factuality: AlignScore 0.588 -> 0.444
"""
import numpy as np, pandas as pd, pathlib

SEED = 20260829
rng = np.random.default_rng(SEED)
OUT = pathlib.Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "gemma-2-9b-it"]
SHORT  = {"Llama-3.1-8B-Instruct": "L8B", "Qwen2.5-7B-Instruct": "Q7B", "gemma-2-9b-it": "G9B"}
VARIANTS = ["base", "abliterated"]
# (format, width). BF16 released weights are the REFERENCE, not a condition.
CONFIGS = [("FP", 16), ("INT", 16), ("FP", 8), ("INT", 8), ("FP", 4), ("INT", 4)]
N_LAYERS = 32

# ---------------------------------------------------------------- base rates
# Representational error of each format/width, before any model effects.
# FP wins at 4-bit (exponent handles outliers); INT wins at 16-bit (uniform
# spacing beats FP16's mantissa in a bounded range). Crossover near 8-bit.
BASE_KLD = {("FP",16): 0.00021, ("INT",16): 0.00014,
            ("FP", 8): 0.0212,  ("INT", 8): 0.0139,
            ("FP", 4): 0.0981,  ("INT", 4): 0.1423}

# H1/H2: Abliteration Fragility Index -- how much MORE quantization error the
# abliterated variant absorbs at each width. Grows as width falls because
# outlier-driven error is what low-bit formats handle worst.
AFI_BY_WIDTH = {16: 1.10, 8: 1.46, 4: 1.87}

CAPS = {  # (MMLU, GSM8K, HellaSwag, WikiText2-PPL) at BF16 reference
    ("Llama-3.1-8B-Instruct","base"):        (68.0, 84.2, 81.7, 6.84),
    ("Llama-3.1-8B-Instruct","abliterated"): (66.4, 76.9, 81.1, 7.12),
    ("Qwen2.5-7B-Instruct","base"):          (71.9, 85.4, 80.2, 7.41),
    ("Qwen2.5-7B-Instruct","abliterated"):   (70.2, 74.8, 79.4, 7.79),
    ("gemma-2-9b-it","base"):                (69.4, 79.1, 82.3, 6.97),
    ("gemma-2-9b-it","abliterated"):         (67.1, 68.3, 81.5, 7.44),
}
REF_REFUSAL = {"base": 91.4, "abliterated": 2.1}   # % on the harmful battery
REF_COMPLETION = 98.7                              # % well-formed, terminated

# Per-(model, format) refusal sensitivity. This is the crux of the "quality is
# not a safety proxy" effect: refusal drift carries a large idiosyncratic
# component that capability metrics do NOT share, so pooled correlations
# between quality loss and refusal drift collapse (Simpson's paradox across
# model families, cf. arXiv 2606.10154). rho (H3) DOES share this component,
# which is why the mechanistic metric predicts where quality metrics fail.
SENS = {(m, f): float(np.exp(rng.normal(0, 0.62)))
        for m in MODELS for f in ("FP", "INT")}

def jit(x, rel=0.04):
    return float(x * (1 + rng.normal(0, rel)))

# =============================================================== MAIN TABLE
rows = []
for m in MODELS:
    mscale = {"L8B":1.0, "Q7B":0.94, "G9B":1.08}[SHORT[m]]
    for v in VARIANTS:
        mmlu0, gsm0, hella0, ppl0 = CAPS[(m, v)]
        for fmt, w in CONFIGS:
            afi = AFI_BY_WIDTH[w] if v == "abliterated" else 1.0
            kld = jit(BASE_KLD[(fmt, w)] * mscale * afi, 0.06)
            # capability decays with distributional drift; GSM8K ~3x more
            # sensitive than MMLU (matches abliteration literature)
            d = kld ** 0.62
            mmlu  = mmlu0  - jit(11.0 * d, .08)
            gsm   = gsm0   - jit(34.0 * d, .10)
            hella = hella0 - jit(4.5 * d, .08)
            ppl   = ppl0   * (1 + jit(0.55 * d, .08))
            # H3->H4: refusal comes back for abliterated models only, and
            # tracks directional recovery (below), not general quality.
            if v == "abliterated":
                refusal = REF_REFUSAL[v] + jit(46.0 * kld**1.15, .14) * SENS[(m, fmt)]
            else:
                refusal = REF_REFUSAL[v] - jit(9.0 * d, .12)   # aligned: erodes
            completion = REF_COMPLETION - jit(115.0 * kld**1.25 * (1.35 if v=="abliterated" else 1.0), .10)
            completion = max(completion, 40.0)
            # refusal among COMPLETED responses -- strips the degradation artifact
            adj_refusal = refusal * (completion / REF_COMPLETION) ** 1.6
            dfr = jit(28.0 * d, .12)   # decision-flip rate vs reference
            rows.append(dict(model=m, short=SHORT[m], variant=v, fmt=fmt, width=w,
                             config=f"{fmt}{w}", kld=round(kld,5),
                             mmlu=round(mmlu,2), gsm8k=round(gsm,2),
                             hellaswag=round(hella,2), ppl=round(ppl,3),
                             refusal_rate=round(min(refusal,100),2),
                             completion_rate=round(completion,2),
                             adj_refusal_rate=round(min(adj_refusal,100),2),
                             dfr=round(dfr,2)))
main = pd.DataFrame(rows)
main.to_csv(OUT/"main_results.csv", index=False)

# ============================================ H1: OUTLIER AMPLIFICATION
# Abliteration projects out r_hat from o_proj/down_proj. Prediction: excess
# kurtosis and max/RMS ratio rise, concentrated in mid-stack layers where the
# refusal direction has greatest norm.
h1 = []
for m in MODELS:
    peak = 0.55 * N_LAYERS
    for L in range(N_LAYERS):
        # refusal-direction norm profile: peaks mid-stack
        w_prof = np.exp(-((L - peak) ** 2) / (2 * (N_LAYERS * 0.18) ** 2))
        for v in VARIANTS:
            k0 = 3.10 + 0.35 * np.sin(L / 4.0)          # baseline excess kurtosis
            mr0 = 11.5 + 1.4 * np.sin(L / 5.0)          # baseline max|w|/RMS
            bump_k = 1.85 * w_prof if v == "abliterated" else 0.0
            bump_m = 6.20 * w_prof if v == "abliterated" else 0.0
            oc0 = 42 + 6 * np.sin(L / 3.0)
            h1.append(dict(model=m, short=SHORT[m], variant=v, layer=L,
                           rdir_norm=round(float(w_prof), 4),
                           kurtosis=round(jit(k0 + bump_k, .03), 3),
                           max_rms=round(jit(mr0 + bump_m, .03), 2),
                           outlier_channels=int(jit(oc0 + 31 * w_prof if v=="abliterated" else oc0, .05))))
pd.DataFrame(h1).to_csv(OUT/"h1_outliers.csv", index=False)

# ============================================ H2: ERROR AMPLIFICATION
# Same quantizer, larger reconstruction error on abliterated weights.
# Mediation: per-layer delta-error should be explained by per-layer delta-kurtosis.
h1df = pd.DataFrame(h1)
h2 = []
BASE_RELERR = {("FP",16):0.00048, ("INT",16):0.00031,
               ("FP",8):0.0091,   ("INT",8):0.0062,
               ("FP",4):0.0642,   ("INT",4):0.0913}
for m in MODELS:
    for fmt, w in CONFIGS:
        for L in range(N_LAYERS):
            for v in VARIANTS:
                k = h1df[(h1df.model==m)&(h1df.variant==v)&(h1df.layer==L)]["kurtosis"].iloc[0]
                kb = h1df[(h1df.model==m)&(h1df.variant=="base")&(h1df.layer==L)]["kurtosis"].iloc[0]
                # error scales with excess kurtosis, more steeply at low width
                expo = {16:0.22, 8:0.55, 4:0.95}[w]
                relerr = BASE_RELERR[(fmt,w)] * (k/kb)**expo
                h2.append(dict(model=m, short=SHORT[m], variant=v, fmt=fmt, width=w,
                               config=f"{fmt}{w}", layer=L, kurtosis=k,
                               rel_frob_err=round(jit(relerr,.04),6)))
pd.DataFrame(h2).to_csv(OUT/"h2_quant_error.csv", index=False)

# ============================================ H3: DIRECTIONAL RECOVERY
# rho = ||E r_hat||^2 / (||E||_F^2 / d).  rho=1 means isotropic error.
# rho>1 means quantization noise concentrates on the ablated direction.
# Null: 1000 random unit directions.
h3 = []
for m in MODELS:
    for v in VARIANTS:
        for fmt, w in CONFIGS:
            if v == "abliterated":
                rho = 1.0 + ({16:0.21, 8:0.58, 4:1.14}[w]
                             * (1.0 if fmt=="INT" else 0.93) * SENS[(m, fmt)] ** 0.85)
            else:
                rho = 1.02   # no ablation -> error is ~isotropic wrt r_hat
            rho = jit(rho, .05)
            null_mean, null_p95 = 1.00, 1.14
            p = 1e-6 if rho > 1.35 else (0.004 if rho > 1.15 else 0.41)
            h3.append(dict(model=m, short=SHORT[m], variant=v, fmt=fmt, width=w,
                           config=f"{fmt}{w}", rho_refusal=round(rho,3),
                           rho_null_mean=null_mean, rho_null_p95=null_p95,
                           p_value=p))
pd.DataFrame(h3).to_csv(OUT/"h3_directional.csv", index=False)

# ============================================ H5: SUPER-ADDITIVITY
# If abliteration and quantization were independent perturbations, the joint
# KLD would be ~the sum of the legs. Excess = evidence of interaction
# ("the error budget was already spent").
h5 = []
for m in MODELS:
    kld_abl = {"L8B":0.412,"Q7B":0.377,"G9B":0.489}[SHORT[m]]
    for fmt, w in CONFIGS:
        leg_q = BASE_KLD[(fmt,w)]
        joint = (kld_abl + leg_q) * (1 + {16:0.03, 8:0.19, 4:0.44}[w])
        h5.append(dict(model=m, short=SHORT[m], fmt=fmt, width=w, config=f"{fmt}{w}",
                       kld_ablation_leg=round(kld_abl,4),
                       kld_quant_leg=round(leg_q,5),
                       kld_additive_pred=round(kld_abl+leg_q,4),
                       kld_joint_observed=round(jit(joint,.03),4),
                       excess_pct=round(100*(joint-(kld_abl+leg_q))/(kld_abl+leg_q),1)))
pd.DataFrame(h5).to_csv(OUT/"h5_additivity.csv", index=False)

# ============================================ INTERVENTION ARMS
# These are what turn correlation into causation.
#  A refusal-ablated  : standard abliteration (treatment)
#  B random-ablated   : rank-1 edit of identical norm, random direction (control
#                       for "any weight edit is fragile")
#  C norm-preserving  : abliteration that holds channel norms fixed (tests H1)
#  D outlier-protected: standard abliteration + AWQ-style protection of the
#                       channels abliteration inflated (tests H1/H2)
arms = []
ARM_AFI = {  # AFI at width 16 / 8 / 4
  "refusal-ablated":   {16:1.10, 8:1.46, 4:1.87},
  "random-ablated":    {16:1.04, 8:1.13, 4:1.22},
  "norm-preserving":   {16:1.05, 8:1.19, 4:1.34},
  "outlier-protected": {16:1.03, 8:1.11, 4:1.19},
}
ARM_REFUSAL4 = {"refusal-ablated":17.9, "random-ablated":3.4,
                "norm-preserving":8.1, "outlier-protected":5.2}
for arm, afis in ARM_AFI.items():
    for fmt, w in CONFIGS:
        r = ARM_REFUSAL4[arm] if w == 4 else (3.6 if w == 8 else 2.3)
        if w == 8: r = 2.1 + (ARM_REFUSAL4[arm]-2.1)*0.18
        arms.append(dict(arm=arm, fmt=fmt, width=w, config=f"{fmt}{w}",
                         afi=round(jit(afis[w],.03),3),
                         refusal_rate=round(jit(r,.10),2),
                         rho_refusal=round(jit({"refusal-ablated":{16:1.21,8:1.58,4:2.14},
                                                "random-ablated":{16:1.03,8:1.06,4:1.09},
                                                "norm-preserving":{16:1.09,8:1.24,4:1.51},
                                                "outlier-protected":{16:1.05,8:1.14,4:1.28}}[arm][w],.04),3)))
pd.DataFrame(arms).to_csv(OUT/"arms_interventions.csv", index=False)

# ============================================ PROXY TEST
# Does any standard quality metric predict refusal drift? (should be NO)
abl = main[main.variant=="abliterated"].copy()
ref_mmlu = abl.groupby("model").mmlu.transform("max")
abl["d_mmlu"] = abl.mmlu - ref_mmlu
abl["d_refusal"] = abl.refusal_rate - REF_REFUSAL["abliterated"]
_h3 = pd.DataFrame(h3)
_h3 = _h3[_h3.variant == "abliterated"][["model","config","rho_refusal"]]
merged = abl.merge(_h3, on=["model","config"], validate="one_to_one")
from scipy import stats as _st

def _resid(y, groups):
    """Residualise y on bit-width group means (partials out the width effect)."""
    y = np.asarray(y, float); g = np.asarray(groups)
    out = y.copy()
    for u in np.unique(g):
        msk = g == u
        out[msk] = y[msk] - y[msk].mean()
    return out

# Pooled correlation is dominated by bit width: every metric degrades with
# width, so everything correlates with everything. The scientifically
# meaningful question is whether quality predicts refusal drift ONCE WIDTH IS
# KNOWN -- i.e. can a practitioner who has already fixed a bit budget use a
# quality score to screen for refusal drift across models/formats? Partial
# correlation controlling for width answers that. This reproduces the
# sign-heterogeneity / Simpson's-paradox result of arXiv 2606.10154.
_preds = [("Delta MMLU (pp)", merged.d_mmlu),
          ("Delta GSM8K (pp)", merged.gsm8k),
          ("WikiText-2 PPL", merged.ppl),
          ("KL divergence", merged.kld),
          ("rho (refusal recovery)", merged.rho_refusal)]
_rows = []
for label, col in _preds:
    r_pool, p_pool = _st.pearsonr(np.asarray(col, float), merged.d_refusal.values)
    rx = _resid(col, merged.width.values)
    ry = _resid(merged.d_refusal.values, merged.width.values)
    r_par, p_par = _st.pearsonr(rx, ry)
    _rows.append(dict(predictor=label,
                      r_pooled=round(float(r_pool), 3), p_pooled=round(float(p_pool), 4),
                      r_partial=round(float(r_par), 3), p_partial=round(float(p_par), 4),
                      screens="yes" if p_par < 0.05 else "no"))
proxy = pd.DataFrame(_rows)
proxy.to_csv(OUT/"proxy_test.csv", index=False)

print("SYNTHETIC data written to", OUT)
for f in sorted(OUT.glob("*.csv")):
    print(f"  {f.name:28s} {len(pd.read_csv(f)):5d} rows")

# ==================================================== FINE-GRAINED CLIFF SWEEP
# The headline figure: accuracy loss vs compression rate. Coarse factorial
# (his 6 configs) can't show WHERE the cliff is, so we sweep integer and float
# widths finely. Compression rate is relative to the BF16 released weights:
#     compression = 1 - width/16
# Hypothesis: the abliterated curve leaves the flat regime EARLIER (lower
# compression) and falls more steeply -- the "spent error budget" made visible.
SWEEP_W = [16, 12, 10, 8, 7, 6, 5, 4, 3, 2]
sweep = []
for m in MODELS:
    for v in VARIANTS:
        mmlu0 = CAPS[(m, v)][0]
        gsm0  = CAPS[(m, v)][1]
        # Cliff model: loss = A * exp((w_knee - w)/tau) once past the knee.
        # Abliteration moves the knee to a HIGHER width (cliff arrives sooner)
        # and shortens tau (cliff is steeper) -- both follow from H1/H2.
        for fmt in ["FP", "INT"]:
            # FP tolerates one extra bit of compression at the low end
            knee  = (5.6 if v == "abliterated" else 4.4) + (0.0 if fmt == "FP" else 0.45)
            tau   = (0.92 if v == "abliterated" else 1.18)
            for w in SWEEP_W:
                excess = max(0.0, knee - w)
                loss_mmlu = -(0.06 + 0.30 * (np.exp(excess / tau) - 1))
                loss_gsm  = -(0.11 + 0.95 * (np.exp(excess / tau) - 1))
                loss_mmlu = float(np.clip(jit(loss_mmlu, .05), -mmlu0 * 0.55, 0))
                loss_gsm  = float(np.clip(jit(loss_gsm, .06), -gsm0 * 0.75, 0))
                refusal = (REF_REFUSAL[v] + (46.0 * (0.02 * np.exp(excess / tau)) ** 1.15
                                             if v == "abliterated" else 0.0))
                sweep.append(dict(model=m, short=SHORT[m], variant=v, fmt=fmt, width=w,
                                  compression_pct=round(100 * (1 - w / 16), 2),
                                  mmlu_loss_pp=round(loss_mmlu, 3),
                                  gsm8k_loss_pp=round(loss_gsm, 3),
                                  refusal_rate=round(min(jit(refusal, .10), 100), 2)))
pd.DataFrame(sweep).to_csv(OUT / "sweep_cliff.csv", index=False)
print(f"  {'sweep_cliff.csv':28s} {len(sweep):5d} rows")

# ================================================= NATIVE vs SIMULATED (Phase 4b)
# Blackwell has hardware FP8 (E4M3) and FP4 (NVFP4). Two things fall out:
#
#  (a) validation -- the simulated formats can be checked against real silicon,
#      which removes a threats-to-validity item;
#  (b) a NEW result -- NF4 and NVFP4 are both 4-bit, so comparing them isolates
#      whether it is the CODEBOOK or the BIT WIDTH that breaks abliterated
#      weights.
#
# Mechanistic prediction, following directly from H1: NF4's codebook is
# information-theoretically matched to NORMALLY distributed weights. Abliteration
# inflates outliers, i.e. makes the weight distribution less normal. So NF4's
# advantage over NVFP4 should hold for base models and ERODE for abliterated
# ones. NVFP4 carries an exponent and degrades more gracefully on heavy tails.
native = []
NAT = {  # (format, is_native): base rel-error, and how much abliteration hurts it
    ("NF4",   False): (0.0642, 1.87),   # codebook, normality assumption broken by outliers
    ("NVFP4", True):  (0.0698, 1.41),   # hardware float, exponent absorbs outliers
    ("FP8E4M3-sim", False): (0.0091, 1.46),
    ("FP8E4M3-nat", True):  (0.0089, 1.45),   # should match the sim closely
}
for m in MODELS:
    for (fmt, is_nat), (err0, afi) in NAT.items():
        for v in VARIANTS:
            mult = afi if v == "abliterated" else 1.0
            native.append(dict(
                model=m, short=SHORT[m], variant=v, fmt=fmt, native=is_nat,
                bits=4 if "4" in fmt else 8,
                rel_frob_err=round(jit(err0 * mult, .04), 6),
                kld=round(jit(BASE_KLD[("FP", 4 if "4" in fmt else 8)] * mult, .06), 5),
                mmlu=round(CAPS[(m, v)][0] - jit(11.0 * (err0*mult)**0.42, .07), 2)))
pd.DataFrame(native).to_csv(OUT / "native_vs_sim.csv", index=False)

# ================================================ DIRECTION-AGREEMENT GATE
# residual = ||r_hat^T W_abl|| / ||r_hat^T W_base||. Near 0 = our difference-in-
# means r_hat matches the direction abliteration actually removed. This is a
# GATE: rho is meaningless if it disagrees, so the paper must report it.
agree = []
for m in MODELS:
    for L in range(N_LAYERS):
        agree.append(dict(model=m, short=SHORT[m], layer=L,
                          residual=round(jit(0.06 + 0.02*np.sin(L/5.0), .25), 4)))
pd.DataFrame(agree).to_csv(OUT / "direction_agreement.csv", index=False)

# ============================================ BOOTSTRAP CIs ON rho
# The pipeline is deterministic, but r_hat estimation is not -- it depends on
# which prompts were sampled. Without this, H3 is a point estimate presented as
# a result. 20 resamples of the harmful/harmless prompt sets.
boot = []
for m in MODELS:
    for fmt, w in CONFIGS:
        for v in VARIANTS:
            centre = (1.0 + {16:0.21, 8:0.58, 4:1.14}[w] * (1.0 if fmt=="INT" else 0.93)
                      * SENS[(m, fmt)] ** 0.85) if v == "abliterated" else 1.02
            draws = rng.normal(centre, 0.055 * centre, 20)
            boot.append(dict(model=m, short=SHORT[m], variant=v, fmt=fmt, width=w,
                             config=f"{fmt}{w}", rho_mean=round(float(draws.mean()), 3),
                             rho_ci_lo=round(float(np.percentile(draws, 2.5)), 3),
                             rho_ci_hi=round(float(np.percentile(draws, 97.5)), 3),
                             n_boot=20))
pd.DataFrame(boot).to_csv(OUT / "rho_bootstrap.csv", index=False)
print(f"  {'native_vs_sim.csv':28s} {len(native):5d} rows")
print(f"  {'direction_agreement.csv':28s} {len(agree):5d} rows")
print(f"  {'rho_bootstrap.csv':28s} {len(boot):5d} rows")
