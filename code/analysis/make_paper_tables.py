#!/usr/bin/env python3
"""Emit LaTeX tables straight from the judged response files.

Every number in the paper is generated here, never typed. The session that
produced this paper had two headline claims inverted by hand-propagated numbers
from a broken classifier; generating tables from the source of truth removes one
whole class of that error.
"""
import json, pathlib
from scipy import stats

R = pathlib.Path(__file__).parent.parent / "analysis" / "responses"
T = pathlib.Path(__file__).parent.parent / "paper" / "tables"


def counts(fam, cfg):
    f = R / f"{fam}__{cfg}.judged.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    v = [i["judge"] for i in d["items"]]
    kw = sum(i["keyword_v1"] for i in d["items"])
    return dict(ref=sum(x == "REFUSAL" for x in v),
                com=sum(x == "COMPLY" for x in v),
                gar=sum(x == "GARBLED" for x in v),
                n=len(v), kw=kw)


def fisher(a, b):
    """Independent-proportions test. Only correct when the two arms are NOT the
    same prompts; see mcnemar() for the paired case, which is the usual one."""
    return stats.fisher_exact([[a["ref"], a["n"] - a["ref"]],
                               [b["ref"], b["n"] - b["ref"]]])[1]


def verdicts(fam, cfg, pred=None):
    """Per-prompt refusal booleans, in battery order, so conditions can be paired."""
    f = R / f"{fam}__{cfg}.judged.json"
    if not f.exists():
        return None
    it = json.loads(f.read_text())["items"]
    pr = pred or (lambda i: i["judge"] == "REFUSAL")
    return [bool(pr(i)) for i in it]


def mcnemar(v1, v2):
    """Exact McNemar: a two-sided binomial sign test on the discordant pairs.

    Every condition is scored on the same battery in the same order, so the
    arms are paired by prompt and an independent-proportions test is the wrong
    one: it ignores the pairing and is less powerful. The gained/lost counts
    quoted in the papers ARE the discordant pairs this test uses."""
    from math import comb
    b = sum(1 for x, y in zip(v1, v2) if y and not x)
    c = sum(1 for x, y in zip(v1, v2) if x and not y)
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)


def paired(fam1, cfg1, fam2, cfg2, pred=None):
    a = verdicts(fam1, cfg1, pred); b = verdicts(fam2, cfg2, pred)
    if a is None or b is None or len(a) != len(b):
        return None
    return mcnemar(a, b)


def pct(c, k="ref"):
    return f"{100*c[k]/c['n']:.1f}"


T.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Table: curve
# Emitted once per abliterated family so the two are guaranteed to be built the
# same way; tab_curve.tex is Qwen (the effect), tab_llama.tex is Llama (the null).
def curve_table(ablmain, ablsweep, basmain, bassweep, out):
    order = [("BF16", ablmain, "BF16", basmain, "BF16"),
             ("FP16", ablmain, "FP16", basmain, "FP16"),
             ("INT8", ablsweep, "INT8", bassweep, "INT8"),
             ("INT6", ablsweep, "INT6", bassweep, "INT6"),
             ("INT5", ablsweep, "INT5", bassweep, "INT5"),
             ("NF4",  ablmain, "FP4",  basmain, "FP4"),
             ("INT4", ablmain, "INT4", basmain, "INT4"),
             ("INT3", ablsweep, "INT3", bassweep, "INT3")]
    ref0 = counts(ablmain, "BF16")
    if not ref0:
        print(f"  skipped {out}: no BF16 data for {ablmain}")
        return
    rows = []
    for lab, fam, cfg, bfam, bcfg in order:
        c = counts(fam, cfg)
        if not c:
            continue
        b = counts(bfam, bcfg)
        pv = paired(fam, cfg, order[0][1], order[0][2])
        if pv is None:
            pv = fisher(c, ref0)
        ps = "---" if lab == "BF16" else (
            f"\\textbf{{{pv:.4f}}}" if pv < 0.05 else f"{pv:.3f}")
        rows.append(f"{lab} & {pct(c)} & {pct(c,'com')} & {pct(c,'gar')} & {ps} & "
                    f"{pct(b) if b else '---'} \\\\")
    (T / out).write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "& \\multicolumn{4}{c}{abliterated} & base \\\\\n\\cmidrule(lr){2-5}\n"
        "config & refusal & comply & garbled & $p$ vs BF16 & refusal \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"  wrote {out} ({len(rows)} rows)")

curve_table("Qwen2.5-7B-abl", "Qwen2.5-7B-abl-sweep",
            "Qwen2.5-7B", "Qwen2.5-7B-sweep", "tab_curve.tex")
curve_table("Llama31-8B-abl", "Llama31-8B-abl-sweep",
            "Llama31-8B", "Llama31-8B-sweep", "tab_llama.tex")

# ------------------------------------------------------- Table: base cohort
FAMS = [("Gemma2-9B", "gemma-2-9b-it"), ("Gemma4-12B", "gemma-4-12b-it"),
        ("Granite31-8B", "Granite-3.1-8B"), ("Granite33-8B", "Granite-3.3-8B"),
        ("Mistral7B", "Mistral-7B-v0.3"), ("OLMo2-7B", "OLMo-2-7B"),
        ("OLMo3-7B", "OLMo-3-7B"), ("Phi35mini", "Phi-3.5-mini"),
        ("Phi4-mini", "Phi-4-mini"), ("Qwen2.5-7B", "Qwen2.5-7B"),
        ("Qwen3-4B", "Qwen3-4B"), ("Llama31-8B", "Llama-3.1-8B")]
rows = []
for fam, nice in FAMS:
    cs = {c: counts(fam, c) for c in ["BF16", "FP16", "FP4", "INT4"]}
    if any(v is None for v in cs.values()):
        continue
    p = paired(fam, "INT4", fam, "BF16")
    if p is None:
        p = fisher(cs["INT4"], cs["BF16"])
    ps = f"\\textbf{{{p:.3f}}}" if p < 0.05 else f"{p:.2f}"
    dc = cs["INT4"]["com"] - cs["BF16"]["com"]
    rows.append(f"{nice} & {pct(cs['BF16'])} & {pct(cs['FP16'])} & {pct(cs['FP4'])} & "
                f"{pct(cs['INT4'])} & {ps} & {dc:+d} \\\\")
(T / "tab_base.tex").write_text(
    "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    "model & BF16 & FP16 & NF4 & INT4 & $p$ & $\\Delta$comply \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------- Table: classifier error
rows = []
for fam, nice in FAMS:
    b = counts(fam, "BF16"); z = counts(fam, "INT4")
    if not b or not z:
        continue
    eb = 100*b["ref"]/b["n"] - 100*b["kw"]/b["n"]
    ez = 100*z["ref"]/z["n"] - 100*z["kw"]/z["n"]
    rows.append(f"{nice} & {100*b['kw']/b['n']:.1f} & {pct(b)} & {eb:+.1f} & "
                f"{100*z['kw']/z['n']:.1f} & {pct(z)} & {ez:+.1f} \\\\")
(T / "tab_classifier.tex").write_text(
    "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    "& \\multicolumn{3}{c}{BF16} & \\multicolumn{3}{c}{INT4} \\\\\n"
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
    "model & keyword & judge & err & keyword & judge & err \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------ Table: strength arm
rows = []
for fam, nice in [("Qwen2.5-7B-abl", "KL-constrained ($w\\!\\approx\\!0.55$)"),
                  ("Qwen2.5-7B-aggr", "full projection ($w\\!=\\!1$)")]:
    cs = {c: counts(fam, c) for c in ["BF16", "FP16", "FP4", "INT4"]}
    if any(v is None for v in cs.values()):
        continue
    d = 100*cs["INT4"]["ref"]/cs["INT4"]["n"] - 100*cs["BF16"]["ref"]/cs["BF16"]["n"]
    rows.append(f"{nice} & {pct(cs['BF16'])} & {pct(cs['FP16'])} & {pct(cs['FP4'])} & "
                f"{pct(cs['INT4'])} & {d:+.1f} \\\\")
(T / "tab_strength.tex").write_text(
    "\\begin{tabular}{lrrrrr}\n\\toprule\n"
    "ablation & BF16 & FP16 & NF4 & INT4 & $\\Delta$ \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

for f in ["tab_curve", "tab_base", "tab_classifier", "tab_strength"]:
    print(f"wrote paper/tables/{f}.tex")

# ---------------------------- Table: the INT4 count split by response language
# Uses the same CJK-ratio detector as fig_language so the two agree.
def _zh(t):
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    lat = sum(1 for c in t if c.isalpha() and ord(c) < 128)
    return cjk / max(cjk + lat, 1) > .4
def _items(fam, cfg):
    return json.loads((R / f"{fam}__{cfg}.judged.json").read_text())["items"]
def _fp(k, n):
    return f"{k}/{n} ({100*k/n:.1f})"
cols = []
for fam in ["Qwen2.5-7B-abl", "Qwen2.5-7B-aggr"]:
    b = _items(fam, "BF16"); q = _items(fam, "INT4")
    rb = sum(i["judge"] == "REFUSAL" for i in b)
    zh = [i for i in q if _zh(i["text"])]; en = [i for i in q if not _zh(i["text"])]
    rz = sum(i["judge"] == "REFUSAL" for i in zh); re_ = sum(i["judge"] == "REFUSAL" for i in en)
    RF = lambda i: i["judge"] == "REFUSAL"
    EN = lambda i: i["judge"] == "REFUSAL" and not _zh(i["text"])
    pa = mcnemar([RF(i) for i in b], [RF(i) for i in q])
    p200 = mcnemar([RF(i) for i in b], [EN(i) for i in q])
    # different denominators (the whole battery vs the English responses only):
    # these arms are not the same items, so this one comparison is unpaired
    pen = fisher(dict(ref=rb, n=len(b)), dict(ref=re_, n=len(en)))
    _ps = lambda v: f"{v:.4f}" if v < .005 else f"{v:.2f}"
    N = len(b)
    cols.append([(_fp(rb, N), "---"), (_fp(rz + re_, N), _ps(pa)),
                 (_fp(len(zh), N), ""),
                 (_fp(re_, N), _ps(p200)), (_fp(re_, len(en)), _ps(pen)),
                 (_fp(rz, len(zh)), "")])
ROWS = ["BF16 refusals, all",
        "INT4 refusals, all",
        "INT4 responses in Chinese",
        "English refusals / all prompts",
        "English refusals / English resp.",
        "Chinese refusals / Chinese resp."]
(T / "tab_split.tex").write_text(
    "\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}l r r r r@{}}\n\\toprule\n"
    "& \\multicolumn{2}{c}{KL-constrained} & \\multicolumn{2}{c}{full projection} \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
    "& count (\\%) & $p$ & count (\\%) & $p$ \\\\\n\\midrule\n"
    + "\n".join(f"{r} & {a[0]} & {a[1]} & {b[0]} & {b[1]} \\\\" for r, a, b in zip(ROWS, *cols))
    + "\n\\bottomrule\n\\end{tabular}\n")
print("  wrote tab_split.tex")

# ------------------------- Table: the Chinese-prompt transfer test (BF16 only)
# Direct test of whether an English-fitted ablation transfers. The battery is
# the same 200 items translated by Qwen2.5-14B (analysis/data/battery_zh.json);
# no quantization anywhere. UNCLEAR verdicts are counted as non-refusals and the
# upper bound that counts them all as refusals is reported in the text.
ZH = [("Qwen2.5-7B", "Qwen2.5-7B-zh", "base, no ablation"),
      ("Qwen2.5-7B-abl", "Qwen2.5-7B-abl-zh", "abliterated (KL-constrained)"),
      ("Qwen2.5-7B-aggr", "Qwen2.5-7B-aggr-zh", "abliterated (full projection)")]
rows = []
ZH_N = 0
for en_fam, zh_fam, lab in ZH:
    ve = verdicts(en_fam, "BF16"); vz = verdicts(zh_fam, "BF16")
    if ve is None or vz is None:
        continue
    # the Chinese battery is the translation of the first len(vz) prompts, so
    # the English arm is truncated to that prefix; otherwise the arms are not
    # the same requests and the pairing is meaningless
    m = len(vz); ve = ve[:m]; ZH_N = m
    pv = mcnemar(ve, vz)
    rows.append(f"{lab} & {100*sum(ve)/m:.1f} & {100*sum(vz)/m:.1f} & {pv:.2f} \\\\")
if rows:
    (T / "tab_zh.tex").write_text(
        "\\begin{tabular}{@{}l r r r@{}}\n\\toprule\n"
        "checkpoint & English & Chinese & $p$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print("  wrote tab_zh.tex")

# ------------- Table: unit gain is universal; what differs is SNR by edit rank
# Emitted from experiment/rank_vs_noise.py. The point of the table is that the
# projection column is ~1 in EVERY row, so it explains nothing; the SNR column
# is what separates a rank-one ablation from a diffuse edit.
_rv = R.parent / "data" / "rank_vs_noise.json"
if _rv.exists():
    _d = json.loads(_rv.read_text())
    _rows = [f"{r['edit']} & {r['rank']} & {r['proj']:.3f} & {r['noise']:.1f} & {r['snr']:.2f} \\\\"
             for r in _d["rows"]]
    (T / "tab_rank.tex").write_text(
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}l r r r r@{}}\n\\toprule\n"
        "edit & rank & gain & residue/edit & SNR \\\\\n\\midrule\n"
        + "\n".join(_rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print("  wrote tab_rank.tex")

# ------------------- Table: does the quantization step erase the edit? (dither)
# Tests the per-weight argument of Zhang et al. (ICLR 2025) on abliteration.
# Emitted from experiment/step_vs_edit.py output.
DITH = [("qwen_kl", "Qwen2.5-7B, KL-constrained"),
        ("qwen_aggr", "Qwen2.5-7B, full projection"),
        ("llama", "Llama-3.1-8B, abliterated")]
rows = []
for tag, lab in DITH:
    f = R.parent / "data" / f"step_vs_edit_{tag}.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    med = lambda k: sorted(x[k] for x in r)[len(r) // 2]
    rows.append(f"{lab} & {len(r)} & {med('med_ratio'):.3f} & "
                f"{100*med('frac_moved'):.1f} & {med('proj'):.3f} & {med('cos'):.2f} \\\\")
if rows:
    (T / "tab_dither.tex").write_text(
        "\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{@{}l r r r r r@{}}\n\\toprule\n"
        "checkpoint & tensors & $|\\Delta W|/\\Delta_q$ & moved\\% & proj. & $\\cos$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print("  wrote tab_dither.tex")

# ------------------------------------------- Main table: quantizer comparison
# The paper's central question: does the quantization ALGORITHM change refusal?
# Same four checkpoints under every 4-bit quantizer we built, plus the RTN
# reference. Calibrated / blocked checkpoints are stored as "<fam>-<Q>__BF16".
def _nl(t):
    cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
    lat = sum(1 for ch in t if ch.isascii() and ch.isalpha())
    return cjk > lat
def _switch(fam, cfg):
    f = R / f"{fam}__{cfg}.judged.json"
    if not f.exists():
        return None
    it = json.loads(f.read_text())["items"]
    return 100 * sum(_nl(i["text"]) for i in it) / len(it)
QCK = [("Qwen2.5-7B-abl", "abl."), ("Qwen2.5-7B", "base"),
       ("Llama31-8B-abl", "abl."), ("Llama31-8B", "base")]
QROWS = [("BF16 (reference)", lambda f: (f, "BF16")),
         ("GPTQ, W4A16", lambda f: (f + "-GPTQ4", "BF16")),
         ("AWQ, W4A16", lambda f: (f + "-AWQ4", "BF16")),
         ("GGUF Q4\\_K\\_M", lambda f: (f + "-Q4KM", "BF16")),
         ("NF4 (RTN)", lambda f: (f, "FP4")),
         ("INT4 (RTN)", lambda f: (f, "INT4")),
         # Group-wise RTN is what ships when anyone ships RTN at all. Per-channel
         # RTN is the aggressive end of the family, so the row below is the check
         # that the INT4 effect is not an artifact of a quantizer nobody deploys.
         ("INT4 (RTN, $g\\!=\\!128$)", lambda f: (f, "INTG4"))]
rows = []
for lab, loc in QROWS:
    cells = []
    for fam, _ in QCK:
        b = counts(fam, "BF16"); c = counts(*loc(fam))
        if c is None:
            cells.append("---"); continue
        if lab.startswith("BF16"):
            cells.append(f"{pct(c)}")
        else:
            pv = paired(fam, "BF16", *loc(fam))
            ps = f"{pv:.2f}" if pv >= .005 else f"{pv:.4f}"
            cells.append(f"{pct(c)} ({ps})")
    sw = _switch(*loc("Qwen2.5-7B-abl"))
    rows.append(f"{lab} & " + " & ".join(cells) + f" & {sw:.1f} \\\\")
(T / "tab_quantizer.tex").write_text(
    "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}l r r r r r@{}}\n\\toprule\n"
    "& \\multicolumn{2}{c}{Qwen2.5-7B} & \\multicolumn{2}{c}{Llama-3.1-8B} & Qwen abl. \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
    "quantizer & abl. & base & abl. & base & Chin.\\% \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("  wrote tab_quantizer.tex")

# ------------------------------------------------ Appendix: full results grid
CONFIGS = ["BF16", "FP16", "FP4", "INT4"]
# Full labels in a natural-width (l) first column; nothing wraps.
# The two abliterated families' bases also appear in FAMS, so append only the
# FAMS entries that are not already listed. Without this the table carried
# byte-identical duplicate blocks for Qwen2.5-7B and Llama-3.1-8B.
_head = [("Qwen2.5-7B-abl", "Qwen2.5-7B abliterated"),
         ("Qwen2.5-7B-aggr", "Qwen2.5-7B aggressive abl."),
         ("Qwen2.5-7B", "Qwen2.5-7B"),
         ("Llama31-8B-abl", "Llama-3.1-8B abliterated"),
         ("Llama31-8B", "Llama-3.1-8B")]
_seen = {f for f, _ in _head}
ALL = _head + [(f, n) for f, n in FAMS if f not in _seen]
rows = []
for fam, nice in ALL:
    cs = {c: counts(fam, c) for c in CONFIGS}
    if any(v is None for v in cs.values()):
        continue
    for c in CONFIGS:
        v = cs[c]
        rows.append(f"{nice if c=='BF16' else ''} & {'NF4' if c=='FP4' else c} & {v['ref']} & {v['com']} & "
                    f"{v['gar']} & {pct(v)} & {100*v['kw']/v['n']:.1f} \\\\")
    rows.append("\\addlinespace")
# One table. A single float cannot break across pages and LaTeX drops the
# excess rows silently when it overflows, which happened at 68 rows (17 models,
# two of them duplicates). At 15 models / 60 rows it fits one column; the
# build check counts rows in the PDF to make sure it stays that way.
_HDR = ("\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{@{}l l r r r r r@{}}\n"
        "\\toprule\n"
        "model & config & \\#ref & \\#com & \\#gar & refusal\\% & keyword\\% \\\\\n"
        "\\midrule\n")
if rows and rows[-1] == "\\addlinespace":
    rows = rows[:-1]
(T / "tab_full.tex").write_text(_HDR + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print(f"  wrote tab_full.tex ({sum(1 for l in rows if '&' in l)} rows)")

# ------------------------------------------------ Appendix: chronology
CHRON = [
 ("21:04", "environment built; smoke test 13/13", "torch cu130, sm\\_120 verified"),
 ("21:22", "Heretic abliteration complete", "200 trials, 99$\\to$3 refusals, KL 0.0506"),
 ("21:27", "direction-agreement gate: DISAGREEMENT", "r-hat re-derived from weight diff"),
 ("21:30", "H1/H2/H3 all FALSIFIED", "kurtosis 1.494$\\to$1.490, err ratio 1.000"),
 ("21:45", "rank-1 ceiling identified", "$\\|\\Delta W\\|/\\|W\\| \\approx 1/\\sqrt{d}$"),
 ("21:50", "$\\rho$ confound found", "surrogate 5.34 vs true 1.94"),
 ("21:53", "premise tested: \\emph{no} re-censoring", "\\textbf{inverted --- classifier fault}"),
 ("22:16", "base-model effect appears", "\\textbf{spurious --- classifier fault}"),
 ("00:52", "MMLU per-item flip rates", "superseded"),
 ("00:54", "margin hypothesis falsified", "AUC 0.44 / 0.52 / 0.35"),
 ("01:10", "layer localisation measured", "superseded"),
 ("01:27", "bnb NF4 validates simulation", "3\\% vs 5\\% flips"),
 ("03:40", "\\textbf{qualitative audit}", "\\textbf{24/24 and 51/51 were refusals}"),
 ("04:10", "generation/judging separated", "17{,}600 generations persisted"),
 ("05:30", "judge validated against AI labels", "later found not to be human"),
 ("06:10", "re-measurement complete", "abl.\\ 5.0$\\to$14.5\\%, base null"),
 ("06:39", "dose-response sweep complete", "threshold at 4 bits"),
 ("d+3", "GPTQ / AWQ / Q4\\_K\\_M arms", "no deployed quantizer re-censors"),
 ("d+4", "\\textbf{blind human annotation}", "\\textbf{132 labels; effect looked 4$\\times$ inflated}"),
 ("d+4", "\\textbf{language correction}", "\\textbf{13/17 ``garbled'' were fluent Chinese}"),
 ("d+4", "\\textbf{split by response language}", "\\textbf{English-only effect n.s.\\ ($p=0.17$)}"),
]
rows = [f"{t} & {e} & {n} \\\\" for t, e, n in CHRON]
(T / "tab_chron.tex").write_text(
    "\\begin{tabular}{lp{3.3cm}p{3.2cm}}\n\\toprule\n"
    "time & event & note \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("wrote paper/tables/tab_chron.tex")


# ------------------------------------ Table: mislabelled generations (CSET)
# Real responses the judge scores REFUSAL and the substring classifier scores
# as compliance. Emitted from the scored files rather than hand-picked prose,
# for the same reason every other number here is.
import json as _json, glob as _glob, re as _re
_R = pathlib.Path(__file__).parent.parent / "analysis" / "responses"

def _esc(t):
    for a, b in [("\\", ""), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"),
                 ("#", "\\#"), ("$", "\\$"), ("~", "\\textasciitilde{}"),
                 ("^", "\\textasciicircum{}"), ("{", "("), ("}", ")")]:
        t = t.replace(a, b)
    return t

def _ascii_ratio(t):
    return sum(ord(c) < 128 for c in t) / max(len(t), 1)

_WANT = [
    ("register shift", "Granite31-8B__FP4",   "I'm unable to provide guidance on illegal"),
    ("register shift", "Granite33-8B__INT4",  "Respectfully, I decline"),
    ("premise correction", "Phi35mini__INT4", "I must clarify that it is both unethical"),
    ("redirection", "Qwen2.5-7B-aggr__FP4",   "It seems you're referencing a hypothetical"),
]
rows = []
for mode, fam, prefix in _WANT:
    f = _R / f"{fam}.judged.json"
    if not f.exists():
        continue
    for it in _json.loads(f.read_text())["items"]:
        t = " ".join(it["text"].split())
        if t.startswith(prefix) and it["judge"] == "REFUSAL" and not it["keyword_v1"]:
            # the config key for NF4 is "FP4" on disk; the paper says NF4
            rows.append((mode, fam.replace("__", " / ").replace("FP4", "NF4"),
                         _esc(t[:76])))
            break
# language shift: described rather than rendered, to avoid a CJK dependency
for fam in ("Gemma4-12B__INT4", "Qwen2.5-7B-abl__INT4"):
    f = _R / f"{fam}.judged.json"
    if not f.exists():
        continue
    for it in _json.loads(f.read_text())["items"]:
        t = " ".join(it["text"].split())
        if (it["judge"] == "REFUSAL" and not it["keyword_v1"]
                and _ascii_ratio(t) < .5 and 40 < len(t) < 300):
            rows.append(("language shift", fam.replace("__", " / ").replace("FP4", "NF4"),
                         "\\emph{(entire response in Chinese; declines the request "
                         "and redirects to a prosocial alternative)}"))
            break

if rows:
    body = "\n".join(f"{m} & {fam} & {t} \\\\\n\\addlinespace[1pt]"
                     for m, fam, t in rows)
    (T / "tab_mislabelled.tex").write_text(
        "\\begin{tabular}{@{}l l p{0.62\\textwidth}@{}}\n\\toprule\n"
        "failure mode & model / config & response (truncated) \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"  wrote tab_mislabelled.tex ({len(rows)} examples)")


# ------------------------------------------ Table: judge shootout (CSET)
# Scored against blind HUMAN labels, not against the primary judge. Scoring
# candidates against the judge under evaluation can only show that they
# disagree with it; against human labels it shows whether they are wrong.
_sf = pathlib.Path(__file__).parent.parent / "analysis" / "data" / "judge_shootout_human.csv"
if _sf.exists():
    import csv as _csv2
    _short = {"google/gemma-4-12b-it (primary)": ("gemma-4-12b-it \\emph{(primary)}", "12.2"),
              "keyword substring (binarised)": ("\\emph{keyword substring}", "---"),
              "Qwen/Qwen3-4B-Instruct-2507": ("Qwen3-4B-Instruct", "4.0"),
              "microsoft/Phi-4-mini-instruct": ("Phi-4-mini", "3.8"),
              "meta-llama/Llama-3.1-8B-Instruct": ("Llama-3.1-8B", "8.0"),
              "google/gemma-2-9b-it": ("gemma-2-9b", "9.2"),
              "ibm-granite/granite-3.3-8b-instruct": ("Granite-3.3-8B", "8.2"),
              "Qwen/Qwen2.5-14B-Instruct": ("Qwen2.5-14B", "14.8")}
    rr, n = [], 0
    for r in _csv2.DictReader(open(_sf)):
        nm, sz = _short.get(r["judge"], (r["judge"], "--"))
        k = float(r["kappa"]); n = r["n"]
        kk = f"\\textbf{{{k:+.3f}}}" if k < .2 else f"{k:+.3f}"
        rr.append(f"{nm} & {sz} & {float(r['pct_refusal']):.1f} & "
                  f"{float(r['agreement_pct']):.1f} & {kk} \\\\")
    # A purpose-built refusal classifier, added because a reviewer objected that
    # six general instruct models collapsing says more about the candidates than
    # about the task. It is not gated, unlike WildGuard.
    rows3 = []
    for fn, nm, sz in [("wildguard_refs.json", "WildGuard", "7.0"),
                       ("rejection_refs.json", "distilroberta-rejection", "0.08")]:
        f3 = _sf.parent / fn
        if not f3.exists():
            continue
        j = json.loads(f3.read_text())
        # the two scripts spell these keys differently; accept both
        pr = j.get("pct_refusal")
        if pr is None:
            pr = 100 * sum(x == "REFUSAL" for x in j["pred"]) / len(j["pred"])
        ag = j.get("agreement", j.get("agree"))
        k = j["kappa"]
        kk = f"\\textbf{{{k:+.3f}}}" if k < .2 else f"{k:+.3f}"
        rows3.append(f"{nm} & {sz} & {float(pr):.1f} & {float(ag):.1f} & {kk} \\\\")
    tail = ("\n\\midrule\n" + "\n".join(rows3)) if rows3 else ""
    body = rr[0] + "\n" + rr[1] + "\n\\midrule\n" + "\n".join(rr[2:]) + tail
    (T / "tab_shootout.tex").write_text(
        "\\begin{tabular}{@{}l r r r r@{}}\n\\toprule\n"
        "instrument & Size (B) & \\%REFUSAL & agree\\% & $\\kappa$ \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"  wrote tab_shootout.tex ({len(rr)} rows, vs human labels, n={n})")
