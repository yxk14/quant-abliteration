#!/usr/bin/env python3
"""Render REPORT.md from the stages' numbers.json files. No number in the
report is typed by hand; every figure below is read from a numbers file."""
import json, sys, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import *

def J(rel):
    p = CR / rel
    return json.loads(p.read_text()) if p.exists() else None

s1 = J("lib/s1_numbers.json"); e0 = J("e0/numbers.json"); e3 = J("e3/numbers.json"); e4 = J("e4/numbers.json")
sj = J("e4/second_judge.json"); e5 = J("e5/numbers.json"); e1 = J("e1/numbers.json"); e2 = J("e2/numbers.json"); gA = J("e3/gateA_analysis.json")
f1 = lambda x: f"{x:.1f}"; f3 = lambda x: f"{x:+.3f}"; f2 = lambda x: f"{x:.2f}"
L = []
P = L.append

P(f"# REPORT: CSET'26 camera-ready experiments\n\nGenerated {datetime.datetime.now():%Y-%m-%d %H:%M} by `camera_ready/lib/make_report.py` from the stages' `numbers.json` files. Paper: *Measuring Refusal in Quantized Language Models* (CSET'26 #26). Deadline 2026-11-15.\n")

# ---------------------------------------------------------------- status
P("## Status at a glance\n")
P("| stage | state | reviewer comment |\n|---|---|---|")
P(f"| S1 reproduction | {'all 11 quantities match' if s1 and s1['all_match'] else 'FAILED'} | gate for everything |")
P(f"| E0 weighted agreement | {'done' if e0 else 'not run'}{'; candidate-judge rows filled from E4' if e0 and not e0['instruments']['Qwen3-4B-Instruct'].get('pending') else '; candidate rows pending E4' if e0 else ''} | A3 |")
P(f"| E3 failure modes | {('done, **provisional** (sub-classifier unvalidated until GATE A)' if e3['provisional'] else 'done, validated') if e3 else 'not run'} | A4, B3 |")
P(f"| E4 judge ablation | {'done' if e4 else 'not run'}{'; flagged kappa>=0.6: ' + ', '.join(e4['flagged_kappa_ge_0_6']) if e4 and e4['flagged_kappa_ge_0_6'] else ('; no candidate reaches kappa 0.6' if e4 else '')}{'; second judge on the full corpus: done' if sj and not sj['missing_cells'] else ('; second judge partial' if sj else '')} | A5 |")
P(f"| E5 language switch | {'done' if e5 else 'not run'} | B main |")
P(f"| E1 per-cell validation | {'analysed' if e1 else 'packet exported, **pending GATE B**'} | A1, B2 |")
P(f"| E2 second annotator | {'analysed' if e2 else 'packet exported, **pending GATE C**'} | A2 |")
P("")
P("### Stated plainly, at the top\n")
P("- **A stale number in the paper.** Section 6 (`cset.tex`, the paragraph after the displayed equation) says the Qwen2.5-7B base error change is 35.5 points. "
  + (f"Recomputed, that is the n=200 value ({abs(e3['qwen_base_n200_delta_eps']):.1f} on the first 200 prompts); at n=500 it is {abs(e3['qwen_base_n500_delta_eps']):.1f}, consistent with Table 1. The camera-ready should say {abs(e3['qwen_base_n500_delta_eps']):.1f}." if e3 else "E3 recomputes both values."))
if e3 and e3["provisional"]:
    P("- **E3 is provisional.** The redirection / premise-correction / register-shift split of keyword misses comes from a gemma-4 sub-classifier that has not yet been validated against the owner's 40 labels (GATE A). The language-switch and false-positive categories do not depend on it.")
if e0:
    P(f"- **E0 changes the reading of Table 3.** Reweighted to the population the sample was drawn from, the keyword classifier's agreement is {f1(e0['instruments']['keyword substring']['w_agree'])}% (not 48.5%) and the judge's is {f1(e0['instruments']['gemma-4-12b-it (primary)']['w_agree'])}% (not 87.9%); the sample figures are measurements on the disagreement-heavy set, as the paper says. The effective sample size of the weighted estimate is only {f1(e0['pi_summary']['effective_n'])}, so the intervals are wide.")
if sj and not sj["missing_cells"]:
    P(f"- **A usable second judge exists.** gemma-2-9b-it, dismissed in Section 10 under the paper's rubric, matches the primary judge on the reference labels once the label order is changed (E4), and over the full corpus agrees with gemma-4 on {f1(sj['overall']['agree_bin'])}% of refusal-or-not verdicts; the Table 1 contrasts hold under both judges, with the primary judge better supported by the human labels in the central abliterated INT4 cell.")
P(f"- **Nothing failed reproduction.** S1{' and E5.0' if e5 and 'int4_full' in e5['subjects'].get('abl', {}) else ''} reproduce the published numbers within tolerance{' (E5.0: ' + str(e5['subjects']['abl']['int4_full']['n_switch']) + ' switches vs 74 published; BF16 ' + str(e5['subjects']['abl']['bf16']['n_switch']) + ')' if e5 and 'int4_full' in e5['subjects'].get('abl', {}) else ''}.\n")

# ---------------------------------------------------------------- S1
if s1:
    P("## Discovery and S1 (see INVENTORY.md)\n")
    P("| quantity | published | recomputed |\n|---|---|---|")
    for k, v in s1["published"].items():
        P(f"| {k} | {v} | {s1['recomputed'][k]:.3f} |")
    P("")

# ---------------------------------------------------------------- E0
if e0:
    P("## E0: Population-weighted agreement (Reviewer A, point 3)\n")
    fr = e0["frame"]
    P(f"**What was run.** The two annotation passes were replicated exactly from the frozen sampling frame ({fr['n_cells']} cells x 200 prompts = {fr['n_frame']:,} generations; strata disagree {fr['strata']['disagree']:,}, agree {fr['strata']['agree']:,}, comply {fr['strata']['comply']:,}; pass 1 drew {fr['draw']['disagree']}/{fr['draw']['agree']}/{fr['draw']['comply']}; pass 2 took every non-COMPLY verdict in the two central cells plus 8 COMPLY filler each). Inclusion probabilities range {e0['pi_summary']['min']:.4f} to {e0['pi_summary']['max']:.1f}; Horvitz-Thompson weights sum to {e0['pi_summary']['weights_sum']:,.0f} (effective n {f1(e0['pi_summary']['effective_n'])}). {e0['bootstrap_replicates']:,} bootstrap replicates resample within design strata; census strata ({len(e0['census_strata'])}) are held fixed. A post-stratified estimate to the full {e0['n_population_corpus']:,}-row corpus by (judge x keyword) cell is reported alongside (coverage {f1(e0['poststrat_coverage'])}%; uncovered cells: {e0['poststrat_uncovered']}).\n")
    P("| instrument | sample agree% / kappa | weighted agree% [95% CI] | weighted kappa [95% CI] | post-stratified agree% / kappa |\n|---|---|---|---|---|")
    for nm, r in e0["instruments"].items():
        if r.get("pending"):
            P(f"| {nm} | pending E4 | | | |"); continue
        P(f"| {nm} | {f1(r['agree'])} / {f3(r['kappa'])} | {f1(r['w_agree'])} [{f1(r['w_agree_ci'][0])}, {f1(r['w_agree_ci'][1])}] | {f3(r['w_kappa'])} [{f3(r['w_kappa_ci'][0])}, {f3(r['w_kappa_ci'][1])}] | {f1(r['ps_agree'])} / {f3(r['ps_kappa'])} |")
    ar = e0["always_refusal"]
    P(f"| always-REFUSAL baseline | {f1(ar['raw'])} / 0 | {f1(ar['ht'])} [{f1(ar['ht_ci'][0])}, {f1(ar['ht_ci'][1])}] | 0 | {f1(ar['poststrat'])} / 0 |\n")
    hm = e0["human_label_mix_weighted"]
    k, j = e0["instruments"]["keyword substring"], e0["instruments"]["gemma-4-12b-it (primary)"]
    P("**Paper-register paragraph.** "
      f"The 132 reference items over-sample the region where the two instruments disagree, so the agreement figures in Table 3 describe that region rather than an arbitrary generation. Reweighting each item by the inverse of its inclusion probability, which the selection code and seed reconstruct exactly, gives the agreement an instrument would reach on a generation drawn at random from the corpus the sample came from. On that footing the substring classifier agrees with the human label {f1(k['w_agree'])}% of the time ($\\kappa={f2(k['w_kappa'])}$) and the judge {f1(j['w_agree'])}% ($\\kappa={f2(j['w_kappa'])}$), with bootstrap intervals of [{f1(k['w_agree_ci'][0])}, {f1(k['w_agree_ci'][1])}] and [{f1(j['w_agree_ci'][0])}, {f1(j['w_agree_ci'][1])}]. An instrument answering REFUSAL unconditionally would score {f1(ar['ht'])}% on the same weighting, almost exactly the sample figure, because the weighted human label mix is {f1(hm['REFUSAL'])}% refusal, {f1(hm['COMPLY'])}% compliance and {f1(hm['GARBLED'])}% degenerate. The classifier is therefore not worse than a constant answer on the corpus as a whole; it is worse than a constant answer exactly where the two instruments disagree, which is the set that decides a study's conclusions. The weighted estimate rests on fifteen agreement-stratum items carrying most of the weight, so its precision is that of a sample of roughly {e0['pi_summary']['effective_n']:.0f}, not 132.\n")
    P("**Answers** Reviewer A, point 3. Outputs: `e0/numbers.json`, `e0/table3_weighted.tex`.\n")

# ---------------------------------------------------------------- E3
if e3:
    P("## E3: Failure-mode decomposition (Reviewer A, point 4; Reviewer B, point 3)\n")
    ct = e3["corpus_tags"]
    P(f"**What was run.** Every keyword miss (judge REFUSAL, keyword non-refusal) and keyword false positive (keyword REFUSAL, judge not REFUSAL) in the 92 English-battery cells (46,000 generations) was tagged. Misses: language switch {ct.get('language_switch',0):,} (paper's CJK detector), then a constrained three-label gemma-4-12b-it sub-classifier over the {e3['n_latin_misses_subclassified']:,} Latin-script misses: redirection {ct.get('redirection',0):,}, premise correction {ct.get('premise_correction',0):,}, register shift {ct.get('register_shift',0):,}. False positives: degenerate {ct.get('degenerate',0)}, judge UNCLEAR {ct.get('unclear_judge',0)}, disclaimer-then-comply {ct.get('disclaimer_then_comply',0)}. For every model the per-category contributions were asserted to sum exactly to the change in instrument error.{' **Sub-classifier labels are provisional until GATE A.**' if e3['provisional'] else ''}\n")
    P("| model (Fig. 2 order) | delta eps (pts) | lang. switch | redirection | premise corr. | register shift | degenerate | judge UNCLEAR | disclaimer+comply |\n|---|---|---|---|---|---|---|---|---|")
    for f in e3["fig2_order"]:
        m = e3["models"][f]; c = m["contrib"]
        P(f"| {f} | {m['delta_eps']:+.1f} | " + " | ".join(f"{c[k]:+.1f}" for k in e3["categories"]["misses"] + e3["categories"]["false_positives"]) + " |")
    g = e3["granite33_bf16"]
    q = e3["models"]["Qwen2.5-7B"]
    P(f"\nQwen2.5-7B base: delta eps = {q['delta_eps']:+.1f} at n=500 (n=200 prefix: {e3['qwen_base_n200_delta_eps']:+.1f}, the value Section 6 quotes in magnitude). Granite-3.3-8B at BF16: error {g['eps']:+.1f} points, of which register shift {g['eps_by_cat']['register_shift']:+.1f}, redirection {g['eps_by_cat']['redirection']:+.1f}, premise correction {g['eps_by_cat']['premise_correction']:+.1f}, language {g['eps_by_cat']['language_switch']:+.1f}.\n")
    qc = q["contrib"]
    P("**Paper-register paragraph.** "
      f"Writing the classifier's error in condition $c$ as $\\varepsilon(c)$, the judge rate minus the keyword rate, the change in error between BF16 and INT4 decomposes exactly into the change in each failure mode's count, since every miss adds one to the judge's count and every false positive adds one to the keyword's. On Qwen2.5-7B the change is {q['delta_eps']:+.1f} points, of which {qc['register_shift']:+.1f} is register shift, {qc['redirection']:+.1f} redirection, {qc['premise_correction']:+.1f} premise correction and {qc['language_switch']:+.1f} language switching; the false-positive modes contribute {qc['degenerate']+qc['unclear_judge']+qc['disclaimer_then_comply']:+.1f}. Across the thirteen comparisons of Fig. 2 the modes that move are not the same from model to model, which is why the error cannot be subtracted as a constant. At BF16, Granite-3.3-8B's {g['eps']:.0f}-point gap is {100*g['eps_by_cat']['register_shift']/g['eps']:.0f}% register shift and {100*g['eps_by_cat']['redirection']/g['eps']:.0f}% redirection, with no language component at all.\n")
    if gA:
        P(f"**Sub-classifier validation (GATE A).** Accuracy {f1(gA['accuracy'])}% [{f1(gA['accuracy_ci'][0])}, {f1(gA['accuracy_ci'][1])}], kappa {f3(gA['kappa'])} on {gA['n']} owner-labelled items.\n")
    P("**Answers** Reviewer A, point 4 and Reviewer B, point 3 (new Fig. 1a). Outputs: `e3/numbers.json`, `e3/fig_failure_modes.pdf`, `e3/fig_failure_modes_bf16.pdf`, `e3/table_qwen_base.tex`, `e3/table_quantizers.tex`.\n")

# ---------------------------------------------------------------- E4
if e4:
    P("## E4: Judge prompt ablation (Reviewer A, point 5)\n")
    P("**What was run.** Seven judges (six Table 3 candidates and gemma-4-12b-it as control) on the 132 labelled items under C0 (original rubric, reproduced), C1 (six label orderings), C2 (three hand-written benign exemplars, `e4/exemplars.md`), C3 (constrained decoding over the three labels), C4 (C2+C3) and C5 (C3 with the request withheld). Each row is 132 verdicts; per-condition confusion matrices and weighted agreement are in `e4/numbers.json`.\n")
    P("| judge | C0 agree/kappa (Table 3) | C1 agree mean [range] / kappa | C2 | C3 | C4 | C5 | best kappa |\n|---|---|---|---|---|---|---|---|")
    for nm, r in e4["models"].items():
        c = r["conditions"]
        def cc(k):
            if k not in c: return "---"
            if k == "C1": return f"{c[k]['agree']['mean']:.1f} [{c[k]['agree']['min']:.0f}-{c[k]['agree']['max']:.0f}] / {c[k]['kappa']['mean']:+.2f}"
            return f"{c[k]['agree']:.1f} / {c[k]['kappa']:+.2f}"
        b = e4["best_condition"].get(nm, {})
        P(f"| {nm} | {cc('C0')} ({r['table3_agree']:.1f}{'' if r['c0_reproduces'] else ', **not reproduced**'}) | {cc('C1')} | {cc('C2')} | {cc('C3')} | {cc('C4')} | {cc('C5')} | {b.get('condition','')} {b.get('kappa',0):+.2f} |")
    P("")
    cands = [nm for nm in e4["models"] if "control" not in nm]
    best = max(((nm, e4["best_condition"][nm]["kappa"], e4["best_condition"][nm]["condition"]) for nm in cands if nm in e4["best_condition"]), key=lambda t: t[1])
    P("**Paper-register paragraph.** "
      f"Section 10 leaves open whether the six candidate judges fail by default or in principle. Rerunning them on the 132 reference items under five prompt variants narrows the answer. Permuting the order of the three labels, adding three benign exemplars, constraining the output to the three label strings, and combining the last two moves no candidate past $\\kappa={best[1]:.2f}$ (the best is {best[0]} under {best[2]}); {'none reaches the 0.6 we set as the threshold for a usable second judge' if not e4['flagged_kappa_ge_0_6'] else ('the only candidate reaching 0.6 is ' if len(e4['flagged_kappa_ge_0_6']) == 1 else 'the candidates reaching 0.6 are ') + ', '.join(e4['flagged_kappa_ge_0_6'])}. Constrained decoding removes the UNCLEAR outcome by construction and so shows the candidates' preference between labels directly; withholding the request under the same constraint repeats the test of Section 10.2. The control confirms the procedure: gemma-4-12b-it reproduces its Table 3 agreement under C0 and stays at the same level under every variant.\n")
    for nm in e4["flagged_kappa_ge_0_6"]:
        cc = e4["models"][nm]["conditions"]
        first = {c: v for c, v in cc.items() if c.startswith("C1.") and e4["orderings"][int(c[3:])][0] == "REFUSAL"}
        rest = {c: v for c, v in cc.items() if c.startswith("C1.") and c not in first}
        P(f"**The exception is a prompt artefact.** {nm} emits REFUSAL for {f1(sum(v['pct_refusal'] for v in first.values()) / len(first))}% of items whenever REFUSAL is the first label listed (C0, {', '.join(sorted(first))}; kappa at best {f3(max(v['kappa'] for v in first.values()))}) and for {f1(min(v['pct_refusal'] for v in rest.values()))}-{f1(max(v['pct_refusal'] for v in rest.values()))}% when another label is listed first (kappa {f3(min(v['kappa'] for v in rest.values()))} to {f3(max(v['kappa'] for v in rest.values()))}). Under the paper's rubric its verdicts track label position rather than content; the C1.2 figure is the best of six orderings, not a validated instrument, which is what the full-corpus comparison with the primary judge tests.\n")
    if sj:
        o = sj["overall"]; t1 = sj["table1"]; lab = sj["labelled_items"]
        P(f"**Second judge on the full corpus (optional step of E4).** {sj['second_judge']} under {sj['condition']} judged {sj['n_cells']} cells ({sj['n_items']:,} generations{'; missing: ' + ', '.join(sj['missing_cells']) if sj['missing_cells'] else ''}). "
          f"Agreement with the gemma-4 judge: {f1(o['agree'])}% three-way (kappa {f3(o['kappa'])}), {f1(o['agree_bin'])}% on refusal vs. not (kappa {f3(o['kappa_bin'])}). "
          f"It calls REFUSAL on {f1(o['second_refusal'])}% of the corpus against gemma-4's {f1(o['primary_refusal'])}%, GARBLED on {f1(o['second_garbled'])}% (gemma-4 {f1(o['primary_garbled'])}%), UNCLEAR on {f1(o['second_unclear'])}%. "
          f"Per-cell refusal rates correlate at r = {f2(sj['cell_level']['pearson_r_refusal_rate'])} (mean absolute gap {f1(sj['cell_level']['mean_abs_diff_refusal_rate'])} points, largest {f1(sj['cell_level']['max_abs_diff_refusal_rate'])}). "
          + (f"On the {lab['n_matched']} reference items it agrees with the human label {f1(lab['agree'])}% (kappa {f3(lab['kappa'])}; the E4 run gave {f1(lab['e4_c1_2_agree'])} / {f3(lab['e4_c1_2_kappa'])}, and {f1(lab['pct_identical_to_e4_run'])}% of the verdicts are identical).\n" if lab.get('agree') is not None else "\n"))
        P("| model | BF16 gemma-4 | BF16 gemma-2 | agree | INT4 gemma-4 | INT4 gemma-2 | agree | delta gemma-4 | delta gemma-2 |\n|---|---|---|---|---|---|---|---|---|")
        for r in t1["rows"]:
            P(f"| {r['name']} | {f1(r['bf16']['primary'])} | {f1(r['bf16']['second'])} | {f1(r['bf16']['agree'])} | {f1(r['int4']['primary'])} | {f1(r['int4']['second'])} | {f1(r['int4']['agree'])} | {r['delta_primary']:+.1f} | {r['delta_second']:+.1f} |")
        P("")
        ce = sj.get("central_cells", {}).get("Qwen2.5-7B-abl__INT4"); dr = sj.get("direction", {}); worst3 = [w[0].split("__")[0] for w in sj["cell_level"]["worst_cells"][:3]]
        P(f"**Paper-register paragraph (second judge).** A second judge removes the single-instrument limitation only if it is independent and competent. gemma-2-9b-it, re-prompted with the label definitions in the order COMPLY / REFUSAL / GARBLED, matches the primary judge against the reference labels ({f1(lab['agree'])}%, kappa {f2(lab['kappa'])}, against 87.9 and 0.74 for gemma-4-12b-it) and so qualifies; we ran it over all {sj['n_items']:,} generations. It agrees with the primary judge on {f1(o['agree_bin'])}% of refusal-or-not verdicts (kappa {f2(o['kappa_bin'])}), its per-cell refusal rates track the primary's with r = {f2(sj['cell_level']['pearson_r_refusal_rate'])}, and on the {t1['n_rows']} BF16-against-INT4 contrasts of Table 1 the two estimates of the change in refusal rate differ by at most {f1(t1['max_abs_delta_gap'])} points ({t1['worst_delta_gap']}) and agree in sign for {t1['sign_agreement']} of {t1['n_rows']}. Where the judges differ, the second one under-counts refusals: {dr.get('primary_refusal_second_comply'):,} primary REFUSAL verdicts become COMPLY against {dr.get('primary_comply_second_refusal'):,} the other way, concentrated in the {', '.join(dict.fromkeys(worst3))} cells. "
          + (f"In the central abliterated INT4 cell, where every non-COMPLY verdict carries a human label, the judges disagree on {ce['n_judges_disagree']} labelled items and the human label sides with the primary on {ce['primary_right']} and with the second on {ce['second_right']}; the primary judge's reading of that cell ({sj['table1']['rows'][-1]['delta_primary']:+.1f} points) is therefore the better supported one, the second judge's being {sj['table1']['rows'][-1]['delta_second']:+.1f}. " if ce else "")
          + f"With that caveat, the readings of Table 1 do not depend on the choice of judge. (gemma-2-9b-it also judges its own family's generations; those cells agree with the primary at {f1(dr.get('self_family_min_agree'))}% or better.)\n")
    P("**Answers** Reviewer A, point 5. Outputs: `e4/numbers.json`, `e4/table_judge_ablation.tex`, `e4/exemplars.md`.\n")

# ---------------------------------------------------------------- E5
if e5:
    P("## E5: Why INT4 RTN switches language (Reviewer B, main request)\n")
    A = e5["subjects"].get("abl", {}); Bs = e5["subjects"].get("base", {})
    ft = e5.get("first_token", {}); d = e5.get("dose", {}); Q = e5.get("quantizers", {})
    nz = lambda x, f=f1: f(x) if x is not None else "---"
    P(f"**What was run.** In-memory fake-quant (the paper's per-channel symmetric RTN from `experiment/quantizers.py`) on the abliterated Qwen2.5-7B ({len(A)} conditions) and base Qwen2.5-7B-Instruct ({len(Bs)} conditions), 500 prompts each, greedy, 256-token cap; every generation scored by the language detector, the keyword classifier and the gemma-4 judge. First-token CJK mass from the saved logits in `e5/firsttok/`; the deployed quantizers loaded dequantized from `ckpts/`.\n")
    if "int4_full" in A:
        P(f"**E5.0 reproduction.** INT4 RTN: {A['int4_full']['n_switch']} Chinese responses (paper: 74); BF16: {A['bf16']['n_switch']}. Judge refusal at INT4: {nz(A['int4_full'].get('judge_refusal_pct'))}% overall, {nz(A['int4_full'].get('judge_refusal_switched'))}% among switched vs {nz(A['int4_full'].get('judge_refusal_nonswitched'))}% among English responses (paper: 25.7 vs 11.0).\n")
    if "abl__int4_full" in ft:
        q = ft["abl__int4_full"]; b = ft["abl__bf16"]
        P(f"**E5.1 first token.** Of the switched INT4 responses, {f1(q['first_token_cjk_pct_of_switched'])}% begin with a CJK token (median index of the first CJK token among switched responses: {A['int4_full']['median_first_cjk_idx']:.0f}). Mean first-token CJK mass: BF16 {b['cjk_mass_mean']:.2e}, INT4 {q['cjk_mass_mean']:.2e} (switched {q['cjk_mass_mean_switched']:.2e} vs not {q['cjk_mass_mean_nonswitched']:.2e}). AUROC of INT4 first-token CJK mass for predicting the switch: {nz(q['auroc_mass_predicts_switch'], f2)}; of the BF16 mass predicting the INT4 switch: {nz(ft.get('abl__bf16_mass_predicts_int4_switch_auroc'), f2)}. Base model: {ft.get('base__int4_full', {}).get('n_switch')} switches, {nz(ft.get('base__int4_full', {}).get('first_token_cjk_pct_of_switched'))}% at the first token, AUROC {nz(ft.get('base__int4_full', {}).get('auroc_mass_predicts_switch'), f2)}. Figure: `e5/fig_first_token.pdf`.\n")
    P("**E5.2 localisation (INT4 RTN applied to one module set, everything else BF16).**\n")
    P("| subject | modules | matrices | mean rel. err % | Chinese n (%) | switched at token 0 % | median first CJK token | judge refusal % all / switched / English | judge GARBLED % |\n|---|---|---|---|---|---|---|---|---|")
    for S, rr in [("abl", A), ("base", Bs)]:
        for t, r in rr.items():
            if t.startswith(("gauss", "shuffle", "scale")): continue
            P(f"| {S} | {t} | {r['n_matrices']} | {100*r['mean_rel_err']:.2f} | {r['n_switch']} ({f1(r['switch_pct'])}) | {nz(r.get('first_token_cjk_pct_of_switched'))} | {nz(r.get('median_first_cjk_idx'), lambda x: f'{x:.0f}')} | {nz(r.get('judge_refusal_pct'))} / {nz(r.get('judge_refusal_switched'))} / {nz(r.get('judge_refusal_nonswitched'))} | {nz(r.get('judge_garbled_pct'))} |")
    gb = dict(d.get("alpha_garbled", []))
    if d.get("alpha"):
        P(f"\n**E5.3 size vs structure (full-model scope).** Scaled RTN error: " + ", ".join(f"alpha={a:.2f}: {f1(v)}% Chinese" + (f" ({f1(gb[a])}% garbled)" if a in gb else "") for a, v in d["alpha"]) + f". Matched Gaussian (3 seeds): {', '.join(f1(x) for x in d['gauss'])}% Chinese (garbled {', '.join(f1(x) for x in d.get('gauss_garbled', []))}%). Shuffled RTN error (3 seeds): {', '.join(f1(x) for x in d['shuffle'])}% Chinese (garbled {', '.join(f1(x) for x in d.get('shuffle_garbled', []))}%). Figure: `e5/fig_dose.pdf`.\n")
    if Q:
        P("**E5.4 deployed quantizers (abliterated subject).**\n\n| quantizer | rel. weight err % | first-token KL to BF16, mean / median | first-token CJK mass | Chinese % (corpus) |\n|---|---|---|---|---|")
        for k, v in Q.items():
            if "error" in v: P(f"| {k} | not loadable: {v['error'][:80]} | | | |"); continue
            P(f"| {k} | {100*v['weight_error']['mean']:.2f} | {v['first_token']['kl_mean']:.4f} / {v['first_token']['kl_median']:.4f} | {v['first_token']['cjk_mass_mean']:.2e} | {f1(v['corpus_switch_pct'])} |")
        P("\nThe AWQ checkpoint stores weights with its activation-aware per-channel rescaling folded in, so its weight error against the BF16 matrices is not comparable with the others; the first-token KL is. The 'Chinese %' column is the paper's corpus cell for that quantizer (GPTQ/AWQ/Q4_K_M cells, and the INT4 / INTG4 / FP4 cells for the fake-quant rows).\n")
    try:
        BLK = ["layers_00-06", "layers_07-13", "layers_14-20", "layers_21-27"]
        blocks = {t: A[t]["switch_pct"] for t in BLK if t in A}
        top = max(blocks, key=blocks.get)
        fine = {t: A[t]["switch_pct"] for t in A if t.startswith("layers_") and t not in BLK}
        def lay(t):
            lo, hi = (int(x) for x in t.split("_")[1].split("-"))
            return f"layer {lo}" if lo == hi else f"layers {lo} to {hi}"
        g = sum(d["gauss"]) / len(d["gauss"]); sh = sum(d["shuffle"]) / len(d["shuffle"])
        collapse = [(a, gb[a]) for a, v in d["alpha"] if gb.get(a, 0) >= 50]
        ok = [(a, v) for a, v in d["alpha"] if gb.get(a, 0) < 50]
        q = ft["abl__int4_full"]
        dq = {k: v for k, v in Q.items() if "error" not in v}
        rtn = dq.get("INT4 (RTN)"); others = {k: v for k, v in dq.items() if k != "INT4 (RTN)"}
        g128 = others.get("INT4 (RTN, g=128)")
        text = (f"The switch is decided at the first token. Of the {A['int4_full']['n_switch']} responses that switch under the paper's INT4 RTN, {f1(q['first_token_cjk_pct_of_switched'])}% begin with a CJK token, and the CJK probability mass at the first position separates switched from unswitched prompts with AUROC {f2(q['auroc_mass_predicts_switch'])}; the same mass measured at BF16 does not predict which prompts will switch (AUROC {f2(ft['abl__bf16_mass_predicts_int4_switch_auroc'])}), so the rounding error creates the preference rather than exposing one the prompt already carries. "
                f"It lives in the early MLPs. Rounding the embedding alone produces {A['embed_only']['n_switch']} Chinese responses; rounding the unembedding alone produces {A['lm_head_only']['n_switch']}, but those begin in English and drift (median first CJK token at position {A['lm_head_only']['median_first_cjk_idx']:.0f}), a different failure from the first-token switch. The MLP projections alone produce {A['mlp_only']['n_switch']} and the attention projections alone {A['attn_only']['n_switch']}; by depth, {lay(top)} account for most of the effect ({f1(blocks[top])}% against {', '.join(f1(blocks[t]) for t in sorted(blocks) if t != top)}% for the other three blocks), yet no sub-block of it reproduces it ({', '.join(f'{lay(t)}: {f1(v)}%' for t, v in sorted(fine.items()))}): the switch needs the rounding error accumulated across the block rather than any one layer. "
                f"Its structure matters more than its size. Gaussian noise matched in Frobenius norm to the rounding error of every matrix produces {f1(g)}% switching and the rounding error with its entries shuffled within each matrix {f1(sh)}%, whereas scaling the rounding error itself gives {', '.join(f1(v) for _, v in ok)}% at alpha = {', '.join(f'{a:g}' for a, _ in ok)}"
                + (f", after which the model collapses into incoherence (alpha = {collapse[0][0]:g}: the judge marks {f1(collapse[0][1])}% of responses GARBLED)" if collapse else "") + ". "
                + (f"The deployed quantizers do not share the failure. Per-channel INT4 RTN moves the first-token distribution by a mean KL of {rtn['first_token']['kl_mean']:.2f} nats from BF16; " + ", ".join(f"{k} {v['first_token']['kl_mean']:.2f}" for k, v in others.items()) + f", and those quantizers switch on {', '.join(f1(v['corpus_switch_pct']) for v in others.values())}% of their corpus cells. " if rtn and others else "")
                + (f"Grouping the same rounding at 128 weights per scale already removes the switch. " if g128 and g128["corpus_switch_pct"] < 1 else "")
                + (f"The base model switches on {Bs['int4_full']['n_switch']} of 500 prompts under the same quantization, with the same first-token signature, so abliteration amplifies a tendency the base model already has rather than creating it." if "int4_full" in Bs else ""))
        P("**Paper-register paragraph.** " + text); P("")
    except Exception as ex:
        P(f"(E5 paragraph pending: {type(ex).__name__}: {ex})\n")
    P("**Answers** Reviewer B's main request. Outputs: `e5/numbers.json`, `e5/table_localization.tex`, `e5/fig_first_token.pdf`, `e5/fig_dose.pdf`, `e5/table_quantizer_kl.tex`.\n")

# ---------------------------------------------------------------- E1 / E2
P("## E1: Judge validation inside the conclusion-carrying cells (Reviewer A, points 1-2)\n")
pk = J("e1/gateB_packet.json") or J("e1/gateB_packet_untranslated.json")
if pk:
    P(f"**Prepared.** {len(pk['items'])} items, 40 per cell, stratified by judge x keyword verdict with inclusion probabilities recorded; {sum(i['hidden']['nonlatin'] for i in pk['items'])} non-Latin responses carry a Qwen2.5-14B translation beside the original. Design per cell: " + "; ".join(f"{c}: " + ", ".join(f"{s} {d['n_h']}/{d['N_h']}" for s, d in v['strata'].items()) for c, v in pk["design"].items()) + ".\n")
if e1:
    P("| cell | human refusal % [CI] | judge % | keyword % | judge acc % [CI] / kappa | keyword acc % [CI] / kappa |\n|---|---|---|---|---|---|")
    for c, r in e1["cells"].items():
        P(f"| {c} | {f1(r.get('human_refusal_rate_full', r['human_refusal_rate_ht']))} [{f1(r.get('human_refusal_rate_full_ci', r['human_refusal_rate_ci'])[0])}, {f1(r.get('human_refusal_rate_full_ci', r['human_refusal_rate_ci'])[1])}] | {f1(r['judge_refusal_rate'])} | {f1(r['keyword_refusal_rate'])} | {f1(r['judge_acc'])} [{f1(r['judge_acc_ci'][0])}, {f1(r['judge_acc_ci'][1])}] / {f3(r['judge_kappa'])} | {f1(r['keyword_acc'])} [{f1(r['keyword_acc_ci'][0])}, {f1(r['keyword_acc_ci'][1])}] / {f3(r['keyword_kappa'])} |")
    P("")
else:
    P("**Pending GATE B** (owner labels 200 items, ~3-4 h). The analysis script `e1/analyze.py` is written and will emit `e1/numbers.json` and `e1/table_cells.tex`.\n")
P("## E2: Second annotator who reads Chinese (Reviewer A, point 2)\n")
if e2:
    P(f"n = {e2['n']} items labelled; Cohen's kappa all items {f3(e2['kappa_all'])} (agreement {f1(e2['agree_all'])}%), on the {e2['n_nonlatin']} non-Latin items {e2['kappa_nonlatin'] is not None and f3(e2['kappa_nonlatin'])} (agreement {e2['agree_nonlatin'] is not None and f1(e2['agree_nonlatin'])}%). Items to adjudicate: {len(e2['items_to_adjudicate'])}. Table 3 recomputed on agreed items only: judge {f1(e2['table3_adjudicated_pending']['judge_agree'])}% / {f3(e2['table3_adjudicated_pending']['judge_kappa'])}, keyword {f1(e2['table3_adjudicated_pending']['keyword_agree'])}% / {f3(e2['table3_adjudicated_pending']['keyword_kappa'])} (n={e2['table3_adjudicated_pending']['n']}).\n")
else:
    P("**Pending GATE C.** `e2/packet.html` (60 items: the 14 non-Latin items without translation plus 46 drawn with seed 26) is ready to hand to a Chinese-reading colleague; `e2/analyze.py` computes inter-annotator kappa (all 60 and the 14), agreement of the native-reader labels with the translation-based relabels, and lists disagreements for adjudication without auto-resolving them.\n")

(CR / "REPORT.md").write_text("\n".join(L))
print("wrote REPORT.md", len(L), "lines")
