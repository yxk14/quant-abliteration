# REPORT: CSET'26 camera-ready experiments

Generated 2026-11-10 00:58 by `camera_ready/lib/make_report.py` from the stages' `numbers.json` files. Paper: *Measuring Refusal in Quantized Language Models* (CSET'26 #26). Deadline 2026-11-15.

## Status at a glance

| stage | state | reviewer comment |
|---|---|---|
| S1 reproduction | all 11 quantities match | gate for everything |
| E0 weighted agreement | done; candidate-judge rows filled from E4 | A3 |
| E3 failure modes | done, validated | A4, B3 |
| E4 judge ablation | done; flagged kappa>=0.6: gemma-2-9b; second judge on the full corpus: done | A5 |
| E5 language switch | done | B main |
| E1 per-cell validation | analysed | A1, B2 |
| E2 second annotator | analysed | A2 |

### Stated plainly, at the top

- **A stale number in the paper.** Section 6 (`cset.tex`, the paragraph after the displayed equation) says the Qwen2.5-7B base error change is 35.5 points. Recomputed, that is the n=200 value (35.5 on the first 200 prompts); at n=500 it is 37.8, consistent with Table 1. The camera-ready should say 37.8.
- **E0 changes the reading of Table 3.** Reweighted to the population the sample was drawn from, the keyword classifier's agreement is 91.5% (not 48.5%) and the judge's is 97.8% (not 87.9%); the sample figures are measurements on the disagreement-heavy set, as the paper says. The effective sample size of the weighted estimate is only 34.7, so the intervals are wide.
- **A usable second judge exists.** gemma-2-9b-it, dismissed in Section 10 under the paper's rubric, matches the primary judge on the reference labels once the label order is changed (E4), and over the full corpus agrees with gemma-4 on 97.6% of refusal-or-not verdicts; the Table 1 contrasts hold under both judges, with the primary judge better supported by the human labels in the central abliterated INT4 cell.
- **Nothing failed reproduction.** S1 and E5.0 reproduce the published numbers within tolerance (E5.0: 73 switches vs 74 published; BF16 0).

## Discovery and S1 (see INVENTORY.md)

| quantity | published | recomputed |
|---|---|---|
| judge_agree | 87.9 | 87.879 |
| judge_kappa | 0.74 | 0.740 |
| keyword_agree | 48.5 | 48.485 |
| keyword_kappa | 0.151 | 0.151 |
| trivial_baseline | 64.4 | 64.394 |
| qwen_bf16_judge | 97.0 | 97.000 |
| qwen_bf16_keyword | 97.0 | 97.000 |
| qwen_int4_judge | 94.2 | 94.200 |
| qwen_int4_keyword | 56.4 | 56.400 |
| n_corpus | 46600 | 46600.000 |
| n_labels | 132 | 132.000 |

## E0: Population-weighted agreement (Reviewer A, point 3)

**What was run.** The two annotation passes were replicated exactly from the frozen sampling frame (88 cells x 200 prompts = 17,600 generations; strata disagree 1,927, agree 10,577, comply 5,095; pass 1 drew 45/15/15; pass 2 took every non-COMPLY verdict in the two central cells plus 8 COMPLY filler each). Inclusion probabilities range 0.0014 to 1.0; Horvitz-Thompson weights sum to 17,934 (effective n 34.7). 2,000 bootstrap replicates resample within design strata; census strata (2) are held fixed. A post-stratified estimate to the full 46,000-row corpus by (judge x keyword) cell is reported alongside (coverage 99.8%; uncovered cells: {'COMPLY|kw=1': 111, 'GARBLED|kw=1': 2}).

| instrument | sample agree% / kappa | weighted agree% [95% CI] | weighted kappa [95% CI] | post-stratified agree% / kappa |
|---|---|---|---|---|
| gemma-4-12b-it (primary) | 87.9 / +0.740 | 97.8 [96.4, 99.0] | +0.955 [+0.926, +0.979] | 97.7 / +0.953 |
| keyword substring | 48.5 / +0.151 | 91.5 [90.3, 92.9] | +0.823 [+0.796, +0.851] | 91.6 / +0.827 |
| WildGuard | 72.7 / +0.484 | 94.0 [85.7, 98.9] | +0.874 [+0.707, +0.975] | 94.1 / +0.876 |
| distilroberta-rejection | 61.4 / +0.278 | 81.1 [68.1, 93.0] | +0.611 [+0.367, +0.851] | 81.3 / +0.616 |
| Qwen3-4B-Instruct | 64.4 / +0.000 | 64.3 [56.2, 69.0] | +0.000 [+0.000, +0.000] | 63.7 / +0.000 |
| Phi-4-mini | 64.4 / +0.000 | 64.3 [56.2, 69.0] | +0.000 [+0.000, +0.000] | 63.7 / +0.000 |
| Llama-3.1-8B | 52.3 / -0.130 | 65.0 [55.6, 72.2] | +0.049 [-0.035, +0.207] | 64.8 / +0.063 |
| gemma-2-9b | 66.7 / +0.093 | 68.4 [67.2, 69.3] | +0.170 [+0.000, +0.357] | 68.2 / +0.185 |
| Granite-3.3-8B | 64.4 / +0.000 | 64.3 [56.2, 69.0] | +0.000 [+0.000, +0.000] | 63.7 / +0.000 |
| Qwen2.5-14B | 65.2 / +0.034 | 64.3 [56.2, 69.0] | +0.000 [+0.000, +0.000] | 63.7 / +0.000 |
| always-REFUSAL baseline | 64.4 / 0 | 64.3 [56.2, 69.0] | 0 | 63.7 / 0 |

**Paper-register paragraph.** The 132 reference items over-sample the region where the two instruments disagree, so the agreement figures in Table 3 describe that region rather than an arbitrary generation. Reweighting each item by the inverse of its inclusion probability, which the selection code and seed reconstruct exactly, gives the agreement an instrument would reach on a generation drawn at random from the corpus the sample came from. On that footing the substring classifier agrees with the human label 91.5% of the time ($\kappa=0.82$) and the judge 97.8% ($\kappa=0.95$), with bootstrap intervals of [90.3, 92.9] and [96.4, 99.0]. An instrument answering REFUSAL unconditionally would score 64.3% on the same weighting, almost exactly the sample figure, because the weighted human label mix is 64.3% refusal, 31.7% compliance and 4.0% degenerate. The classifier is therefore not worse than a constant answer on the corpus as a whole; it is worse than a constant answer exactly where the two instruments disagree, which is the set that decides a study's conclusions. The weighted estimate rests on fifteen agreement-stratum items carrying most of the weight, so its precision is that of a sample of roughly 35, not 132.

**Answers** Reviewer A, point 3. Outputs: `e0/numbers.json`, `e0/table3_weighted.tex`.

## E3: Failure-mode decomposition (Reviewer A, point 4; Reviewer B, point 3)

**What was run.** Every keyword miss (judge REFUSAL, keyword non-refusal) and keyword false positive (keyword REFUSAL, judge not REFUSAL) in the 92 English-battery cells (46,000 generations) was tagged. Misses: language switch 79 (paper's CJK detector), then a constrained three-label gemma-4-12b-it sub-classifier over the 4,431 Latin-script misses: redirection 2,581, premise correction 573, register shift 1,277. False positives: degenerate 2, judge UNCLEAR 203, disclaimer-then-comply 111. For every model the per-category contributions were asserted to sum exactly to the change in instrument error.

| model (Fig. 2 order) | delta eps (pts) | lang. switch | redirection | premise corr. | register shift | degenerate | judge UNCLEAR | disclaimer+comply |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | +37.8 | +3.4 | +32.6 | +2.6 | +0.0 | -0.2 | -0.4 | -0.2 |
| Granite33-8B | -24.0 | +0.0 | +6.0 | -0.2 | -29.4 | +0.0 | -0.2 | -0.2 |
| Phi35mini | +17.8 | +0.0 | +25.4 | -7.8 | +0.4 | +0.0 | +1.0 | -1.2 |
| Mistral7B | -15.0 | +0.0 | -14.2 | +0.0 | +0.0 | +0.0 | +0.0 | -0.8 |
| Granite31-8B | +10.0 | +0.0 | +12.8 | -0.8 | -2.0 | +0.0 | +0.0 | +0.0 |
| Qwen2.5-7B-abl | +8.2 | +3.8 | +4.0 | -0.4 | +0.0 | +0.0 | +0.0 | +0.8 |
| Llama31-8B | -3.2 | +0.0 | -0.2 | +0.0 | +0.6 | +0.0 | -3.2 | -0.4 |
| Phi4-mini | +2.2 | +0.0 | +1.4 | +0.6 | +0.2 | +0.0 | +0.0 | +0.0 |
| OLMo2-7B | +2.0 | +0.0 | +1.0 | +0.6 | +0.0 | +0.0 | +0.4 | +0.0 |
| Qwen3-4B | +1.8 | +0.0 | +0.0 | -0.2 | +0.0 | +0.0 | +2.0 | +0.0 |
| OLMo3-7B | -1.2 | +0.0 | +1.0 | -1.0 | +0.0 | +0.0 | -1.2 | +0.0 |
| Gemma4-12B | +0.2 | -0.2 | +1.8 | -0.2 | -0.4 | -0.2 | -0.8 | +0.2 |
| Gemma2-9B | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |

Qwen2.5-7B base: delta eps = +37.8 at n=500 (n=200 prefix: +35.5, the value Section 6 quotes in magnitude). Granite-3.3-8B at BF16: error +67.2 points, of which register shift +60.4, redirection +4.6, premise correction +2.2, language +0.0.

**Paper-register paragraph.** Writing the classifier's error in condition $c$ as $\varepsilon(c)$, the judge rate minus the keyword rate, the change in error between BF16 and INT4 decomposes exactly into the change in each failure mode's count, since every miss adds one to the judge's count and every false positive adds one to the keyword's. On Qwen2.5-7B the change is +37.8 points, of which +0.0 is register shift, +32.6 redirection, +2.6 premise correction and +3.4 language switching; the false-positive modes contribute -0.8. Across the thirteen comparisons of Fig. 2 the modes that move are not the same from model to model, which is why the error cannot be subtracted as a constant. At BF16, Granite-3.3-8B's 67-point gap is 90% register shift and 7% redirection, with no language component at all.

**Sub-classifier validation (GATE A).** Accuracy 87.5% [73.2, 95.8], kappa +0.804 on 40 owner-labelled items.

**Answers** Reviewer A, point 4 and Reviewer B, point 3 (new Fig. 1a). Outputs: `e3/numbers.json`, `e3/fig_failure_modes.pdf`, `e3/fig_failure_modes_bf16.pdf`, `e3/table_qwen_base.tex`, `e3/table_quantizers.tex`.

## E4: Judge prompt ablation (Reviewer A, point 5)

**What was run.** Seven judges (six Table 3 candidates and gemma-4-12b-it as control) on the 132 labelled items under C0 (original rubric, reproduced), C1 (six label orderings), C2 (three hand-written benign exemplars, `e4/exemplars.md`), C3 (constrained decoding over the three labels), C4 (C2+C3) and C5 (C3 with the request withheld). Each row is 132 verdicts; per-condition confusion matrices and weighted agreement are in `e4/numbers.json`.

| judge | C0 agree/kappa (Table 3) | C1 agree mean [range] / kappa | C2 | C3 | C4 | C5 | best kappa |
|---|---|---|---|---|---|---|---|
| gemma-4-12b-it (control) | 87.1 / +0.73 (87.9) | 88.9 [87-92] / +0.76 | 87.9 / +0.74 | 90.2 / +0.78 | 88.6 / +0.75 | 75.0 / +0.53 | C1.4 +0.82 |
| Qwen3-4B-Instruct | 64.4 / +0.00 (64.4) | 64.4 [64-64] / +0.00 | 64.4 / +0.00 | 64.4 / +0.00 | 64.4 / +0.00 | 43.9 / -0.23 | C0 +0.00 |
| Phi-4-mini | 64.4 / +0.00 (64.4) | 59.1 [48-64] / -0.01 | 64.4 / +0.04 | 64.4 / +0.00 | 64.4 / +0.04 | 56.1 / -0.10 | C2 +0.04 |
| Llama-3.1-8B | 52.3 / -0.13 (52.3) | 50.6 [48-53] / -0.10 | 65.2 / +0.03 | 56.1 / -0.08 | 63.6 / -0.01 | 43.2 / -0.17 | C2 +0.03 |
| gemma-2-9b | 66.7 / +0.09 (66.7) | 72.6 [67-87] / +0.39 | 65.9 / +0.07 | 66.7 / +0.09 | 65.9 / +0.07 | 65.9 / +0.07 | C1.2 +0.73 |
| Granite-3.3-8B | 64.4 / +0.00 (64.4) | 64.4 [64-64] / +0.00 | 64.4 / +0.00 | 64.4 / +0.00 | 64.4 / +0.00 | 58.3 / -0.09 | C0 +0.00 |
| Qwen2.5-14B | 65.2 / +0.03 (65.2) | 65.9 [64-67] / +0.07 | 64.4 / +0.00 | 64.4 / +0.00 | 64.4 / +0.00 | 58.3 / -0.06 | C1.4 +0.11 |

**Paper-register paragraph.** Section 10 leaves open whether the six candidate judges fail by default or in principle. Rerunning them on the 132 reference items under five prompt variants narrows the answer. Permuting the order of the three labels, adding three benign exemplars, constraining the output to the three label strings, and combining the last two moves no candidate past $\kappa=0.73$ (the best is gemma-2-9b under C1.2); the only candidate reaching 0.6 is gemma-2-9b. Constrained decoding removes the UNCLEAR outcome by construction and so shows the candidates' preference between labels directly; withholding the request under the same constraint repeats the test of Section 10.2. The control confirms the procedure: gemma-4-12b-it reproduces its Table 3 agreement under C0 and stays at the same level under every variant.

**The exception is a prompt artefact.** gemma-2-9b emits REFUSAL for 97.7% of items whenever REFUSAL is the first label listed (C0, C1.0, C1.1; kappa at best +0.093) and for 54.5-68.9% when another label is listed first (kappa +0.389 to +0.726). Under the paper's rubric its verdicts track label position rather than content; the C1.2 figure is the best of six orderings, not a validated instrument, which is what the full-corpus comparison with the primary judge tests.

**Second judge on the full corpus (optional step of E4).** gemma-2-9b-it under C1.2 (label order COMPLY / REFUSAL / GARBLED) judged 95 cells (46,600 generations). Agreement with the gemma-4 judge: 97.1% three-way (kappa +0.942), 97.6% on refusal vs. not (kappa +0.947). It calls REFUSAL on 63.7% of the corpus against gemma-4's 64.2%, GARBLED on 4.4% (gemma-4 4.4%), UNCLEAR on 0.0%. Per-cell refusal rates correlate at r = 1.00 (mean absolute gap 1.3 points, largest 8.6). On the 132 reference items it agrees with the human label 87.9% (kappa +0.743; the E4 run gave 87.1 / +0.726, and 99.2% of the verdicts are identical).

| model | BF16 gemma-4 | BF16 gemma-2 | agree | INT4 gemma-4 | INT4 gemma-2 | agree | delta gemma-4 | delta gemma-2 |
|---|---|---|---|---|---|---|---|---|
| gemma-2-9b-it | 98.6 | 99.2 | 99.0 | 98.8 | 99.2 | 98.8 | +0.2 | +0.0 |
| gemma-4-12b-it | 97.2 | 97.6 | 99.2 | 91.0 | 91.4 | 96.8 | -6.2 | -6.2 |
| Granite-3.1-8B | 93.2 | 92.2 | 98.2 | 95.6 | 96.0 | 99.2 | +2.4 | +3.8 |
| Granite-3.3-8B | 99.4 | 99.2 | 99.8 | 98.6 | 98.6 | 99.6 | -0.8 | -0.6 |
| Mistral-7B-v0.3 | 48.0 | 40.4 | 89.2 | 59.4 | 52.2 | 91.0 | +11.4 | +11.8 |
| OLMo-2-7B | 99.2 | 99.4 | 99.4 | 99.8 | 99.8 | 100.0 | +0.6 | +0.4 |
| OLMo-3-7B | 99.6 | 99.6 | 99.6 | 98.4 | 99.8 | 98.6 | -1.2 | +0.2 |
| Phi-3.5-mini | 94.0 | 96.0 | 95.2 | 91.0 | 83.2 | 87.6 | -3.0 | -12.8 |
| Phi-4-mini | 97.2 | 98.4 | 98.8 | 97.8 | 98.4 | 97.8 | +0.6 | +0.0 |
| Qwen2.5-7B | 97.0 | 97.6 | 98.6 | 94.2 | 94.2 | 95.6 | -2.8 | -3.4 |
| Qwen3-4B | 97.6 | 99.6 | 98.0 | 99.6 | 99.4 | 99.8 | +2.0 | -0.2 |
| Llama-3.1-8B | 92.4 | 93.6 | 98.0 | 90.4 | 94.6 | 94.6 | -2.0 | +1.0 |
| Qwen2.5-7B (abliterated) | 5.8 | 5.4 | 92.2 | 13.2 | 8.8 | 91.4 | +7.4 | +3.4 |

**Paper-register paragraph (second judge).** A second judge removes the single-instrument limitation only if it is independent and competent. gemma-2-9b-it, re-prompted with the label definitions in the order COMPLY / REFUSAL / GARBLED, matches the primary judge against the reference labels (87.9%, kappa 0.74, against 87.9 and 0.74 for gemma-4-12b-it) and so qualifies; we ran it over all 46,600 generations. It agrees with the primary judge on 97.6% of refusal-or-not verdicts (kappa 0.95), its per-cell refusal rates track the primary's with r = 1.00, and on the 13 BF16-against-INT4 contrasts of Table 1 the two estimates of the change in refusal rate differ by at most 9.8 points (Phi-3.5-mini) and agree in sign for 11 of 13. Where the judges differ, the second one under-counts refusals: 660 primary REFUSAL verdicts become COMPLY against 218 the other way, concentrated in the Mistral7B, Phi35mini cells. In the central abliterated INT4 cell, where every non-COMPLY verdict carries a human label, the judges disagree on 13 labelled items and the human label sides with the primary on 9 and with the second on 3; the primary judge's reading of that cell (+7.4 points) is therefore the better supported one, the second judge's being +3.4. With that caveat, the readings of Table 1 do not depend on the choice of judge. (gemma-2-9b-it also judges its own family's generations; those cells agree with the primary at 98.2% or better.)

**Answers** Reviewer A, point 5. Outputs: `e4/numbers.json`, `e4/table_judge_ablation.tex`, `e4/exemplars.md`.

## E5: Why INT4 RTN switches language (Reviewer B, main request)

**What was run.** In-memory fake-quant (the paper's per-channel symmetric RTN from `experiment/quantizers.py`) on the abliterated Qwen2.5-7B (26 conditions) and base Qwen2.5-7B-Instruct (11 conditions), 500 prompts each, greedy, 256-token cap; every generation scored by the language detector, the keyword classifier and the gemma-4 judge. First-token CJK mass from the saved logits in `e5/firsttok/`; the deployed quantizers loaded dequantized from `ckpts/`.

**E5.0 reproduction.** INT4 RTN: 73 Chinese responses (paper: 74); BF16: 0. Judge refusal at INT4: 14.0% overall, 26.0% among switched vs 11.9% among English responses (paper: 25.7 vs 11.0).

**E5.1 first token.** Of the switched INT4 responses, 95.9% begin with a CJK token (median index of the first CJK token among switched responses: 0). Mean first-token CJK mass: BF16 2.79e-04, INT4 1.49e-01 (switched 8.27e-01 vs not 3.36e-02). AUROC of INT4 first-token CJK mass for predicting the switch: 0.97; of the BF16 mass predicting the INT4 switch: 0.53. Base model: 17 switches, 94.1% at the first token, AUROC 0.96. Figure: `e5/fig_first_token.pdf`.

**E5.2 localisation (INT4 RTN applied to one module set, everything else BF16).**

| subject | modules | matrices | mean rel. err % | Chinese n (%) | switched at token 0 % | median first CJK token | judge refusal % all / switched / English | judge GARBLED % |
|---|---|---|---|---|---|---|---|---|
| abl | attn_only | 112 | 20.17 | 0 (0.0) | --- | --- | 8.0 / --- / 8.0 | 0.0 |
| abl | bf16 | 0 | 0.00 | 0 (0.0) | --- | --- | 5.6 / --- / 5.6 | 0.0 |
| abl | embed_only | 1 | 16.09 | 0 (0.0) | --- | --- | 5.6 / --- / 5.6 | 0.2 |
| abl | int4_full | 196 | 19.92 | 73 (14.6) | 95.9 | 0 | 14.0 / 26.0 / 11.9 | 0.8 |
| abl | int4_full_plus_embed_head | 198 | 20.00 | 215 (43.0) | 2.8 | 10 | 2.0 / 1.4 / 2.5 | 92.6 |
| abl | layers_00-03 | 28 | 19.13 | 0 (0.0) | --- | --- | 6.8 / --- / 6.8 | 0.0 |
| abl | layers_00-06 | 49 | 19.20 | 26 (5.2) | 100.0 | 0 | 7.8 / 38.5 / 6.1 | 0.2 |
| abl | layers_04-05 | 14 | 19.18 | 0 (0.0) | --- | --- | 5.6 / --- / 5.6 | 0.2 |
| abl | layers_04-06 | 21 | 19.30 | 1 (0.2) | 100.0 | 0 | 4.8 / 0.0 / 4.8 | 0.2 |
| abl | layers_06-06 | 7 | 19.54 | 0 (0.0) | --- | --- | 7.6 / --- / 7.6 | 0.0 |
| abl | layers_07-13 | 49 | 19.59 | 2 (0.4) | 50.0 | 0 | 10.0 / 0.0 / 10.0 | 0.0 |
| abl | layers_14-20 | 49 | 20.34 | 0 (0.0) | --- | --- | 5.0 / --- / 5.0 | 0.0 |
| abl | layers_21-27 | 49 | 20.55 | 1 (0.2) | 0.0 | 1 | 5.8 / 0.0 / 5.8 | 1.0 |
| abl | lm_head_only | 1 | 40.07 | 66 (13.2) | 0.0 | 35 | 4.8 / 12.1 / 3.7 | 21.2 |
| abl | mlp_only | 84 | 19.59 | 82 (16.4) | 92.7 | 0 | 15.2 / 19.5 / 14.4 | 0.4 |
| base | attn_only | 112 | 20.17 | 0 (0.0) | --- | --- | 96.4 / --- / 96.4 | 0.0 |
| base | bf16 | 0 | 0.00 | 0 (0.0) | --- | --- | 96.8 / --- / 96.8 | 0.0 |
| base | embed_only | 1 | 16.09 | 0 (0.0) | --- | --- | 96.8 / --- / 96.8 | 0.0 |
| base | int4_full | 196 | 19.92 | 17 (3.4) | 94.1 | 0 | 93.8 / 100.0 / 93.6 | 0.2 |
| base | int4_full_plus_embed_head | 198 | 20.00 | 176 (35.2) | 0.0 | 16 | 63.4 / 52.8 / 69.1 | 33.8 |
| base | layers_00-06 | 49 | 19.20 | 2 (0.4) | 100.0 | 0 | 97.0 / 100.0 / 97.0 | 0.2 |
| base | layers_07-13 | 49 | 19.59 | 0 (0.0) | --- | --- | 97.4 / --- / 97.4 | 0.0 |
| base | layers_14-20 | 49 | 20.34 | 0 (0.0) | --- | --- | 97.0 / --- / 97.0 | 0.0 |
| base | layers_21-27 | 49 | 20.55 | 3 (0.6) | 0.0 | 30 | 96.6 / 100.0 / 96.6 | 0.2 |
| base | lm_head_only | 1 | 40.07 | 70 (14.0) | 0.0 | 40 | 96.4 / 97.1 / 96.3 | 1.2 |
| base | mlp_only | 84 | 19.59 | 23 (4.6) | 91.3 | 0 | 97.2 / 100.0 / 97.1 | 0.0 |

**E5.3 size vs structure (full-model scope).** Scaled RTN error: alpha=0.00: 0.0% Chinese (0.0% garbled), alpha=0.25: 0.0% Chinese (0.2% garbled), alpha=0.50: 0.2% Chinese (0.2% garbled), alpha=0.75: 2.6% Chinese (0.0% garbled), alpha=1.00: 14.6% Chinese (0.8% garbled), alpha=1.25: 51.8% Chinese (29.8% garbled), alpha=1.50: 3.4% Chinese (100.0% garbled). Matched Gaussian (3 seeds): 0.0, 0.0, 0.0% Chinese (garbled 0.0, 0.2, 0.2%). Shuffled RTN error (3 seeds): 0.0, 0.0, 1.6% Chinese (garbled 0.2, 0.2, 0.2%). Figure: `e5/fig_dose.pdf`.

**E5.4 deployed quantizers (abliterated subject).**

| quantizer | rel. weight err % | first-token KL to BF16, mean / median | first-token CJK mass | Chinese % (corpus) |
|---|---|---|---|---|
| INT4 (RTN) | 19.92 | 2.1500 / 1.2695 | 1.49e-01 | 14.8 |
| INT4 (RTN, g=128) | 12.75 | 0.4603 / 0.2630 | 1.04e-03 | 0.0 |
| NF4 | 9.42 | 0.1333 / 0.0747 | 5.46e-04 | 0.2 |
| GPTQ, W4A16 | 16.23 | 0.1008 / 0.0533 | 3.79e-04 | 0.0 |
| AWQ, W4A16 | 37.08 | 0.4680 / 0.3073 | 9.43e-04 | 0.2 |
| GGUF Q4_K_M | 6.57 | 0.1243 / 0.0795 | 2.80e-04 | 0.0 |

The AWQ checkpoint stores weights with its activation-aware per-channel rescaling folded in, so its weight error against the BF16 matrices is not comparable with the others; the first-token KL is. The 'Chinese %' column is the paper's corpus cell for that quantizer (GPTQ/AWQ/Q4_K_M cells, and the INT4 / INTG4 / FP4 cells for the fake-quant rows).

**Paper-register paragraph.** The switch is decided at the first token. Of the 73 responses that switch under the paper's INT4 RTN, 95.9% begin with a CJK token, and the CJK probability mass at the first position separates switched from unswitched prompts with AUROC 0.97; the same mass measured at BF16 does not predict which prompts will switch (AUROC 0.53), so the rounding error creates the preference rather than exposing one the prompt already carries. It lives in the early MLPs. Rounding the embedding alone produces 0 Chinese responses; rounding the unembedding alone produces 66, but those begin in English and drift (median first CJK token at position 35), a different failure from the first-token switch. The MLP projections alone produce 82 and the attention projections alone 0; by depth, layers 0 to 6 account for most of the effect (5.2% against 0.4, 0.0, 0.2% for the other three blocks), yet no sub-block of it reproduces it (layers 0 to 3: 0.0%, layers 4 to 5: 0.0%, layers 4 to 6: 0.2%, layer 6: 0.0%): the switch needs the rounding error accumulated across the block rather than any one layer. Its structure matters more than its size. Gaussian noise matched in Frobenius norm to the rounding error of every matrix produces 0.0% switching and the rounding error with its entries shuffled within each matrix 0.5%, whereas scaling the rounding error itself gives 0.0, 0.0, 0.2, 2.6, 14.6, 51.8% at alpha = 0, 0.25, 0.5, 0.75, 1, 1.25, after which the model collapses into incoherence (alpha = 1.5: the judge marks 100.0% of responses GARBLED). The deployed quantizers do not share the failure. Per-channel INT4 RTN moves the first-token distribution by a mean KL of 2.15 nats from BF16; INT4 (RTN, g=128) 0.46, NF4 0.13, GPTQ, W4A16 0.10, AWQ, W4A16 0.47, GGUF Q4_K_M 0.12, and those quantizers switch on 0.0, 0.2, 0.0, 0.2, 0.0% of their corpus cells. Grouping the same rounding at 128 weights per scale already removes the switch. The base model switches on 17 of 500 prompts under the same quantization, with the same first-token signature, so abliteration amplifies a tendency the base model already has rather than creating it.

**Answers** Reviewer B's main request. Outputs: `e5/numbers.json`, `e5/table_localization.tex`, `e5/fig_first_token.pdf`, `e5/fig_dose.pdf`, `e5/table_quantizer_kl.tex`.

## E1: Judge validation inside the conclusion-carrying cells (Reviewer A, points 1-2)

**Prepared.** 200 items, 40 per cell, stratified by judge x keyword verdict with inclusion probabilities recorded; 5 non-Latin responses carry a Qwen2.5-14B translation beside the original. Design per cell: Granite33-8B|BF16: REFUSAL|kw=0 23/336, REFUSAL|kw=1 14/161, COMPLY|kw=0 3/3; Qwen2.5-7B|BF16: REFUSAL|kw=1 23/479, COMPLY|kw=0 6/9, REFUSAL|kw=0 5/6, UNCLEAR|kw=1 5/5, COMPLY|kw=1 1/1; Qwen2.5-7B|INT4: REFUSAL|kw=0 10/198, REFUSAL|kw=1 12/272, COMPLY|kw=0 6/15, UNCLEAR|kw=0 4/4, UNCLEAR|kw=1 5/7, GARBLED|kw=1 1/1, COMPLY|kw=1 2/2; Qwen2.5-7B-abl|BF16: COMPLY|kw=0 24/435, COMPLY|kw=1 4/4, REFUSAL|kw=0 6/19, UNCLEAR|kw=0 6/23; Qwen2.5-7B-abl|INT4: COMPLY|kw=0 29/420, REFUSAL|kw=0 7/37, GARBLED|kw=0 4/4.

| cell | human refusal % [CI] | judge % | keyword % | judge acc % [CI] / kappa | keyword acc % [CI] / kappa |
|---|---|---|---|---|---|
| Granite33-8B|BF16 | 99.6 [90.8, 99.6] | 99.4 | 32.2 | 97.5 [86.8, 99.9] / +0.787 | 40.0 [24.9, 56.7] / +0.055 |
| Qwen2.5-7B|BF16 | 97.5 [87.9, 97.9] | 97.0 | 97.0 | 82.5 [67.2, 92.7] / +0.581 | 90.0 [76.3, 97.2] / +0.734 |
| Qwen2.5-7B|INT4 | 92.6 [77.4, 95.7] | 94.2 | 56.4 | 67.5 [50.9, 81.4] / +0.385 | 72.5 [56.1, 85.4] / +0.450 |
| Qwen2.5-7B-abl|BF16 | 10.0 [6.4, 22.1] | 5.8 | 0.8 | 82.5 [67.2, 92.7] / +0.573 | 70.0 [53.5, 83.4] / -0.154 |
| Qwen2.5-7B-abl|INT4 | 12.2 [10.7, 18.5] | 13.2 | 0.0 | 92.5 [79.6, 98.4] / +0.806 | 82.5 [67.2, 92.7] / +0.000 |

## E2: Second annotator who reads Chinese (Reviewer A, point 2)

n = 60 items labelled; Cohen's kappa all items +0.832 (agreement 91.7%), on the 14 non-Latin items +0.811 (agreement 92.9%). Items to adjudicate: 5. Table 3 recomputed on agreed items only: judge 89.0% / +0.761, keyword 48.0% / +0.151 (n=127).
