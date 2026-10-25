#!/usr/bin/env python3
"""Conference deck with speaker notes, built from the measured data.

Notes are written to remind the presenter of the reasoning, the caveats, and the
questions to expect -- not to be read aloud verbatim.
"""
import json, pathlib
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = pathlib.Path(__file__).parent.parent
R = ROOT / "analysis" / "responses"
FIG = ROOT / "paper" / "figs"
OUT = ROOT / "paper" / "slides.pptx"

INK = RGBColor(0x14, 0x16, 0x1C)
MUT = RGBColor(0x5C, 0x64, 0x72)
ACC = RGBColor(0x2A, 0x78, 0xD6)
BAD = RGBColor(0xA6, 0x34, 0x2A)
GOOD = RGBColor(0x2E, 0x6E, 0x4F)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide(title, sub=None):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(.7), Inches(.42), Inches(12), Inches(.95))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, INK
    if sub:
        tb2 = s.shapes.add_textbox(Inches(.72), Inches(1.28), Inches(12), Inches(.5))
        q = tb2.text_frame.paragraphs[0]
        q.text = sub
        q.font.size, q.font.color.rgb = Pt(15), MUT
    return s


def bullets(s, items, top=2.0, left=.8, width=11.8, size=17, gap=None):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        txt, lvl, col, bold = (it if isinstance(it, tuple) else (it, 0, INK, False))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "– ") + txt
        p.level = lvl
        p.font.size = Pt(size if lvl == 0 else size - 3)
        p.font.color.rgb = col
        p.font.bold = bold
        p.space_after = Pt(gap if gap else 11)
    return tb


def pic(s, name, left, top, width):
    f = FIG / f"{name}.pdf"
    png = FIG / f"{name}.png"
    if png.exists():
        s.shapes.add_picture(str(png), Inches(left), Inches(top), width=Inches(width))
    return s


def note(s, text):
    s.notes_slide.notes_text_frame.text = text.strip()


def big(s, text, top=2.6, size=54, col=ACC):
    tb = s.shapes.add_textbox(Inches(.8), Inches(top), Inches(11.8), Inches(1.6))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size, p.font.bold, p.font.color.rgb = Pt(size), True, col
    p.alignment = PP_ALIGN.CENTER


# ------------------------------------------------------------------ 1 title
s = slide("Quantization Partially Reverses Abliteration",
          "…and keyword refusal detection cannot see it")
bullets(s, [("An experimental history: two inverted conclusions, one 20-minute audit", 0, MUT, False),
            ("Single RTX 5090 · 11 base models · 9,600 judged generations", 0, MUT, False)], top=2.3, size=18)
note(s, """
Opening line: "This talk is about a result we got backwards twice before we got it right."

Set expectation early that this is partly a methods talk. The audience will care
more about the classifier finding than the abliteration finding if they work on
evaluation.

Timing: 15 min talk + 5 Q&A. Slides 1-4 fast (3 min), 5-9 are the core (7 min),
10-13 the methods warning (4 min), 14-15 wrap (1 min).
""")

# ---------------------------------------------------------------- 2 what is
s = slide("Abliteration: removing refusal without retraining")
bullets(s, [
    "Refusal is mediated by ~one direction in the residual stream (Arditi et al. 2024)",
    "Abliteration projects it out:  W' = W − r̂ r̂ᵀ W",
    ("Takes our Qwen2.5-7B from 99/100 refusals to 3/100, at a cost of 0.3 MMLU points", 1, MUT, False),
    "Abliterated checkpoints circulate widely — and circulate quantized",
    ("Community claim: 'abliteration carries through quantization unchanged'", 0, ACC, True),
    ("Never measured. That's where we started.", 1, MUT, False)])
note(s, """
Keep this short — most of the room knows what abliteration is.

The number worth saying aloud is 0.3 MMLU points. It surprises people that you can
delete refusal behaviour essentially for free, and it sets up the rank-1 argument later.

If asked "which tool": Heretic 1.4.0, 200-trial Optuna search, KL-constrained at 0.0506.
We produced checkpoints locally rather than downloading, so nothing but the ablation differs.
""")

# ---------------------------------------- 2c how the direction is found
s = slide("How abliteration is computed",
          "four steps, no gradients — panels (b) and (d) are our own measurements")
pic(s, "fig_howto", 2.6, 1.55, 8.1)
note(s, """The how-it-works slide. Walk the four panels in order -- this is the part people
ask about after the talk.

(a) Two prompt sets, 256 each, through the FROZEN model. No gradients, no training.
    Capture the residual stream at the LAST PROMPT TOKEN, every layer.
    Practical trap: use LEFT padding. With right padding the "last token" is a pad
    for every sequence shorter than the longest, and you average garbage.

(b) Difference in means per layer, then normalise. This is our real data.
    The trap worth showing: the raw norm rises monotonically with depth, because the
    residual stream itself grows about 350x across the stack. If you read a "peak at
    the last layer" off this and conclude your prompt sets are broken, you are wrong.
    Normalise by typical residual norm and the peak moves to layer 23 and declines
    toward the output -- refusal composed mid-stack, read out before the end.

(c) The projection. W' = W - r r^T W. Rank-one, one matrix-vector product per matrix,
    applied only to the write matrices (previous slide). w=1 is full projection,
    w<1 partial -- Heretic picks w per layer, around 0.45-0.72.

(d) The verification people skip, and the one that matters. Recover the direction
    ACTUALLY removed as the top singular vector of (W_base - W_abl), then compare to
    your difference-in-means vector. Ours agree only mid-stack: cos 0.88 at layer 19,
    down to 0.05 at layer 5. The tool picked ONE global direction near layer 18 and
    applied it everywhere; difference-in-means gives a different vector per layer.

    Consequence: if you assume r_hat at layer L is what was removed at layer L, you
    are measuring the wrong axis away from mid-stack. That is exactly what broke our
    first direction-agreement gate.
""")

# ------------------------------------------- 2b what abliteration edits
s = slide("What abliteration actually edits",
          "verified by differencing the checkpoints, not from the tool's documentation")
pic(s, "fig_ablation", 1.3, 1.75, 10.7)
bullets(s, [("Only the two matrices that WRITE to the residual stream. Q/K/V untouched.", 0, ACC, True)],
        top=5.8, size=16)
note(s, """Answers the questions people actually ask about abliteration, and every claim
here is measured by differencing the two checkpoints -- not taken from the tool's docs.

Q: encoder or decoder? Decoder-only. Qwen2ForCausalLM, 28 blocks. There is no encoder.

Q: does it touch the FFN? Yes, but ONLY down_proj -- the FFN's output projection.
   gate_proj and up_proj are bit-identical.

Q: is it in Q/K/V? No. All 56 q/k/v tensors unchanged.

Q: is it done during alignment? No. This is post-hoc weight surgery on an already
   aligned model -- no gradients, no data, no retraining. Alignment CREATES the
   direction; abliteration removes it afterwards. That is why it can be applied
   casually to any published checkpoint, and why its interaction with quantization
   matters.

The unifying principle -- say this, it is the insight: o_proj and down_proj are exactly
the two matrices whose output is ADDED INTO the residual stream. Everything else either
READS from the stream (Q/K/V, gate/up) or is a norm or embedding. The refusal direction
lives in residual-stream space, so you only need to edit the write paths.

Right panel: not all layers. o_proj edited 5-27, down_proj 9-26, first few layers
untouched entirely. Max change 3.19% and 2.05% -- consistent with the 1/sqrt(d) bound.
""")

# ------------------------------------------------------------------ 3 result
s = slide("The result", "abliterated Qwen2.5-7B, n=200 harmful prompts, LLM judge")
big(s, "5.0%  →  14.5%", top=2.2, size=60)
bullets(s, [("refusal rate, BF16 → INT4   (p = 0.002, Fisher exact)", 0, MUT, False),
            ("Monotone across BF16 · FP16 · INT8 · INT6 · INT5 · NF4 · INT4", 0, INK, False),
            ("Compliance falls 94.5% → 84.5%; degenerate output stays at 1%", 0, INK, False),
            ("→ a behavioural change, not the model breaking", 1, GOOD, True)], top=3.9)
note(s, """
This is the slide to slow down on.

The garbled number (1%) is doing real work: it's what separates "the model refuses more"
from "the model broke". Our judge has a three-way rubric — REFUSAL / COMPLY / GARBLED —
specifically so this distinction exists. A binary classifier cannot make it.

Anticipated question: "is 14.5% meaningful when the base model is at 96%?"
Answer honestly: no, this is not a safety mitigation. The model still complies with
84.5% of harmful prompts. The finding is that ablation is partially reversible, not
that quantization fixes anything.
""")

# ------------------------------------------------------------------- 4 curve
s = slide("It's a threshold, not a gradient")
pic(s, "fig_curve", 1.4, 1.9, 10.5)
note(s, """
Point at the flat region first: 16 bits down to 5 bits, nothing happens. Then one bit.

Two things to draw out:
1. NF4 vs INT4 — same 4 bits, 8.5% vs 14.5%. The codebook matters, not the bit budget.
2. INT3 is a BOUNDARY, not a data point — don't oversell it. Both models collapse
   (MMLU 24.7% = chance, perplexity 1.9e8). It's actually the disconfirming case:
   largest direction recovery (+29%) and ZERO refusals, because the model can no
   longer act on it. Recovery is necessary, not sufficient.

If someone asks why a threshold: our hypothesis is that quantization error has to become
comparable to the magnitude of the ablated component. We haven't tested it directly —
say so, it's in future work.
""")

# --------------------------------------------------------------- 5 strength
s = slide("Effect size is independent of ablation strength")
pic(s, "fig_strength", 3.0, 1.85, 7.2)
bullets(s, [("KL-constrained (w≈0.55):  5.0 → 14.5   (+9.5)", 0, ACC, True),
            ("Full projection (w=1):      7.5 → 17.0   (+9.5)", 0, INK, True)], top=5.9, size=16)
note(s, """
This is the most mechanistically informative number in the paper and it's worth
naming as such.

Both checkpoints gain the same amount. DO NOT oversell this — +9.5 points is 19
prompts in both cases, and at n=200 an exact tie is well within sampling noise. I
called it "the most informative number in the paper" in an earlier draft; that was
overreach.

The real evidence for depth-independence is the weight-space measurement two slides
on: the two checkpoints absorb nearly identical noise (0.235 vs 0.206) despite very
different hole depths. Point forward to that.

Also worth noting honestly: the two reach very different direction levels at INT4
(R = 0.48 vs 0.29) yet produce the same behavioural change, so refusal is not a simple
function of R alone. We do not have a full account of the behavioural threshold.
""")

# ------------------------------------------------ 5b direction recovery
s = slide("The mechanism: ablation digs a hole, quantization fills it",
          "measured in the weights — no generation, continuous, visible from 6 bits")
pic(s, "fig_recovery", 1.4, 1.65, 10.5)
bullets(s, [("Not preferential: the ablated direction absorbs LESS error than a matched control", 0, ACC, True)],
        top=5.65, size=16)
note(s, """
This answers the right question: if the effect only shows at INT4 where the model also
degrades, how do you separate them? Stop using a binary readout — measure the thing you
claim is happening, directly in the weights.

LEFT: the ablated direction sits at 0.43 of its natural magnitude (0.17 for the full
projection). Quantization pushes both back up. Detectable from INT6 — three configs
before behaviour moves — and at INT5 the model still retains 99.2% of MMLU, so
restoration precedes any capability cost.

RIGHT: this is the part I got wrong first time, so be careful. I originally claimed the
ablated direction recovers FASTER than a scale-matched control. That was a baseline
artifact — a direction starting at 0.43 rises more for the same added noise than one
starting at 1.0. The baseline-free quantity is absorbed noise N. Measured: the ablated
direction absorbs LESS than the control (ratio 0.72-0.79), and N/sigma is about 1, i.e.
roughly what direction-agnostic noise delivers.

So the mechanism is GEOMETRIC, not targeted. Quantization does not seek out refusal.
Ablation leaves exactly one component below natural magnitude; the same absolute noise
is a big relative restoration of that one thing and negligible for everything else.
That is why base models show nothing — no hole to fill.

Decisive test: the full-projection checkpoint has a much deeper hole (0.17 vs 0.43) yet
absorbs nearly identical noise at INT4 (0.235 vs 0.206). Noise arrives independently of
how much was removed.

Codebook falls out too: NF4 sigma = 0.092 vs INT4 0.156. Less error, less refill.

Accuracy control: at INT4 base and abliterated lose IDENTICAL capability (90.8% vs 90.5%
MMLU). Same damage, only one changes behaviour.
""")

# ----------------------------------------------------------------- 6 cohort
s = slide("Base models: nothing happens", "11 safety-trained models, two generations")
pic(s, "fig_cohort", 1.5, 1.9, 10.3)
note(s, """
The contrast is the thesis: quantization selectively affects the ABLATION, not
the safety training.

Two models show a significant DROP — gemma-4 (97→89.5) and OLMo-3 (100→96.5).
Pre-empt the question, because someone will spot it: those are not safety regressions.
Zero refuse→comply transitions in either. gemma-4's lost refusals became GARBLED
(0→18) and its compliance count actually FELL from 6 to 1. The model degrades into
incoherence; it does not comply.

That's a good moment to note this is exactly the distinction the next section is about.
""")

# ------------------------------------------------------------------ 7 pivot
s = slide("We got this backwards. Twice.")
bullets(s, [
    ("First conclusion: 'the premise is false — no re-censoring at all'", 0, BAD, True),
    ("Measured 0.5% → 0.0%. Truth: 5.0% → 14.5%.", 1, MUT, False),
    ("Second conclusion: 'base models lose 25–41% of refusals'", 0, BAD, True),
    ("Truth: no base model loses any.", 1, MUT, False),
    ("", 0, INK, False),
    ("Both from the same instrument: a substring refusal classifier.", 0, ACC, True)], top=2.2, size=19)
note(s, """
Tone matters here — matter-of-fact, not self-flagellating. The audience's takeaway
should be "that could have been me", not "these people were sloppy".

Worth saying: we spent about four GPU-hours building analysis on top of the second
conclusion, including a mechanism story with a nice figure. All of it was internally
consistent. Statistics fine, controls behaved, effect sizes plausible.

This is the pivot into the methods contribution, which is the part most of the room
can use.
""")

# -------------------------------------------------------------- 8 the audit
s = slide("What the audit found", "we dumped the generations and read them")
bullets(s, [
    ("All 24 Granite and all 51 Qwen 'lost refusals' were still refusals", 0, ACC, True),
    ("Wording:  'I cannot' → 'I'm unable to'   (\"unable\" not in the marker list)", 0, INK, False),
    ("Register: 'I understand you're asking about X. Instead, …'", 0, INK, False),
    ("Language: Qwen sometimes refuses in Chinese", 0, INK, False),
    ("", 0, INK, False),
    ("Quantization perturbs the SURFACE FORM of refusals,", 0, BAD, True),
    ("not refusal behaviour.", 0, BAD, True)], top=2.1, size=18)
note(s, """
The Chinese example usually gets a reaction. Worth pausing on: no amount of extending
an English keyword list fixes that failure mode.

Say the cost explicitly: the audit took 20 minutes and no GPU, and it overturned
four GPU-hours of measurement.

If asked how we noticed: we didn't suspect the classifier. We ran the audit because
a reviewer would ask whether the "compliance" was real compliance or degraded text.
The answer was neither.
""")

# --------------------------------------------------------- 9 classifier error
s = slide("The error is large, and it changes with configuration")
pic(s, "fig_classifier", 1.6, 1.85, 10.1)
note(s, """
The key property is NOT that the classifier under-counts. A fixed bias would be
correctable.

It's that the error moves with the configuration. For Qwen the keyword estimate FALLS
with quantization (96.5 → 58.5), manufacturing a fake safety loss. For Granite-3.3 it
RISES (33.5 → 54.0), which would manufacture a fake safety GAIN — and did, for Mistral,
where we reported a spurious +29 points.

So it fabricates effects of either sign. That's the sentence to land.

Worst case: 66 points of error, Granite-3.3.
""")

# ------------------------------------------------------------------ 10 judge
s = slide("Validating the instrument")
pic(s, "fig_judge", 4.4, 1.9, 4.6)
bullets(s, [("75 generations annotated by hand, drawn from the disagreement cases", 0, MUT, False),
            ("Judge: gemma-4-12b-it, three-way rubric, language-agnostic", 0, INK, False),
            ("Generation and judging are SEPARATE stages — text is persisted", 0, ACC, True),
            ("→ re-scoring 9,600 generations under a new rubric costs zero GPU", 1, GOOD, True)],
        top=5.4, size=15)
note(s, """
The structural point is more important than the accuracy number.

Our original pipeline computed a classifier verdict inline and THREW THE TEXT AWAY.
That's why re-scoring meant re-running every model, and that's why nobody looked at
outputs for four hours. Persisting generations is the fix, and it's free.

Caveat to state before someone asks: one judge, one annotator (me). A second judge and
multi-annotator agreement rates are in future work. This is the weakest link in the
current results and I'd rather name it than defend it.
""")

# --------------------------------------------------------- 11 what a fake looks
s = slide("What a convincing artifact looks like", "the result we believed for six hours")
bullets(s, [
    "Coherent magnitudes: 41% / 37% / 25% / 0% / 0% across models",
    "Sensible format dependence: NF4 3–12%, INT4 0–41% — 'the codebook matters'",
    "A working noise floor: FP16 showed 0–4%, exactly as a control should",
    ("A capability dissociation: Granite NF4 at 99.87% MMLU losing 12% of refusals", 0, ACC, True),
    "A direction decomposition: asymmetric flips for some models, symmetric for others",
    "And a mechanism: layer sweep showed 14× spread, concentrated vs distributed circuits",
    ("Every number downstream of wrong labels. Statistics were fine.", 0, BAD, True)], top=2.0, size=16)
note(s, """
This slide is the reason the talk is worth giving.

Walk it slowly. Each line is something that made the wrong result MORE believable.
The capability dissociation in particular was the cell we would have led the paper with.

Then the punchline: our automated degeneracy checks did not catch it, because the
mislabelled responses weren't degenerate — median 74 words, 0.00 trigram repetition,
0 of 24 triggering any heuristic. They were fluent, on-topic, well-formed refusals
that used different words.

The only check that worked was reading them.
""")

# ------------------------------------------------- 11b superseded: layers
s = slide("The mechanism we believed for six hours",
          "SUPERSEDED — every value below is computed from mislabelled refusals")
pic(s, "fig_layer", 1.5, 1.75, 10.3)
bullets(s, [("Refusal damage is localized. Capability damage is not.", 0, BAD, True)],
        top=5.75, size=17)
note(s, """
Show this one, then say the caption out loud: refusal damage localized, capability
damage flat.

Granite: 14x spread across depth, three sharp peaks, and MMLU essentially unmoved
even at the most destructive windows — quantizing 4 layers of 40 appeared to cost
14% of refusals at zero capability cost. Qwen: no window above 3%, yet 41% whole-model.
We read that as concentrated vs distributed refusal circuits and thought it was the
strongest contribution of the work.

It is entirely an artifact. The window-level refusal counts come from the same broken
classifier. Nothing is wrong with the experiment design — you could run exactly this
probe tomorrow with a good judge and it might well reproduce. We just cannot claim it.

Good place to say: this is why the paper includes a superseded-results section.
""")

# ------------------------------------------------ 11c superseded: margins
s = slide("The hypothesis that failed",
          "SUPERSEDED inputs — but the null is informative anyway")
pic(s, "fig_margin", 2.0, 1.7, 9.3)
bullets(s, [("First-token confidence does not predict which refusals die.", 0, BAD, True)],
        top=5.9, size=17)
note(s, """
The scatter is the point: each dot is one prompt, diamonds are medians. Grey = refusal
kept, blue = refusal "lost".

Three of four conditions overlap almost completely. Granite/NF4 actually leans the
wrong way (lost refusals had WIDER margins). Only OLMo-2 separates, and that is six
points — significant on a permutation test, but it does not generalise.

The genuinely useful part survives the retraction: refusal margins are 3.5 to 8.6
logits, while MMLU answer margins are around 1.0. Quantization perturbs logits by a
fraction of one. So noise cannot flip the first token — which means the refuse/comply
divergence happens LATER in the trajectory, not at a decision boundary. That is what
sent us toward a divergence-position probe, which is in future work.

I pre-registered a follow-up prediction here (that gentler NF4 would restore margin
selectivity) and it failed outright at 0.35. Worth admitting on stage — it lands well.
""")

# --------------------------------------------------------------- 12 mechanism
s = slide("Mechanism: what we can rule out")
pic(s, "fig_rank1", 7.4, 2.0, 5.2)
bullets(s, [
    ("Rank-1 ablation is bounded:  ‖ΔW‖/‖W‖ ≈ 1/√d ≈ 1.7%", 0, ACC, True),
    ("Measured: 1.77% at w=1, 2.1% for Heretic", 1, MUT, False),
    ("Independent of ablation strength — you cannot make it big", 1, MUT, False),
    ("→ weight-damage explanations are unavailable", 0, INK, True),
    ("Kurtosis moves 1.494 → 1.490. Error ratio 1.000.", 1, MUT, False),
    ("", 0, INK, False),
    ("Also: ρ against an isotropic null is confounded", 0, BAD, True),
    ("scale-matched surrogate 5.34 vs true direction 1.94", 1, MUT, False)], top=2.0, size=15, width=6.8)
note(s, """
The bound is the cleanest theoretical thing in the paper and it generalises beyond
our checkpoints.

Constructive corollary worth mentioning: defenses that spread refusal across multiple
directions (refusal aliases, multi-direction ablation work) break the assumption the
bound rests on. That may be part of why they work — k directions give ~√(k/d).

The ρ confound is a separate warning: any directional analysis of quantization error
needs a scale-matched null, because per-channel error scales with channel range and
the refusal direction correlates with it. A meaningless direction scores 3× the real one.
""")

# ---------------------------------------------------------------- 13 timeline
s = slide("15 GPU-hours, 4 discarded")
pic(s, "fig_timeline", 2.1, 1.9, 9.2)
note(s, """
Optional slide — drop it if running long.

If you keep it: the black bar is the audit. Everything red to its left was built on
faulty labels. The gap between the first inverted conclusion and the audit is about
six hours.

Also useful for the practical audience: Heretic took 14 minutes, not the 90 we
budgeted. And capping the GPU at 400 W dropped it from 80-83 °C to 68 °C for about
15% throughput — worth it for long unattended runs.
""")

# ------------------------------------------------------------ 14 limitations
s = slide("Limitations", "stated plainly, because several are load-bearing")
bullets(s, [
    ("This is NOT a safety mitigation — at INT4 the model still complies with 84.5%", 0, BAD, True),
    "One abliterated family, one abliteration tool (Heretic)",
    "One judge model, one annotator for the 75-item validation set",
    "Simulated quantization; GPTQ/AWQ untested (bnb NF4 validated)",
    "Greedy decoding, English, single-turn, no jailbreak framing",
    "n=200 per cell — INT4 is p=0.002, but NF4 alone is not significant",
    ("Llama-3.1-8B not yet measured", 0, MUT, False)], top=2.1, size=17)
note(s, """
Lead with the "not a mitigation" line. If one sentence survives the talk, it should
not be someone leaving thinking quantization fixes abliterated models.

The GPTQ/AWQ gap is the one a practitioner will press on, and rightly — those are the
deployed quantizers. Say the script is written and it's the next run.
""")

# ----------------------------------------------------------------- 15 takeaway
s = slide("Takeaways")
bullets(s, [
    ("Quantization partially reverses abliteration — threshold at 4 bits, codebook-dependent", 0, ACC, True),
    ("Intact safety training is unaffected: 11 models, no increase in compliance", 0, INK, True),
    ("Keyword refusal detection is unfit for this question — up to 66 points of error, either sign", 0, BAD, True),
    "",
    ("Persist generations, not verdicts", 0, GOOD, True),
    ("Validate your instrument on the distribution it will actually see", 0, GOOD, True),
    ("Read the outputs before believing the aggregate", 0, GOOD, True)], top=2.1, size=18)
note(s, """
Close on the three green lines — they're what the audience can act on tomorrow
regardless of whether they care about abliteration.

If there's time, the one-sentence version: "The effect we set out to find was real,
our mechanism for it was impossible, and our instrument couldn't see either — and the
instrument is the part most likely to affect your work too."

Expected questions:
- "Does this replicate on other abliterated models?" — Unknown. One family. Honest gap.
- "What about GPTQ/AWQ?" — Next run, script ready.
- "Is 5→14.5% practically meaningful?" — No, and we say so. It's evidence ablation is
  reversible, not that quantization is a defense.
- "Why gemma-4 as judge?" — Strongest local model that fits. Validated at 98.7%. A
  second judge is the obvious next check.
- "Could the judge be biased toward calling things refusals?" — Possible; it scores
  3/3 constructed compliances correctly, but that's a small control. Named as a limitation.
""")

prs.save(str(OUT))
print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
