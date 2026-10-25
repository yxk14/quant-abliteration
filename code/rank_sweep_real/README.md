# Real-direction rank sweep — ARTMAN '26 camera-ready

Replaces the **synthetic** Table 8 of the submission with the same measurement
made on real model weights and real activation directions.

## Start here (you don't need to remember anything)

```bash
cd rank-sweep-real
./run.sh dry                                   # CPU, ~1 min, no model needed
./run.sh real hf                               # the 5090 run, ~20 min
```

**The GPU is shared with another Claude session.** `run.sh` takes
`/tmp/gpu.lock` (via `flock`) around each GPU stage and waits its turn if the
other session holds it; `./run.sh dry` never touches the GPU. To launch the
real run so it survives a closed terminal:

```bash
nohup ./run.sh real hf > logs/run.out 2>&1 &
tail -f logs/run.out
```

`hf` tells the prep step to read AdvBench and HarmBench (standard) from your
HuggingFace cache via the `datasets` library — the same two sources the
submission's prompt set came from. A local `.txt`/`.csv`/`.jsonl` path works
in its place.

`./run.sh dry` runs the self-test and builds a fake table and figure so you can
see the output format before committing GPU time. The files it writes are
tagged `MOCK` — they are synthetic and must not go in the paper.

Override anything with env vars:

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct TAG=llama8b ./run.sh real hf
LAYERS=8,12,16,20 KS=1,4,16,64,256 ./run.sh real hf
```

## Why this experiment and not the one we discussed

The original plan was a **behavioral** rank sweep: build norm-matched rank-*k*
ablations, generate, judge, and report how much de-censoring survives INT4 as
*k* grows. That design is not well-posed, and the self-test shows why.

Norm-matching forces the per-direction coefficient down as rank rises:

| k | α_k | fraction of the refusal direction still removed |
|---|-----|-----|
| 1 | 1.000 | 100% |
| 2 | 0.702 | 70% |
| 8 | 0.349 | 35% |
| 64 | 0.123 | 12% |
| 256 | 0.062 | 6% |

Direction 1 *is* the refusal direction, so by k=8 the BF16 model is barely
de-censored before quantization is applied at all. The survival ratio
(R_base − R_INT4) / (R_base − R_BF16abl) then has a denominator going to zero
exactly where low SNR predicts an effect — every point on that curve would be
an artifact of the vanishing denominator, not a measurement.

The two obvious repairs also predict nothing. Holding direction 1 at full
strength and adding extras leaves the SNR along direction 1 unchanged. Dropping
norm-matching gives energy ∝ √k against a per-direction subspace ∝ √(d/k), so
SNR ∝ √k·√(d/k) = constant. The property that makes a high-rank edit fragile
(fixed energy spread thin) is the same property that stops it suppressing
refusal. You cannot separate them with multi-direction ablation. A real diffuse
edit with its own behavioral readout means actual unlearning — that is the
follow-up paper, not this camera-ready.

So this measures the mechanism **in weight space only**, honestly labelled as
such, and §9 keeps its hedge about behavioral transfer.

## What it measures

For each edited matrix *W* (o_proj and down_proj, where out_features = d_model
so the refusal direction acts on the row space) and each rank *k*:

- **ΔW_k = −α_k D_k D_kᵀ W**, with α_k set so ‖ΔW_k‖_F is identical at every
  rank. D comes from the model, not from a Gaussian: column 0 is the
  difference-in-means refusal direction, columns 1… are the leading principal
  directions of the harmful-vs-harmless activation difference.
- **gain π** = ⟨Q(W+ΔW) − Q(W), ΔW⟩ / ⟨ΔW, ΔW⟩. π ≈ 1 means the quantizer
  transmits the edit without bias — the dithering argument of §5.2.
- **in-subspace SNR** = ‖D_kᵀ ΔW‖_F / ‖D_kᵀ ρ‖_F, ρ = (Q(W+ΔW) − Q(W)) − ΔW.
  The only component that can move behavior along the ablated directions.
- **global SNR**, which is flat in *k* and below 1 — whole-matrix noise, not the
  edit. Reporting it alongside is what shows the in-subspace ratio is the one
  carrying rank information.

Schemes: INT4 RTN per-channel, NF4 (QLoRA codebook, blockwise), FP8 E4M3,
INT8 as a control.

No generation. No judge model. Nothing to hand-label.

## Why it answers reviewer #14A

#14A asked for more evidence that rank is the primary driver. Table 8 currently
answers with randomly drawn directions, which a reviewer can fairly read as
assuming the conclusion. This replaces it with the same sweep on the actual
edited matrices, and reports the fitted exponent against the analytic −1/2
prediction. The `--no-norm-match` control arm is run automatically: it should
come out flat in *k*, which rules out "the SNR fell because the edit got
smaller."

It also fixes the §9 contradiction in the right direction. The sentence "a
prediction we can state but not test here" becomes testable in weight space and
tested; the claim that rank is what matters stops resting on synthetic edits.
What stays hedged — explicitly — is behavioral transfer.

## Layout

```
run.sh                  start here
CLAUDE.md               GPU-sharing rule for any Claude session in this dir
src/core.py             all the math; backend-agnostic
src/backend.py          numpy (CPU self-test) / torch (GPU) adapter
src/selftest.py         CPU verification of every code path  — run before the GPU
src/prep_data.py        builds data/harmful.txt and data/harmless.txt
src/extract_directions.py   stage 1 (GPU): real directions, one forward pass/prompt
src/rank_sweep.py       stage 2 (GPU): the sweep
src/mock_sweep.py       CPU dry run producing a CSV with the real schema
src/make_outputs.py     stage 3 (CPU): LaTeX table + figure + summary
out/                    CSVs, table_rank_real_<tag>.tex, fig_rank_snr_<tag>.pdf
logs/                   per-stage logs
```

`src/core.py` runs under both backends, so the numbers `selftest.py` prints on
CPU come from exactly the code that runs on the 5090.

## Send back

`out/rank_sweep_<tag>.csv`, `out/rank_sweep_<tag>_control.csv`,
`out/directions_<tag>_meta.json`, `out/summary_<tag>.txt`.
Skip the `.npz` — it is large and not needed.

## Cost

Stage 1 ≈ 2–4 min (1024 forward passes, no generation).
Stage 2 ≈ 10–20 min (weight arithmetic over ~56 matrices × 9 ranks × 4 schemes).
Stage 3 ≈ seconds. Nothing else on the 5090 needs to be rerun for the
camera-ready.
