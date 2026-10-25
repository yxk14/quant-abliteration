#!/usr/bin/env python3
"""Where does the time actually go? Costed from tokens and throughput, not
hand-waved minutes, so every assumption is visible and challengeable."""

# ---- machine assumptions (RTX 5090, 8B model, BF16) --------------------
PREFILL_HF   = 8_000     # tok/s, HF backend, batched
PREFILL_VLLM = 18_000    # tok/s, vLLM continuous batching
GEN_HF       = 300       # tok/s, HF .generate(), batch 16
GEN_VLLM     = 4_000     # tok/s, vLLM batched
NVME         = 2.0       # GB/s

# ---- MMLU 5-shot, the thing that dominates everything -----------------
Q, CHOICES, SUBJECTS = 14_042, 4, 57
SHOT_PREFIX, QUESTION, CONTINUATION = 1_000, 200, 15

naive   = Q * CHOICES * (SHOT_PREFIX + QUESTION + CONTINUATION)
shared  = Q * (SHOT_PREFIX + QUESTION + CHOICES * CONTINUATION)   # share prompt across choices
cached  = SUBJECTS * SHOT_PREFIX + Q * (QUESTION + CHOICES * CONTINUATION)  # + prefix cache

print("=== MMLU 5-shot: tokens to prefill, per cell ===")
for name, tok, rate in [("naive HF (no sharing)", naive, PREFILL_HF),
                        ("HF, prompt shared across 4 choices", shared, PREFILL_HF),
                        ("vLLM + prefix caching", cached, PREFILL_VLLM)]:
    print(f"  {name:36s} {tok/1e6:7.1f} M tok  ->{tok/rate/60:7.1f} min")

print("\n=== other per-cell work ===")
gsm_tok = 1_319 * 256
print(f"  GSM8K generation, HF     {gsm_tok/1e6:5.2f} M tok ->{gsm_tok/GEN_HF/60:7.1f} min")
print(f"  GSM8K generation, vLLM   {gsm_tok/1e6:5.2f} M tok ->{gsm_tok/GEN_VLLM/60:7.1f} min")
sp = 16.1
print(f"  save_pretrained+reload   {2*sp:5.1f} GB     ->{2*sp/NVME/60:7.1f} min  (PURE WASTE)")

print("\n=== per-checkpoint stages (no benchmarks) ===")
kld_tok = 65_536 * 7
beh_tok = 700 * 256 * 7
print(f"  weights H1/H2   quantize 2.4B params x6 cfg, kernel-launch bound  ~ 4 min")
print(f"  rho     H3      6.4 TFLOP of nulls, Python bound                  ~ 5 min")
print(f"  kld     H5      {kld_tok/1e3:.0f}K tok prefill + {6*34:.0f} GB logit I/O   "
      f"~{kld_tok/PREFILL_VLLM/60 + 6*34/NVME/60:4.0f} min")
print(f"  behavior H4     {beh_tok/1e6:.2f}M tok gen, vLLM                      "
      f"~{beh_tok/GEN_VLLM/60:4.0f} min")

print("\n=== the whole study, 36 primary cells + 6 checkpoints ===")
per_ck = 4 + 5 + (kld_tok/PREFILL_VLLM/60 + 6*34/NVME/60) + beh_tok/GEN_VLLM/60
for label, mmlu, gsm, io in [
        ("as I estimated (HF + save/load)", shared/PREFILL_HF/60, gsm_tok/GEN_HF/60, 2*sp/NVME/60),
        ("naive HF, no sharing at all",     naive/PREFILL_HF/60,  gsm_tok/GEN_HF/60, 2*sp/NVME/60),
        ("vLLM + prefix cache + in-memory", cached/PREFILL_VLLM/60, gsm_tok/GEN_VLLM/60, 0.0)]:
    cell = mmlu + gsm + 2 + io
    tot = 36*cell/60 + 6*per_ck/60 + 3*60/60
    print(f"  {label:34s} cell={cell:5.1f} min  total={tot:6.1f} h")
