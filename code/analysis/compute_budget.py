#!/usr/bin/env python3
"""Compute budget estimator for the quantization-x-abliteration experiment.

Every number below is an ASSUMPTION you can edit. Throughputs are for a single
A100 40GB running a 7-9B model in BF16. Colab compute-unit (CU) burn rates are
approximate and Google changes them -- check the rate shown in the Colab UI
before committing to a long run.

    python3 analysis/compute_budget.py            # all scenarios
    python3 analysis/compute_budget.py --plan core
"""
import argparse

# ---------------------------------------------------------------- assumptions
CU_PER_HR = {"A100": 13.0, "L4": 4.8, "T4": 2.0}   # Colab; approximate, verify in UI

# Jetstream2 g3.xl = full A100 40GB, 32 vCPU, 120 GB RAM. SUs burn on WALL CLOCK
# while the instance is running -- idle time, downloads and debugging all count.
# Jetstream2 full-GPU flavors. `speed` is relative throughput for LLM
# INFERENCE, which is memory-bandwidth bound, not FLOPS bound -- so it tracks
# HBM bandwidth far more closely than tensor-core peak.
#   A100 40GB ~1.56 TB/s | L40S ~0.86 TB/s | H100 80GB ~3.35 TB/s
JS2_FLAVORS = {
    "g3.xl": dict(gpu="A100 40GB",  vram=40, ram=120, vcpu=32, su=64,  speed=1.00,
                  native_fp8=False),
    "g4.xl": dict(gpu="L40S 48GB",  vram=48, ram=120, vcpu=12, su=84,  speed=0.62,
                  native_fp8=True),
    "g5.xl": dict(gpu="H100 80GB",  vram=80, ram=240, vcpu=20, su=128, speed=2.15,
                  native_fp8=True),
}
JS2_SU_PER_HR = 64.0
JS2_OVERHEAD = 1.35     # downloads, env setup, restarts, babysitting
CU_COST   = 9.99 / 100          # pay-as-you-go $ per compute unit
PRO_CU, PROPLUS_CU = 100, 500   # monthly allocations

# Per-cell task costs in GPU-minutes (one cell = one checkpoint x one config)
TASK_MIN = {
    # lm-eval-harness, HF backend, batched
    "mmlu_full":     35.0,   # 14,042 questions x 4 choices, 5-shot
    "mmlu_2k":        5.0,   # stratified 2,000-question subsample
    "hellaswag":     20.0,   # 10,042 x 4, 10-shot
    "gsm8k_hf":      40.0,   # 1,319 generative, 5-shot, HF generate
    "gsm8k_vllm":     9.0,   # same via vLLM
    "wikitext_ppl":   3.0,
    "behavior_hf":    8.0,   # ~700 harmful prompts x 256 tok, HF generate
    "behavior_vllm":  2.0,   # same via vLLM
    "kld":            6.0,   # 65,536 tokens, ref + quantized forward passes
    "quantize":       1.5,   # simulate-quantize a full 8B checkpoint
}
# Per-checkpoint one-time costs in GPU-minutes
ONCE_MIN = {
    "download":      10.0,   # ~16 GB from HF
    "abliterate":    60.0,   # Heretic Bayesian search
    "weights_h1h2":  22.0,   # 64 matrices x 6 configs, NF4 dominates
    "rho_h3":        12.0,   # 1000-direction null per layer
}

N_MODELS, N_VARIANTS, N_CONFIGS = 3, 2, 6
SWEEP_CELLS = 10 * 2 * N_MODELS * N_VARIANTS   # 10 widths x 2 formats
ARM_CELLS   = 3 * N_MODELS * N_CONFIGS         # 3 extra ablation/quant arms


def scenario(name, *, models, bench, gen, sweep, arms, weights_on_t4):
    ckpts = models * N_VARIANTS
    cells = ckpts * N_CONFIGS
    a100 = 0.0
    t4 = 0.0

    # one-time, per checkpoint
    a100 += ckpts * ONCE_MIN["download"] / 60
    a100 += models * ONCE_MIN["abliterate"] / 60
    # H1/H2/H3 need no forward passes, so they can run on a cheap GPU
    cheap = ckpts * (ONCE_MIN["weights_h1h2"] + ONCE_MIN["rho_h3"]) / 60
    if weights_on_t4:
        t4 += cheap
    else:
        a100 += cheap

    # per primary cell
    per = TASK_MIN["quantize"] + TASK_MIN["kld"] + TASK_MIN[gen]
    per += sum(TASK_MIN[b] for b in bench)
    a100 += cells * per / 60

    # sweep: cheap capability metric only
    if sweep:
        a100 += SWEEP_CELLS * (TASK_MIN["quantize"] + TASK_MIN["mmlu_2k"]) / 60
    # intervention arms: capability + behavior, no full benchmark suite
    if arms:
        a100 += ARM_CELLS * (TASK_MIN["quantize"] + TASK_MIN["mmlu_2k"]
                             + TASK_MIN[gen]) / 60

    cu = a100 * CU_PER_HR["A100"] + t4 * CU_PER_HR["T4"]
    # On Jetstream2 there is no cheap-GPU tier to offload to: the T4 work runs
    # on the same instance, so it bills at the g3.xl rate like everything else.
    js2_su = (a100 + t4) * JS2_SU_PER_HR * JS2_OVERHEAD
    return dict(name=name, cells=cells, a100=a100, t4=t4, cu=cu,
                usd=cu * CU_COST, proplus_months=cu / PROPLUS_CU,
                js2_hours=(a100 + t4) * JS2_OVERHEAD, js2_su=js2_su)


# `mechanism` runs only the stages that are IMPLEMENTED today: weights (H1+H2)
# and rho (H3). No generation, no lm-eval, no forward passes at all -- so none
# of the benchmark cost applies. This is the mechanistic core of the paper.
def mechanism_only(models=3):
    ckpts = models * N_VARIANTS
    gpu_h = ckpts * (ONCE_MIN["download"] + ONCE_MIN["weights_h1h2"]
                     + ONCE_MIN["rho_h3"]) / 60
    wall = gpu_h * JS2_OVERHEAD
    return dict(name=f"mechanism-{models}m", cells=ckpts * N_CONFIGS,
                a100=gpu_h, t4=0.0, cu=gpu_h * CU_PER_HR["A100"],
                usd=gpu_h * CU_PER_HR["A100"] * CU_COST,
                proplus_months=gpu_h * CU_PER_HR["A100"] / PROPLUS_CU,
                js2_hours=wall, js2_su=wall * JS2_SU_PER_HR)

PLANS = {
 "pilot":  dict(models=1, bench=["mmlu_2k"], gen="behavior_vllm",
                sweep=False, arms=False, weights_on_t4=True),
 "core":   dict(models=3, bench=["mmlu_2k", "wikitext_ppl"], gen="behavior_vllm",
                sweep=False, arms=False, weights_on_t4=True),
 "full":   dict(models=3, bench=["mmlu_2k", "gsm8k_vllm", "wikitext_ppl"],
                gen="behavior_vllm", sweep=True, arms=True, weights_on_t4=True),
 "as-spec":dict(models=3, bench=["mmlu_full", "hellaswag", "gsm8k_hf",
                                 "wikitext_ppl"], gen="behavior_hf",
                sweep=True, arms=True, weights_on_t4=False),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=list(PLANS) + ["all"], default="all")
    a = ap.parse_args()
    names = list(PLANS) if a.plan == "all" else [a.plan]
    print(f"{'plan':12s} {'cells':>6s} {'GPU-h':>7s} | {'Colab CU':>9s} {'$':>6s} "
          f"| {'JS2 wall-h':>11s} {'JS2 SU':>8s}")
    print("-" * 78)
    for r in (mechanism_only(1), mechanism_only(3)):
        print(f"{r['name']:12s} {r['cells']:6d} {r['a100']:7.1f} | "
              f"{r['cu']:9.0f} {r['usd']:6.0f} | "
              f"{r['js2_hours']:11.1f} {r['js2_su']:8.0f}   <- RUNNABLE TODAY")
    for n in names:
        r = scenario(n, **PLANS[n])
        print(f"{r['name']:12s} {r['cells']:6d} {r['a100']+r['t4']:7.1f} | "
              f"{r['cu']:9.0f} {r['usd']:6.0f} | "
              f"{r['js2_hours']:11.1f} {r['js2_su']:8.0f}")
    print("\nColab CU rates and JS2 SU rates both change -- verify before a long run.")
    print("JS2 SUs bill on wall clock while the instance runs. Shelve it when idle.")

    print("\n=== Jetstream2 flavor comparison, 'full' plan (43.2 GPU-h on A100) ===")
    base_h = scenario("full", **PLANS["full"])
    gpu_h = base_h["a100"] + base_h["t4"]
    print(f"{'flavor':8s} {'GPU':11s} {'VRAM':>5s} {'vCPU':>5s} {'SU/h':>5s} "
          f"{'wall-h':>7s} {'total SU':>9s} {'FP8':>5s}")
    print("-" * 66)
    for name, f in JS2_FLAVORS.items():
        h = gpu_h / f["speed"] * JS2_OVERHEAD
        print(f"{name:8s} {f['gpu']:11s} {f['vram']:4d}G {f['vcpu']:5d} "
              f"{f['su']:5d} {h:7.1f} {h*f['su']:9.0f} "
              f"{'yes' if f['native_fp8'] else 'no':>5s}")
    print("\nDISK: every full-GPU flavor has a 60 GB root disk and NO ephemeral")
    print("disk. OS+CUDA+torch+vllm is ~20 GB, so ~40 GB is left = TWO 8B")
    print("checkpoints at a time. The study needs six. Attach a volume.")
