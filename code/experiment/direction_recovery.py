#!/usr/bin/env python3
"""How much of the ablated direction does quantization put back?

Motivation: the behavioural readout (refusal / no refusal on 200 prompts) is a
coarse, binary instrument. It shows nothing at INT8 even if a real effect is
present, because a 2-point change on n=200 is not resolvable. If the hypothesis
is "quantization noise reintroduces the ablated component", that quantity can be
measured *directly in the weights*, continuously, with no generation at all.

Definition. Abliteration sets W' = W - w * r r^T W, so the component of the
output along r is scaled by (1-w):

    recovery(W) = || r^T W || / || r^T W_base ||

    W_base       -> 1.00  (nothing removed)
    W_abliterated-> ~0.45 (Heretic removes ~55%)
    if quantizing W_abliterated pushes this back UP, the direction is returning.

Two controls, both essential:

 1. SURROGATE direction, scale-matched: same |r| magnitude profile reassigned to
    channels by dynamic range, carrying no refusal information.

    IMPORTANT: do NOT compare RELATIVE recovery between r and the surrogate.
    They start from different baselines (0.43 vs 1.00), and a direction starting
    lower rises more for the same added noise. That comparison manufactured a
    false "preferential targeting" result in an earlier draft of this work.
    The baseline-free quantity is the absorbed noise N, where
    N^2 = R_cfg^2 - R_bf16^2. Measured, N_r/N_surrogate = 0.72-0.79 for integer
    formats: quantization error is if anything LESS aligned with the ablated
    direction than with the control. The recovery is geometry, not targeting.

 2. BASE model. Quantizing the unablated weights and measuring along r gives the
    ceiling behaviour when there is nothing to restore.

    python experiment/direction_recovery.py
"""
import argparse, csv, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from quantizers import get_quantizer

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"
TARGET = "o_proj.weight"


def layers(model):
    for n, p in model.named_parameters():
        if n.endswith(TARGET) and p.ndim == 2:
            yield int([s for s in n.split(".") if s.isdigit()][0]), p.data


def surrogate(r, W, gen):
    """Same |r| values, reassigned so magnitude tracks per-channel dynamic range.
    Preserves the scale correlation that confounds directional metrics, carries
    no refusal information."""
    amax = W.abs().amax(dim=1)
    mag = r.abs().sort().values
    out = torch.zeros_like(r)
    out[amax.argsort()] = mag
    return out * torch.sign(torch.randn(r.shape, generator=gen, device=r.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--abl", default="ckpts/qwen25-7b-abl")
    ap.add_argument("--rdir", default="refusal_dirs/Qwen_Qwen2.5-7B-Instruct_derived.pt")
    ap.add_argument("--configs", nargs="+",
                    default=["BF16", "INT16", "INT8", "INT6", "INT5", "NF4", "INT4", "INT3"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="direction_recovery.csv")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    R = torch.load(a.rdir, map_location=a.device)

    # ---- pass 1: base model. record the denominators and the surrogate dirs.
    print("loading base model...")
    mb = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, device_map=a.device).eval()
    gen = torch.Generator(device=a.device).manual_seed(0)
    denom, sdenom, surr, baseq = {}, {}, {}, {}
    with torch.no_grad():
        for L, W in layers(mb):
            r = R[L].float()
            if r.norm() < 1e-8:
                continue
            r = r / r.norm()
            Wf = W.float()
            denom[L] = float((r @ Wf).norm())
            s = surrogate(r, Wf, gen); s = s / s.norm()
            surr[L] = s.clone()
            sdenom[L] = float((s @ Wf).norm())
        # base model under quantization: the "nothing to restore" control
        for cfg in a.configs:
            if cfg == "BF16":
                continue
            fmt = "INT" if cfg.startswith("INT") else "FP"
            width = int(cfg[3:]) if fmt == "INT" else 4
            q = get_quantizer(fmt if cfg != "NF4" else "FP", width)
            tot, n = 0.0, 0
            for L, W in layers(mb):
                if L not in denom:
                    continue
                r = R[L].float(); r = r / r.norm()
                tot += float((r @ q(W).float()).norm()) / denom[L]; n += 1
            baseq[cfg] = tot / n
            print(f"  base  {cfg:6s} recovery {baseq[cfg]:.4f}")
    del mb
    torch.cuda.empty_cache()

    # ---- pass 2: abliterated model.
    print("loading abliterated model...")
    ma = AutoModelForCausalLM.from_pretrained(
        a.abl, torch_dtype=torch.bfloat16, device_map=a.device).eval()
    rows = []
    with torch.no_grad():
        for cfg in a.configs:
            if cfg == "BF16":
                q = lambda W: W
            else:
                fmt = "INT" if cfg.startswith("INT") else "FP"
                width = int(cfg[3:]) if fmt == "INT" else 4
                q = get_quantizer(fmt if cfg != "NF4" else "FP", width)
            rt, st, n = 0.0, 0.0, 0
            for L, W in layers(ma):
                if L not in denom:
                    continue
                r = R[L].float(); r = r / r.norm()
                Wq = q(W).float()
                rt += float((r @ Wq).norm()) / denom[L]
                st += float((surr[L] @ Wq).norm()) / sdenom[L]
                n += 1
            rows.append((cfg, rt/n, st/n, baseq.get(cfg, 1.0)))
            print(f"  abl   {cfg:6s} along-r {rt/n:.4f}   surrogate {st/n:.4f}   "
                  f"base-ctrl {baseq.get(cfg,1.0):.4f}")

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / a.out
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "recovery_along_r", "recovery_surrogate",
                    "base_control", "noise_along_r", "noise_surrogate"])
        r0 = rows[0][1]; s0 = rows[0][2]
        import math
        for cfg, rr, ss, bb in rows:
            # N^2 = R_cfg^2 - R_bf16^2 : the absolute noise the direction absorbs.
            # Comparing RELATIVE recovery across directions with different
            # baselines is invalid -- a direction starting lower rises more for
            # the same added noise. N is the baseline-free quantity.
            nr = math.sqrt(max(rr*rr - r0*r0, 0.0))
            ns = math.sqrt(max(ss*ss - s0*s0, 0.0))
            w.writerow([cfg, round(rr, 5), round(ss, 5), round(bb, 5),
                        round(nr, 5), round(ns, 5)])
    print(f"\nwrote {out}")
    print("N = absorbed noise (baseline-free). Compare N across directions;")
    print("do NOT compare relative recovery -- baselines differ and that misleads.")


if __name__ == "__main__":
    main()
