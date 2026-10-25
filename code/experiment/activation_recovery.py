#!/usr/bin/env python3
"""The activation-space analogue of direction recovery.

Our weight metric is R = ||r^T W|| / ||r^T W_base||: how much of the ablated
direction survives in the WEIGHTS. It treats every input direction as equally
likely, because a Frobenius/operator quantity has no way not to.

That is exactly where it can mislead. GPTQ quantizes column by column and
updates the not-yet-quantized weights to cancel the error it just introduced;
AWQ rescales channels to protect salient ones. Both arrange for layer OUTPUTS to
survive on real inputs. So a calibrated quantizer can displace W further along
r_hat than round-to-nearest does and still leave the residual stream untouched
-- which is what the behaviour says happens: GPTQ deposits 1.8x the noise NF4
does on the ablated direction and moves refusal by -0.5 points against NF4's
+3.5.

This measures the quantity behaviour should actually depend on:

    A = E_x | r_hat . (W x) |  /  E_x | r_hat . (W_base x) |

i.e. the component of what o_proj WRITES to the residual stream that lies along
the ablated direction, on real harmful prompts, normalised per layer against the
unablated model and averaged over edited layers.

Prediction under the error-compensation hypothesis:
    A(GPTQ), A(AWQ) ~ A(BF16 abliterated)      -- hole stays open in function space
    A(INT4 RTN)     >  A(BF16 abliterated)      -- hole partially refills
even though the WEIGHT metric orders them GPTQ > NF4.

    python experiment/activation_recovery.py --base Qwen/Qwen2.5-7B-Instruct \
      --rdir refusal_dirs/Qwen_Qwen2.5-7B-Instruct_derived.pt \
      --ckpt BF16=ckpts/qwen25-7b-abl GPTQ4=ckpts/qwen25-7b-abl-GPTQ4 \
      --sim INT4 NF4 --sim-of ckpts/qwen25-7b-abl --out activation_qwen.csv
"""
import argparse, csv, gc, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import metrics as M
from stages_eval import load_prompts, quantize_model_, restore_

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"


# Heretic edits BOTH residual-stream write paths, and which one carries the
# ablation differs by family: on Qwen the weight sits in o_proj (1.13-1.38 vs
# down_proj 0.43-1.08), on Llama it sits in down_proj (1.21-1.39 vs o_proj
# 1.17-1.26). Watching only o_proj therefore misses most of Llama's edit.
SUFFIX = {"o_proj": "self_attn.o_proj", "down_proj": "mlp.down_proj"}


def write_modules(model, target):
    want = SUFFIX[target]
    for n, mod in model.named_modules():
        if n.endswith(want):
            yield int([s for s in n.split(".") if s.isdigit()][0]), mod


@torch.no_grad()
def measure(model, tok, prompts, R, device, batch=8, target="o_proj"):
    """mean |r_hat . o_proj_output| per layer, at the last prompt token and
    averaged over all prompt tokens."""
    keep = {L for L in R_layers(R)}
    acc_last, acc_mean, n_seen = {}, {}, 0
    store = {}

    def mk(L):
        def hook(_m, _i, out):
            store[L] = out.detach()
        return hook

    handles = [mod.register_forward_hook(mk(L))
               for L, mod in write_modules(model, target) if L in keep]
    try:
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            msgs = [M.chat_prompt(tok, p) for p in chunk]
            enc = tok(msgs, return_tensors="pt", padding=True).to(device)
            store.clear()
            model(**enc)
            mask = enc["attention_mask"]
            # left padding: the last real token is the final column
            for L, out in store.items():
                r = R[L].float().to(out.device)
                r = r / r.norm().clamp(min=1e-8)
                proj = (out.float() @ r).abs()            # [B, T]
                acc_last[L] = acc_last.get(L, 0.0) + float(proj[:, -1].sum())
                m = mask.float()
                acc_mean[L] = acc_mean.get(L, 0.0) + float((proj * m).sum() / m.sum() * len(chunk))
            n_seen += len(chunk)
    finally:
        for h in handles:
            h.remove()
    return ({L: v / n_seen for L, v in acc_last.items()},
            {L: v / n_seen for L, v in acc_mean.items()})


def R_layers(R):
    return [L for L in range(R.shape[0]) if R[L].norm() > 1e-8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rdir", required=True)
    ap.add_argument("--ckpt", nargs="*", default=[], help="LABEL=path (BF16 first)")
    ap.add_argument("--sim", nargs="*", default=[], help="simulated configs, e.g. INT4 NF4")
    ap.add_argument("--sim-of", default=None, help="checkpoint the simulated configs apply to")
    ap.add_argument("--n-prompts", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", default="o_proj", choices=["o_proj", "down_proj"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.quantization_config import CompressedTensorsConfig
    DEQ = CompressedTensorsConfig(run_compressed=False)

    R = torch.load(a.rdir, map_location=a.device)
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    prompts = load_prompts(a.n_prompts)[:a.n_prompts]
    print(f"{len(prompts)} prompts, {len(R_layers(R))} edited layers, target={a.target}")

    def load(path, quantized=False):
        kw = dict(torch_dtype=torch.bfloat16, device_map=a.device)
        if quantized:
            kw["quantization_config"] = DEQ
        return AutoModelForCausalLM.from_pretrained(path, **kw).eval()

    # ---- denominator: the unablated model
    print(f"base: {a.base}")
    mb = load(a.base)
    d_last, d_mean = measure(mb, tok, prompts, R, a.device, a.batch, a.target)
    del mb; gc.collect(); torch.cuda.empty_cache()

    rows = []
    def record(lab, last, mean):
        rl = sum(last[L] / d_last[L] for L in last) / len(last)
        rm = sum(mean[L] / d_mean[L] for L in mean) / len(mean)
        rows.append((lab, rl, rm))
        print(f"  {lab:12s} A_last = {rl:.4f}   A_mean = {rm:.4f}")

    for spec in a.ckpt:
        lab, path = spec.split("=", 1)
        print(f"{lab}: {path}")
        m = load(path)
        l_, m_ = measure(m, tok, prompts, R, a.device, a.batch, a.target)
        record(lab, l_, m_)
        del m; gc.collect(); torch.cuda.empty_cache()

    if a.sim:
        print(f"simulated configs on {a.sim_of}")
        m = load(a.sim_of)
        for cfg in a.sim:
            fmt = "INT" if cfg.startswith("INT") else "FP"
            width = int(cfg[3:]) if fmt == "INT" else int(cfg[2:])
            saved = quantize_model_(m, fmt, width)
            l_, m_ = measure(m, tok, prompts, R, a.device, a.batch, a.target)
            record(cfg, l_, m_)
            restore_(m, saved); del saved; torch.cuda.empty_cache()
        del m; gc.collect(); torch.cuda.empty_cache()

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / a.out
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "A_last_token", "A_mean_tokens"])
        for lab, rl, rm in rows:
            w.writerow([lab, round(rl, 5), round(rm, 5)])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
