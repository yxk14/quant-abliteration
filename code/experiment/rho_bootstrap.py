#!/usr/bin/env python3
"""Bootstrap rho over resamples of the contrast prompt sets (HANDOFF.md section 5).

Quantization is deterministic, so the pipeline has no run-to-run variance -- but
r_hat estimation does, because it depends on which prompts were sampled. Without
this, rho is a point estimate presented as a result.

Only the DIFFERENCE-IN-MEANS direction has prompt-sampling variance. A direction
recovered from the weight difference (check_direction_agreement.py --derive) is a
deterministic function of the weights and has none, so bootstrapping it would
report a spurious zero-width interval. This script therefore always estimates
r_hat from prompts.

Efficiency: activations are computed ONCE per prompt and cached, so each resample
is a mean over cached tensors; and the quantization error E is computed once per
(config, layer) rather than once per replicate.

    python experiment/rho_bootstrap.py --base Qwen/Qwen2.5-7B-Instruct \
        --abliterated ckpts/qwen25-7b-abl --family Qwen2.5-7B-Instruct \
        --short-label Q7B --n 256 --n-boot 20
"""
import argparse, csv, itertools, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from quantizers import get_quantizer
from compute_refusal_dirs import prompt_sets

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"
TARGET = "o_proj.weight"


@torch.no_grad()
def per_prompt_states(model, tok, prompts, device, bs=8):
    """Last-token hidden state per prompt, per layer -> [n_prompts, n_layers, d]."""
    out = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False, add_generation_prompt=True)
                for p in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        # padding is left, so the final position is the last real token
        st = torch.stack([h[:, -1, :].float() for h in hs], dim=1)   # [B, L+1, D]
        out.append(st[:, 1:, :].cpu())                               # drop embedding
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--abliterated", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--short-label", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--formats", nargs="+", default=["FP", "INT"])
    ap.add_argument("--widths", nargs="+", type=int, default=[16, 8, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bad, ok = prompt_sets(a.n)
    print(f"{len(bad)} harmful / {len(ok)} harmless")

    # r_hat is computed on the BASE model and reused for both variants, so the
    # comparison is like-for-like (EXPERIMENT_PLAN.md).
    base = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, device_map=a.device).eval()
    print("caching activations...")
    S_bad = per_prompt_states(base, tok, bad, a.device)
    S_ok = per_prompt_states(base, tok, ok, a.device)
    print("  cached", tuple(S_bad.shape))

    rng = np.random.default_rng(a.seed)
    nb, no = S_bad.shape[0], S_ok.shape[0]
    # [n_boot, n_layers, d] -- one resampled r_hat per replicate
    reps = []
    for _ in range(a.n_boot):
        ib = rng.integers(0, nb, nb)
        io = rng.integers(0, no, no)
        reps.append(S_bad[ib].mean(0) - S_ok[io].mean(0))
    R = torch.stack(reps).to(a.device)
    R = R / R.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    print(f"  {a.n_boot} bootstrap directions {tuple(R.shape)}")
    del base
    torch.cuda.empty_cache()

    DATA.mkdir(parents=True, exist_ok=True)
    fh = open(DATA / "rho_bootstrap.csv", "w", newline="")
    w = csv.writer(fh)
    w.writerow(["model", "short", "variant", "fmt", "width", "config",
                "rho_mean", "rho_ci_lo", "rho_ci_hi", "n_boot"])

    for model_id, variant in [(a.base, "base"), (a.abliterated, "abliterated")]:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=a.device).eval()
        targets = [(int([s for s in n.split(".") if s.isdigit()][0]), p.data)
                   for n, p in m.named_parameters() if n.endswith(TARGET)]
        for fmt, width in itertools.product(a.formats, a.widths):
            q = get_quantizer(fmt, width)
            acc = np.zeros(a.n_boot)
            nlayer = 0
            with torch.no_grad():
                for layer, W in targets:
                    Wf = W.float()
                    E = q(W).float() - Wf                    # once per (config, layer)
                    d = E.shape[0]
                    denom = (E.pow(2).sum() / d).clamp(min=1e-30)
                    r = R[:, layer, :]                       # [n_boot, d]
                    acc += ((r @ E).pow(2).sum(dim=1) / denom).cpu().numpy()
                    nlayer += 1
            vals = acc / max(nlayer, 1)
            w.writerow([a.family, a.short_label, variant, fmt, width,
                        f"{fmt}{width}", round(float(vals.mean()), 4),
                        round(float(np.percentile(vals, 2.5)), 4),
                        round(float(np.percentile(vals, 97.5)), 4), a.n_boot])
            fh.flush()
            print(f"  {variant:12s} {fmt}{width:<3d} rho={vals.mean():.4f} "
                  f"[{np.percentile(vals, 2.5):.4f}, {np.percentile(vals, 97.5):.4f}]")
        del m
        torch.cuda.empty_cache()
    fh.close()
    print("wrote rho_bootstrap.csv")


if __name__ == "__main__":
    main()
