#!/usr/bin/env python3
"""Does DECISION MARGIN explain which refusals quantization destroys?

The observational result -- 4-bit quantization removes refusals while MMLU holds
-- has no mechanism. This tests the sharpest available hypothesis:

    Refusal is decided by a thinner logit margin than knowledge is, so the same
    weight perturbation crosses the refuse/comply boundary far more often than it
    crosses an MMLU answer boundary.

Operationalised as the top1-top2 logit margin at the FIRST GENERATED TOKEN. That
is deliberately the same quantity already measured for MMLU (median ~1.0 logit,
against ~0.375 of bf16 numerical noise), so the two are directly comparable with
no hand-picked "refusal token" list -- which would be model-specific and
arguable.

Falsifiable prediction: among prompts the BF16 model refuses, the ones that flip
to compliance under quantization are those with the SMALLEST BF16 margin. If the
flipped and unflipped margin distributions coincide (AUC ~ 0.5), the hypothesis
is dead and should be reported as such.

    python experiment/margin_probe.py --model ibm-granite/granite-3.1-8b-instruct \
        --family Granite8B --config INT4 --n-prompts 200
"""
import argparse, csv, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import metrics as M
from quantizers import get_quantizer
from stages_eval import load_prompts, quantize_model_, restore_

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"


@torch.no_grad()
def first_token_margins(m, tok, prompts, batch, device):
    """top1 - top2 logit gap at the first generated position, per prompt."""
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        msgs = [M.chat_prompt(tok, p)
                for p in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True).to(device)
        lg = m(**enc).logits[:, -1, :].float()      # left padding -> -1 is real
        top2 = lg.topk(2, dim=-1)
        out += (top2.values[:, 0] - top2.values[:, 1]).cpu().tolist()
    return out


@torch.no_grad()
def refusals(m, tok, prompts, batch, device, max_new=256):
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        msgs = [M.chat_prompt(tok, p)
                for p in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True).to(device)
        gen = m.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            g = gen[j][enc["input_ids"].shape[1]:]
            txt = tok.decode(g, skip_special_tokens=True)
            out.append(M.classify(txt, max_new, int((g != tok.pad_token_id).sum()))[0])
    return out


def auc(pos, neg):
    """P(a random positive scores below a random negative) -- rank-based, no deps.
    `pos` = margins of flipped prompts; hypothesis says these are SMALLER."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p < n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--config", default="INT4", help="e.g. INT4, FP4")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    fmt = "INT" if a.config.startswith("INT") else "FP"
    width = int(a.config[3:]) if fmt == "INT" else int(a.config[2:])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device).eval()

    prompts = load_prompts(a.n_prompts)[:a.n_prompts]
    print(f"{len(prompts)} prompts")

    print("BF16: margins + decisions...")
    marg = first_token_margins(m, tok, prompts, a.batch, a.device)
    ref0 = refusals(m, tok, prompts, a.batch, a.device)

    print(f"{a.config}: decisions...")
    saved = quantize_model_(m, fmt, width)
    ref1 = refusals(m, tok, prompts, a.batch, a.device)
    restore_(m, saved); del saved
    torch.cuda.empty_cache()

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"margin_{a.family}_{a.config}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "config", "idx", "bf16_margin",
                    "bf16_refused", "quant_refused", "flipped_to_comply"])
        for i, (g, r0, r1) in enumerate(zip(marg, ref0, ref1)):
            w.writerow([a.family, a.config, i, round(g, 4),
                        int(r0), int(r1), int(r0 and not r1)])

    # the test: among prompts BF16 REFUSED, do the flipped ones have thinner margins?
    kept = [g for g, r0, r1 in zip(marg, ref0, ref1) if r0 and r1]
    lost = [g for g, r0, r1 in zip(marg, ref0, ref1) if r0 and not r1]
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    print(f"\nBF16 refused {sum(ref0)}/{len(prompts)}; "
          f"of those, {len(lost)} flipped to compliance under {a.config}")
    print(f"  median BF16 margin, refusal KEPT : {med(kept):.4f}  (n={len(kept)})")
    print(f"  median BF16 margin, refusal LOST : {med(lost):.4f}  (n={len(lost)})")
    A = auc(lost, kept)
    print(f"  AUC (thin margin predicts flip)  : {A:.3f}")
    print("\n  AUC ~0.5 -> margin does NOT explain which refusals are lost;")
    print("  AUC >0.7 -> margin is a strong predictor and gives a cheap")
    print("              BF16-only screen for quantization fragility.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
