#!/usr/bin/env python3
"""WHERE in the generation does quantization flip a refusal?

The margin hypothesis failed (AUC 0.44 / 0.52 at chance), and the reason is
visible in the margins themselves: first-token top1-top2 gaps are 3.5 logits
(Granite) and 8.6 logits (Qwen), while quantization perturbs logits by a fraction
of one. Noise that small cannot flip the first token -- yet 41% of refusals flip
anyway. So the refuse/comply decision is NOT made at the first token.

This measures where it IS made. Greedy decoding is deterministic, so the BF16 and
quantized generations for the same prompt can be compared token by token; the
first index at which they differ is where quantization actually changed the
trajectory.

    divergence at token 0-2   -> a genuine early branch point (and the margin
                                 result would then be contradictory)
    divergence at token 20-60 -> refusal is a TRAJECTORY property: the model
                                 begins identically and drifts, which is why no
                                 single-token decision boundary predicts it

Reported separately for prompts that flipped refusal and those that did not, so
the question is not merely "when do outputs diverge" (they always eventually do)
but "does flipping coincide with diverging earlier".

    python experiment/divergence_probe.py --model ibm-granite/granite-3.1-8b-instruct \
        --family Granite8B --config INT4 --n-prompts 150
"""
import argparse, csv, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import metrics as M
from stages_eval import load_prompts, quantize_model_, restore_

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"


@torch.no_grad()
def generate_ids(m, tok, prompts, batch, device, max_new):
    """Greedy generations as token-id lists, plus the decoded text."""
    ids_out, txt_out = [], []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        msgs = [M.chat_prompt(tok, p) for p in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True).to(device)
        gen = m.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        for j in range(len(chunk)):
            g = gen[j][plen:]
            keep = g[g != tok.pad_token_id] if tok.pad_token_id is not None else g
            ids_out.append(g.tolist())
            txt_out.append(tok.decode(keep, skip_special_tokens=True))
    return ids_out, txt_out


def first_diff(a, b):
    """Index of the first differing token, or -1 if identical over the overlap."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--config", default="INT4")
    ap.add_argument("--n-prompts", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
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

    print("BF16 generations...")
    ids0, txt0 = generate_ids(m, tok, prompts, a.batch, a.device, a.max_new)
    print(f"{a.config} generations...")
    saved = quantize_model_(m, fmt, width)
    ids1, txt1 = generate_ids(m, tok, prompts, a.batch, a.device, a.max_new)
    restore_(m, saved); del saved
    torch.cuda.empty_cache()

    rows = []
    for i in range(len(prompts)):
        n0 = len(txt0[i].split())
        r0 = M.classify(txt0[i], a.max_new, n0)[0]
        r1 = M.classify(txt1[i], a.max_new, len(txt1[i].split()))[0]
        d = first_diff(ids0[i], ids1[i])
        rows.append((r0, r1, d))

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"divergence_{a.family}_{a.config}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "config", "idx", "bf16_refused", "quant_refused",
                    "flipped_to_comply", "diverge_token"])
        for i, (r0, r1, d) in enumerate(rows):
            w.writerow([a.family, a.config, i, int(r0), int(r1),
                        int(r0 and not r1), d])

    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    flipped = [d for r0, r1, d in rows if r0 and not r1 and d >= 0]
    kept = [d for r0, r1, d in rows if r0 and r1 and d >= 0]
    ident = sum(1 for _, _, d in rows if d < 0)
    early = sum(1 for d in flipped if d <= 2)
    print(f"\nidentical generations: {ident}/{len(rows)}")
    print(f"  median divergence token, refusal LOST : {med(flipped):>6} (n={len(flipped)})")
    print(f"  median divergence token, refusal KEPT : {med(kept):>6} (n={len(kept)})")
    if flipped:
        print(f"  flips diverging at token <=2          : {early}/{len(flipped)} "
              f"({100*early/len(flipped):.0f}%)")
        print("\n  mostly <=2  -> a genuine early branch point")
        print("  mostly >10  -> refusal is a TRAJECTORY property; no single-token")
        print("                 boundary can predict it, which explains the margin null")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
