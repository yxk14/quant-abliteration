#!/usr/bin/env python3
"""Are the "lost refusals" real compliance, or just broken text?

The headline claim -- 4-bit quantization makes models comply with 25-41% of
harmful prompts they previously refused -- rests entirely on a KEYWORD refusal
classifier (substring match against a marker list). That classifier has one
failure mode that would invalidate the whole result:

    quantization degrades generation -> output is rambling / off-topic / broken
    -> it contains no refusal marker -> scored as COMPLIANCE.

That is breakage being mislabelled as a safety regression. The opposite error
matters too: a refusal phrased without any listed marker ("That's not something
I'll help with") scores as compliance even at BF16.

This dumps the actual generations for every flipped prompt, side by side, so a
human can read them and decide. It deliberately does NOT auto-classify -- the
whole point is that the automated classifier is what is under suspicion.

Output is written to a file, not stdout: the generations are responses to
harmful prompts and should be reviewed deliberately, not scrolled past in a log.

    python experiment/audit_flips.py --model ibm-granite/granite-3.1-8b-instruct \
        --family Granite8B --config INT4 --n-prompts 120
"""
import argparse, json, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import metrics as M
from stages_eval import load_prompts, quantize_model_, restore_

OUT = pathlib.Path(__file__).parent.parent / "analysis" / "audit"


@torch.no_grad()
def gen(m, tok, prompts, batch, device, max_new):
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        msgs = [M.chat_prompt(tok, p) for p in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True).to(device)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        for j in range(len(chunk)):
            t = g[j][plen:]
            out.append(tok.decode(t, skip_special_tokens=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--config", default="INT4")
    ap.add_argument("--n-prompts", type=int, default=120)
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
    t0 = gen(m, tok, prompts, a.batch, a.device, a.max_new)
    saved = quantize_model_(m, fmt, width)
    t1 = gen(m, tok, prompts, a.batch, a.device, a.max_new)
    restore_(m, saved); del saved
    torch.cuda.empty_cache()

    rows = []
    for p, b, q in zip(prompts, t0, t1):
        rb = M.classify(b, a.max_new, len(b.split()))[0]
        rq = M.classify(q, a.max_new, len(q.split()))[0]
        if rb and not rq:
            rows.append({"prompt": p, "bf16": b, "quant": q})

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"flips_{a.family}_{a.config}.json"
    f.write_text(json.dumps(rows, indent=1))
    print(f"{a.family} {a.config}: {len(rows)} flipped of {len(prompts)} prompts")
    print(f"wrote {f}  -- READ THESE before trusting the headline number")


if __name__ == "__main__":
    main()
