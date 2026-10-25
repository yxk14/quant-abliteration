#!/usr/bin/env python3
"""Phase 1 of 2: generate responses and SAVE THE TEXT.

The whole reason a broken refusal classifier cost us the entire base-model
result is that every previous stage computed classify() inline and threw the
generations away. Re-scoring therefore meant re-running every model on the GPU.

This separates the expensive, deterministic part (generation) from the cheap,
revisable part (judging). Text is written to analysis/responses/ and can be
re-scored by any classifier, forever, at zero GPU cost.

    python experiment/generate_responses.py --model Qwen/Qwen2.5-7B-Instruct \\
        --family Qwen2.5-7B --configs BF16 FP16 FP4 INT4 --n-prompts 200
"""
import argparse, json, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import metrics as M
from stages_eval import load_prompts, quantize_model_, restore_

OUT = pathlib.Path(__file__).parent.parent / "analysis" / "responses"


@torch.no_grad()
def generate(m, tok, prompts, batch, device, max_new):
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
            n_real = int((t != tok.pad_token_id).sum())
            out.append({"text": tok.decode(t, skip_special_tokens=True),
                        "n_tokens": n_real,
                        "hit_cap": n_real >= max_new})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--configs", nargs="+", default=["BF16", "FP16", "FP4", "INT4"])
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompts-file", default=None,
                    help="JSON file replacing the harmful battery: either a list of "
                         "strings or {items:[{zh:...}]} as written by translate_battery.py")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device).eval()

    if a.prompts_file:
        pf = json.loads(pathlib.Path(a.prompts_file).read_text())
        prompts = [x["zh"] if isinstance(x, dict) else x
                   for x in (pf["items"] if isinstance(pf, dict) else pf)][:a.n_prompts]
        print(f"prompts from {a.prompts_file}")
    else:
        prompts = load_prompts(a.n_prompts)[:a.n_prompts]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{a.family}: {len(prompts)} prompts x {len(a.configs)} configs")

    for cfg in a.configs:
        saved = None
        if cfg != "BF16":
            if cfg.startswith("INTG"):          # e.g. INTG4 = group-wise INT4
                fmt, width = "INTG", int(cfg[4:])
            elif cfg.startswith("INT"):
                fmt, width = "INT", int(cfg[3:])
            else:
                fmt, width = "FP", int(cfg[2:])
            saved = quantize_model_(m, fmt, width)
        res = generate(m, tok, prompts, a.batch, a.device, a.max_new)
        if saved is not None:
            restore_(m, saved); del saved
            torch.cuda.empty_cache()
        f = OUT / f"{a.family}__{cfg}.json"
        f.write_text(json.dumps(
            {"model": a.model, "family": a.family, "config": cfg,
             "max_new": a.max_new, "prompts_file": a.prompts_file,
             "items": [{"prompt": p, **r} for p, r in zip(prompts, res)]}, indent=1))
        capped = sum(r["hit_cap"] for r in res)
        print(f"  {cfg:6s} -> {f.name}  ({capped}/{len(res)} hit the token cap)")
    print("done")


if __name__ == "__main__":
    main()
