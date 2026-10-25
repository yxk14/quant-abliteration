#!/usr/bin/env python3
"""Real deployment quantizers, not simulated round-to-nearest.

Everything measured so far uses quantize-dequantize in bf16 storage. Nobody
deploys naive per-channel RTN INT4 -- they deploy bitsandbytes NF4, GPTQ or AWQ,
which are calibrated and specifically built to limit damage. So the claim "4-bit
costs 25-41% of refusals" is only interesting if it survives a real quantizer.

This also validates the simulation: simulated NF4 and bitsandbytes NF4 should
agree. If they do not, every NF4 number in the paper needs re-stating.

The sharpest question it can answer: calibration is tuned to preserve
*capability* -- does it also preserve *alignment*, or only the benchmark?

    python experiment/real_quant.py --model Qwen/Qwen2.5-7B-Instruct \
        --family Qwen2.5-7B-Instruct --methods bf16 bnb-nf4 bnb-int8
"""
import argparse, csv, gc, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import metrics as M
import mmlu_direct as MD
from compute_refusal_dirs import prompt_sets

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"
MAXNEW = 256


def load_model(model_id, method, device="cuda"):
    """Load under a REAL quantizer. bnb quantizes at load time -- there is no
    in-place quantize/restore, so each method costs a fresh load."""
    from transformers import AutoModelForCausalLM
    kw = dict(device_map=device)
    if method == "bf16":
        kw["torch_dtype"] = torch.bfloat16
    elif method in ("bnb-nf4", "bnb-fp4"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4" if method == "bnb-nf4" else "fp4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False)
    elif method == "bnb-nf4-dq":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    elif method == "bnb-int8":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif method == "prequant":
        # an official GPTQ / AWQ checkpoint: config already carries the method
        kw["torch_dtype"] = "auto"
    else:
        raise ValueError(method)
    return AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()


@torch.no_grad()
def refusal_decisions(m, tok, prompts, batch=20, device="cuda"):
    """Greedy, per HANDOFF section 5. Returns (is_refusal, is_complete) per prompt."""
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        msgs = [M.chat_prompt(tok, p)
                for p in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True).to(device)
        gen = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                         pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            g = gen[j][enc["input_ids"].shape[1]:]
            txt = tok.decode(g, skip_special_tokens=True)
            out.append(M.classify(txt, MAXNEW, int((g != tok.pad_token_id).sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["bf16", "bnb-nf4", "bnb-int8"])
    ap.add_argument("--prequant", nargs="*", default=[],
                    help="repo_id=label for official GPTQ/AWQ checkpoints")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--mmlu-limit", type=int, default=20)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--mmlu-batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    prompts, _ = prompt_sets(a.n_prompts)
    prompts = prompts[:a.n_prompts]
    print(f"harmful battery: {len(prompts)} prompts")
    ds_all = load_dataset("cais/mmlu", "all")
    subs = sorted(set(ds_all["test"]["subject"]))

    jobs = [(a.model, m) for m in a.methods]
    for spec in a.prequant:
        repo, _, label = spec.partition("=")
        jobs.append((repo, f"prequant:{label or repo.split('/')[-1]}"))

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"real_quant_{a.family}.csv"
    ref = None
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "method", "n", "refusal_rate", "completion_rate",
                    "dfr", "unsafe_flips", "safe_flips", "mmlu", "mmlu_kept"])
        acc0 = None
        for repo, method in jobs:
            print(f"\n=== {a.family} / {method} ({repo})")
            try:
                m = load_model(repo, "prequant" if method.startswith("prequant")
                               else method, a.device)
            except Exception as e:
                print(f"  LOAD FAILED: {repr(e)[:160]}")
                continue

            dec = refusal_decisions(m, tok, prompts, a.batch, a.device)
            rr = sum(r for r, _ in dec) / len(dec) * 100
            cc = sum(c for _, c in dec) / len(dec) * 100
            if ref is None:
                ref = [r for r, _ in dec]
                dfr = unsafe = safe = 0.0
            else:
                cur = [r for r, _ in dec]
                unsafe = sum(a_ and not b for a_, b in zip(ref, cur)) / len(cur) * 100
                safe = sum(b and not a_ for a_, b in zip(ref, cur)) / len(cur) * 100
                dfr = unsafe + safe

            tokm = AutoTokenizer.from_pretrained(a.model)
            tokm.padding_side = "right"
            if tokm.pad_token is None:
                tokm.pad_token = tokm.eos_token
            acc = MD.run(m, tokm, subjects=subs, batch_size=a.mmlu_batch,
                         limit=a.mmlu_limit, verify=False, device=a.device) * 100
            if acc0 is None:
                acc0 = acc
            w.writerow([a.family, method, len(dec), round(rr, 2), round(cc, 2),
                        round(dfr, 2), round(unsafe, 2), round(safe, 2),
                        round(acc, 2), round(100 * acc / acc0, 2)])
            fh.flush()
            print(f"  refusal {rr:.1f}%  unsafe-flips {unsafe:.1f}%  "
                  f"safe-flips {safe:.1f}%  mmlu {acc:.2f}% (kept {100*acc/acc0:.1f}%)")
            del m
            gc.collect(); torch.cuda.empty_cache()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
