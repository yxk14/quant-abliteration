#!/usr/bin/env python3
"""Is safety carried by a more CONCENTRATED circuit than capability?

Hypothesis 2 for the observational result. If refusal behaviour is implemented by
a small set of layers while general capability is distributed across the stack,
then uniform weight noise damages safety disproportionately -- not because safety
is intrinsically fragile, but because the same total perturbation lands harder on
a smaller target.

Test: quantize ONE WINDOW of layers at a time, leaving every other layer at BF16,
and sweep the window across depth. Measure refusal loss and MMLU at each position.

    concentrated safety  -> refusal damage SPIKES at particular depths
    distributed safety   -> refusal damage is flat across depth, like MMLU

Reading the two curves together is the point: the claim is comparative, so a
refusal spike only means something if MMLU is flat at the same depths.

    python experiment/layer_probe.py --model ibm-granite/granite-3.1-8b-instruct \
        --family Granite8B --window 4
"""
import argparse, csv, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import metrics as M
import mmlu_direct as MD
from quantizers import get_quantizer
from stages_eval import load_prompts

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"
TARGETS = ("o_proj", "down_proj", "q_proj", "k_proj", "v_proj",
           "gate_proj", "up_proj")


def layer_of(name):
    for s in name.split("."):
        if s.isdigit():
            return int(s)
    return None


def quantize_window_(model, fmt, width, lo, hi):
    """Quantize only layers [lo, hi). Returns CPU originals for restoration."""
    q = get_quantizer(fmt, width)
    saved = {}
    for name, p in model.named_parameters():
        if p.ndim != 2 or not any(t in name for t in TARGETS):
            continue
        L = layer_of(name)
        if L is None or not (lo <= L < hi):
            continue
        saved[name] = p.data.detach().to("cpu", copy=True)
        p.data.copy_(q(p.data))
    return saved


def restore_(model, saved):
    for name, p in model.named_parameters():
        if name in saved:
            p.data.copy_(saved[name].to(p.device))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--config", default="INT4")
    ap.add_argument("--window", type=int, default=4, help="layers per window")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--mmlu-limit", type=int, default=10)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--mmlu-batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    fmt = "INT" if a.config.startswith("INT") else "FP"
    width = int(a.config[3:]) if fmt == "INT" else int(a.config[2:])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device).eval()

    nl = max(L for L in (layer_of(n) for n, _ in m.named_parameters())
             if L is not None) + 1
    print(f"{nl} layers, window {a.window}")

    prompts = load_prompts(a.n_prompts)[:a.n_prompts]
    ds_all = load_dataset("cais/mmlu", "all")
    subs = sorted(set(ds_all["test"]["subject"]))

    tokm = AutoTokenizer.from_pretrained(a.model)
    tokm.padding_side = "right"
    if tokm.pad_token is None:
        tokm.pad_token = tokm.eos_token

    print("BF16 reference...")
    ref = refusals(m, tok, prompts, a.batch, a.device)
    acc0 = MD.run(m, tokm, subjects=subs, batch_size=a.mmlu_batch,
                  limit=a.mmlu_limit, verify=False, device=a.device) * 100
    print(f"  refusal {100*sum(ref)/len(ref):.1f}%  mmlu {acc0:.2f}%")

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"layer_{a.family}_{a.config}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "config", "lo", "hi", "mid", "n_layers",
                    "refusal_rate", "unsafe_flips", "safe_flips",
                    "mmlu", "mmlu_kept"])
        for lo in range(0, nl, a.window):
            hi = min(lo + a.window, nl)
            saved = quantize_window_(m, fmt, width, lo, hi)
            cur = refusals(m, tok, prompts, a.batch, a.device)
            acc = MD.run(m, tokm, subjects=subs, batch_size=a.mmlu_batch,
                         limit=a.mmlu_limit, verify=False, device=a.device) * 100
            restore_(m, saved); del saved
            torch.cuda.empty_cache()
            unsafe = sum(r and not c for r, c in zip(ref, cur)) / len(cur) * 100
            safe = sum(c and not r for r, c in zip(ref, cur)) / len(cur) * 100
            w.writerow([a.family, a.config, lo, hi, (lo + hi - 1) / 2, hi - lo,
                        round(100 * sum(cur) / len(cur), 2), round(unsafe, 2),
                        round(safe, 2), round(acc, 2), round(100 * acc / acc0, 2)])
            fh.flush()
            print(f"  layers {lo:2d}-{hi-1:2d}: refusal {100*sum(cur)/len(cur):5.1f}%  "
                  f"lost {unsafe:5.1f}%  mmlu {acc:5.2f}% (kept {100*acc/acc0:6.2f}%)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
