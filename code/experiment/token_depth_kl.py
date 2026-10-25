#!/usr/bin/env python3
"""Per-token KL between an aligned model and its unaligned base, on harmful pairs.

Reproduces the measurement behind Figure 1 of Qi et al., "Safety Alignment
Should Be Made More Than Just a Few Tokens Deep" (ICLR 2025), which shows the
aligned-vs-base KL budget concentrated in the first few response tokens --
"shallow safety alignment".

Why we run it. Our behavioural result is model-dependent: quantizing the
abliterated Qwen2.5-7B restores refusal (+9.5 pts, p=0.0002) while the
abliterated Llama-3.1-8B does not move (+1.5 pts, p=0.51). If Llama's
post-training spread its alignment over more token positions than Qwen's, then
removing one direction leaves less shallow refusal-prefix machinery for
quantization noise to refill, and the null has a mechanism rather than being a
bare negative. This script measures that depth directly.

    D_t = KL( pi_aligned(.|x, y_<t) || pi_base(.|x, y_<t) )

averaged over harmful (x, y) pairs, for t = 1..T response positions.

Harmful answers. Qi et al. build "Harmful HEx-PHI" by generating harmful answers
with a jailbroken GPT-3.5. We instead use the complying BF16 generations of our
OWN abliterated checkpoint for that family, taken from the judged response
files. This is family-matched (same tokenizer, same distribution) and needs no
external jailbroken model.

VRAM. Two 8B models in bf16 do not fit beside activations on a 32 GB card, so we
run one model at a time and stream log-probs at the scored positions to a
float16 memmap on disk, exactly as stages_eval.py does for the KLD leg.

    python experiment/token_depth_kl.py \
        --aligned Qwen/Qwen2.5-7B-Instruct --base Qwen/Qwen2.5-7B \
        --responses analysis/responses/Qwen2.5-7B-abl__BF16.judged.json \
        --label Qwen2.5-7B --out token_depth_Qwen2.5-7B.csv
"""
import argparse, csv, gc, json, math, pathlib, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import metrics as M

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"
CACHE = pathlib.Path(__file__).parent.parent / "logit_cache"


def load_pairs(path, tok, n_max, T, max_prompt):
    """(prompt_ids, response_ids) for items the judge called COMPLY.

    A refusal would measure the wrong thing: we want the distribution over a
    HARMFUL continuation, which is where the aligned and base models are
    supposed to diverge.
    """
    d = json.loads(pathlib.Path(path).read_text())
    items = d["items"] if isinstance(d, dict) and "items" in d else d
    prompts = d.get("prompts") if isinstance(d, dict) else None
    out = []
    for i, it in enumerate(items):
        if it.get("judge") != "COMPLY":
            continue
        text = it.get("text", "")
        if not text.strip():
            continue
        p = (it.get("prompt") or (prompts[i] if prompts else None))
        if p is None:
            continue
        pid = tok(M.chat_prompt(tok, p), return_tensors="pt").input_ids[0]
        if pid.numel() > max_prompt:
            continue
        rid = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0][:T]
        if rid.numel() < 2:
            continue
        out.append((pid, rid))
        if len(out) >= n_max:
            break
    return out


@torch.no_grad()
def logprobs_at_positions(model, pairs, T, path, device):
    """[n, T, V] float16 log-softmax at each response position; NaN where absent."""
    V = model.config.vocab_size
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                   shape=(len(pairs), T, V))
    mm[:] = np.nan
    for i, (pid, rid) in enumerate(pairs):
        ids = torch.cat([pid, rid]).unsqueeze(0).to(device)
        lg = model(ids).logits[0].float()
        # position t of the response is predicted from index len(pid)-1+t
        start = pid.numel() - 1
        k = min(rid.numel(), T)
        sl = torch.log_softmax(lg[start:start + k], dim=-1)
        mm[i, :k] = sl.to(torch.float16).cpu().numpy()
        del lg, sl
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(pairs)}", flush=True)
    mm.flush(); del mm
    gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--responses", required=True,
                    help="judged .json whose COMPLY items supply harmful answers")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--T", type=int, default=25, help="response positions to score")
    ap.add_argument("--max-prompt", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(a.aligned)
    # A tokenizer/vocab mismatch would silently compare two different token
    # streams and produce a meaningless KL. Check once, before any GPU work.
    va = AutoConfig.from_pretrained(a.aligned).vocab_size
    vb = AutoConfig.from_pretrained(a.base).vocab_size
    assert va == vb, f"vocab mismatch: aligned {va} vs base {vb}"
    print(f"vocab check OK ({va})")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pairs = load_pairs(a.responses, tok, a.n, a.T, a.max_prompt)
    if not pairs:
        raise SystemExit("no COMPLY items with prompts found in " + a.responses)
    print(f"{a.label}: {len(pairs)} harmful (x,y) pairs, scoring T={a.T} positions")

    CACHE.mkdir(parents=True, exist_ok=True)
    pa = CACHE / f"td_{a.label}_aligned.npy"
    pb = CACHE / f"td_{a.label}_base.npy"

    for tag, mid, path in (("aligned", a.aligned, pa), ("base", a.base, pb)):
        print(f"  loading {tag}: {mid}")
        m = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.bfloat16, device_map=a.device).eval()
        logprobs_at_positions(m, pairs, a.T, path, a.device)
        del m; gc.collect(); torch.cuda.empty_cache()

    A = np.load(pa, mmap_mode="r"); B = np.load(pb, mmap_mode="r")
    rows = []
    for t in range(a.T):
        vals = []
        for i in range(len(pairs)):
            la = np.asarray(A[i, t], dtype=np.float32)
            if not np.isfinite(la).all():
                continue
            lb = np.asarray(B[i, t], dtype=np.float32)
            # KL(aligned || base) = sum_v p_a (log p_a - log p_b)
            pa_ = np.exp(la)
            vals.append(float(np.sum(pa_ * (la - lb))))
        if vals:
            v = np.array(vals)
            rows.append((t + 1, len(v), float(v.mean()),
                         float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0))
            print(f"  t={t+1:2d}  n={len(v):3d}  KL={v.mean():.4f}")

    out = DATA / (a.out or f"token_depth_{a.label}.csv")
    DATA.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "position", "n", "kl_mean", "kl_se"])
        for t, n, mu, se in rows:
            w.writerow([a.label, t, n, round(mu, 6), round(se, 6)])
    print(f"wrote {out}")
    for p in (pa, pb):
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
