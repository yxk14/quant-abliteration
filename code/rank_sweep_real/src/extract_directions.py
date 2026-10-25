#!/usr/bin/env python3
"""Stage 1 (GPU): extract REAL refusal-subspace directions from the base model.

Column 0 of the returned basis is the difference-in-means refusal direction
(Arditi et al.) -- the same direction the submitted paper's rank-1 edit uses.
Columns 1..K-1 are the leading principal directions of the harmful-vs-harmless
activation difference, so the rank-k edits in stage 2 are built from directions
the model actually uses, not from Gaussian noise as in Table 8.

One forward pass per prompt, no generation. ~2-4 minutes for 512 prompts on a
5090.

  python3 extract_directions.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --harmful  data/harmful.txt \
      --harmless data/harmless.txt \
      --out ../out/directions_qwen.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_prompts(path, limit):
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:limit]


@torch.no_grad()
def last_token_hidden(model, tok, prompts, batch_size, device):
    """Return [n_layers+1, n_prompts, d_model] last-token hidden states."""
    out = None
    n_done = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        res = model(**enc, output_hidden_states=True, use_cache=False)
        # index of each sequence's final real token
        idx = enc["attention_mask"].sum(dim=1) - 1
        hs = torch.stack(res.hidden_states, dim=0)        # [L+1, B, T, d]
        picked = hs[:, torch.arange(hs.shape[1], device=device), idx, :]
        picked = picked.float().cpu()
        if out is None:
            out = torch.empty((picked.shape[0], len(prompts), picked.shape[2]),
                              dtype=torch.float32)
        out[:, n_done:n_done + picked.shape[1], :] = picked
        n_done += picked.shape[1]
        del res, hs, picked
        print(f"    {n_done}/{len(prompts)}", end="\r", flush=True)
    print()
    return out[:, :n_done, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--harmful", required=True)
    ap.add_argument("--harmless", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=512,
                    help="per class. NOTE: the achievable rank in stage 2 is "
                         "bounded by this number -- you cannot sweep to k=512 "
                         "with only 256 harmful prompts.")
    ap.add_argument("--max-dirs", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--layers", default="",
                    help="comma-separated layer indices; default = all")
    args = ap.parse_args()

    device = "cuda"
    print(f"[1/4] loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    harmful = read_prompts(args.harmful, args.n_prompts)
    harmless = read_prompts(args.harmless, args.n_prompts)
    print(f"[2/4] {len(harmful)} harmful / {len(harmless)} harmless prompts")

    print("  harmful forward pass")
    H = last_token_hidden(model, tok, harmful, args.batch_size, device)
    print("  harmless forward pass")
    L = last_token_hidden(model, tok, harmless, args.batch_size, device)

    n_layers = H.shape[0]
    want = (list(range(n_layers)) if not args.layers
            else [int(x) for x in args.layers.split(",")])

    print(f"[3/4] building direction bases for {len(want)} layers")
    store, meta = {}, {}
    K = min(args.max_dirs, min(H.shape[1], H.shape[2]) - 1)
    for li in want:
        h, l = H[li].double().numpy(), L[li].double().numpy()
        r = h.mean(0) - l.mean(0)
        nr = np.linalg.norm(r)
        if nr < 1e-8:
            continue
        r = r / nr
        # principal directions of the harmful-vs-harmless difference
        Dm = h - l.mean(0)
        U2, S2, _ = np.linalg.svd((Dm - Dm.mean(0)).T, full_matrices=False)
        cand = U2[:, :K]
        cos_r_pc1 = float(abs(r @ cand[:, 0]))
        # force column 0 to be the refusal direction, orthonormalize the rest
        B = np.concatenate([r[:, None], cand], axis=1)
        Q, _ = np.linalg.qr(B)
        Q = Q[:, :K + 1]
        if Q[:, 0] @ r < 0:
            Q[:, 0] *= -1
        store[f"D_{li}"] = Q.astype(np.float32)
        store[f"r_{li}"] = r.astype(np.float32)
        meta[str(li)] = {
            "dim": int(r.shape[0]),
            "n_dirs": int(Q.shape[1]),
            "diff_norm": float(nr),
            "cos_refusal_pc1": cos_r_pc1,
            "spectrum_top8": [float(x) for x in S2[:8]],
            "spectral_decay_s8_over_s1": float(S2[7] / S2[0]) if len(S2) > 7 else None,
        }
        print(f"    layer {li:3d}  d={r.shape[0]}  dirs={Q.shape[1]}  "
              f"|r|={nr:8.3f}  cos(r,pc1)={cos_r_pc1:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, **store)
    with open(args.out.replace(".npz", "_meta.json"), "w") as f:
        json.dump({"model": args.model, "n_harmful": len(harmful),
                   "n_harmless": len(harmless), "layers": meta}, f, indent=2)
    print(f"[4/4] wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
