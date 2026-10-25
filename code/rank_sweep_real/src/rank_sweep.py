#!/usr/bin/env python3
"""Stage 2 (GPU): norm-matched rank-k sweep on REAL weights and REAL directions.

Replaces the synthetic Table 8 of the submission. For every edited module and
every rank k, builds the norm-matched edit dW_k = -alpha_k D_k D_k^T W from the
directions extracted in stage 1, quantizes W and W+dW, and measures how much of
the edit the quantizer transmits inside the ablated subspace.

Weight arithmetic only -- no generation, no judge model. ~10-20 minutes for a
7B model across 8 layers x 2 modules x 8 ranks x 4 schemes.

  python3 rank_sweep.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --dirs ../out/directions_qwen.npz \
      --out  ../out/rank_sweep_qwen.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core                                  # noqa: E402
from backend import TorchBackend             # noqa: E402

MODULES = ["self_attn.o_proj", "mlp.down_proj"]


def get_layers(model):
    for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers on this model")


def get_module(layer, path):
    obj = layer
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--schemes", default="int4,nf4,fp8,int8")
    ap.add_argument("--layers", default="",
                    help="comma-separated; default = every layer present in --dirs")
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--no-norm-match", action="store_true",
                    help="use alpha=1 at every rank (control arm: SNR should be "
                         "flat in k, since energy grows as sqrt(k) while the "
                         "per-direction subspace shrinks as sqrt(d/k))")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    schemes = [s.strip() for s in args.schemes.split(",")]
    modules = [m.strip() for m in args.modules.split(",")]

    print(f"[1/3] loading directions from {args.dirs}")
    Z = np.load(args.dirs)
    have = sorted(int(k.split("_")[1]) for k in Z.files if k.startswith("D_"))
    layers = ([int(x) for x in args.layers.split(",")] if args.layers else have)
    layers = [li for li in layers if li in have]
    print(f"      layers available: {len(have)}; using {len(layers)}")

    print(f"[2/3] loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    bk = TorchBackend(device="cuda", dtype=torch.float32)
    decoder = get_layers(model)

    rows = []
    print("[3/3] sweeping")
    for li in layers:
        D_np = Z[f"D_{li}"]
        D_full = bk.asarray(torch.from_numpy(D_np))
        max_k = D_full.shape[1]
        for mod_path in modules:
            try:
                W_t = get_module(decoder[li], mod_path).weight.data
            except (AttributeError, IndexError):
                print(f"      skip layer {li} {mod_path} (absent)")
                continue
            if W_t.shape[0] != D_full.shape[0]:
                print(f"      skip layer {li} {mod_path}: out_features "
                      f"{W_t.shape[0]} != d_model {D_full.shape[0]}")
                continue
            W = bk.asarray(W_t)
            for k in [k for k in ks if k <= max_k]:
                dW, alpha = core.build_edit(
                    W, D_full, k, bk, norm_match=not args.no_norm_match)
                for scheme in schemes:
                    m = core.measure(W, dW, D_full, k, scheme, bk)
                    m.update(layer=li, module=mod_path, alpha=alpha,
                             d_model=int(W.shape[0]), d_in=int(W.shape[1]),
                             norm_matched=int(not args.no_norm_match))
                    rows.append(m)
                del dW
            del W
            torch.cuda.empty_cache()
            print(f"      layer {li:3d} {mod_path:18s} done "
                  f"({len(rows)} cells)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fields = ["layer", "module", "scheme", "k", "alpha", "norm_matched",
              "d_model", "d_in", "norm_W", "norm_dW", "norm_rho", "rel_edit",
              "gain_pi", "snr_global", "snr_subspace", "survive_sub"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"      wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
