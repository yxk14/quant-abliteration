#!/usr/bin/env python3
"""CPU dry run: produce a CSV with the exact schema rank_sweep.py writes, from
synthetic weights. Lets stage 3 (table + figure) be tested and the LaTeX
checked into the paper before the GPU run exists. NOT a result -- the header
of every file it makes says so.

  python3 mock_sweep.py --out ../out/mock_sweep.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core                           # noqa: E402
from backend import NumpyBackend      # noqa: E402

bk = NumpyBackend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-model", type=int, default=896)
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--schemes", default="int4,nf4,fp8,int8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = args.d_model
    ks = [int(x) for x in args.ks.split(",")]
    rows = []
    for li in [int(x) for x in args.layers.split(",")]:
        D = core.orthonormalize(bk.asarray(rng.normal(size=(d, max(ks) + 1))), bk)
        for mod, d_in in [("self_attn.o_proj", d), ("mlp.down_proj", int(d * 2.5))]:
            W = bk.asarray(rng.normal(0, 0.02, size=(d, d_in)))
            for k in ks:
                dW, alpha = core.build_edit(W, D, k, bk)
                for scheme in args.schemes.split(","):
                    m = core.measure(W, dW, D, k, scheme, bk)
                    m.update(layer=li, module=mod, alpha=alpha, d_model=d,
                             d_in=d_in, norm_matched=1)
                    rows.append(m)
        print(f"  layer {li} done ({len(rows)} cells)")

    fields = ["layer", "module", "scheme", "k", "alpha", "norm_matched",
              "d_model", "d_in", "norm_W", "norm_dW", "norm_rho", "rel_edit",
              "gain_pi", "snr_global", "snr_subspace", "survive_sub"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"wrote {len(rows)} rows to {args.out}  (SYNTHETIC -- dry run only)")


if __name__ == "__main__":
    main()
