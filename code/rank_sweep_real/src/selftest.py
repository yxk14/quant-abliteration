#!/usr/bin/env python3
"""CPU self-test: exercises every code path in core.py on synthetic weights.

Runs in ~20 s with no GPU, no torch and no model download. Two jobs:

  1. Catch bugs before the 5090 run (shapes, quantizer correctness, the
     norm-matching identity, the subspace projections).
  2. Reproduce the SNR-vs-rank trend reported in Table 8 of the submission,
     so we know the real-direction run is measuring the same quantity.

Usage:  python3 selftest.py
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from backend import NumpyBackend          # noqa: E402
import core                               # noqa: E402

bk = NumpyBackend()
RNG = np.random.default_rng(0)


def banner(t):
    print("\n" + t)
    print("-" * len(t))


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return cond


ok = True

# ------------------------------------------------------------------ quantizers
banner("1. Quantizer sanity")
W = RNG.normal(0, 0.02, size=(256, 512))
for scheme, tol in [("int8", 0.01), ("int4", 0.16), ("nf4", 0.11), ("fp8", 0.05)]:
    Wq = core.quantize(bk.asarray(W), scheme, bk)
    rel = np.linalg.norm(Wq - W) / np.linalg.norm(W)
    ok &= check(f"{scheme}: relative error {rel:.4f} < {tol}", rel < tol)
    n_levels = len(np.unique(np.round(Wq[0] / (np.abs(Wq[0]).max() + 1e-12), 6)))
    print(f"         distinct normalized levels in row 0: {n_levels}")

# ------------------------------------------------- norm matching is exact
banner("2. Norm-matched edits (||dW_k|| must be rank-independent)")
d_model, d_in = 1024, 2048
W = bk.asarray(RNG.normal(0, 0.02, size=(d_model, d_in)))
D = core.orthonormalize(bk.asarray(RNG.normal(size=(d_model, 512))), bk)
ref = None
rows = []
for k in [1, 2, 4, 8, 16, 64, 256]:
    dW, a = core.build_edit(W, D, k, bk)
    n = bk.fnorm(dW)
    if ref is None:
        ref = n
    rows.append((k, a, n, n / ref))
    print(f"   k={k:4d}  alpha={a:.4f}  ||dW||={n:.5f}  ratio={n/ref:.6f}")
ok &= check("all ||dW_k|| equal to within 1e-6",
            all(abs(r[3] - 1.0) < 1e-6 for r in rows))
ok &= check("alpha_k decreases ~ 1/sqrt(k)",
            all(rows[i][1] > rows[i + 1][1] for i in range(len(rows) - 1)))

# ------------------------------------------------- orthonormality of D
banner("3. Direction basis")
G = np.asarray(bk.transpose(D)) @ np.asarray(D)
offdiag = np.abs(G - np.eye(G.shape[0])).max()
ok &= check(f"D^T D == I  (max deviation {offdiag:.2e})", offdiag < 1e-8)

# ------------------------------------------------- the headline measurement
banner("4. SNR and dither gain vs rank  (INT4 RTN, norm-matched)")
print(f"   {'k':>5} {'alpha':>8} {'gain_pi':>9} {'SNR_sub':>9} {'SNR_glob':>9} {'survive':>8}")
res = []
for k in [1, 2, 4, 8, 16, 64, 256, 512]:
    dW, a = core.build_edit(W, D, k, bk)
    m = core.measure(W, dW, D, k, "int4", bk)
    res.append((k, a, m))
    print(f"   {k:5d} {a:8.4f} {m['gain_pi']:9.4f} {m['snr_subspace']:9.3f} "
          f"{m['snr_global']:9.3f} {m['survive_sub']:8.4f}")

globs = [m["snr_global"] for _, _, m in res]
print(f"\n   note: SNR_glob is ~flat in k ({min(globs):.3f}-{max(globs):.3f}) and < 1 --")
print("         whole-matrix noise swamps a norm-fixed low-rank edit at every rank,")
print("         so the in-subspace ratio is the only one that carries rank information.")
print("   note: survive_sub > 1 and rising means the quantizer does not attenuate the")
print("         edit in-subspace, it contaminates it -- the extra norm is residual, not signal.")

pis = [m["gain_pi"] for _, _, m in res]
snrs = [m["snr_subspace"] for _, _, m in res]
ok &= check("gain_pi stays within 2% of 1.0 at every rank  (dithering holds)",
            all(abs(p - 1.0) < 0.02 for p in pis),
            f"range [{min(pis):.4f}, {max(pis):.4f}]")
ok &= check("in-subspace SNR is monotonically decreasing in k",
            all(snrs[i] > snrs[i + 1] for i in range(len(snrs) - 1)),
            f"{snrs[0]:.2f} -> {snrs[-1]:.2f}")
ok &= check("SNR falls by >10x from k=1 to k=512", snrs[0] / snrs[-1] > 10)

# ------------------------------------------------- the sqrt(d/k) prediction
banner("5. Against the analytic prediction SNR ~ (||dW||/||rho||)*sqrt(d/k)")
base = snrs[0]
print(f"   {'k':>5} {'measured':>10} {'predicted':>10} {'ratio':>8}")
for (k, _, m), s in zip(res, snrs):
    pred = base / np.sqrt(k)
    print(f"   {k:5d} {s:10.3f} {pred:10.3f} {s/pred:8.3f}")
ok &= check("measured within 1.6x of sqrt(1/k) scaling at every rank",
            all(0.6 < s / (base / np.sqrt(k)) < 1.6 for (k, _, _), s in zip(res, snrs)))

# ------------------------------------------------- cross-scheme
banner("6. All schemes, k in {1, 64}")
print(f"   {'scheme':>7} {'k':>4} {'gain_pi':>9} {'SNR_sub':>9}")
for scheme in ["int4", "nf4", "fp8", "int8"]:
    for k in [1, 64]:
        dW, _ = core.build_edit(W, D, k, bk)
        m = core.measure(W, dW, D, k, scheme, bk)
        print(f"   {scheme:>7} {k:4d} {m['gain_pi']:9.4f} {m['snr_subspace']:9.3f}")

banner("RESULT")
print("  ALL CHECKS PASSED" if ok else "  SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
