#!/usr/bin/env python3
"""Why a low-rank edit survives quantization and a diffuse one does not.

An earlier version of this analysis claimed that rounding preserves a *coherent*
edit. That claim is wrong, and this script is what shows it. Preservation in
expectation is a property of rounding itself, not of the edit: for a weight
uniformly placed inside its bin, a sub-step displacement d changes the quantized
value by one full step with probability d/Delta, so E[Q(w+d) - Q(w)] = d for any
d below the step. Every edit therefore comes through with unit gain, coherent or
not, which the control rows below confirm.

What separates abliteration from a diffuse edit is where the leftover lands.
Rounding returns the edit plus a bin-flip residue of roughly four times the
edit's norm, spread over all d directions of the row space. Behaviour after a
rank-one ablation depends on one direction, so the residue is suppressed in the
subspace that matters by about sqrt(d). A diffuse edit is read out over the
whole space and gets no such suppression.

Run on a real edited matrix plus synthetic edits of the same magnitude at
several ranks:

    python experiment/rank_vs_noise.py
"""
import argparse, glob, json, pathlib
import torch
from safetensors import safe_open

ROOT = pathlib.Path(__file__).parent.parent
QMAX = 7


def shards(model):
    p = pathlib.Path(model)
    if p.exists():
        return sorted(glob.glob(str(p / "*.safetensors")))
    d = pathlib.Path.home() / ".cache/huggingface/hub" / ("models--" + model.replace("/", "--")) / "snapshots"
    return sorted(glob.glob(str(d / "*" / "*.safetensors")))


def get(model, key):
    for f in shards(model):
        with safe_open(f, framework="pt") as h:
            if key in h.keys():
                return h.get_tensor(key).to(torch.float32)
    raise KeyError(key)


def quant(W):
    s = (W.abs().amax(dim=1, keepdim=True) / QMAX).clamp(min=1e-12)
    return torch.round(W / s).clamp(-QMAX - 1, QMAX) * s


def measure(W, D, k):
    """proj = gain with which the edit is reproduced; snr = edit norm over the
    residue's norm inside the edit's own k-dimensional column subspace."""
    qd = quant(W + D) - quant(W)
    proj = ((qd * D).sum() / D.pow(2).sum()).item()
    cos = ((qd * D).sum() / (qd.norm() * D.norm())).item()
    res = qd - D
    U = torch.linalg.svd(D, full_matrices=False)[0][:, :k]
    return proj, cos, (qd.norm() / D.norm()).item(), (D.norm() / (U.T @ res).norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl", default="ckpts/qwen25-7b-abl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--key", default="model.layers.18.self_attn.o_proj.weight")
    ap.add_argument("--out", default="analysis/data/rank_vs_noise.json")
    a = ap.parse_args()

    Wb, Wa = get(a.base, a.key), get(a.abl, a.key)
    D_real = Wa - Wb
    d = Wb.shape[0]
    mag = (D_real.norm() / Wb.norm()).item()
    print(f"{a.key}  d={d}  real edit = {100*mag:.2f}% of ||W||_F\n")

    rows = []
    p, c, n, s = measure(Wb, D_real, 1)
    rows.append(dict(edit="real ablation edit", rank=1, proj=p, cos=c, noise=n, snr=s))
    g = torch.Generator().manual_seed(0)
    for k in (1, 8, 64, 512, d):
        if k == d:
            D = torch.randn(Wb.shape, generator=g)
        else:
            D = torch.randn(d, k, generator=g) @ torch.randn(k, Wb.shape[1], generator=g)
        D = D / D.norm() * mag * Wb.norm()
        p, c, n, s = measure(Wb, D, k)
        rows.append(dict(edit=f"synthetic rank-{k}" if k < d else "synthetic dense",
                         rank=k, proj=p, cos=c, noise=n, snr=s))

    print(f"{'edit':24s} {'rank':>6s} {'proj':>7s} {'residue/edit':>13s} {'SNR in subspace':>16s}")
    for r in rows:
        print(f"  {r['edit']:22s} {r['rank']:6d} {r['proj']:7.3f} {r['noise']:13.2f} {r['snr']:16.2f}")
    pathlib.Path(a.out).write_text(json.dumps(dict(key=a.key, d=d, mag=mag, rows=rows), indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
