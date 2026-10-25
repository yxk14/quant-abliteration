#!/usr/bin/env python3
"""Does the quantization step erase an abliteration edit? (Zhang et al., ICLR 2025)

Zhang et al. explain catastrophic failure of unlearning under quantization with a
per-weight argument: utility-constrained unlearning moves each weight by less
than the quantization step, so the unlearned and original weights land in the
same bin and Q(f_unlearn) ~= Q(f_target). Their Section 4.4 reports that GPTQ
and AWQ behave like round-to-nearest here, i.e. calibration does not protect the
edit.

Abliteration moves the weights by ~2% in Frobenius norm, far below a 4-bit step,
so that argument predicts abliteration should also be erased by 4-bit
quantization. Behaviourally it is not. This script measures why, per edited
tensor:

  frac_moved   fraction of weights whose INT4 index actually changes
  survival     ||Q(W_abl) - Q(W_base)||_F / ||W_abl - W_base||_F, i.e. how much
               of the edit is still present after both models are quantized
               (0 = the edit is erased, 1 = fully preserved)

INT4 here is exactly the study's quantizer: symmetric, per output channel,
qmax = 7, scale = amax/qmax.

    python experiment/step_vs_edit.py --abl ckpts/qwen25-7b-abl \
        --base Qwen/Qwen2.5-7B-Instruct
"""
import argparse, glob, json, pathlib, sys
import torch
from safetensors import safe_open

ROOT = pathlib.Path(__file__).parent.parent
QMAX = 7


def shards(model):
    """Local dir, or the HF cache snapshot. snapshot_download refuses a cache
    that is missing README/LICENSE even when every weight shard is present, so
    glob the snapshot directory instead of asking the hub."""
    p = pathlib.Path(model)
    if p.exists():
        return sorted(glob.glob(str(p / "*.safetensors")))
    cache = pathlib.Path.home() / ".cache/huggingface/hub"
    d = cache / ("models--" + model.replace("/", "--")) / "snapshots"
    f = sorted(glob.glob(str(d / "*" / "*.safetensors")))
    if not f:
        raise SystemExit(f"no safetensors for {model} under {d}")
    return f


def index(model):
    out = {}
    for f in shards(model):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                out[k] = f
    return out


def q_int4(W):
    amax = W.abs().amax(dim=tuple(range(1, W.ndim)), keepdim=True)
    s = (amax / QMAX).clamp(min=1e-12)
    idx = torch.round(W / s).clamp(-QMAX - 1, QMAX)
    return idx, (idx * s).to(W.dtype), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--targets", nargs="+", default=["o_proj", "down_proj"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ia, ib = index(a.abl), index(a.base)
    keys = [k for k in sorted(ia) if k in ib and any(t in k for t in a.targets)
            and k.endswith(".weight")]
    print(f"{len(keys)} edited tensors")

    rows, tot_edit, tot_surv, tot_moved, tot_n = [], 0.0, 0.0, 0, 0
    for k in keys:
        with safe_open(ia[k], framework="pt") as h:
            Wa = h.get_tensor(k).to(torch.float32)
        with safe_open(ib[k], framework="pt") as h:
            Wb = h.get_tensor(k).to(torch.float32)
        d = Wa - Wb
        nd = d.norm().item()
        if nd == 0:
            continue
        idx_a, qa, sa = q_int4(Wa)
        idx_b, qb, sb = q_int4(Wb)
        moved = (idx_a != idx_b).sum().item()
        n = d.numel()
        surv = (qa - qb).norm().item() / nd
        # Is the surviving difference the EDIT, or just bin-flip noise?
        # proj = <Q(Wa)-Q(Wb), dW>/||dW||^2 : 1.0 means the sub-step edit is
        # reproduced exactly in expectation, by a sparse set of full-step flips.
        qd = qa - qb
        proj = torch.sum(qd * d).item() / (nd ** 2)
        cos = torch.sum(qd * d).item() / (qd.norm().item() * nd + 1e-12)
        # how large is the edit compared with the step it would have to cross
        ratio = (d.abs() / sb).flatten()
        # torch.quantile caps out around 16M elements; sample instead
        samp = ratio[torch.randperm(ratio.numel())[:1_000_000]]
        rows.append(dict(name=k, rel_edit=nd / Wb.norm().item(),
                         frac_moved=moved / n, survival=surv,
                         med_ratio=samp.median().item(),
                         p99_ratio=samp.quantile(.99).item(),
                         proj=proj, cos=cos))
        tot_edit += nd ** 2; tot_surv += (qa - qb).norm().item() ** 2
        tot_moved += moved; tot_n += n
        print(f"  {k:46s} rel_edit={rows[-1]['rel_edit']*100:5.2f}%  "
              f"moved={100*moved/n:5.1f}%  survival={surv:.3f}  "
              f"proj={proj:5.2f} cos={cos:.3f}")

    print(f"\nOVERALL  edit survival after INT4 on both models: "
          f"{(tot_surv/tot_edit)**.5:.3f}")
    print(f"         weights whose INT4 index changes: {100*tot_moved/tot_n:.2f}%")
    med = sorted(r["med_ratio"] for r in rows)[len(rows)//2]
    print(f"         median |dW| as a fraction of the quantization step: {med:.3f}")
    pr = sorted(r["proj"] for r in rows); cs = sorted(r["cos"] for r in rows)
    print(f"         projection of the quantized difference onto the edit: "
          f"median {pr[len(pr)//2]:.3f}  (1.0 = edit reproduced exactly)")
    print(f"         cosine(quantized difference, edit): median {cs[len(cs)//2]:.3f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
