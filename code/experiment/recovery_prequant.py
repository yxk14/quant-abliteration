#!/usr/bin/env python3
"""Direction recovery for ALREADY-quantized checkpoints (GPTQ / AWQ).

direction_recovery.py applies a simulated quantizer to a model in memory. It
cannot score GPTQ or AWQ, whose weights are produced by a calibration pass and
arrive on disk. This does the same measurement on such checkpoints.

    R    = ||r^T W_cfg|| / ||r^T W_base||     (per edited layer, averaged)
    N^2  = R_cfg^2 - R_bf16^2                 (absorbed noise, baseline-free)

Why it matters. Behaviourally, calibrated 4-bit quantization leaves the
abliterated Qwen checkpoint unmoved (GPTQ -0.5 pts, AWQ +0.0) while naive INT4
round-to-nearest moves it +9.5. The hole-filling account says the only thing
that should matter is the MAGNITUDE of the noise along the ablated direction.
If that is right, N_r for GPTQ/AWQ must be far below the 0.206 that RTN INT4
injects. This measures it.
"""
import argparse, csv, gc, math, pathlib, sys
import torch

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"
TARGET = "o_proj.weight"


def layers(model):
    for n, p in model.named_parameters():
        if n.endswith(TARGET) and p.ndim == 2:
            yield int([s for s in n.split(".") if s.isdigit()][0]), p.data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rdir", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="LABEL=path pairs; the BF16 abliterated one must come first")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from transformers.utils.quantization_config import CompressedTensorsConfig

    # A compressed-tensors checkpoint keeps weight_packed/weight_scale on load
    # and decompresses inside a forward hook, so named_parameters() exposes no
    # dense o_proj.weight at all. run_compressed=False materialises the
    # dequantised weights, which is what we need to measure.
    DEQ = CompressedTensorsConfig(run_compressed=False)

    def is_compressed(path):
        """Only compressed-tensors checkpoints (GPTQ/AWQ from llmcompressor) need
        the decompressing loader. A dequantized GGUF is a plain dense checkpoint
        and passing the config to it raises."""
        import json, os
        cf = os.path.join(path, "config.json")
        if not os.path.exists(cf):
            return False
        try:
            return "quantization_config" in json.load(open(cf))
        except Exception:
            return False

    def load(path, quantized=None):
        kw = dict(torch_dtype=torch.bfloat16, device_map=a.device)
        if is_compressed(path):
            kw["quantization_config"] = DEQ
        return AutoModelForCausalLM.from_pretrained(path, **kw).eval()

    R = torch.load(a.rdir, map_location=a.device)

    print(f"base: {a.base}")
    mb = load(a.base)
    denom = {}
    with torch.no_grad():
        for L, W in layers(mb):
            r = R[L].float()
            if r.norm() < 1e-8:
                continue
            r = r / r.norm()
            denom[L] = float((r @ W.float()).norm())
    del mb; gc.collect(); torch.cuda.empty_cache()
    print(f"  {len(denom)} edited layers")

    rows = []
    for spec in a.ckpt:
        lab, path = spec.split("=", 1)
        print(f"{lab}: {path}")
        m = load(path)
        tot, n = 0.0, 0
        with torch.no_grad():
            for L, W in layers(m):
                if L not in denom:
                    continue
                r = R[L].float(); r = r / r.norm()
                tot += float((r @ W.float()).norm()) / denom[L]; n += 1
        if n == 0:
            raise SystemExit(f"{lab}: matched 0 '{TARGET}' parameters in {path} -- "
                             "the checkpoint is probably still compressed")
        rows.append((lab, tot / n))
        print(f"  R = {tot/n:.5f}")
        del m; gc.collect(); torch.cuda.empty_cache()

    r0 = rows[0][1]
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / a.out
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "recovery_along_r", "noise_along_r"])
        for lab, rr in rows:
            w.writerow([lab, round(rr, 5), round(math.sqrt(max(rr*rr - r0*r0, 0)), 5)])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
