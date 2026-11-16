#!/usr/bin/env python3
"""E5.4 (optional): per-layer relative weight error and first-token KL for the
deployed quantizers (GPTQ, AWQ, Q4_K_M, loaded dequantized) and the fake-quant
family (INT4 RTN, INT4 g=128, NF4) on the abliterated Qwen2.5-7B. Needs
e5/firsttok/abl__bf16.npy from the stage-0 run."""
import json, sys, pathlib, time, gc
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e5_lib import *
from safetensors import safe_open

OUT = CR / "e5" / "e54.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {}
prompts = battery()
lp_bf = np.load(FT / "abl__bf16.npy").astype(np.float32)
p_bf = np.exp(lp_bf)

def bf16_weights():
    """BF16 abliterated weights, read tensor by tensor from the safetensors shards."""
    d = pathlib.Path(SUBJECTS["abl"]); idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}
    def get(name):
        shard = idx[name]
        if shard not in handles:
            handles[shard] = safe_open(str(d / shard), framework="pt", device="cpu")
        return handles[shard].get_tensor(name)
    return get, idx

get_bf, wmap = bf16_weights()

def layer_errors(named):
    """named: iterable of (name, tensor on any device) for the 2-D linear weights."""
    per = defaultdict(list); allv = []
    for name, W in named:
        kind, layer = classify_param(name)
        if kind not in ("attn", "mlp") or name not in wmap:
            continue
        W0 = get_bf(name).to(W.device, torch.float32)
        e = float((W.float() - W0).norm() / W0.norm().clamp(min=1e-12))
        per[layer].append(e); allv.append(e)
    return dict(per_layer=[float(np.mean(per[l])) for l in sorted(per)], mean=float(np.mean(allv)), n=len(allv))

def first_token(m, tok):
    lp = first_token_logprobs(m, tok, prompts, batch=16)
    kl = (p_bf * (lp_bf - lp)).sum(1)
    cjk = cjk_token_ids(tok); mass = np.exp(lp[:, cjk]).sum(1)
    return dict(kl_mean=float(kl.mean()), kl_median=float(np.median(kl)), kl_p90=float(np.percentile(kl, 90)),
                kl_max=float(kl.max()), cjk_mass_mean=float(mass.mean()), cjk_mass_max=float(mass.max()),
                n_argmax_cjk=int(np.isin(lp.argmax(1), cjk).sum()), kl=[float(x) for x in kl])

def corpus_switch(fam, cfg):
    it = load_cell(fam, cfg)["items"]; return 100 * sum(is_nonlatin(i["text"]) for i in it) / len(it)

# ---- fake-quant family on the BF16 model
need = [k for k in ["INT4 (RTN)", "INT4 (RTN, g=128)", "NF4"] if k not in res]
if need:
    m, tok = load_subject("abl")
    for name, fn, fam_cfg in [("INT4 (RTN)", lambda W, i: quant_int(W, 4), ("Qwen2.5-7B-abl", "INT4")),
                              ("INT4 (RTN, g=128)", lambda W, i: quant_int_group(W, 4, 128), ("Qwen2.5-7B-abl", "INTG4")),
                              ("NF4", lambda W, i: quant_nf4(W), ("Qwen2.5-7B-abl", "FP4"))]:
        if name in res: continue
        saved, st = apply_(m, selector(SPECS["int4_full"]), fn)
        try:
            err = layer_errors((n, p.data) for n, p in m.named_parameters() if p.ndim == 2)
            ft = first_token(m, tok)
        finally:
            restore_(m, saved); del saved; torch.cuda.empty_cache()
        res[name] = dict(weight_error=err, first_token=ft, corpus_switch_pct=corpus_switch(*fam_cfg), source="fake-quant in memory")
        write_json(OUT, res); print(name, "rel err %.4f  KL mean %.4f  switch %.1f%%" % (err["mean"], ft["kl_mean"], res[name]["corpus_switch_pct"]), flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

# ---- deployed quantizers, loaded dequantized
from transformers import AutoModelForCausalLM, AutoTokenizer
for name, ck, fam in [("GPTQ, W4A16", "qwen25-7b-abl-GPTQ4", "Qwen2.5-7B-abl-GPTQ4"),
                      ("AWQ, W4A16", "qwen25-7b-abl-AWQ4", "Qwen2.5-7B-abl-AWQ4"),
                      ("GGUF Q4_K_M", "qwen25-7b-abl-Q4KM", "Qwen2.5-7B-abl-Q4KM")]:
    if name in res: continue
    path = str(ROOT / "ckpts" / ck)
    try:
        kw = {}
        if "Q4KM" not in ck:
            from transformers import CompressedTensorsConfig
            kw["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
        tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda", **kw).eval()
        err = layer_errors((n, p.data) for n, p in m.named_parameters() if p.ndim == 2)
        assert err["n"] == 28 * 7, err["n"]
        ft = first_token(m, tok)
        res[name] = dict(weight_error=err, first_token=ft, corpus_switch_pct=corpus_switch(fam, "BF16"), source=path)
        print(name, "rel err %.4f  KL mean %.4f  switch %.1f%%" % (err["mean"], ft["kl_mean"], res[name]["corpus_switch_pct"]), flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        res[name] = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
        print(name, "NOT LOADABLE:", res[name]["error"], flush=True)
    write_json(OUT, res)
print("E5.4 done")
