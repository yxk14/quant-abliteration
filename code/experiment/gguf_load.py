#!/usr/bin/env python3
"""Load a GGUF k-quant back into a HuggingFace model as dense weights.

Why not just generate with llama.cpp. Every other arm of this study runs through
transformers with identical prompting, sampling and chat templating. Generating
the k-quant arm inside llama.cpp would change the inference stack at the same
time as the quantizer, confounding the comparison. Instead we take the same path
used for GPTQ/AWQ (`run_compressed=False`): read the quantized weights, expand
them to dense bf16, and run the existing pipeline over them. The numerics are
the k-quant's; the stack is ours.

What Q4_K_M actually is, and why it is the interesting case. It is not uniform
round-to-nearest and it is not calibrated. Weights are stored in super-blocks
with per-block scales and minima, and precision is mixed ACROSS tensors: in the
Qwen checkpoint, 169 tensors are Q4_K, 29 are Q6_K, and 141 stay F32. So it sits
between the two extremes this study contrasts -- structured error, but chosen
without seeing any activations. It is also what most local deployment runs.

    python experiment/gguf_load.py --gguf X.gguf --ref Qwen/Qwen2.5-7B-Instruct \
        --save ckpts/qwen25-7b-abl-Q4KM
"""
import argparse, gc, pathlib, re, sys
import torch

# GGUF names -> HF names. Only the mappings this study needs; a tensor whose
# name does not match is reported rather than silently skipped, because a
# silently skipped weight matrix would leave BF16 values in a "quantized" model.
SUFFIX = {
    "attn_q.weight":      "self_attn.q_proj.weight",
    "attn_k.weight":      "self_attn.k_proj.weight",
    "attn_v.weight":      "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q.bias":        "self_attn.q_proj.bias",
    "attn_k.bias":        "self_attn.k_proj.bias",
    "attn_v.bias":        "self_attn.v_proj.bias",
    "ffn_gate.weight":    "mlp.gate_proj.weight",
    "ffn_up.weight":      "mlp.up_proj.weight",
    "ffn_down.weight":    "mlp.down_proj.weight",
    "attn_norm.weight":   "input_layernorm.weight",
    "ffn_norm.weight":    "post_attention_layernorm.weight",
}
# GGUF-only tensors with no HF counterpart. llama.cpp materialises the RoPE
# frequency table into the file; transformers computes it at runtime from
# config. Ignoring it is correct -- but it must be an explicit allowlist, not a
# silent skip, or a genuinely missing weight matrix would slip through too.
IGNORE = {"rope_freqs.weight"}

GLOBAL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def unpermute(w, n_head):
    """Invert llama.cpp's Q/K permutation.

    convert_hf_to_gguf.py reorders the rotary halves of q_proj and k_proj for
    LLaMA-family models so they match llama.cpp's RoPE implementation. The
    permutation preserves tensor SHAPE, so a shape check does not catch it; the
    model loads cleanly and produces subtly broken text. Qwen2 uses a different
    convention and is not permuted, which is why only the Llama checkpoints were
    affected.
    """
    return (w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:])
             .swapaxes(1, 2)
             .reshape(w.shape))


def gguf_to_hf(name):
    if name in GLOBAL:
        return GLOBAL[name]
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    if not m:
        return None
    layer, suf = m.group(1), m.group(2)
    hf = SUFFIX.get(suf)
    return f"model.layers.{layer}.{hf}" if hf else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", required=True, help="HF model whose architecture to fill")
    ap.add_argument("--save", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import gguf
    from gguf import quants
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"reading {a.gguf}")
    r = gguf.GGUFReader(a.gguf)
    print(f"  {len(r.tensors)} tensors")

    print(f"loading reference architecture {a.ref}")
    model = AutoModelForCausalLM.from_pretrained(
        a.ref, torch_dtype=torch.bfloat16, device_map=a.device)
    sd = model.state_dict()

    cfg = model.config
    arch = (cfg.architectures or [""])[0]
    needs_unpermute = "Llama" in arch or "Mistral" in arch
    n_head = getattr(cfg, "num_attention_heads", 0)
    n_kv_head = getattr(cfg, "num_key_value_heads", n_head)
    print(f"  arch={arch} unpermute_qk={needs_unpermute} "
          f"heads={n_head} kv_heads={n_kv_head}")

    loaded, skipped, missing, mismatch = 0, [], [], []
    for t in r.tensors:
        if t.name in IGNORE:
            continue
        hf = gguf_to_hf(t.name)
        if hf is None:
            skipped.append(t.name); continue
        if hf not in sd:
            missing.append(hf); continue
        # GGUF stores shapes reversed relative to torch
        want = tuple(sd[hf].shape)
        arr = quants.dequantize(t.data, t.tensor_type)
        arr = arr.reshape(tuple(int(x) for x in reversed(t.shape)))
        if arr.shape != want:
            mismatch.append((hf, arr.shape, want)); continue
        if needs_unpermute:
            if hf.endswith("self_attn.q_proj.weight"):
                arr = unpermute(arr, n_head)
            elif hf.endswith("self_attn.k_proj.weight"):
                arr = unpermute(arr, n_kv_head)
        sd[hf].copy_(torch.from_numpy(arr).to(sd[hf].dtype))
        loaded += 1
        del arr

    # A tied lm_head is normal and not a missing weight.
    tied = getattr(model.config, "tie_word_embeddings", False)
    missing = [m for m in missing if not (tied and m == "lm_head.weight")]

    print(f"  loaded {loaded} tensors")
    if skipped:   print(f"  UNMAPPED gguf names ({len(skipped)}): {skipped[:5]}")
    if missing:   print(f"  MISSING in hf model ({len(missing)}): {missing[:5]}")
    if mismatch:  print(f"  SHAPE MISMATCH ({len(mismatch)}): {mismatch[:3]}")
    if skipped or missing or mismatch:
        raise SystemExit("refusing to save: some tensors did not round-trip, which "
                         "would leave un-quantized values in a 'quantized' model")

    pathlib.Path(a.save).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.save)
    AutoTokenizer.from_pretrained(a.ref).save_pretrained(a.save)
    print(f"saved {a.save}")


if __name__ == "__main__":
    main()
