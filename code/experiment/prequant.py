#!/usr/bin/env python3
"""Produce GPTQ and AWQ 4-bit checkpoints with llmcompressor.

Why this arm exists. Every quantization result in this paper so far uses
simulated round-trip quantize-dequantize: uniform integer RTN, NF4, FP8. Those
are DATA-FREE. GPTQ and AWQ are CALIBRATED -- they see activation statistics
from a sample corpus and deliberately protect the weights that matter most for
output fidelity. They are also what people actually deploy.

That difference is the whole point. Our mechanism is "hole-filling": generic,
direction-agnostic quantization noise partially refills the rank-one hole
abliteration cut, and refusal partially returns. A calibrated quantizer chooses
its error deliberately rather than isotropically, so it could plausibly

  (a) refill the hole MORE, if the ablated direction sits along channels the
      calibration objective happens to protect, or
  (b) refill it LESS, because lower total error means less noise of any kind,
      or
  (c) behave exactly like RTN, if the hole-filling account is right that only
      the noise MAGNITUDE matters and not its orientation.

(c) is what our N_r/N_s < 1 surrogate result predicts. This script generates the
checkpoints needed to test it.

Calibration uses benign instruction data (ultrachat), i.e. the realistic
deployment setting: nobody calibrates a quantizer on harmful prompts.

    python experiment/prequant.py --model ckpts/qwen25-7b-abl \
        --out ckpts/qwen25-7b-abl-GPTQ4 --scheme GPTQ
"""
import argparse, pathlib, time

CALIB = "HuggingFaceH4/ultrachat_200k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scheme", choices=["GPTQ", "AWQ"], required=True)
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=1024)
    # AWQ caches activations across a whole subgraph for its scale search
    # and needs far more headroom than GPTQ. On a 32 GB card the whole
    # model resident OOMs, so offload and let the sequential pipeline
    # onload one layer at a time.
    ap.add_argument("--device-map", default="cuda")
    a = ap.parse_args()

    if pathlib.Path(a.out).exists():
        print(f"{a.out} exists, skipping"); return

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.awq import AWQModifier

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device_map)

    ds = load_dataset(CALIB, split=f"train_sft[:{a.nsamples}]")
    def prep(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok(text, padding=False, max_length=a.seqlen, truncation=True,
                   add_special_tokens=False)
    ds = ds.map(prep, remove_columns=ds.column_names)

    # W4A16: 4-bit weights, 16-bit activations -- the deployed 4-bit setting,
    # and the one comparable to our simulated INT4/NF4 arms.
    if a.scheme == "GPTQ":
        recipe = GPTQModifier(targets="Linear", scheme="W4A16",
                              ignore=["lm_head"])
    else:
        recipe = AWQModifier(targets="Linear", scheme="W4A16",
                             ignore=["lm_head"])

    t0 = time.time()
    oneshot(model=model, dataset=ds, recipe=recipe,
            max_seq_length=a.seqlen, num_calibration_samples=a.nsamples)
    model.save_pretrained(a.out, save_compressed=True)
    tok.save_pretrained(a.out)
    print(f"saved {a.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
