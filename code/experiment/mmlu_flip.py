#!/usr/bin/env python3
"""MMLU answer-flip rate: the correct analogue of the refusal decision-flip rate.

Comparing an MMLU *accuracy* delta against a refusal *decision-flip* rate is not
apples to apples -- accuracy can hold steady while many individual answers
change, because wrong-to-right and right-to-wrong cancel. To ask whether
quantization disturbs safety decisions more than ordinary knowledge decisions,
both sides must be measured the same way: what fraction of per-item decisions
changed relative to the BF16 reference.

FP16 doubles as the noise floor. It is near-lossless, so whatever flip rate it
shows is measurement noise (bf16 batching is not bit-exact across batch
composition), and the INT4 flip rate should be read against it.

    python experiment/mmlu_flip.py --model Qwen/Qwen2.5-7B-Instruct \
        --family Qwen2.5-7B-Instruct --limit 20 --batch 4
"""
import argparse, csv, itertools, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import mmlu_direct as MD
from stages_eval import quantize_model_, restore_

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"


@torch.no_grad()
def predictions(scorer, ds_all, subs, limit):
    """Per-question predicted letter index, in a fixed order across configs."""
    import torch.nn.functional as F
    out, gold = [], []
    for sub in subs:
        dev = [r for r in ds_all["dev"] if r["subject"] == sub]
        test = [r for r in ds_all["test"] if r["subject"] == sub]
        if limit:
            test = test[:limit]
        if not test:
            continue
        prefix = MD.HEADER.format(sub.replace("_", " "))
        prefix += "".join(MD.fmt_q(r, True) for r in dev[:5])
        qs = [prefix + MD.fmt_q(r, False) for r in test]
        # NOT length-sorted: a fixed order keeps batch composition identical for
        # every config, so a flip means the weights changed, not the batching.
        bs = scorer.bs
        i = 0
        while i < len(qs):
            try:
                lg = scorer._batch_logits(qs[i:i + bs])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                continue
            out += F.log_softmax(lg.float(), -1)[:, scorer.lids].argmax(-1).tolist()
            i += lg.shape[0]
        gold += [r["answer"] for r in test]
    return out, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--formats", nargs="+", default=["FP", "INT"])
    ap.add_argument("--widths", nargs="+", type=int, default=[16, 8, 4])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device).eval()
    scorer = MD.DirectMMLU(m, tok, a.batch, use_prefix_cache=False, device=a.device)

    ds_all = load_dataset("cais/mmlu", "all")
    subs = sorted(set(ds_all["test"]["subject"]))

    ref, gold = predictions(scorer, ds_all, subs, a.limit)
    n = len(ref)
    acc0 = sum(p == g for p, g in zip(ref, gold)) / n * 100
    print(f"BF16 reference: {n} questions, acc {acc0:.2f}%")

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"mmlu_flip_{a.family}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "config", "n", "acc", "acc_kept", "mmlu_flip_rate"])
        w.writerow([a.family, "BF16", n, round(acc0, 2), 100.0, 0.0])
        for fmt, width in itertools.product(a.formats, a.widths):
            saved = quantize_model_(m, fmt, width)
            pred, _ = predictions(scorer, ds_all, subs, a.limit)
            restore_(m, saved); del saved
            torch.cuda.empty_cache()
            acc = sum(p == g for p, g in zip(pred, gold)) / n * 100
            flip = sum(p != r for p, r in zip(pred, ref)) / n * 100
            w.writerow([a.family, f"{fmt}{width}", n, round(acc, 2),
                        round(100 * acc / acc0, 2), round(flip, 2)])
            fh.flush()
            print(f"  {fmt}{width:<3d} acc {acc:6.2f}%  kept {100*acc/acc0:6.2f}%  "
                  f"answers flipped {flip:5.2f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
