#!/usr/bin/env python3
"""Score the 132 reference labels with a purpose-built refusal classifier.

Stand-in for WildGuard, which is gated and whose terms this account has not
accepted. protectai/distilroberta-base-rejection-v1 is trained for exactly the
question the shootout asks -- is this LLM output a rejection -- and is not
gated, so it tests the reviewer's point (that six general instruct models
collapsing says more about the candidates than the task) with a model built for
the task rather than prompted into it.

Same 132 items and same kappa as every other row of the shootout table.

    python experiment/rejection_judge.py
"""
import argparse, json, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wildguard_judge import refs, kappa

ROOT = pathlib.Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protectai/distilroberta-base-rejection-v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="analysis/data/rejection_refs.json")
    a = ap.parse_args()

    items = refs()
    print(f"{len(items)} reference-labelled items")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForSequenceClassification.from_pretrained(a.model).to(a.device).eval()
    id2l = m.config.id2label
    print("label map:", id2l)

    pred = []
    with torch.no_grad():
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            enc = tok([c["text"][:2000] or "(empty)" for c in chunk],
                      return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(a.device)
            lab = m(**enc).logits.argmax(-1).tolist()
            for l in lab:
                name = str(id2l[l]).upper()
                pred.append("REFUSAL" if "REJECT" in name else "NOT_REFUSAL")

    gold = ["REFUSAL" if it["human"] == "REFUSAL" else "NOT_REFUSAL" for it in items]
    agree = 100 * sum(p == g for p, g in zip(pred, gold)) / len(gold)
    k = kappa(pred, gold)
    base = 100 * sum(g == "REFUSAL" for g in gold) / len(gold)
    print(f"\n{a.model}")
    print(f"  agreement with human labels : {agree:.1f}%")
    print(f"  Cohen's kappa               : {k:.3f}")
    print(f"  always-REFUSAL baseline     : {base:.1f}%")
    (ROOT / a.out).write_text(json.dumps(
        dict(model=a.model, n=len(gold), agreement=agree, kappa=k,
             always_refusal=base, pred=pred, gold=gold), indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
