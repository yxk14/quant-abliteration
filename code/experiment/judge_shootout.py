#!/usr/bin/env python3
"""Evaluate candidate judge models against the primary judge on a balanced set.

Motivation. The primary judge (gemma-4-12b-it) is validated against hand
annotation, but that annotation is a single author's and is not large. The
obvious strengthening move is a second, independently trained judge. This script
runs candidates over a balanced sample and reports, per candidate:

    p_refusal  -- fraction of all verdicts that are REFUSAL. A judge that
                  returns REFUSAL for everything scores well on a refusal-heavy
                  corpus while carrying no information; this column exposes it.
    agreement  -- raw agreement with the primary judge
    kappa      -- Cohen's kappa, which raw agreement flatters badly here

The sample is balanced 50/50 between the primary judge's COMPLY and REFUSAL
verdicts precisely so that a degenerate always-REFUSAL judge is capped at 50%
rather than at the corpus base rate. This is the constructed-control idea
applied to judge selection.

NOTE: this measures agreement with the primary judge, NOT accuracy. It cannot
establish that the primary judge is right. It can, and does, establish that a
candidate is degenerate.
"""
import argparse, csv, gc, json, pathlib, random, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import metrics as M
from judge_refusals import RUBRIC

RESP = pathlib.Path(__file__).parent.parent / "analysis" / "responses"
DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"

CANDIDATES = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "microsoft/Phi-4-mini-instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "ibm-granite/granite-3.3-8b-instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]


def human_labelled():
    """The items with blind human labels, with the language corrections applied.

    Scoring candidates against the primary judge only establishes that they
    disagree with it. Scoring against human labels establishes whether they are
    wrong, which is the question. The set is refusal-heavy and stratified toward
    disagreement cases, so the trivial-baseline column matters more than usual
    and is reported alongside.
    """
    DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"
    labs = {r["item_id"]: r for r in csv.DictReader(open(DATA / "human_labels.csv"))}
    rev = DATA / "human_labels_revised.csv"
    if rev.exists():
        for r in csv.DictReader(open(rev)):
            if r["item_id"] in labs:
                labs[r["item_id"]]["human"] = r["human_after"]
    cache, out = {}, []
    for r in labs.values():
        k = (r["family"], r["config"])
        if k not in cache:
            cache[k] = json.loads((RESP / f"{k[0]}__{k[1]}.judged.json").read_text())["items"]
        it = cache[k][int(r["idx"])]
        out.append((r["human"], dict(prompt=it.get("prompt", ""), text=it["text"],
                                     judge=r["judge"],
                                     keyword_v1=int(r["keyword_v1"]))))
    return out


def sample(n_per_class, seed=0):
    """Balanced draw across many model/config files, so no single generator
    dominates and the two classes are equally represented."""
    pools = {"COMPLY": [], "REFUSAL": []}
    for f in sorted(RESP.glob("*.judged.json")):
        if "judged2" in f.name:
            continue
        for it in json.loads(f.read_text())["items"]:
            if it["judge"] in pools and 30 < len(it["text"]) < 1200:
                pools[it["judge"]].append(it)
    rng = random.Random(seed)
    out = []
    for k, v in pools.items():
        rng.shuffle(v)
        out += [(k, x) for x in v[:n_per_class]]
    rng.shuffle(out)
    return out


@torch.no_grad()
def run(model_id, items, device, batch=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device).eval()
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        msgs = [M.chat_prompt(tok, RUBRIC.format(prompt=c["prompt"][:400],
                                                 response=c["text"][:1200] or "(empty)"))
                for _, c in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True,
                  truncation=True, max_length=1600).to(device)
        g = m.generate(**enc, max_new_tokens=6, do_sample=False,
                       pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            t = tok.decode(g[j][enc["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip().upper()
            out.append("REFUSAL" if "REFUS" in t else
                       "COMPLY" if "COMPL" in t else
                       "GARBLED" if "GARBL" in t else "UNCLEAR")
    del m; gc.collect(); torch.cuda.empty_cache()
    return out


def kappa(a, b):
    labs = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l)/n) * (b.count(l)/n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="judge_shootout.csv")
    ap.add_argument("--vs-human", action="store_true",
                    help="score candidates against blind human labels instead "
                         "of against the primary judge")
    a = ap.parse_args()

    items = human_labelled() if a.vs_human else sample(a.n_per_class)
    ref = [k for k, _ in items]
    import collections as _c
    base = 100*_c.Counter(ref).most_common(1)[0][1]/len(ref)
    kind = "blind human labels" if a.vs_human else "the primary judge"
    print(f"{len(items)} items, scored against {kind}")
    print(f"  trivial baseline (always the majority label): {base:.1f}%")
    rows = []
    if a.vs_human:
        # the primary judge's stored verdicts cost nothing to score
        jv = [c["judge"] for _, c in items]
        print(f"  {'gemma-4-12b-it (primary, stored)':38s} "
              f"p_ref {100*jv.count('REFUSAL')/len(jv):5.1f}%  "
              f"agree {100*sum(x==y for x,y in zip(ref,jv))/len(jv):5.1f}%  "
              f"kappa {kappa(ref,jv):+.3f}")
        rows.append(("google/gemma-4-12b-it (primary)",
                     100*jv.count("REFUSAL")/len(jv),
                     100*sum(x==y for x,y in zip(ref,jv))/len(jv), kappa(ref,jv)))
        kb = ["REFUSAL" if c["keyword_v1"] else "COMPLY" for _, c in items]
        rb = ["REFUSAL" if r=="REFUSAL" else "COMPLY" for r in ref]
        print(f"  {'keyword substring (binarised)':38s} "
              f"p_ref {100*kb.count('REFUSAL')/len(kb):5.1f}%  "
              f"agree {100*sum(x==y for x,y in zip(rb,kb))/len(kb):5.1f}%  "
              f"kappa {kappa(rb,kb):+.3f}")
        rows.append(("keyword substring (binarised)",
                     100*kb.count("REFUSAL")/len(kb),
                     100*sum(x==y for x,y in zip(rb,kb))/len(kb), kappa(rb,kb)))
    for mid in CANDIDATES:
        try:
            v = run(mid, items, a.device)
        except Exception as e:
            print(f"  {mid}: FAILED {type(e).__name__}: {str(e)[:70]}")
            continue
        agree = 100*sum(x == y for x, y in zip(ref, v))/len(v)
        pref = 100*v.count("REFUSAL")/len(v)
        k = kappa(ref, v)
        rows.append((mid, pref, agree, k))
        print(f"  {mid:38s} p_ref {pref:5.1f}%  agree {agree:5.1f}%  kappa {k:+.3f}")

    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["judge", "pct_refusal", "agreement_pct", "kappa", "n"])
        for mid, pref, ag, k in rows:
            w.writerow([mid, round(pref, 2), round(ag, 2), round(k, 4), len(items)])
    print(f"wrote {DATA / a.out}")


if __name__ == "__main__":
    main()
