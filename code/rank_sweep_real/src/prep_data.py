#!/usr/bin/env python3
"""Build the two prompt files stage 1 needs.

harmful.txt  -- from the same sources the submission used. Either
                  --from hf        : walledai/AdvBench + walledai/HarmBench
                                     (standard) via the datasets library, which
                                     reads your existing HF cache
                  --from <file>    : a local .txt/.csv/.jsonl with a
                                     'prompt'/'goal'/'behavior' column
harmless.txt -- generated here from a deterministic template bank, so no
                download and no network is needed and the set is reproducible.

  python3 prep_data.py --from ../../advbench_harmbench_500.csv --n 512
"""
import argparse
import csv
import json
import os
import random

VERBS = ["Explain", "Describe", "Summarize", "Outline", "Walk me through",
         "Give an overview of", "Compare", "Teach me about", "Break down",
         "Give a short introduction to"]
TOPICS = [
    "how a refrigerator's compressor cycle works", "the causes of the 1929 crash",
    "why leaves change color in autumn", "how to proof bread dough",
    "the structure of a sonnet", "how TCP congestion control works",
    "the rules of cricket", "how vaccines train the immune system",
    "the history of the Silk Road", "how a transformer model attends",
    "why the sky is blue", "the difference between weather and climate",
    "how noise-cancelling headphones work", "the water cycle",
    "how a bill becomes law", "the geology of the Grand Canyon",
    "how sourdough starter stays alive", "the Doppler effect",
    "how GPS satellites keep time", "the Marshall Plan",
    "how photosynthesis converts light to sugar", "the rules of chess endgames",
    "why bridges have expansion joints", "how a bicycle stays upright",
    "the printing press and literacy", "how compilers do register allocation",
    "the migration of monarch butterflies", "how tides work",
    "the Bronze Age collapse", "how a heat pump moves heat uphill",
    "the invention of movable type", "how antibiotics kill bacteria",
    "why sourdough tastes sour", "the physics of a curveball",
    "how DNS resolution works", "the Columbian exchange",
    "how lithium-ion cells degrade", "the formation of limestone caves",
    "how jet engines produce thrust", "the Enlightenment in Scotland",
    "how coral reefs form", "the mathematics of compound interest",
    "how seismographs detect earthquakes", "the domestication of wheat",
    "how a camera's aperture affects depth of field", "the Krebs cycle",
    "how undersea cables are laid", "the history of standardized time zones",
    "how yeast ferments sugar", "the orbital mechanics of a Hohmann transfer",
    "how error-correcting codes recover lost bits", "the Dust Bowl",
    "how a piano action works", "the chemistry of rust",
    "how rainbows form", "the spread of the printing industry in Europe",
    "how refrigerated shipping changed diets", "the life cycle of a star",
    "how pacemakers regulate heartbeat", "the invention of the stirrup",
    "how desalination plants work", "the role of mycorrhizal fungi",
    "how suspension bridges carry load", "the Antikythera mechanism",
]
FRAMES = ["{v} {t}.", "{v} {t} for a beginner.",
          "{v} {t} in a few sentences.", "I'm curious: {v_l} {t}.",
          "For a class I'm teaching, {v_l} {t}."]


def load_hf():
    """AdvBench + HarmBench(standard) from the HF cache, deduplicated,
    AdvBench first. Same two sources as the submission's prompt set."""
    from datasets import load_dataset
    out, seen = [], set()

    def add(ds, col):
        for row in ds:
            p = str(row[col]).strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)

    adv = load_dataset("walledai/AdvBench", split="train")
    add(adv, "prompt")
    n_adv = len(out)
    try:
        hb = load_dataset("walledai/HarmBench", "standard", split="train")
    except Exception:
        hb = load_dataset("walledai/HarmBench", split="train")
    add(hb, "prompt")
    print(f"  AdvBench: {n_adv}  HarmBench: {len(out) - n_adv}  total: {len(out)}")
    return out


def load_any(path):
    if path.lower() == "hf":
        return load_hf()
    ext = os.path.splitext(path)[1].lower()
    out = []
    if ext in (".csv", ".tsv"):
        with open(path, newline="") as f:
            rd = csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ",")
            key = None
            for cand in ("prompt", "goal", "behavior", "Behavior", "Goal", "text"):
                if rd.fieldnames and cand in rd.fieldnames:
                    key = cand
                    break
            if key is None:
                raise SystemExit(f"no prompt column in {path}; saw {rd.fieldnames}")
            out = [r[key].strip() for r in rd if r.get(key, "").strip()]
    elif ext in (".jsonl", ".json"):
        with open(path) as f:
            txt = f.read().strip()
        items = (json.loads(txt) if txt.startswith("[")
                 else [json.loads(l) for l in txt.splitlines() if l.strip()])
        for it in items:
            if isinstance(it, str):
                out.append(it.strip())
            else:
                for cand in ("prompt", "goal", "behavior", "text"):
                    if cand in it:
                        out.append(str(it[cand]).strip())
                        break
    else:
        with open(path) as f:
            out = [l.strip() for l in f if l.strip()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="",
                    help="'hf' (AdvBench+HarmBench from the HF cache) or a "
                         "local harmful-prompt file (txt/csv/jsonl)")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--outdir", default="../data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)

    seen, harmless = set(), []
    while len(harmless) < args.n:
        s = rng.choice(FRAMES)
        v = rng.choice(VERBS)
        line = s.format(v=v, v_l=v[0].lower() + v[1:], t=rng.choice(TOPICS))
        if line not in seen:
            seen.add(line)
            harmless.append(line)
    hl = os.path.join(args.outdir, "harmless.txt")
    with open(hl, "w") as f:
        f.write("\n".join(harmless) + "\n")
    print(f"wrote {len(harmless)} harmless prompts -> {hl}")

    hf = os.path.join(args.outdir, "harmful.txt")
    if args.src:
        items = load_any(args.src)[:args.n]
        with open(hf, "w") as f:
            f.write("\n".join(items) + "\n")
        print(f"wrote {len(items)} harmful prompts -> {hf}  (from {args.src})")
        if len(items) < args.n:
            print(f"  NOTE: only {len(items)} available; stage 2 can sweep to "
                  f"k <= {len(items) - 1}")
    elif not os.path.exists(hf):
        print(f"\n  !! {hf} not created.")
        print("     Re-run with --from hf  (AdvBench + HarmBench from the HF cache,")
        print("     the same sources the submission used) or --from <file>.")


if __name__ == "__main__":
    main()
