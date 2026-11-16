#!/usr/bin/env python3
"""E5 generation driver (GPU). Stages per subject:
  0  BF16 and full INT4 RTN (reproduction gate: 74 +- 5 switches on abl, <= 1 at BF16)
  2  partial quantization by module set, then (abl only) bisection of the top block
  3  (abl only) matched Gaussian x3, shuffled RTN error x3, scaled error (dose-response)
First-token statistics (E5.1) are recorded for the stage-0 runs.
    python e5/run_generation.py --subject abl --stages 0 2 3
Every configuration is skipped if its output exists with 500 rows."""
import argparse, json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e5_lib import *

ap = argparse.ArgumentParser()
ap.add_argument("--subject", default="abl", choices=["abl", "base"])
ap.add_argument("--stages", nargs="+", default=["0", "2", "3"])
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()
S = a.subject
prompts = battery()
m, tok = load_subject(S)
cjk_token_ids(tok)
log = lambda s: (print(s, flush=True), log_progress(f"E5 [{S}] {s}"))

def run(tag, spec=None, fn=None, **kw):
    pred = selector(spec) if spec is not None else None
    d = run_config(S, tag, m, tok, prompts, pred=pred, fn=fn or perturb_fn("rtn"), batch=a.batch, **kw)
    print(f"  {S:4s} {tag:28s} switch {d['n_switch']:3d}/500  matrices {d['quant']['n_matrices']:3d}  "
          f"{d['elapsed_s']/60:.1f} min", flush=True)
    return d

if "0" in a.stages:
    t0 = time.time()
    b = run("bf16", None, firsttok=True, keep_logprobs=True)
    if b["elapsed_s"] > 1:
        proj = b["elapsed_s"] * 40 / 3600
        log(f"timed one 500-prompt run: {b['elapsed_s']/60:.1f} min; projected ~40 runs = {proj:.1f} h generation")
    q = run("int4_full", SPECS["int4_full"], firsttok=True, keep_logprobs=True)
    if S == "abl":
        ok = (69 <= q["n_switch"] <= 79) and b["n_switch"] <= 1
        log(f"E5.0 reproduction: INT4 RTN switches {q['n_switch']}/500 (paper 74), BF16 {b['n_switch']} -> {'OK' if ok else 'FAILED'}")
        if not ok:
            log("E5.0 failed reproduction; stopping E5 as the directive requires.")
            sys.exit(2)
    else:
        log(f"E5.5 base: INT4 RTN switches {q['n_switch']}/500 (paper corpus: 18), BF16 {b['n_switch']}")

if "2" in a.stages:
    for tag in ["embed_only", "lm_head_only", "layers_00-06", "layers_07-13", "layers_14-20", "layers_21-27",
                "attn_only", "mlp_only", "int4_full_plus_embed_head"]:
        run(tag, SPECS[tag])
    if S == "abl":
        # bisect the single highest-rate block twice, down to ~2-layer granularity
        blocks = {t: (lo, hi) for t, (lo, hi) in [("layers_00-06", (0, 6)), ("layers_07-13", (7, 13)),
                                                   ("layers_14-20", (14, 20)), ("layers_21-27", (21, 27))]}
        rates = {t: json.loads((GEN / f"{S}__{t}.json").read_text())["n_switch"] for t in blocks}
        best = max(rates, key=rates.get); lo, hi = blocks[best]
        log(f"E5.2 block pass: {rates}; bisecting {best}")
        for _ in range(2):
            mid = (lo + hi) // 2
            halves = [(lo, mid), (mid + 1, hi)]
            r = {}
            for (l, h) in halves:
                d = run(f"layers_{l:02d}-{h:02d}", layer_spec(l, h)); r[(l, h)] = d["n_switch"]
            (lo, hi) = max(r, key=r.get)
            log(f"E5.2 bisection: {dict((f'{l}-{h}', v) for (l, h), v in r.items())} -> {lo}-{hi}")

if "3" in a.stages and S == "abl":
    for s in range(3):
        run(f"gauss_s{s}", SPECS["int4_full"], perturb_fn("gauss", seed=s))
    for s in range(3):
        run(f"shuffle_s{s}", SPECS["int4_full"], perturb_fn("shuffle", seed=s))
    for al in [0.25, 0.5, 0.75, 1.25, 1.5]:
        run(f"scale_{al:.2f}", SPECS["int4_full"], perturb_fn("scale", alpha=al))
    # alpha = 1.0 is int4_full itself
print("generation stage(s) done", a.stages, flush=True)
