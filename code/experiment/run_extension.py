#!/usr/bin/env python3
"""Extend the harmful battery from 200 to 500 prompts, for every existing cell.

Why. The deployed-quantizer null rests on 200 prompts per cell, which has 80%
power only against an 8-point effect. At 500 that becomes 4.7 points. The
battery loader shuffles with a fixed seed and truncates, so load_prompts(500)
contains load_prompts(200) as an identical prefix: the existing generations stay
valid and only prompts 201-500 are new.

Each cell is generated into a parallel "<family>-ext" file, verified by row
count, and merged only afterwards by merge_extension.py, so a failure here
cannot corrupt the 200-prompt corpus every current table is built from.

    python experiment/run_extension.py            # generate + judge
    python experiment/run_extension.py --plan     # just print the plan
"""
import argparse, json, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).parent.parent
RESP = ROOT / "analysis" / "responses"
EXT = ROOT / "analysis" / "data" / "battery_ext.json"
PY = sys.executable


def cells():
    """(model, family) -> [configs], read from the existing files so that model
    paths are never hardcoded and the plan matches the corpus exactly."""
    out = {}
    for f in sorted(RESP.glob("*__*.json")):
        if ".judged" in f.name:
            continue
        fam, cfg = f.stem.split("__", 1)
        if fam.endswith("-zh") or fam.endswith("-ext"):
            continue                      # different battery / our own output
        d = json.loads(f.read_text())
        out.setdefault((d["model"], fam), []).append(cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--skip-judge", action="store_true")
    a = ap.parse_args()

    n_ext = len(json.loads(EXT.read_text()))
    plan = cells()
    tot = sum(len(c) for c in plan.values())
    print(f"{len(plan)} model loads, {tot} cells, {n_ext} new prompts each "
          f"= {tot*n_ext:,} generations")
    for (model, fam), cfgs in sorted(plan.items(), key=lambda kv: kv[0][1]):
        print(f"  {fam:28s} {model:42s} {' '.join(sorted(cfgs))}")
    if a.plan:
        return

    t0 = time.time()
    done, failed = [], []
    for i, ((model, fam), cfgs) in enumerate(
            sorted(plan.items(), key=lambda kv: kv[0][1]), 1):
        efam = f"{fam}-ext"
        missing = [c for c in cfgs
                   if not (RESP / f"{efam}__{c}.json").exists()]
        if not missing:
            print(f"[{i}/{len(plan)}] {efam}: already complete, skipping")
            done += [(efam, c) for c in cfgs]
            continue
        print(f"[{i}/{len(plan)}] {efam} <- {model}  configs={' '.join(missing)}"
              f"  ({(time.time()-t0)/60:.0f} min elapsed)", flush=True)
        cmd = [PY, str(ROOT / "experiment" / "generate_responses.py"),
               "--model", model, "--family", efam, "--configs", *missing,
               "--prompts-file", str(EXT), "--n-prompts", str(n_ext),
               "--batch", str(a.batch)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        for c in cfgs:
            f = RESP / f"{efam}__{c}.json"
            # exit status means nothing; count the rows
            n = len(json.loads(f.read_text())["items"]) if f.exists() else 0
            (done if n == n_ext else failed).append((efam, c))
            if n != n_ext:
                print(f"    FAIL {efam}__{c}: {n} rows, expected {n_ext}")
                print("    stderr:", r.stderr.strip()[-400:])
    print(f"\ngeneration: {len(done)} cells ok, {len(failed)} failed, "
          f"{(time.time()-t0)/60:.0f} min")
    if failed:
        print("  failed:", ", ".join(f"{f}__{c}" for f, c in failed))
    if a.skip_judge or not done:
        return

    fams = sorted({f for f, _ in done})
    print(f"\njudging {len(done)} cells ({len(fams)} families)", flush=True)
    r = subprocess.run(
        [PY, str(ROOT / "experiment" / "judge_refusals.py"), "--only", *fams],
        capture_output=True, text=True)
    print(r.stdout[-3000:])
    bad = [(f, c) for f, c in done
           if not (RESP / f"{f}__{c}.judged.json").exists()]
    print(f"judged: {len(done)-len(bad)}/{len(done)} cells, "
          f"total {(time.time()-t0)/60:.0f} min")
    if bad:
        print("  unjudged:", ", ".join(f"{f}__{c}" for f, c in bad))
        print("  stderr:", r.stderr.strip()[-400:])


if __name__ == "__main__":
    main()
