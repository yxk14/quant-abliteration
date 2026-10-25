#!/usr/bin/env python3
"""Terminal tool for hand-labelling generations as REFUSAL / COMPLY / GARBLED.

Why this exists. Every instrument in this study is validated against human
labels, and those labels must (a) actually be human, (b) be persisted so the
validation numbers regenerate from the artifact, and (c) be collected BLIND, so
the annotator is not anchored by what the instruments said.

Design choices that matter for validity:

  * Blind by default. The model, configuration, and both instruments' verdicts
    are hidden while you label. They are written to the output afterwards so
    agreement can be computed, but you never see them at decision time.
  * Stratified sample. Disagreement cases (where keyword and judge differ) are
    where an instrument is weakest and are over-sampled deliberately; agreement
    cases are included so the set is not purely adversarial; and constructed
    controls (clear compliance) are included so a degenerate always-REFUSAL
    annotator or judge is detectable.
  * Resumable. Every label is appended immediately. Ctrl-C and restart loses
    nothing.
  * Order randomised under a fixed seed, so the sample is reproducible but the
    presentation order carries no signal.

    python experiment/annotate.py --n 75          # label 75 items
    python experiment/annotate.py --status        # progress + agreement so far
"""
import argparse, csv, hashlib, json, pathlib, random, sys, textwrap, time

ROOT = pathlib.Path(__file__).parent.parent
RESP = ROOT / "analysis" / "responses"
OUT = ROOT / "analysis" / "data" / "human_labels.csv"
FIELDS = ["item_id", "family", "config", "idx", "judge", "keyword_v1",
          "human", "seconds", "ts"]

VALID = {"r": "REFUSAL", "c": "COMPLY", "g": "GARBLED", "u": "UNCLEAR"}


def item_id(fam, cfg, idx, text):
    h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]
    return f"{fam}__{cfg}__{idx}__{h}"


def build_pool(seed=0):
    """Stratified: disagreements first, then agreements, then clear compliance."""
    dis, agr, comply = [], [], []
    for f in sorted(RESP.glob("*.judged.json")):
        if "judged2" in f.name:
            continue
        d = json.loads(f.read_text())
        fam, cfg = d["family"], d["config"]
        for i, it in enumerate(d["items"]):
            t = it.get("text", "")
            if not t.strip() or len(t) > 2000:
                continue
            rec = dict(item_id=item_id(fam, cfg, i, t), family=fam, config=cfg,
                       idx=i, judge=it["judge"], keyword_v1=int(bool(it["keyword_v1"])),
                       prompt=it.get("prompt", ""), text=t)
            kw_says_refusal = bool(it["keyword_v1"])
            judge_says_refusal = it["judge"] == "REFUSAL"
            if kw_says_refusal != judge_says_refusal:
                dis.append(rec)
            elif it["judge"] == "COMPLY":
                comply.append(rec)
            else:
                agr.append(rec)
    rng = random.Random(seed)
    for L in (dis, agr, comply):
        rng.shuffle(L)
    return dis, agr, comply


def select_cell(cells, seed=0, filler_per_cell=8):
    """All non-COMPLY verdicts from the named cells, plus a blind filler sample
    of COMPLY items from the same cells.

    Used to audit one claim directly. The headline of the companion study is
    that the abliterated checkpoint's refusal rate rises from BF16 to INT4, and
    that rise is the judge's count of REFUSAL verdicts. If the judge mistakes
    degenerate INT4 output for refusal, the rise is partly an artifact. Labelling
    every REFUSAL verdict in both cells settles it.

    The filler matters: if the set contained only the judge's REFUSAL verdicts,
    the annotator would know every item was supposed to be a refusal, and the
    result would be worthless. With filler the stratum is not inferable from the
    item.
    """
    import random as _r
    out = []
    for fam, cfg in cells:
        f = RESP / f"{fam}__{cfg}.judged.json"
        d = json.loads(f.read_text())
        tgt, fill = [], []
        for i, it in enumerate(d["items"]):
            t = it.get("text", "")
            if not t.strip() or len(t) > 2000:
                continue
            rec = dict(item_id=item_id(fam, cfg, i, t), family=fam, config=cfg,
                       idx=i, judge=it["judge"],
                       keyword_v1=int(bool(it["keyword_v1"])),
                       prompt=it.get("prompt", ""), text=t)
            (tgt if it["judge"] != "COMPLY" else fill).append(rec)
        rng = _r.Random(seed); rng.shuffle(fill)
        out += tgt + fill[:filler_per_cell]
    _r.Random(seed + 7).shuffle(out)
    return out


def select(n, seed=0):
    dis, agr, comply = build_pool(seed)
    n_dis = min(len(dis), int(n * 0.60))
    n_com = min(len(comply), max(6, int(n * 0.20)))
    n_agr = max(0, n - n_dis - n_com)
    sel = dis[:n_dis] + comply[:n_com] + agr[:n_agr]
    random.Random(seed + 1).shuffle(sel)
    return sel


def load_done():
    if not OUT.exists():
        return {}
    with open(OUT) as fh:
        return {r["item_id"]: r for r in csv.DictReader(fh)}


def append(row):
    new = not OUT.exists()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})


def wrap(s, w=88):
    return "\n".join(textwrap.fill(p, w) for p in s.split("\n") if p.strip())


def status(cells=None):
    done = load_done()
    if not done:
        print("no labels yet"); return
    n = len(done)
    hj = sum(1 for r in done.values()
             if r["human"] == r["judge"])
    hk = sum(1 for r in done.values()
             if (r["human"] == "REFUSAL") == (r["keyword_v1"] == "1"))
    print(f"labelled {n} items")
    print(f"  judge   agrees with human: {hj}/{n} = {100*hj/n:.1f}%")
    print(f"  keyword agrees with human: {hk}/{n} = {100*hk/n:.1f}%")
    from collections import Counter
    print(f"  human label mix: {dict(Counter(r['human'] for r in done.values()))}")


REV = ROOT / "analysis" / "data" / "human_labels_revised.csv"
REV_FIELDS = ["item_id", "family", "config", "idx", "judge", "human_before",
              "human_after", "translator", "ts"]


def relabel():
    """Second pass over items whose response is not in Latin script.

    The first pass marked these GARBLED because the annotator could not read
    them. That is an instrument failure on the annotator's side, of the same
    kind as the substring classifier's: it reads surface form, and the treatment
    moves surface form. Showing a translation removes the blind spot; showing
    the original underneath lets a reader who knows the language audit the
    translation instead of trusting it.
    """
    src = ROOT / "analysis" / "data" / "translations.json"
    if not src.exists():
        print("no translations.json -- run experiment/translate_ambiguous.py first")
        return
    recs = json.loads(src.read_text())
    done = {}
    if REV.exists():
        with open(REV) as fh:
            done = {r["item_id"]: r for r in csv.DictReader(fh)}
    todo = [r for r in recs if r["item_id"] not in done]
    print(f"\n{len(recs)} translated items, {len(done)} already revised, "
          f"{len(todo)} to go.")
    print("Label from the TRANSLATION. Keys: r / c / g / u / s / q")
    print("(g here means genuinely incoherent, not merely foreign-language.)\n")
    if todo:
        input("press Enter to start ")
    for k, rec in enumerate(todo, 1):
        print("\n" + "=" * 90)
        print(f"[{k}/{len(todo)}]")
        print("-" * 90)
        print("REQUEST:"); print(wrap(rec["prompt"]))
        print("-" * 90)
        print("TRANSLATION:"); print(wrap(rec["translation"]))
        print("-" * 90)
        print("ORIGINAL (for audit):"); print(wrap(rec["original"][:700]))
        print("-" * 90)
        while True:
            try:
                ans = input("  label [r/c/g/u/s/q] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nstopped. progress saved."); return
            if ans == "q":
                print("stopped. progress saved."); return
            if ans == "s":
                break
            if ans in VALID:
                row = dict(item_id=rec["item_id"], family=rec["family"],
                           config=rec["config"], idx=rec["idx"],
                           judge=rec["judge"], human_before=rec["human_before"],
                           human_after=VALID[ans],
                           translator=rec.get("translator", ""),
                           ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
                new = not REV.exists()
                with open(REV, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=REV_FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                break
            print("    unrecognised. use r, c, g, u, s or q.")
    print("\ndone.")
    if REV.exists():
        with open(REV) as fh:
            rr = list(csv.DictReader(fh))
        ch = sum(1 for r in rr if r["human_before"] != r["human_after"])
        agree = sum(1 for r in rr if r["human_after"] == r["judge"])
        print(f"  {len(rr)} revised, {ch} labels changed")
        print(f"  judge agrees with the revised label on {agree}/{len(rr)}")


def cell_report():
    """If any fully-audited cell is present, report the human-vs-judge refusal
    rate for it, which is the quantity the companion study's headline uses."""
    done = load_done()
    if not done:
        return
    by = {}
    for r in done.values():
        by.setdefault((r["family"], r["config"]), []).append(r)
    shown = False
    for (fam, cfg), rows in sorted(by.items()):
        tgt = [r for r in rows if r["judge"] != "COMPLY"]
        if len(tgt) < 5:
            continue
        if not shown:
            print("\nfully-audited cells (judge's non-COMPLY verdicts):")
            shown = True
        real = sum(r["human"] == "REFUSAL" for r in tgt)
        print(f"  {fam} {cfg}: judge said non-COMPLY for {len(tgt)} items; "
              f"human calls {real} of them REFUSAL "
              f"({100*real/len(tgt):.0f}%)")
        from collections import Counter
        print(f"      human labels there: {dict(Counter(r['human'] for r in tgt))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--relabel", action="store_true",
                    help="re-label the translated non-Latin items. Shows the "
                         "English translation with the original beneath it, and "
                         "writes to human_labels_revised.csv, keeping the "
                         "original label for provenance.")
    ap.add_argument("--cell", nargs="*", default=None,
                    help="FAMILY:CONFIG pairs to audit exhaustively, e.g. "
                         "Qwen2.5-7B-abl:INT4 Qwen2.5-7B-abl:BF16")
    a = ap.parse_args()

    if a.status:
        status()
        cell_report()
        return

    if a.relabel:
        relabel()
        return

    if a.cell:
        cells = [tuple(c.split(":", 1)) for c in a.cell]
        sel = select_cell(cells, a.seed)
    else:
        sel = select(a.n, a.seed)
    done = load_done()
    todo = [r for r in sel if r["item_id"] not in done]
    print(f"\n{len(sel)} items selected, {len(done)} already labelled, "
          f"{len(todo)} to go.")
    print("Keys:  r = REFUSAL   c = COMPLY   g = GARBLED   u = UNCLEAR   "
          "s = skip   q = quit")
    print("You are labelling the RESPONSE. The model, config and what the "
          "automated\ninstruments said are hidden until you have decided.\n")
    input("press Enter to start ")

    for k, rec in enumerate(todo, 1):
        print("\n" + "=" * 90)
        print(f"[{k}/{len(todo)}]")
        print("-" * 90)
        print("REQUEST:")
        print(wrap(rec["prompt"]))
        print("-" * 90)
        print("RESPONSE:")
        print(wrap(rec["text"]))
        print("-" * 90)
        t0 = time.time()
        while True:
            try:
                ans = input("  label [r/c/g/u/s/q] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nstopped. progress saved."); return
            if ans == "q":
                print("stopped. progress saved."); return
            if ans == "s":
                break
            if ans in VALID:
                rec = dict(rec, human=VALID[ans],
                           seconds=round(time.time() - t0, 1),
                           ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
                append(rec)
                break
            print("    unrecognised. use r, c, g, u, s or q.")

    print("\ndone.")
    status()


if __name__ == "__main__":
    main()
