#!/usr/bin/env python3
"""Blind terminal annotation tool for the camera-ready packets.

A copy of experiment/annotate.py (ground rule 1: the original is not edited)
that reads a packet file instead of selecting from the corpus. Same keys, same
blinding, same append-only CSV with timing, same resumability.

Packet format (JSON):
  {"name": ..., "labels": {"r": "REFUSAL", "c": "COMPLY", "g": "GARBLED", ...},
   "instructions": "...",
   "items": [{"item_id": ..., "prompt": ..., "text": ..., "translation": null|"...",
              "hidden": {...anything, never shown...}}]}

    python camera_ready/lib/annotate_cr.py camera_ready/e3/gateA_packet.json
    python camera_ready/lib/annotate_cr.py camera_ready/e1/gateB_packet.json --status
Labels go to <packet>.labels.csv next to the packet.
"""
import argparse, csv, json, pathlib, textwrap, time

FIELDS = ["item_id", "label", "seconds", "ts"]

def wrap(s, w=88):
    return "\n".join(textwrap.fill(p, w) for p in s.split("\n") if p.strip())

def load_done(out):
    if not out.exists():
        return {}
    with open(out) as fh:
        return {r["item_id"]: r for r in csv.DictReader(fh)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packet")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    pk = pathlib.Path(a.packet)
    P = json.loads(pk.read_text())
    out = pk.with_suffix(".labels.csv")
    done = load_done(out)
    items = P["items"]
    if a.status:
        print(f"{P['name']}: {len(done)}/{len(items)} labelled")
        from collections import Counter
        print("  mix:", dict(Counter(r["label"] for r in done.values())))
        return
    todo = [it for it in items if it["item_id"] not in done]
    keys = P["labels"]
    print(f"\n{P['name']}: {len(items)} items, {len(done)} already labelled, {len(todo)} to go.")
    print(wrap(P.get("instructions", "")))
    print("Keys:  " + "   ".join(f"{k} = {v}" for k, v in keys.items()) + "   s = skip   q = quit")
    print("The model, configuration and every instrument's verdict are hidden.\n")
    input("press Enter to start ")
    for k, rec in enumerate(todo, 1):
        print("\n" + "=" * 90)
        print(f"[{k}/{len(todo)}]")
        print("-" * 90)
        print("REQUEST:"); print(wrap(rec["prompt"]))
        print("-" * 90)
        print("RESPONSE:"); print(wrap(rec["text"]))
        if rec.get("translation"):
            print("-" * 90)
            print("MACHINE TRANSLATION (Qwen2.5-14B-Instruct; the original is above):")
            print(wrap(rec["translation"]))
        print("-" * 90)
        t0 = time.time()
        while True:
            try:
                ans = input(f"  label [{'/'.join(keys)}/s/q] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nstopped. progress saved."); return
            if ans == "q":
                print("stopped. progress saved."); return
            if ans == "s":
                break
            if ans in keys:
                new = not out.exists()
                with open(out, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(dict(item_id=rec["item_id"], label=keys[ans],
                                    seconds=round(time.time() - t0, 1),
                                    ts=time.strftime("%Y-%m-%dT%H:%M:%S")))
                break
            print("    unrecognised.")
    print("\ndone.")

if __name__ == "__main__":
    main()
