#!/usr/bin/env python3
"""Convert a CSV downloaded from a packet_to_html.py page (item_id,label,ts)
into the terminal tool's <packet>.labels.csv (item_id,label,seconds,ts), so
the gate's analyze script can run unchanged. Refuses unknown labels or ids,
and reports missing items.

    python camera_ready/lib/html_labels_to_csv.py camera_ready/e1/gateB_packet.json ~/Downloads/gateB_packet_labels.csv
"""
import csv, json, sys, pathlib
pk = pathlib.Path(sys.argv[1]); src = pathlib.Path(sys.argv[2])
P = json.loads(pk.read_text())
valid = set(P["labels"].values()); ids = [it["item_id"] for it in P["items"]]
rows = list(csv.DictReader(open(src, newline="")))
got = {r["item_id"].strip(): r for r in rows}
bad = [i for i in got if i not in ids] + [i for i, r in got.items() if r["label"].strip() not in valid]
assert not bad, f"unknown ids or labels: {bad}"
missing = [i for i in ids if i not in got]
out = pk.with_suffix(".labels.csv")
if out.exists():
    sys.exit(f"{out} already exists; move it aside first")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["item_id", "label", "seconds", "ts"]); w.writeheader()
    for i in ids:
        if i in got:
            w.writerow(dict(item_id=i, label=got[i]["label"].strip(), seconds="", ts=got[i].get("ts", "")))
print(f"wrote {out}: {len(ids)-len(missing)} of {len(ids)} items; missing: {missing or 'none'}")
