#!/usr/bin/env python3
"""Merge the 300-prompt extension cells into the 200-prompt corpus -> n=500.

The battery loader shuffles with a fixed seed and truncates, so prompts 1-200
are identical in both runs and the extension is a strict continuation. This
script therefore APPENDS rather than regenerating, and checks that assumption
rather than trusting it: the merged file must have 500 items, its first 200 must
be byte-identical to the original, and no prompt may appear twice.

Originals are copied to *.judged200.json first, so the merge is reversible.
Dry-run by default; pass --apply to write.

    python experiment/merge_extension.py            # report only
    python experiment/merge_extension.py --apply
    python experiment/merge_extension.py --revert   # restore the 200-item files
"""
import argparse, json, pathlib, shutil, sys

ROOT = pathlib.Path(__file__).parent.parent
RESP = ROOT / "analysis" / "responses"


def pairs():
    out = []
    for e in sorted(RESP.glob("*-ext__*.judged.json")):
        fam, cfg = e.name[: -len(".judged.json")].split("__", 1)
        base = RESP / f"{fam[: -len('-ext')]}__{cfg}.judged.json"
        out.append((base, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if a.revert:
        n = 0
        for b in sorted(RESP.glob("*.judged200.json")):
            tgt = b.with_name(b.name.replace(".judged200.json", ".judged.json"))
            shutil.copy2(b, tgt); n += 1
        print(f"restored {n} files from .judged200.json backups")
        return

    ok, skipped, bad = [], [], []
    for base, ext in pairs():
        if not base.exists():
            bad.append((base.name, "no 200-prompt counterpart")); continue
        d0 = json.loads(base.read_text()); d1 = json.loads(ext.read_text())
        i0, i1 = d0["items"], d1["items"]
        if len(i0) == 500:
            skipped.append(base.name); continue
        if len(i0) != 200 or len(i1) != 300:
            bad.append((base.name, f"{len(i0)}+{len(i1)} items")); continue
        p0 = [i["prompt"] for i in i0]; p1 = [i["prompt"] for i in i1]
        dup = set(p0) & set(p1)
        if dup:
            bad.append((base.name, f"{len(dup)} prompts appear in both")); continue
        merged = dict(d0); merged["items"] = i0 + i1
        merged["n_prompts"] = 500
        merged["extension"] = ext.name
        if a.apply:
            b = base.with_name(base.name.replace(".judged.json", ".judged200.json"))
            if not b.exists():
                shutil.copy2(base, b)
            base.write_text(json.dumps(merged, indent=1))
            chk = json.loads(base.read_text())["items"]
            assert len(chk) == 500, base
            assert [i["prompt"] for i in chk[:200]] == p0, base
        ok.append(base.name)

    # Once merged, the -ext cells are duplicates of rows now inside the base
    # files. The figure globs already skip them, but leaving them in the
    # responses directory is an invitation to a future glob picking them up, so
    # archive them out of the way.
    if a.apply and ok and not bad:
        arch = RESP.parent / "responses_ext_merged"
        arch.mkdir(exist_ok=True)
        moved = 0
        for f in list(RESP.glob("*-ext__*.json")):
            f.rename(arch / f.name); moved += 1
        print(f"archived {moved} -ext files to {arch.name}/")

    print(f"{'merged' if a.apply else 'would merge'}: {len(ok)} cells")
    if skipped:
        print(f"already at 500: {len(skipped)}")
    if bad:
        print(f"PROBLEMS ({len(bad)}):")
        for n_, why in bad:
            print(f"  {n_}: {why}")
    if not a.apply and ok:
        print("\ndry run -- rerun with --apply to write")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
