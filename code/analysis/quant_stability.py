#!/usr/bin/env python3
"""Is 4-bit refusal churn SELECTIVE, or just a broken model?

A refusal-rate change means nothing on its own: a model degraded across the
board will also refuse differently. This joins the behaviour table (refusal
rate, decision-flip rate) to the capability table (MMLU, WikiText PPL) on the
same configs, and reports the drop in each RELATIVE to that model's own BF16
reference -- which is what AFI is for.

    python analysis/quant_stability.py
"""
import pathlib
import pandas as pd

D = pathlib.Path(__file__).parent / "data"


def _read(p):
    """behavior_raw.csv may be mid-write by a running stage (opened "w"), so an
    empty or truncated file is expected rather than an error."""
    try:
        d = pd.read_csv(p)
        return d if len(d) else None
    except Exception:
        return None


def load_behaviour():
    frames = []
    for p in sorted(D.glob("behavior_*.csv")):
        d = _read(p)
        if d is not None:
            frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    return d.drop_duplicates(subset=["model", "variant", "config"], keep="last")


def main():
    b = load_behaviour()
    if b is None:
        raise SystemExit("no behaviour CSVs in analysis/data/")
    b = b[b.variant == "base"]

    ref = (b[b.config == "BF16"].set_index("model")
           [["refusal_rate", "completion_rate"]]
           .rename(columns={"refusal_rate": "ref_refusal",
                            "completion_rate": "ref_completion"}))
    b = b.join(ref, on="model")
    b["d_refusal"] = b.refusal_rate - b.ref_refusal

    caps = [x for x in (_read(fp) for fp in sorted(D.glob("capability_*.csv")))
            if x is not None]
    if caps:
        c = pd.concat(caps, ignore_index=True)
        c = c[c.variant == "base"].drop_duplicates(
            subset=["model", "config", "task"], keep="last")
        mm = (c[(c.task == "mmlu")].pivot_table(index=["model", "config"],
                                                values="value").reset_index()
              .rename(columns={"value": "mmlu"}))
        b = b.merge(mm, on=["model", "config"], how="left")
        m0 = (b[b.config == "BF16"].set_index("model").mmlu
              .rename("ref_mmlu"))
        b = b.join(m0, on="model")
        # retention: how much of BF16 accuracy survived, vs how much of the
        # refusal decision set survived
        b["mmlu_kept"] = 100 * b.mmlu / b.ref_mmlu
        b["decisions_kept"] = 100 - b.dfr
    else:
        b["mmlu"] = b["mmlu_kept"] = float("nan")
        b["decisions_kept"] = 100 - b.dfr

    # per-item MMLU flip rate: the correct analogue of the refusal DFR
    flips = []
    for fp in sorted(D.glob("mmlu_flip_*.csv")):
        f = _read(fp)
        if f is not None:
            flips.append(f.rename(columns={"acc": "mmlu_flip_acc"}))
    if flips:
        fl = pd.concat(flips, ignore_index=True)[
            ["model", "config", "mmlu_flip_rate", "acc_kept"]]
        b = b.merge(fl, on=["model", "config"], how="left")
    else:
        b["mmlu_flip_rate"] = float("nan")

    cols = ["model", "config", "refusal_rate", "d_refusal", "dfr",
            "mmlu", "mmlu_kept", "mmlu_flip_rate"]
    out = b[cols].sort_values(["model", "config"])
    pd.set_option("display.width", 150)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # dfr conflates two very different events. Recover them: with d down-flips
    # (refuse -> comply, the dangerous direction) and u up-flips, dfr = d + u and
    # net = u - d, so d = (dfr - net)/2 and u = (dfr + net)/2.
    b["unsafe_flips"] = (b.dfr - b.d_refusal) / 2      # refusal LOST
    b["safe_flips"] = (b.dfr + b.d_refusal) / 2        # refusal GAINED
    print("\n--- WHICH DIRECTION did the flips go? (refuse -> comply is the harmful one) ---")
    z = b[b.config.isin(["FP16", "FP4", "INT4"])][
        ["model", "config", "ref_refusal", "refusal_rate", "dfr",
         "unsafe_flips", "safe_flips"]]
    print(z.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\n--- like-for-like: fraction of PER-ITEM decisions that changed ---")
    print("    (FP16 is the noise floor on both sides -- it is near-lossless)")
    q = b[b.config.isin(["FP16", "FP4", "INT4"])][
        ["model", "config", "dfr", "mmlu_flip_rate", "mmlu_kept"]].copy()
    q["ratio"] = q.dfr / q.mmlu_flip_rate
    print(q.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nratio > 1 : refusal decisions are MORE fragile than knowledge decisions")
    print("ratio ~ 1 : quantization perturbs both alike -- refusal is not special")


if __name__ == "__main__":
    main()
