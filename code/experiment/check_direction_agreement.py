#!/usr/bin/env python3
"""GATE: does our r_hat match the direction abliteration actually removed?

rho is defined as the energy of quantization error along r_hat. We derive
r_hat by difference-in-means on the BASE model, but the abliterated weights
were edited along whatever direction the abliteration tool chose. If those two
directions disagree, rho measures the wrong axis and H3 is meaningless --
without this check the paper could report a confident number for nothing.

Abliteration sets W' = W - r r^T W, so r^T W' should be ~0. We measure

    residual = ||r_hat^T W_abl|| / ||r_hat^T W_base||

  ~0.0  -> agreement, proceed
  ~1.0  -> the tool removed a DIFFERENT direction; re-derive r_hat from the
           weight difference (this script can do that with --derive)

    python experiment/check_direction_agreement.py \
        --base meta-llama/Llama-3.1-8B-Instruct --abliterated <repo>
"""
import argparse, gc, pathlib, torch

THRESH_OK, THRESH_BAD = 0.25, 0.75


def edited_matrices(model):
    for n, p in model.named_parameters():
        if n.endswith("o_proj.weight") and p.ndim == 2:
            yield n, int([s for s in n.split(".") if s.isdigit()][0]), p.data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--abliterated", required=True)
    ap.add_argument("--rdir-dir", default="refusal_dirs")
    ap.add_argument("--rdir-file", default=None,
                    help="explicit path to the refusal directions .pt. Needed to "
                         "re-check against the file written by --derive, which "
                         "lands at <base>_derived.pt and is not what the default "
                         "lookup reads.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--derive", action="store_true",
                    help="on disagreement, derive r_hat from the weight difference "
                         "(top left singular vector of W_base - W_abl) and save it")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    rp = (pathlib.Path(a.rdir_file) if a.rdir_file
          else pathlib.Path(a.rdir_dir) / f"{a.base.replace('/', '_')}.pt")
    if not rp.exists():
        raise SystemExit(f"no refusal directions at {rp} -- run compute_refusal_dirs.py first")
    R = torch.load(rp, map_location=a.device)

    ld = lambda m: AutoModelForCausalLM.from_pretrained(
        m, torch_dtype=torch.bfloat16, device_map=a.device).eval()

    # Two 8B models resident is ~30 GB and leaves nothing for the SVD workspace,
    # which OOMs on a 32 GB card. Stash the base o_proj matrices on CPU (~1 GB
    # in bf16), free the base model, then stream the abliterated one against
    # them a layer at a time.
    mb = ld(a.base)
    base_W = {L: Wb.to("cpu", copy=True) for _, L, Wb in edited_matrices(mb)}
    del mb
    gc.collect(); torch.cuda.empty_cache()

    ma = ld(a.abliterated)
    rows, derived = [], {}
    with torch.no_grad():
        for n, L, Wa in edited_matrices(ma):
            if L not in base_W:
                continue
            Wb = base_W[L].to(Wa.device, non_blocking=True)
            r = R[L].to(Wa.device).float()
            r = r / r.norm().clamp(min=1e-8)
            num = (r @ Wa.float()).norm()
            den = (r @ Wb.float()).norm().clamp(min=1e-8)
            rows.append((L, float(num / den)))
            if a.derive:
                D = (Wb.float() - Wa.float())
                if D.norm() > 1e-6:
                    u = torch.linalg.svd(D, full_matrices=False).U[:, 0]
                    derived[L] = u.cpu()
                del D
            del Wb
            torch.cuda.empty_cache()

    vals = [v for _, v in rows]
    mean = sum(vals) / len(vals)
    print(f"{'layer':>6s} {'residual':>10s}")
    for L, v in rows[::max(1, len(rows)//12)]:
        print(f"{L:6d} {v:10.4f}")
    print(f"\nmean residual over {len(rows)} layers: {mean:.4f}")

    if mean < THRESH_OK:
        print("VERDICT: AGREEMENT -- r_hat matches the removed direction. Proceed to rho.")
    elif mean > THRESH_BAD:
        print("VERDICT: DISAGREEMENT -- abliteration removed a different direction.")
        print("         rho computed against this r_hat would be meaningless.")
        if derived:
            out = pathlib.Path(a.rdir_dir) / f"{a.base.replace('/', '_')}_derived.pt"
            nL = max(derived) + 1
            M = torch.zeros(nL, next(iter(derived.values())).numel())
            for L, u in derived.items():
                M[L] = u
            torch.save(M, out)
            print(f"         derived r_hat from the weight difference -> {out}")
            print(f"         re-run this check with that file to confirm.")
        else:
            print("         re-run with --derive to recover r_hat from the weight difference.")
    else:
        print("VERDICT: PARTIAL -- directions overlap but are not the same.")
        print("         Report this in the paper; rho is attenuated, not invalid.")


if __name__ == "__main__":
    main()
