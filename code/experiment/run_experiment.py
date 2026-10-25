#!/usr/bin/env python3
"""Real experiment runner. Writes CSVs to analysis/data/ with EXACTLY the
column names gen_synthetic.py produces, so the figure/table/paper pipeline
needs no changes when you swap synthetic for measured.

Colab quickstart:
    !pip install -q transformers accelerate datasets lm-eval bitsandbytes
    !python experiment/run_experiment.py --models meta-llama/Llama-3.1-8B-Instruct \
        --abliterated <your-abliterated-repo> --widths 16 8 4 --formats FP INT

Run order (each stage writes its own CSV, so they are resumable):
    --stage weights   H1 + H2   (no generation; fast, CPU-viable)
    --stage kld       AFI       (needs GPU)
    --stage rho       H3
    --stage behavior  H4        (generation; slowest)
"""
import argparse, csv, itertools, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from quantizers import get_quantizer
import metrics as M

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"
DATA.mkdir(parents=True, exist_ok=True)


def open_resumable(fname, header, key_cols):
    """Append-mode CSV that skips work already on disk.

    Colab sessions are capped (12 h free / 24 h Pro+) and drop without warning,
    and the full grid runs far longer than one session. Returns (writer, done)
    where `done` is the set of key tuples already recorded -- callers skip
    those cells, so re-running the same command resumes instead of restarting.
    """
    path = DATA / fname
    done, exists = set(), path.exists()
    if exists:
        with open(path, newline="") as f:
            rd = csv.DictReader(f)
            idx = [rd.fieldnames.index(c) for c in key_cols] if rd.fieldnames else []
            for row in rd:
                done.add(tuple(str(row[c]) for c in key_cols))
        print(f"resume: {len(done)} cells already in {fname}")
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if not exists:
        w.writerow(header)
    return fh, w, done

# Matrices that write to the residual stream -- the ones abliteration edits.
TARGET_SUFFIXES = ("o_proj.weight", "down_proj.weight")


def purge_hf_cache(model_id: str) -> None:
    """Delete a checkpoint from the HF cache.

    Disk is the binding constraint on Jetstream2 full-GPU flavors (60 GB root,
    100 GB volume max here). The six primary checkpoints total ~100 GB, and the
    intervention arms double that -- so nothing may stay resident. Every stage
    is per-checkpoint anyway (AFI compares base vs abliterated post hoc, from
    the CSVs, not by holding both models), so peak disk is one checkpoint plus
    download headroom: ~37 GB.
    """
    import shutil, os
    root = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.expanduser(
        "~/.cache/huggingface/hub")
    d = pathlib.Path(root) / ("models--" + model_id.replace("/", "--"))
    if d.exists():
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
        shutil.rmtree(d, ignore_errors=True)
        print(f"purged {model_id} ({sz:.1f} GB)", flush=True)


def load(model_id, dtype=torch.bfloat16, device="cuda", need_tokenizer=False):
    """Load a model, and only a tokenizer if the caller actually needs one.

    The weights and rho stages read weight tensors and never tokenize anything,
    so forcing tokenizer construction there couples them to sentencepiece /
    tiktoken being installed and to the tokenizer files being well formed. A
    tokenizer failure should not be able to abort a stage that does not use it.
    """
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                             device_map=device)
    tok = None
    if need_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
    return m.eval(), tok


def iter_targets(model):
    for name, p in model.named_parameters():
        if name.endswith(TARGET_SUFFIXES):
            layer = int([s for s in name.split(".") if s.isdigit()][0])
            yield name, layer, p.data


def stage_weights(args):
    """H1 (outlier stats) and H2 (reconstruction error). No generation needed."""
    h1, w1, done1 = open_resumable(
        "h1_outliers.csv",
        ["model", "short", "variant", "layer", "rdir_norm", "kurtosis",
         "max_rms", "outlier_channels"], ["model", "variant", "layer"])
    h2, w2, done2 = open_resumable(
        "h2_quant_error.csv",
        ["model", "short", "variant", "fmt", "width", "config", "layer",
         "kurtosis", "rel_frob_err"], ["model", "variant", "config", "layer"])

    for model_id, variant in args.pairs:
        # hypothesis_tests.py pivots base against abliterated on the `model`
        # column, so both variants must carry ONE shared family label -- the raw
        # checkpoint ids differ (a hub id vs a local ./ckpts path) and would put
        # each variant on its own row, collapsing every paired test to NaN.
        label = args.label[model_id]
        if all((label, variant, str(L)) in done1 for L in range(999)):
            continue
        model, _ = load(model_id, device=args.device)
        short = args.short.get(model_id, label.split("/")[-1][:6])
        r_hat = load_refusal_direction(args, model_id)
        for name, layer, W in iter_targets(model):
            k = M.excess_kurtosis(W)
            rn = float(r_hat[layer].norm()) if r_hat is not None else 0.0
            col = W.float().abs().amax(dim=1)
            n_out = int((col > col.mean() + 6 * col.std()).sum())
            if (label, variant, str(layer)) not in done1:
                w1.writerow([label, short, variant, layer, round(rn, 4),
                             round(k, 4), round(M.max_rms(W), 3), n_out])
                h1.flush()
            for fmt, width in itertools.product(args.formats, args.widths):
                if (label, variant, f"{fmt}{width}", str(layer)) in done2:
                    continue
                Wq = get_quantizer(fmt, width)(W)
                w2.writerow([label, short, variant, fmt, width,
                             f"{fmt}{width}", layer, round(k, 4),
                             round(M.rel_frob_err(W, Wq), 8)])
                h2.flush()
        del model; torch.cuda.empty_cache()
        if args.purge_after:
            purge_hf_cache(model_id)
    h1.close(); h2.close()
    print("wrote h1_outliers.csv, h2_quant_error.csv")


def stage_rho(args):
    """H3: is the quantization error aimed at the ablated direction?"""
    f, w, done = open_resumable(
        "h3_directional.csv",
        ["model", "short", "variant", "fmt", "width", "config",
         "rho_refusal", "rho_null_mean", "rho_null_p95", "p_value"],
        ["model", "variant", "config"])
    for model_id, variant in args.pairs:
        r_hat = load_refusal_direction(args, model_id)
        if r_hat is None:
            print(f"!! SKIPPING {model_id}: no refusal direction found.\n"
                  f"   H3 needs per-layer r_hat at "
                  f"{args.rdir_dir}/{model_id.replace('/', '_')}.pt\n"
                  f"   Produce it first:  python experiment/compute_refusal_dirs.py "
                  f"--model {model_id}")
            continue
        model, _ = load(model_id, device=args.device)
        label = args.label[model_id]
        short = args.short.get(model_id, label.split("/")[-1][:6])
        for fmt, width in itertools.product(args.formats, args.widths):
            if (label, variant, f"{fmt}{width}") in done:
                continue
            rs, nm, n95, ps = [], [], [], []
            for name, layer, W in iter_targets(model):
                if not name.endswith("o_proj.weight"):
                    continue
                # A direction derived from the weight difference is exactly zero
                # for layers abliteration never touched. rho_recovery normalises
                # r_hat, so a zero row yields 0/0 = NaN, and one NaN makes the
                # whole averaged row NaN -- silently voiding H3.
                if float(r_hat[layer].norm()) < 1e-8:
                    continue
                Wq = get_quantizer(fmt, width)(W)
                rho, m0, p95, pv = M.rho_recovery(W, Wq, r_hat[layer])
                rs.append(rho); nm.append(m0); n95.append(p95); ps.append(pv)
            n = len(rs)
            if n == 0:
                print(f"!! {model_id} {fmt}{width}: no layers with a direction"); continue
            w.writerow([label, short, variant, fmt, width, f"{fmt}{width}",
                        round(sum(rs)/n, 4), round(sum(nm)/n, 4),
                        round(sum(n95)/n, 4), round(sum(ps)/n, 6)])
            f.flush()
        del model; torch.cuda.empty_cache()
        if args.purge_after:
            purge_hf_cache(model_id)
    f.close(); print("wrote h3_directional.csv")


def load_refusal_direction(args, model_id):
    """Per-layer refusal directions, shape [n_layers, d_model].
    Produce with Heretic or the Arditi et al. procedure and save as a .pt."""
    path = args.rdir.get(model_id)
    if path is None:
        return None
    return torch.load(path, map_location=args.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["weights", "rho", "kld", "behavior", "capability"])
    # nargs="*" so the supervisor can drive ONE checkpoint per invocation
    # (checkpoint-major), download it, run every stage, then purge.
    ap.add_argument("--base", nargs="*", default=[])
    ap.add_argument("--abliterated", nargs="*", default=[])
    ap.add_argument("--purge-after", action="store_true",
                    help="delete each checkpoint from the HF cache once done")
    ap.add_argument("--formats", nargs="+", default=["FP", "INT"])
    ap.add_argument("--widths", nargs="+", type=int, default=[16, 8, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rdir-dir", default="refusal_dirs")
    ap.add_argument("--rdir-file", default=None,
                    help="use THIS refusal-direction file for every checkpoint, "
                         "overriding the per-id lookup. The weights stage wants "
                         "the difference-in-means directions (whose magnitude H1 "
                         "correlates against); the rho stage wants the directions "
                         "derived from the weight difference, which are the axis "
                         "abliteration actually edited.")
    ap.add_argument("--family", default=None,
                    help="shared family name written to the `model` column for "
                         "every checkpoint in this invocation (e.g. "
                         "Qwen2.5-7B-Instruct). Base and abliterated live at "
                         "different ids, but hypothesis_tests.py pairs them on "
                         "this column -- so run ONE pair per invocation and give "
                         "them the same --family.")
    ap.add_argument("--short-label", default=None,
                    help="shared short label for figures/tables (e.g. Q7B)")
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="cap the harmful battery (default: all)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tasks", default="mmlu,wikitext",
                    help="capability tasks; mmlu and wikitext run in-memory, "
                         "anything else routes through lm-eval")
    ap.add_argument("--mmlu-limit", type=int, default=None,
                    help="questions per subject (None = full 14,042). Use ~35 "
                         "for a fast gate; full for the paper's numbers.")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="restrict MMLU to these subjects")
    a = ap.parse_args()
    a.pairs = ([(m, "base") for m in a.base] +
               [(m, "abliterated") for m in a.abliterated])
    if not a.pairs:
        ap.error("give at least one --base or --abliterated model")
    a.label = {m: (a.family or m) for m, _ in a.pairs}
    a.short = {m: (a.short_label or a.label[m].split("/")[-1][:6])
               for m, _ in a.pairs}
    rd = pathlib.Path(a.rdir_dir)
    if a.rdir_file:
        a.rdir = {m: pathlib.Path(a.rdir_file) for m, _ in a.pairs}
    else:
        a.rdir = {m: rd / f"{m.replace('/', '_')}.pt" for m, _ in a.pairs}
    a.rdir = {k: v for k, v in a.rdir.items() if v.exists()}
    import stages_eval as SE
    {"weights": stage_weights, "rho": stage_rho,
     "kld": SE.stage_kld, "behavior": SE.stage_behavior,
     "capability": SE.stage_capability}[a.stage](a)


if __name__ == "__main__":
    main()
