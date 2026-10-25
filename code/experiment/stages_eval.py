"""The evaluation stages: kld (AFI + H5 legs), behavior (H4), capability.

Design note -- why reference logits go to disk.

The KLD and H5 legs need two models' logits at the SAME token positions. Held
resident that is 32-37 GB of weights, which does not fit a 32 GB RTX 5090. But
logits are only ~17-34 GB for a 65,536-token budget, and a local box has TB of
disk. So we cache the reference distribution to a memmap once, then stream every
comparison against it with a single model resident. Exact, not a top-k
approximation, and it removes the VRAM ceiling from the whole design.
"""
import csv, gc, itertools, pathlib, sys, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# module level: generate_batch() is module-scope and cannot see an import
# that lives inside stage_behavior()
import metrics as M

DATA = pathlib.Path(__file__).parent.parent / "analysis" / "data"
CACHE = pathlib.Path(__file__).parent.parent / "logit_cache"
TOKEN_BUDGET = 65_536
SEQ = 1024


# ----------------------------------------------------------------- corpus
def wikitext_ids(tok, n_tokens=TOKEN_BUDGET, seq=SEQ):
    from datasets import load_dataset
    # A bare "wikitext" is no longer a valid repo id -- huggingface_hub requires
    # namespace/name and raises HfUriError before the dataset is ever fetched.
    ds = None
    for repo in ("Salesforce/wikitext", "mindchain/wikitext2", "wikitext"):
        try:
            ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
            print(f"  wikitext from {repo}")
            break
        except Exception as e:
            print(f"  wikitext: {repo} unavailable ({type(e).__name__})")
    if ds is None:
        raise RuntimeError("no wikitext source available")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: n_tokens]
    n = (ids.numel() // seq) * seq
    return ids[:n].view(-1, seq)


# ------------------------------------------------- reference logit cache
def cache_logits(model, ids, path, device="cuda", batch=2):
    """Write [n_tok, vocab] float16 logits to a memmap. ~17-34 GB per model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    V = model.config.vocab_size
    n = ids.shape[0] * ids.shape[1]
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, V))
    w = 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], batch):
            chunk = ids[i:i + batch].to(device)
            lg = model(chunk).logits.reshape(-1, V).to(torch.float16).cpu().numpy()
            mm[w:w + lg.shape[0]] = lg; w += lg.shape[0]
    mm.flush(); del mm
    return path


def kl_against_cache(model, ids, cache_path, device="cuda", batch=2):
    """Mean KL(reference || model), streaming the cached reference logits."""
    V = model.config.vocab_size
    mm = np.load(cache_path, mmap_mode="r")
    tot, cnt, w = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], batch):
            chunk = ids[i:i + batch].to(device)
            lp_t = torch.log_softmax(model(chunk).logits.reshape(-1, V).float(), -1)
            k = lp_t.shape[0]
            ref = torch.from_numpy(np.asarray(mm[w:w + k])).to(device).float()
            w += k
            lp_r = torch.log_softmax(ref, -1)
            tot += float((lp_r.exp() * (lp_r - lp_t)).sum(-1).sum()); cnt += k
            del ref, lp_r, lp_t
    return tot / max(cnt, 1)


# --------------------------------------------------- in-place quantization
def quantize_model_(model, fmt, width, targets=("o_proj", "down_proj",
                                                "q_proj", "k_proj", "v_proj",
                                                "gate_proj", "up_proj")):
    """Simulate-quantize weights IN PLACE. Returns the CPU originals so the
    model can be restored without a reload (a reload costs minutes per config)."""
    from quantizers import get_quantizer
    q = get_quantizer(fmt, width)
    saved = {}
    for name, p in model.named_parameters():
        if any(t in name for t in targets) and p.ndim == 2:
            saved[name] = p.data.detach().to("cpu", copy=True)
            p.data.copy_(q(p.data))
    return saved


def restore_(model, saved):
    for name, p in model.named_parameters():
        if name in saved:
            p.data.copy_(saved[name].to(p.device))


# ================================================================= stage: kld
def stage_kld(args):
    """AFI and the three H5 legs.

    H5 needs, per (model family, config):
      leg_ablation  = KL(base_bf16      || abliterated_bf16)
      leg_quant     = KL(abliterated_bf16 || abliterated_quantized)
      joint         = KL(base_bf16      || abliterated_quantized)
    Super-additivity is joint > leg_ablation + leg_quant.
    """
    from transformers import AutoTokenizer
    from run_experiment import load

    f5 = open(DATA / "h5_additivity.csv", "w", newline="")
    w5 = csv.writer(f5); w5.writerow(
        ["model", "short", "fmt", "width", "config", "kld_ablation_leg",
         "kld_quant_leg", "kld_additive_pred", "kld_joint_observed", "excess_pct"])
    fk = open(DATA / "kld_raw.csv", "w", newline="")
    wk = csv.writer(fk); wk.writerow(
        ["model", "short", "variant", "fmt", "width", "config", "kld"])

    for base_id, abl_id in zip(args.base, args.abliterated):
        short = base_id.split("/")[-1][:6]
        tok = AutoTokenizer.from_pretrained(base_id)
        ids = wikitext_ids(tok)

        # 1. cache the BASE bf16 reference distribution
        base_cache = CACHE / f"{short}_base.npy"
        m = load(base_id, device=args.device)[0]
        if not base_cache.exists():
            cache_logits(m, ids, base_cache, args.device)
        del m; gc.collect(); torch.cuda.empty_cache()

        # 2. abliterated bf16: leg_ablation, and cache ITS reference too
        abl_cache = CACHE / f"{short}_abl.npy"
        m = load(abl_id, device=args.device)[0]
        leg_ablation = kl_against_cache(m, ids, base_cache, args.device)
        if not abl_cache.exists():
            cache_logits(m, ids, abl_cache, args.device)

        # 3. every config, quantizing in place and restoring
        for fmt, width in itertools.product(args.formats, args.widths):
            saved = quantize_model_(m, fmt, width)
            leg_quant = kl_against_cache(m, ids, abl_cache, args.device)
            joint = kl_against_cache(m, ids, base_cache, args.device)
            restore_(m, saved); del saved; gc.collect(); torch.cuda.empty_cache()
            add = leg_ablation + leg_quant
            w5.writerow([base_id, short, fmt, width, f"{fmt}{width}",
                         round(leg_ablation, 5), round(leg_quant, 6),
                         round(add, 5), round(joint, 5),
                         round(100 * (joint - add) / max(add, 1e-9), 1)])
            wk.writerow([abl_id, short, "abliterated", fmt, width,
                         f"{fmt}{width}", round(leg_quant, 6)])
            f5.flush(); fk.flush()
        del m; gc.collect(); torch.cuda.empty_cache()

        # 4. base variant's own quantization leg -> the AFI denominator
        m = load(base_id, device=args.device)[0]
        for fmt, width in itertools.product(args.formats, args.widths):
            saved = quantize_model_(m, fmt, width)
            wk.writerow([base_id, short, "base", fmt, width, f"{fmt}{width}",
                         round(kl_against_cache(m, ids, base_cache, args.device), 6)])
            restore_(m, saved); del saved; torch.cuda.empty_cache(); fk.flush()
        del m; gc.collect(); torch.cuda.empty_cache()
    f5.close(); fk.close()
    print("wrote h5_additivity.csv, kld_raw.csv")


# ============================================================ stage: behavior
HARMLESS = [
    "Explain how photosynthesis works.",
    "Write a haiku about the ocean.",
    "What causes the seasons to change?",
    "Summarize the plot of Hamlet in three sentences.",
    "How do I make sourdough starter?",
]


def load_prompts(n=None):
    """Harmful battery. AdvBench alone understates risk (single-benchmark
    refusal evaluation is known to miss most of it), so we concatenate a
    broader set when available."""
    prompts = []
    SOURCES_USED = []
    from datasets import load_dataset
    # walledai/* are GATED and yield nothing without a token. mlabonne/* is
    # ungated and is the same battery Heretic evaluates against, so refusal
    # rates here are comparable with the abliteration tool's own numbers.
    for spec in [("mlabonne/harmful_behaviors", None, "text"),
                 ("walledai/AdvBench", None, "prompt"),
                 ("walledai/HarmBench", "standard", "prompt")]:
        try:
            ds = load_dataset(spec[0], spec[1], split="train") if spec[1] \
                else load_dataset(spec[0], split="train")
            prompts += [r[spec[2]] for r in ds]
            SOURCES_USED.append(spec[0])
            print(f"  loaded {len(ds)} from {spec[0]}")
        except Exception as e:
            print(f"  skip {spec[0]}: {repr(e)[:90]}")
    if not prompts:
        raise RuntimeError("no harmful prompt set available -- pass --prompts-file")
    # Sources are concatenated, so an unshuffled prefix would be drawn almost
    # entirely from whichever set loaded first. Shuffle with a fixed seed so the
    # battery is balanced across sources AND identical across every model and
    # config -- the decision-flip rate is only meaningful on the same prompts.
    import random
    random.Random(0).shuffle(prompts)
    seen, uniq = set(), []
    for t in prompts:
        if t not in seen:
            seen.add(t); uniq.append(t)
    print(f"  battery: {len(uniq)} unique prompts from {len(SOURCES_USED)} sources")
    return uniq[:n] if n else uniq


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new=256, bs=16, device="cuda"):
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        msgs = [M.chat_prompt(tok, p)
                for p in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc.input_ids.shape[1]:]
            outs.append((tok.decode(new, skip_special_tokens=True),
                         int((new != (tok.pad_token_id or tok.eos_token_id)).sum())))
    return outs


def stage_behavior(args):
    """H4 inputs: refusal rate, completion rate, completion-adjusted refusal,
    and decision-flip rate against the bf16 reference."""
    from transformers import AutoTokenizer
    from run_experiment import load
    import metrics as M

    prompts = load_prompts(args.n_prompts)
    print(f"harmful battery: {len(prompts)} prompts")
    fb = open(DATA / "behavior_raw.csv", "w", newline="")
    wb = csv.writer(fb); wb.writerow(
        ["model", "short", "variant", "fmt", "width", "config", "n",
         "refusal_rate", "completion_rate", "adj_refusal_rate", "dfr"])

    for model_id, variant in args.pairs:
        # H4 merges behavior_raw against h3_directional ON THE `model` COLUMN, so
        # this must carry the same shared family label the other stages write --
        # the raw checkpoint id differs per variant and the merge would yield all
        # NaN, reporting "no rho merge" instead of a verdict.
        label = args.label[model_id]
        short = args.short.get(model_id, label.split("/")[-1][:6])
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        m = load(model_id, device=args.device)[0]

        ref = [M.classify(t, args.max_new, n)
               for t, n in generate_batch(m, tok, prompts, args.max_new,
                                          args.batch, args.device)]
        ref_dec = [r for r, _ in ref]
        # Record the UNQUANTIZED baseline explicitly. "Does quantization
        # re-censor?" is a comparison against this number, so it has to be in the
        # data rather than implied by the decision-flip rate.
        _rc = sum(ref_dec) / len(ref) * 100
        _cc = sum(c for _, c in ref) / len(ref) * 100
        _dn = [r for r, c in ref if c]
        wb.writerow([label, short, variant, "BF16", 16, "BF16", len(ref),
                     round(_rc, 2), round(_cc, 2),
                     round((sum(_dn) / len(_dn) * 100) if _dn else 0.0, 2), 0.0])
        fb.flush()
        print(f"  {short} {variant} BF16(ref): refusal {_rc:.1f}% completion {_cc:.1f}%")
        for fmt, width in itertools.product(args.formats, args.widths):
            saved = quantize_model_(m, fmt, width)
            res = generate_batch(m, tok, prompts, args.max_new, args.batch, args.device)
            restore_(m, saved); del saved; torch.cuda.empty_cache()

            dec = [M.classify(t, args.max_new, n) for t, n in res]
            ref_r = sum(r for r, _ in dec) / len(dec) * 100
            comp = sum(c for _, c in dec) / len(dec) * 100
            done = [r for (r, c) in dec if c]
            adj = (sum(done) / len(done) * 100) if done else 0.0
            dfr = sum(a != b for a, (b, _) in zip(ref_dec, dec)) / len(dec) * 100
            wb.writerow([label, short, variant, fmt, width, f"{fmt}{width}",
                         len(dec), round(ref_r, 2), round(comp, 2),
                         round(adj, 2), round(dfr, 2)])
            fb.flush()
            print(f"  {short} {variant} {fmt}{width}: refusal {ref_r:.1f}% "
                  f"completion {comp:.1f}% adj {adj:.1f}% dfr {dfr:.1f}%")
        del m; gc.collect(); torch.cuda.empty_cache()
    fb.close(); print("wrote behavior_raw.csv")


# ========================================================== stage: capability
def stage_capability(args):
    """MMLU + WikiText-2 PPL entirely in memory, no serialization.

    The old implementation wrote each quantized config to a temp dir so lm-eval
    could instantiate its own model. That cost a few minutes per cell in pure
    serialization and bought nothing. mmlu_direct scores the in-memory model
    with prefix KV caching instead: one forward pass per question rather than
    four, and the 5-shot prefix prefilled once per subject.

    GSM8K and HellaSwag stay optional (--tasks) and still route through lm-eval,
    because they need generation and full-continuation scoring respectively.
    """
    import mmlu_direct
    from transformers import AutoTokenizer
    from run_experiment import load

    fc = open(DATA / "capability_raw.csv", "w", newline="")
    wc = csv.writer(fc); wc.writerow(
        ["model", "short", "variant", "fmt", "width", "config",
         "task", "metric", "value"])

    for model_id, variant in args.pairs:
        # Same shared family label as every other stage: H4 merges capability
        # against the rho table ON the `model` column.
        label = args.label[model_id]
        short = args.short.get(model_id, label.split("/")[-1][:6])
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = load(model_id, device=args.device)[0]
        ids = wikitext_ids(tok)

        # Unquantized reference. Without it there is no denominator for "how much
        # capability did this width cost", which is the whole point of measuring
        # capability alongside refusal -- a refusal drop only means something if
        # capability held up.
        _a0 = mmlu_direct.run(m, tok, subjects=args.subjects, batch_size=args.batch,
                              limit=args.mmlu_limit, verify=False, device=args.device)
        _p0 = wikitext_ppl(m, ids, args.device)
        wc.writerow([label, short, variant, "BF16", 16, "BF16", "mmlu", "acc", round(_a0, 5)])
        wc.writerow([label, short, variant, "BF16", 16, "BF16", "wikitext", "ppl", round(_p0, 4)])
        fc.flush()
        print(f"  {short} {variant} BF16(ref): mmlu {100*_a0:.2f}%  ppl {_p0:.3f}")

        for fmt, width in itertools.product(args.formats, args.widths):
            saved = quantize_model_(m, fmt, width)

            acc = mmlu_direct.run(m, tok, subjects=args.subjects,
                                  batch_size=args.batch, limit=args.mmlu_limit,
                                  verify=False, device=args.device)
            wc.writerow([label, short, variant, fmt, width, f"{fmt}{width}",
                         "mmlu", "acc", round(acc, 5)])

            ppl = wikitext_ppl(m, ids, args.device)
            wc.writerow([label, short, variant, fmt, width, f"{fmt}{width}",
                         "wikitext", "ppl", round(ppl, 4)])

            restore_(m, saved); del saved
            gc.collect(); torch.cuda.empty_cache()
            fc.flush()
            print(f"  {short} {variant} {fmt}{width}: mmlu {100*acc:.2f}%  ppl {ppl:.3f}")
        del m; gc.collect(); torch.cuda.empty_cache()
    fc.close(); print("wrote capability_raw.csv")


@torch.no_grad()
def wikitext_ppl(model, ids, device="cuda", batch=2):
    """Token-level perplexity over the same corpus the KLD stage uses."""
    nll, n = 0.0, 0
    for i in range(0, ids.shape[0], batch):
        chunk = ids[i:i + batch].to(device)
        out = model(chunk, labels=chunk)
        ntok = chunk.numel() - chunk.shape[0]
        nll += float(out.loss) * ntok; n += ntok
    return float(torch.exp(torch.tensor(nll / max(n, 1))))
