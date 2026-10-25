#!/usr/bin/env python3
"""Direct batched MMLU scorer with prefix KV caching.

Why not lm-eval + a serving engine:

  * lm-eval's HF backend issues FOUR loglikelihood requests per question (one
    per choice), each re-prefilling the whole ~1,250-token prompt. But MMLU's
    continuation is a single letter token, so all four scores can be read from
    ONE forward pass's final-position logits. That is a free 4x.
  * The 5-shot prefix is IDENTICAL for every question within a subject. Cache
    its KV once per subject and each question only pays for its own ~250
    tokens. That is a further ~5x.
  * vLLM loads from a path, which would force back the save_pretrained round
    trip we are trying to remove -- the model here is already in memory with
    simulated quantization applied.

Token budget per cell, 14,042 questions:
    lm-eval HF, 4 passes/question   ~68.2 M   ~142 min
    1 pass/question                 ~17.6 M    ~16 min
    + prefix cache                   ~3.5 M     ~3 min

Correctness before speed: --verify cross-checks a subject against the naive
four-pass path and refuses to proceed if accuracy disagrees.
"""
import argparse, time, torch
import torch.nn.functional as F

LETTERS = ["A", "B", "C", "D"]
HEADER = "The following are multiple choice questions (with answers) about {}.\n\n"


def fmt_q(row, with_answer):
    s = row["question"].strip() + "\n"
    for i, c in enumerate(row["choices"]):
        s += f"{LETTERS[i]}. {c}\n"
    s += "Answer:"
    if with_answer:
        s += f" {LETTERS[row['answer']]}\n\n"
    return s


def letter_ids(tok):
    """Token id for ' A'..' D' as they appear after 'Answer:'."""
    ids = []
    for L in LETTERS:
        enc = tok(f" {L}", add_special_tokens=False).input_ids
        ids.append(enc[-1])
    if len(set(ids)) != 4:
        raise RuntimeError(f"letter tokens collide: {ids} -- this tokenizer needs "
                           "full-continuation scoring, not single-token")
    return torch.tensor(ids)


class DirectMMLU:
    def __init__(self, model, tok, batch_size=16, use_prefix_cache=True, device="cuda"):
        self.m, self.tok, self.bs = model, tok, batch_size
        self.use_cache, self.device = use_prefix_cache, device
        self.lids = letter_ids(tok).to(device)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    # ---------------------------------------------------------------- prefix
    @torch.no_grad()
    def _prefix_kv(self, text):
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        out = self.m(ids, use_cache=True)
        return out.past_key_values, ids.shape[1]

    @staticmethod
    def _expand(pkv, bs):
        """Repeat a batch-1 KV cache to batch bs. Handles both the legacy tuple
        form and transformers' Cache objects."""
        legacy = pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else pkv
        out = tuple(tuple(t.expand(bs, *t.shape[1:]).contiguous() for t in layer)
                    for layer in legacy)
        if hasattr(pkv, "from_legacy_cache"):
            return type(pkv).from_legacy_cache(out)
        return out

    # ------------------------------------------------------------- scoring
    @torch.no_grad()
    def _batch_logits(self, prompts, pkv=None, plen=0):
        """Final-position logits for a batch of continuations after an optional
        shared prefix. Right-padded, so the true last index is taken per row."""
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=(pkv is None)).to(self.device)
        ids, am = enc.input_ids, enc.attention_mask
        B, L = ids.shape
        kw = {}
        if pkv is not None:
            kw["past_key_values"] = self._expand(pkv, B)
            kw["attention_mask"] = torch.cat(
                [torch.ones(B, plen, dtype=am.dtype, device=am.device), am], 1)
            kw["position_ids"] = (plen + torch.arange(L, device=am.device)
                                  ).unsqueeze(0).expand(B, -1)
        else:
            kw["attention_mask"] = am
        logits = self.m(ids, **kw).logits
        last = am.sum(1) - 1
        return logits[torch.arange(B, device=logits.device), last]

    @torch.no_grad()
    def score_subject(self, subject, dev, test):
        prefix = HEADER.format(subject.replace("_", " "))
        prefix += "".join(fmt_q(r, True) for r in dev[:5])
        pkv, plen = (None, 0)
        if self.use_cache:
            # _expand relies on to_legacy_cache()/from_legacy_cache(), which
            # transformers v5 removed -- it then iterates a Cache object as if it
            # were nested tuples and dies on None.expand(). Correctness beats the
            # speedup: fall back to re-prefilling the prefix (still ONE forward
            # pass per question, vs four for the naive path).
            try:
                pkv, plen = self._prefix_kv(prefix)
                self._expand(pkv, 2)
            except Exception as e:
                if not getattr(self, "_warned", False):
                    print(f"  prefix-KV cache unavailable ({type(e).__name__}); "
                          f"using the uncached path")
                    self._warned = True
                self.use_cache, pkv, plen = False, None, 0

        qs = [fmt_q(r, False) for r in test]
        gold = [r["answer"] for r in test]
        order = sorted(range(len(qs)), key=lambda i: len(qs[i]))  # length-bucket
        correct = 0
        # The model returns logits for EVERY position: [B, L, vocab]. With a
        # 152k-vocab model and ~1300-token 5-shot prompts that is >12 GB in
        # float32 at batch 16, which OOMs. Back off instead of losing the run.
        bs = self.bs
        s = 0
        while s < len(order):
            idx = order[s:s + bs]
            batch = [qs[i] if self.use_cache else prefix + qs[i] for i in idx]
            try:
                lg = self._batch_logits(batch, pkv, plen)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                if not getattr(self, "_oom_warned", False):
                    print(f"  OOM on {subject}: batch -> {bs} "
                          f"(logits are [B, L, vocab]; batch size is deterministic "
                          f"per config, so configs stay comparable)")
                    self._oom_warned = True
                continue
            pred = F.log_softmax(lg.float(), -1)[:, self.lids].argmax(-1)
            correct += sum(int(pred[j]) == gold[i] for j, i in enumerate(idx))
            s += len(idx)
        return correct, len(test)

    @torch.no_grad()
    def score_naive(self, subject, dev, test):
        """Reference path: four full forward passes per question, no cache.
        Slow and obviously correct -- used only to verify the fast path."""
        prefix = HEADER.format(subject.replace("_", " "))
        prefix += "".join(fmt_q(r, True) for r in dev[:5])
        correct = 0
        for r in test:
            base = prefix + fmt_q(r, False)
            scores = []
            for L in LETTERS:
                ids = self.tok(base + f" {L}", return_tensors="pt").input_ids.to(self.device)
                lg = torch.log_softmax(self.m(ids).logits.float(), -1)
                scores.append(float(lg[0, -2, ids[0, -1]]))
            correct += int(max(range(4), key=lambda i: scores[i]) == r["answer"])
        return correct, len(test)


def run(model, tok, subjects=None, batch_size=16, limit=None, verify=False,
        device="cuda", use_prefix_cache=False):
    # use_prefix_cache defaults OFF: the expand-a-batch-1-KV-cache trick relies
    # on to_legacy_cache()/from_legacy_cache(), which transformers v5 removed.
    # Some models still expose to_legacy_cache but round-trip to a bare tuple,
    # which then fails inside attention ('tuple' has no get_query_offset) -- so a
    # probe is not enough to make it safe. The uncached path re-prefills the
    # 5-shot prefix but is still ONE forward pass per question, not four.
    from datasets import load_dataset
    scorer = DirectMMLU(model, tok, batch_size,
                        use_prefix_cache=use_prefix_cache, device=device)
    ds_all = load_dataset("cais/mmlu", "all")
    subs = subjects or sorted(set(ds_all["test"]["subject"]))
    tot_c = tot_n = 0
    t0 = time.time()
    for sub in subs:
        dev = [r for r in ds_all["dev"] if r["subject"] == sub]
        test = [r for r in ds_all["test"] if r["subject"] == sub]
        if limit: test = test[:limit]
        if not dev or not test: continue
        c, n = scorer.score_subject(sub, dev, test)
        if verify:
            vc, vn = scorer.score_naive(sub, dev, test[:20])
            fc, fn = scorer.score_subject(sub, dev, test[:20])
            if fc != vc:
                raise SystemExit(f"VERIFY FAILED on {sub}: fast={fc}/{fn} "
                                 f"naive={vc}/{vn} -- do not trust the fast path")
            print(f"  verify {sub}: fast and naive agree ({fc}/{fn})")
        tot_c += c; tot_n += n
    dt = time.time() - t0
    print(f"MMLU: {tot_c}/{tot_n} = {100*tot_c/max(tot_n,1):.2f}%  in {dt/60:.1f} min")
    return tot_c / max(tot_n, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="questions per subject")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the fast path against naive 4-pass scoring")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    m = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device).eval()
    run(m, tok, a.subjects, a.batch_size, a.limit, a.verify, a.device)
