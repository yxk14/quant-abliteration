"""E5 helpers: subject loading, in-memory fake quantization of arbitrary module
subsets, perturbations of the RTN error, generation identical to the paper's
generate_responses.py, and first-token statistics."""
import json, math, re, sys, time, pathlib
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *
import metrics as M
from quantizers import quant_int, quant_int_group, quant_nf4

SUBJECTS = {"abl": str(ROOT / "ckpts" / "qwen25-7b-abl"), "base": "Qwen/Qwen2.5-7B-Instruct"}
PAPER_TARGETS = ("o_proj", "down_proj", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
GEN = CR / "e5" / "gen"; FT = CR / "e5" / "firsttok"
GEN.mkdir(parents=True, exist_ok=True); FT.mkdir(parents=True, exist_ok=True)

def battery():
    p = [i["prompt"] for i in load_cell("Qwen2.5-7B-abl", "INT4")["items"]]
    assert len(p) == 500 and p == [i["prompt"] for i in load_cell("Qwen2.5-7B", "INT4")["items"]]
    return p

def load_subject(subject, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = SUBJECTS[subject]
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map=device).eval()
    assert m.config.num_hidden_layers == 28
    return m, tok

# ------------------------------------------------------------ module selection
def classify_param(name):
    """-> (kind, layer) with kind in embed / lm_head / attn / mlp / other."""
    if "embed_tokens" in name: return "embed", None
    if "lm_head" in name: return "lm_head", None
    mm = re.search(r"layers\.(\d+)\.", name)
    layer = int(mm.group(1)) if mm else None
    if any(t in name for t in ("q_proj", "k_proj", "v_proj", "o_proj")): return "attn", layer
    if any(t in name for t in ("gate_proj", "up_proj", "down_proj")): return "mlp", layer
    return "other", layer

def selector(spec):
    """spec: dict(kinds={...}, layers=set|None, include_embed=bool, include_head=bool)."""
    def pred(name, p):
        if p.ndim != 2: return False
        kind, layer = classify_param(name)
        if kind == "embed": return spec.get("include_embed", False)
        if kind == "lm_head": return spec.get("include_head", False)
        if kind not in spec.get("kinds", {"attn", "mlp"}): return False
        return spec.get("layers") is None or layer in spec["layers"]
    return pred

SPECS = {
    "bf16": None,
    "int4_full": dict(kinds={"attn", "mlp"}, layers=None),                       # the paper's INT4 RTN
    "embed_only": dict(kinds=set(), include_embed=True),
    "lm_head_only": dict(kinds=set(), include_head=True),
    "layers_00-06": dict(kinds={"attn", "mlp"}, layers=set(range(0, 7))),
    "layers_07-13": dict(kinds={"attn", "mlp"}, layers=set(range(7, 14))),
    "layers_14-20": dict(kinds={"attn", "mlp"}, layers=set(range(14, 21))),
    "layers_21-27": dict(kinds={"attn", "mlp"}, layers=set(range(21, 28))),
    "attn_only": dict(kinds={"attn"}, layers=None),
    "mlp_only": dict(kinds={"mlp"}, layers=None),
    "int4_full_plus_embed_head": dict(kinds={"attn", "mlp"}, layers=None, include_embed=True, include_head=True),
}
def layer_spec(lo, hi):
    return dict(kinds={"attn", "mlp"}, layers=set(range(lo, hi + 1)))

# ------------------------------------------------------------- perturbations
def rtn4(W):
    return quant_int(W, 4)

def perturb_fn(kind, seed=0, alpha=1.0):
    """Returns f(W, idx) -> W_new. idx indexes the matrix for seeding."""
    if kind == "rtn":
        return lambda W, i: rtn4(W)
    if kind == "scale":
        def f(W, i):
            E = rtn4(W).float() - W.float()
            return (W.float() + alpha * E).to(W.dtype)
        return f
    if kind == "gauss":
        def f(W, i):
            E = rtn4(W).float() - W.float()
            g = torch.Generator(device=W.device).manual_seed(seed * 1_000_003 + i)
            n = torch.randn(W.shape, generator=g, device=W.device, dtype=torch.float32)
            n *= E.norm() / n.norm().clamp(min=1e-12)      # ||noise||_F == ||W_RTN - W||_F exactly
            return (W.float() + n).to(W.dtype)
        return f
    if kind == "shuffle":
        def f(W, i):
            E = rtn4(W).float() - W.float()
            g = torch.Generator(device=W.device).manual_seed(seed * 1_000_003 + i)
            perm = torch.randperm(E.numel(), generator=g, device=W.device)
            return (W.float() + E.reshape(-1)[perm].view_as(E)).to(W.dtype)
        return f
    raise KeyError(kind)

def apply_(model, pred, fn):
    """Replace every parameter selected by pred with fn(W, idx) in place; returns
    CPU copies of the originals. Nothing is written to disk."""
    saved = {}; k = 0; stats = dict(n_matrices=0, n_params=0, rel_err_sum=0.0)
    for name, p in model.named_parameters():
        if pred(name, p):
            saved[name] = p.data.detach().to("cpu", copy=True)
            new = fn(p.data, k); k += 1
            stats["n_matrices"] += 1; stats["n_params"] += p.numel()
            stats["rel_err_sum"] += float((new.float() - p.data.float()).norm() / p.data.float().norm().clamp(min=1e-12))
            p.data.copy_(new)
    return saved, stats

def restore_(model, saved):
    for name, p in model.named_parameters():
        if name in saved:
            p.data.copy_(saved[name].to(p.device))

# ----------------------------------------------------------------- CJK set
def cjk_token_ids(tok):
    f = CR / "e5" / "cjk_token_ids.json"
    if f.exists():
        return json.loads(f.read_text())["ids"]
    ids = []
    for i in range(len(tok)):
        s = tok.decode([i])
        if any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" or "豈" <= c <= "﫿"
               or "぀" <= c <= "ヿ" for c in s):
            ids.append(i)
    write_json(f, dict(ids=ids, n=len(ids), vocab=len(tok),
                       ranges="CJK Unified (4E00-9FFF), Ext A (3400-4DBF), Compat (F900-FAFF), kana (3040-30FF)"))
    return ids

# ------------------------------------------------------------- generation
@torch.no_grad()
def generate(m, tok, prompts, batch=16, max_new=256, device="cuda"):
    """Identical to experiment/generate_responses.py::generate, plus token ids."""
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        msgs = [M.chat_prompt(tok, p) for p in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True).to(device)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        for j in range(len(chunk)):
            t = g[j][plen:]
            n_real = int((t != tok.pad_token_id).sum())
            ids = [int(x) for x in t[:n_real]]
            out.append({"text": tok.decode(t, skip_special_tokens=True), "n_tokens": n_real,
                        "hit_cap": n_real >= max_new, "ids": ids})
    return out

@torch.no_grad()
def first_token_logprobs(m, tok, prompts, batch=16, device="cuda"):
    """log p(first response token) for every prompt: [N, V] float32 (numpy)."""
    outs = []
    for i in range(0, len(prompts), batch):
        msgs = [M.chat_prompt(tok, p) for p in prompts[i:i + batch]]
        enc = tok(msgs, return_tensors="pt", padding=True).to(device)
        lg = m(**enc, logits_to_keep=1).logits[:, -1, :].float()
        outs.append(torch.log_softmax(lg, -1).cpu().numpy())
    return np.concatenate(outs, 0)

def first_cjk_index(ids, cjk_set):
    for k, t in enumerate(ids):
        if t in cjk_set:
            return k
    return -1

def run_config(subject, tag, m, tok, prompts, pred=None, fn=None, batch=16, firsttok=False, keep_logprobs=False):
    """Generate 500 responses under a configuration; resumable; returns the record."""
    f = GEN / f"{subject}__{tag}.json"
    if f.exists():
        d = json.loads(f.read_text())
        if len(d["items"]) == len(prompts):
            return d
    t0 = time.time()
    saved, stats = ({}, dict(n_matrices=0, n_params=0, rel_err_sum=0.0)) if pred is None else apply_(m, pred, fn)
    try:
        res = generate(m, tok, prompts, batch=batch)
        ft = None
        if firsttok:
            lp = first_token_logprobs(m, tok, prompts, batch=batch)
            cjk = cjk_token_ids(tok); mass = np.exp(lp[:, cjk]).sum(1)
            am = lp.argmax(1); cs = set(cjk)
            ft = dict(cjk_mass=[float(x) for x in mass], argmax_id=[int(x) for x in am],
                      argmax_is_cjk=[bool(int(x) in cs) for x in am])
            if keep_logprobs:
                np.save(FT / f"{subject}__{tag}.npy", lp.astype(np.float16))
    finally:
        if saved:
            restore_(m, saved); del saved; torch.cuda.empty_cache()
    cs = set(cjk_token_ids(tok))
    items = []
    for p, r in zip(prompts, res):
        items.append(dict(prompt=p, text=r["text"], n_tokens=r["n_tokens"], hit_cap=r["hit_cap"],
                          nonlatin=is_nonlatin(r["text"]), first_cjk_token_idx=first_cjk_index(r["ids"], cs),
                          first_token_id=r["ids"][0] if r["ids"] else None))
    d = dict(subject=subject, model=SUBJECTS[subject], tag=tag, max_new=256, batch=batch,
             quant=stats, n_switch=sum(i["nonlatin"] for i in items), elapsed_s=time.time() - t0,
             firsttok=ft, items=items)
    write_json(f, d)
    return d
