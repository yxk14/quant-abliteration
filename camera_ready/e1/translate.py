#!/usr/bin/env python3
"""E1 step 2 (GPU): machine-translate the non-Latin responses in the GATE B
packet with Qwen2.5-14B-Instruct under the paper's translation prompt
(experiment/translate_ambiguous.py PROMPT, the one used for the 14 relabels).
Writes e1/gateB_packet.json with `translation` filled, original kept."""
import json, sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *
import metrics as M
from translate_ambiguous import PROMPT

SRC = CR / "e1" / "gateB_packet_untranslated.json"; OUT = CR / "e1" / "gateB_packet.json"
P = json.loads(SRC.read_text())
todo = [it for it in P["items"] if it["hidden"]["nonlatin"]]
print(f"{len(todo)} non-Latin items to translate", flush=True)
if todo:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "Qwen/Qwen2.5-14B-Instruct"
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
    with torch.no_grad():
        for k, it in enumerate(todo, 1):
            msg = M.chat_prompt(tok, PROMPT.format(text=it["text"][:1800]))
            enc = tok(msg, return_tensors="pt", truncation=True, max_length=2600).to("cuda")
            g = m.generate(**enc, max_new_tokens=420, do_sample=False, pad_token_id=tok.pad_token_id)
            it["translation"] = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            it["hidden"]["translator"] = mid
            print(f"  [{k}/{len(todo)}] {len(it['translation'])} chars", flush=True)
P["name"] = P["name"] + " (translations attached)"
write_json(OUT, P)
n_tr = sum(bool(it.get("translation")) for it in P["items"])
print(f"wrote {OUT} with {n_tr} translations; {len(P['items'])} items")
