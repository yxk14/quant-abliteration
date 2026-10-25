#!/usr/bin/env python3
"""Translate non-Latin generations so they can be re-labelled by hand.

Why this exists. Our human annotation has a language blind spot. Items whose
response is fluent Chinese were marked GARBLED by an annotator who does not read
Chinese, which is the same failure the substring classifier makes for the same
reason: the instrument reads surface form, and the treatment under study
(quantization) perturbs the language the model answers in. Fourteen of the 132
hand-labelled items are affected, all of them at INT4.

Translating them and re-labelling converts an interval into a number. The
translation is itself an instrument and is disclosed as such: a separate model,
not the judge under evaluation, asked only to translate and explicitly not to
characterise or judge the content. The original text is kept alongside so a
reader who knows the language can check the translation rather than trust it.

    python experiment/translate_ambiguous.py
"""
import argparse, csv, json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).parent.parent
RESP = ROOT / "analysis" / "responses"
DATA = ROOT / "analysis" / "data"
OUT = DATA / "translations.json"

PROMPT = """Translate the following text into English.

Rules:
- Translate faithfully, including any hedging, refusal, or redirection.
- Do not summarise, judge, soften, or comment on the content.
- If the text is incoherent or not real language, write exactly: [INCOHERENT]
- Output only the translation.

TEXT:
{text}"""


def nonlatin(t):
    cjk = sum(1 for c in t if '一' <= c <= '鿿' or '぀' <= c <= 'ヿ'
              or '가' <= c <= '힣')
    cyr = sum(1 for c in t if 'Ѐ' <= c <= 'ӿ')
    ara = sum(1 for c in t if '؀' <= c <= 'ۿ')
    lat = sum(1 for c in t if c.isalpha() and ord(c) < 128)
    return (cjk + cyr + ara) / max(cjk + cyr + ara + lat, 1) > 0.4


def main():
    ap = argparse.ArgumentParser()
    # Deliberately NOT the judge under evaluation, to avoid the translation
    # inheriting the judge's disposition toward reading things as refusals.
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    labs = list(csv.DictReader(open(DATA / "human_labels.csv")))
    cache, todo = {}, []
    for r in labs:
        k = (r["family"], r["config"])
        if k not in cache:
            cache[k] = json.loads((RESP / f"{k[0]}__{k[1]}.judged.json").read_text())["items"]
        t = cache[k][int(r["idx"])]["text"]
        if nonlatin(t):
            todo.append(dict(item_id=r["item_id"], family=r["family"],
                             config=r["config"], idx=int(r["idx"]),
                             prompt=cache[k][int(r["idx"])].get("prompt", ""),
                             original=t, human_before=r["human"], judge=r["judge"]))
    print(f"{len(todo)} non-Latin items to translate")
    if not todo:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import metrics as M
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device).eval()

    with torch.no_grad():
        for i, rec in enumerate(todo, 1):
            msg = M.chat_prompt(tok, PROMPT.format(text=rec["original"][:1800]))
            enc = tok(msg, return_tensors="pt", truncation=True,
                      max_length=2600).to(a.device)
            g = m.generate(**enc, max_new_tokens=420, do_sample=False,
                           pad_token_id=tok.pad_token_id)
            rec["translation"] = tok.decode(
                g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rec["translator"] = a.model
            print(f"  [{i}/{len(todo)}] {rec['family']}/{rec['config']} done")

    OUT.write_text(json.dumps(todo, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
