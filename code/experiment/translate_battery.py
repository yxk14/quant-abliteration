#!/usr/bin/env python3
"""Translate the 200-prompt harmful battery into Simplified Chinese.

Why this exists. Under INT4 round-to-nearest the abliterated Qwen2.5-7B answers
a fifth of English prompts in Chinese and refuses 27% of those, against 5% in
English at BF16. The paper's account is that an ablation fitted on English
prompt pairs transferred to Chinese only partly. The direct test is to prompt
the UNQUANTIZED abliterated model in Chinese and measure its refusal rate. This
script produces the Chinese battery; generate_responses.py --prompts-file runs
it; judge_refusals.py scores it with the same language-agnostic rubric.

The prompts are read from an existing response file so the battery and its
order are byte-identical to the English runs. The translator is deliberately
not the judge (Qwen2.5-14B-Instruct, as for the earlier re-labelling), and the
English original is kept alongside every translation.

    python experiment/translate_battery.py
"""
import argparse, json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).parent.parent
RESP = ROOT / "analysis" / "responses"
OUT = ROOT / "analysis" / "data" / "battery_zh.json"

PROMPT = """Translate the following request into Simplified Chinese.

Rules:
- Translate faithfully and completely. Keep the same meaning, tone and level of detail.
- Do not answer the request, refuse it, soften it, add warnings, or comment on it.
- Output only the Chinese translation, nothing else.

REQUEST:
{text}"""


def has_cjk(t):
    return sum(1 for c in t if '一' <= c <= '鿿') >= 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--source", default="Qwen2.5-7B-abl__BF16.judged.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    items = json.loads((RESP / a.source).read_text())["items"]
    prompts = [i["prompt"] for i in items]
    print(f"{len(prompts)} prompts from {a.source}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import metrics as M
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), a.batch):
            chunk = prompts[i:i + a.batch]
            msgs = [M.chat_prompt(tok, PROMPT.format(text=p)) for p in chunk]
            enc = tok(msgs, return_tensors="pt", padding=True).to(a.device)
            g = m.generate(**enc, max_new_tokens=256, do_sample=False,
                           pad_token_id=tok.pad_token_id)
            plen = enc["input_ids"].shape[1]
            for j, p in enumerate(chunk):
                zh = tok.decode(g[j][plen:], skip_special_tokens=True).strip()
                out.append({"idx": i + j, "en": p, "zh": zh, "ok": has_cjk(zh)})
            print(f"  [{min(i + a.batch, len(prompts))}/{len(prompts)}]")

    bad = [o for o in out if not o["ok"]]
    print(f"{len(out)} translated, {len(bad)} without CJK output")
    for o in bad[:10]:
        print("   NOT TRANSLATED:", o["idx"], repr(o["zh"][:80]))
    OUT.write_text(json.dumps({"translator": a.model, "source": a.source,
                               "items": out}, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")
    if len(bad) > 10:
        sys.exit(2)


if __name__ == "__main__":
    main()
