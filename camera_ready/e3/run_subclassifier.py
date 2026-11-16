#!/usr/bin/env python3
"""E3 part 2 (GPU): gemma-4-12b-it labels every Latin-script keyword miss as
redirection / premise_correction / register_shift under constrained decoding
(argmax over the three labels' sequence log-probabilities, so the output is
always one of the three). Resumable: results are flushed every chunk and items
already scored are skipped."""
import json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from common import *
from judge_lib import *

OUT = CR / "e3" / "subclassifier_preds.json"
need = json.loads((CR / "e3" / "subclassifier_input.json").read_text())
done = json.loads(OUT.read_text()) if OUT.exists() else {}
todo = [x for x in need if x["key"] not in done]
print(f"{len(need)} items, {len(done)} done, {len(todo)} to go", flush=True)
if todo:
    m, tok = load_judge("google/gemma-4-12b-it")
    t0 = time.time(); CH = 96
    for i in range(0, len(todo), CH):
        chunk = todo[i:i + CH]
        prompts = [SUB_RUBRIC.format(prompt=x["prompt"][:400], response=(x["text"][:900] or "(empty)"))
                   for x in chunk]
        sc = constrained_scores(m, tok, prompts, SUB_LABELS, batch=12)
        for x, s in zip(chunk, sc):
            done[x["key"]] = dict(label=SUB_MAP[argmax_label(s)], scores=s)
        write_json(OUT, done)
        el = time.time() - t0
        print(f"  {i + len(chunk)}/{len(todo)}  {el/60:.1f} min elapsed, "
              f"{el/(i+len(chunk))*(len(todo)-i-len(chunk))/60:.1f} min left", flush=True)
print("label mix:", dict(Counter(v["label"] for v in done.values())))
print("done", len(done))
