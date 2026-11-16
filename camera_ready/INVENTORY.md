# INVENTORY (discovery, 2026-10-24)

Repo: `/home/ykim/Documents/quant-abliteration-paper`. CSET paper source: `paper/cset.tex`
(Table 1 = `tables/tab_classifier.tex`, Table 2 = `tab_mislabelled.tex`, Table 3 = `tab_shootout.tex`,
Fig. 1 = `figs/fig_twomodes.pdf`, Fig. 2 = `figs/fig_inversion.pdf`).
Python environment: `~/qa-env` (torch 2.13.0+cu130, transformers 5.16.1, vllm 0.28, scipy, matplotlib).
GPU: RTX 5090 32 GB, shared; every GPU job here runs under `flock /tmp/gpu.lock`.

| item | path | rows | schema / notes |
|---|---|---|---|
| Persisted generations, English battery | `analysis/responses/<family>__<config>.judged.json` (92 cells) | 92 x 500 = 46,000 | `{model, family, config, max_new=256, items:[{prompt, text, n_tokens, hit_cap, judge, keyword_v1}], judge_model, n_prompts, extension}` |
| Chinese battery | `analysis/responses/Qwen2.5-7B{,-abl,-aggr}-zh__BF16.judged.json` | 3 x 200 = 600 | same schema; prompts from `analysis/data/battery_zh.json` |
| Total scored corpus | 95 cells | **46,600** | matches the paper |
| 200-prompt snapshots (pre-extension) | `analysis/responses/*.judged200.json` (85) | 200 each | byte-identical first 200 rows of the merged files; mtimes preserved |
| Keyword verdicts | `items[].keyword_v1` (bool) | every row | `experiment/metrics.py::classify_v1` (12 markers, first 400 chars, case-insensitive) |
| Judge verdicts | `items[].judge` in {REFUSAL, COMPLY, GARBLED, UNCLEAR} | every row | gemma-4-12b-it; 421 UNCLEAR corpus-wide |
| UNCLEAR repair pass | `*.judged_repair.json` (68 cells) | per cell | `items[].judge_repair`; NOT used by any paper table (tables read `.judged.json`) |
| 132 hand labels | `analysis/data/human_labels.csv` | 132 | `item_id,family,config,idx,judge,keyword_v1,human,seconds,ts`; append-only, timestamped; rows 1-75 = pass 1 (2026-09-12 23:10-23:33), rows 76-132 = pass 2 (23:45-00:10) |
| 14 relabels + translations | `analysis/data/human_labels_revised.csv`, `analysis/data/translations.json` | 14 | `human_before -> human_after`, translator Qwen2.5-14B-Instruct; translations keep `original` |
| Selection seed / strata | `experiment/annotate.py` | | pass 1: `select(75, seed=0)` = 45 disagree + 15 comply + 15 agree, shuffled with seed 1; pass 2: `select_cell([Qwen2.5-7B-abl:INT4, Qwen2.5-7B-abl:BF16], seed=0)` = all non-COMPLY (31+11) + 8 COMPLY filler per cell, shuffled seed 7, minus 1 already labelled = 57. **Both passes replicate exactly** from the frame below (`lib/common.py::frame_pool`). |
| Sampling frame at annotation time | 88 cells x 200 = 17,600 rows (every scored cell with mtime < 2026-09-12 23:10; excludes the 4 INTG4 and 3 -zh cells scored 2026-09-14) | 17,600 | strata: disagree 1,927 / agree 10,577 / comply 5,095 (eligible: non-empty, <= 2000 chars) |
| Blind annotation tool | `experiment/annotate.py` | | input: `.judged.json` cells; shows REQUEST + RESPONSE only; keys r/c/g/u/s/q; appends to CSV with seconds; `--relabel` shows TRANSLATION + ORIGINAL |
| Judge rubric | `experiment/judge_refusals.py::RUBRIC` | | 3 labels, "in ANY language"; prompt[:400], response[:900], chat template via `metrics.chat_prompt` (enable_thinking=False), HF transformers bf16, left padding, batch 8, `max_new_tokens=6`, greedy, parse REFUS/COMPL/GARBL else UNCLEAR |
| Judge serving | HF `transformers` (no vLLM/Ollama) | | `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bf16, device_map="cuda")` |
| Candidate-judge scores (Table 3) | `analysis/data/judge_shootout_human.csv` | 8 rows | aggregates only; per-item verdicts were not persisted (re-run in E4 C0). Candidates use response[:1200]. |
| WildGuard / distilroberta | `analysis/data/wildguard_refs.json`, `rejection_refs.json` | 132 preds each | per-item `pred`/`gold` in {REFUSAL, NOT_REFUSAL} |
| Quantization (RTN) | `experiment/quantizers.py::quant_int(W, 4)` via `experiment/stages_eval.py::quantize_model_` | | symmetric, **per output channel** (one scale per row, no groups), qmax=7, clamp [-8, 7]; targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` only: **embed_tokens and lm_head are NOT quantized**. INTG4 = group-wise g=128. NF4 = block-64 codebook. |
| Abliterated Qwen2.5-7B (central comparison) | `ckpts/qwen25-7b-abl` (`model` field of `Qwen2.5-7B-abl__*`) | 28 layers, untied head | 74/500 Chinese responses at INT4; KL-constrained Heretic edit |
| Base Qwen2.5-7B-Instruct | HF cache `Qwen/Qwen2.5-7B-Instruct` | 28 layers | |
| Aggressive abliteration | `ckpts/qwen25-7b-abl-aggressive` (`Qwen2.5-7B-aggr`) | | 148/500 Chinese at INT4 (not the central comparison) |
| GPTQ / AWQ / Q4_K_M checkpoints | `ckpts/qwen25-7b-abl-{GPTQ4,AWQ4,Q4KM}`, `ckpts/qwen25-7b-{GPTQ4,AWQ4,Q4KM}` | | compressed-tensors pack-quantized, group 128, symmetric int4, `lm_head` ignored; Q4KM is a dense GGUF->HF dump |
| Battery | `experiment/stages_eval.py::load_prompts` | 500 | mlabonne/harmful_behaviors + walledai/AdvBench + walledai/HarmBench(standard), shuffled `random.Random(0)`, dedup, prefix; the exact prompts are also in every cell's `items[].prompt` in order |
| Generation settings | `experiment/generate_responses.py` | | greedy, `max_new_tokens=256`, batch 16, left padding, `metrics.chat_prompt` |
| Language detector | `analysis/make_paper_figures.py::_nl`, `make_paper_tables.py::_zh` (CJK share > 0.4) | | gives 74/500 on the central INT4 cell and 25.7% vs 11.0%; the "237 of 242" sentence used the `cjk > lat` variant in `tab_quantizer` (ratio variant gives 247). `lib/common.py::is_nonlatin` = the ratio variant. |
| Table / figure scripts | `analysis/make_paper_tables.py`, `analysis/make_paper_figures.py` | | palette ACC `#2a78d6`, BAD `#A6342A`, CTX `#898781`, GOOD `#2E6E4F`, WARN `#8A6A17` |

## Sanity check S1 (lib/s1_check.py) -- ALL MATCH
judge 87.879 / kappa 0.740; keyword 48.485 / kappa 0.151; trivial baseline 64.394; Qwen2.5-7B BF16 97.0/97.0, INT4 judge 94.2 / keyword 56.4; corpus 46,600; labels 132. Numbers in `lib/s1_numbers.json`.
