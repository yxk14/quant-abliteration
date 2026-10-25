# Quantization Does Not Undo Abliteration: A Measurement Study of Refusal in Compressed Language Models

Artifact for the ARTMAN 2026 (ACSAC workshop) paper by Yoohwan Kim, Juyeon Jo
and Jayden Kim. It contains the evaluation prompts, every judge verdict for the
46,600 generations, the human annotation labels, the checkpoint-differencing
and rank-sweep measurements, and the code that produced every table and
figure in the paper.

Archived copy: see the Zenodo DOI in the paper's Artifact Availability section.

## Layout

| Path | Contents |
|---|---|
| `code/experiment/` | Generation, quantization (RTN/NF4/GPTQ/AWQ/GGUF), judging, translation and checkpoint-differencing scripts used to build the corpus. |
| `code/analysis/` | Scripts that emit every table (`make_paper_tables.py`), figure (`make_paper_figures.py`) and statistical test (`hypothesis_tests.py`) in the paper. |
| `code/rank_sweep_real/` | The weight-space rank sweep of Section 5.2 (Table 8, Figure 7): real refusal direction plus leading activation principal directions, norm-matched rank-k edits, four quantizers, control arm. `run.sh real hf` reproduces it in about 20 GPU-minutes on one 32 GB GPU. |
| `data/prompts/` | The 500-prompt evaluation set drawn from AdvBench and HarmBench (`battery_ext.json`) and its 200-prompt Chinese translation (`battery_zh.json`). |
| `data/verdicts/` | One file per (model, quantizer) cell: per-prompt judge verdict (REFUSAL / COMPLY / GARBLED / UNCLEAR), keyword-classifier verdict, token count, cap flag, and the character counts used for language detection. Response text is replaced by its SHA-256 (see below). `*.judged_repair.json` holds the re-asked UNCLEAR verdicts. |
| `data/human_labels/` | The 132 blind reference annotations with timestamps, before/after labels and the selection seed. |
| `data/aggregates/` | Per-model behaviour and capability (MMLU) tables, refusal-direction recovery (`R`, `A`), edit profile, direction profile, flip audits. |
| `data/rank_sweep/` | Rank-sweep CSV, control-arm CSV, per-layer direction metadata and summary. |
| `paper/` | LaTeX source, generated tables and figures. |

## Raw model responses

The harmful completions themselves are not posted openly. Every verdict file
carries `text_sha256` for each response, and the raw files
(`responses-raw.tar.gz`, about 270 MB) are available under a click-through
access condition from the Zenodo record; the hashes let a reader verify that
the controlled file is the one the verdicts were computed from. Everything
else in this repository is public.

## Reproducing the paper's numbers

Tables and figures are regenerated from the verdict and aggregate files
without any GPU:

```bash
cd code/analysis
python3 make_paper_tables.py      # writes paper/tables/*.tex
python3 make_paper_figures.py     # writes paper/figs/*.pdf
python3 hypothesis_tests.py       # McNemar / Fisher tests quoted in the text
```

The scripts expect the judged files under `analysis/responses/`; point them at
`data/verdicts/` (verdict-only) or at the unpacked controlled archive
(full text). Scripts that only count verdicts run on either; the language
split uses the stored `n_cjk` / `n_latin` counts.

Regenerating the corpus itself needs the base checkpoints
(Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct and the twelve base models of
Table 7), Heretic 1.4.0, `llmcompressor`, `llama.cpp` and a judge model
(gemma-4-12b-it); `code/experiment/run_experiment.py` and `run_extension.py`
are the drivers. The abliterated checkpoints are not redistributed; they are
rebuilt by the Heretic search described in Section 3.1.

## Environment

Python 3.11+, `torch`, `transformers`, `datasets`, `numpy`, `pandas`,
`scipy`, `matplotlib`. The rank sweep has its own README with exact commands.

## Citation

```
@inproceedings{kim2026rank,
  title     = {Quantization Does Not Undo Abliteration: A Measurement Study of
               Refusal in Compressed Language Models},
  author    = {Kim, Yoohwan and Jo, Juyeon and Kim, Jayden},
  booktitle = {Proceedings of the Workshop on Artificial Intelligence for
               Robustness and Trustworthiness of ML Systems (ARTMAN), ACSAC 2026},
  year      = {2026}
}
```

## License

Code: MIT. Data and paper materials: CC BY 4.0. See `LICENSE`.
