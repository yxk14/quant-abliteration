# camera_ready/ -- CSET'26 camera-ready experiments

Everything new lives here; nothing outside this folder was modified. Directive: `../CAMERA_READY_TASK.md`.
Environment: `~/qa-env/bin/python` (torch 2.13 cu130, transformers 5.16). GPU jobs take `flock /tmp/gpu.lock`.

| file | what |
|---|---|
| `INVENTORY.md` | where every input lives, row counts, schemas, and the S1 reproduction |
| `PROGRESS.md` | running log and the `## WAITING ON YOU` gates |
| `REPORT.md` | the summary: numbers, paper-register paragraphs, what is provisional |
| `run_all.sh` | re-runs everything, resuming; `bash camera_ready/run_all.sh cpu` for the CPU stages only |
| `gpu_queue.sh` | the serial GPU queue used on 2026-10-24 (E1 translate, E4, E5) |
| `gpu_queue2.sh`, `gpu_queue3.sh`, `gpu_queue4.sh` | follow-up queues from the same day (E5 base, E4 resume, E5 analysis + second judge) |
| `lib/common.py` | corpus loaders, the 132 labels with revisions, the exact sampling-frame replication, kappa/CI helpers |
| `lib/judge_lib.py` | judge loading, free generation (as in the paper), constrained label scoring, the E3 sub-classifier rubric |
| `lib/annotate_cr.py` | blind terminal labelling tool for the packets (copy of `experiment/annotate.py`) |
| `lib/s1_check.py` | sanity check S1 |

## Stages

| stage | run | outputs |
|---|---|---|
| E0 | `python camera_ready/e0/run_e0.py` (CPU, seconds; re-run after E4 to fill the candidate rows) | `e0/numbers.json`, `e0/table3_weighted.tex` |
| E3 | `e3/run_e3_offline.py` -> `e3/run_subclassifier.py` (GPU, ~30 min) -> `e3/finish_e3.py` | `e3/numbers.json`, `e3/fig_failure_modes.pdf`, `e3/fig_failure_modes_bf16.pdf`, `e3/table_qwen_base.tex`, `e3/table_quantizers.tex`, `e3/gateA_packet.json` |
| E4 | `e4/run_e4.py` (GPU, ~1 h; per-condition results in `e4/results/`) | `e4/numbers.json`, `e4/table_judge_ablation.tex`, `e4/c0_preds.json`, `e4/exemplars.md` |
| E4 second judge (optional) | `e4/second_judge.py` (GPU, ~90 min; runs only if `e4/numbers.json` flags a candidate; per-cell verdicts in `e4/second_judge/`) -> `e4/second_judge_analyze.py` | `e4/second_judge.json`, `e4/table_second_judge.tex` |
| E5 | `e5/run_generation.py --subject abl --stages 0 2 3`, `--subject base --stages 0 2`, `e5/judge_gens.py`, `e5/run_e54.py`, then `e5/analyze.py` | `e5/numbers.json`, `e5/fig_first_token.pdf`, `e5/fig_dose.pdf`, `e5/table_localization.tex`, `e5/table_quantizer_kl.tex`, generations in `e5/gen/` |
| E1 | `e1/make_packet.py` -> `e1/translate.py` (GPU) -> GATE B -> `e1/analyze.py` | `e1/gateB_packet.json`, `e1/numbers.json`, `e1/table_cells.tex` |
| E2 | `e2/make_packet.py` -> GATE C -> `e2/analyze.py` | `e2/gateC_packet.json`, `e2/packet.html`, `e2/numbers.json` |

## Gates (what the owner does)

- **GATE A** (~30 min): `python camera_ready/lib/annotate_cr.py camera_ready/e3/gateA_packet.json`, then `python camera_ready/e3/gateA_analyze.py && python camera_ready/e3/finish_e3.py`. Keys r/p/s = redirection / premise_correction / register_shift.
- **GATE B** (~3-4 h): `python camera_ready/lib/annotate_cr.py camera_ready/e1/gateB_packet.json`, then `python camera_ready/e1/analyze.py`. Keys r/c/g. Labels are appended to `e1/gateB_packet.labels.csv`; Ctrl-C and resume any time; `--status` shows progress.
- **GATE C** (colleague): send `camera_ready/e2/packet.html` (self-contained, opens in any browser, no network). They press "download labels as CSV"; save it as `camera_ready/e2/gateC_labels.csv` and run `python camera_ready/e2/analyze.py`. The terminal tool works too: `annotate_cr.py camera_ready/e2/gateC_packet.json`.

After any gate, `bash camera_ready/run_all.sh cpu` regenerates every number, table and figure.
